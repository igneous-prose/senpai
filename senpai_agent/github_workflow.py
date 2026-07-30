"""Idempotent GitHub workflow transitions over the GitHub API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, cast
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit

from pydantic import SecretStr

from senpai_agent.models import (
    AssignmentRecord,
    ExperimentResult,
    ResultMarkerError,
    RevisionRecord,
    parse_assignment_markers,
    parse_result_markers,
    render_assignment_marker,
    render_result_comment,
    render_revision_marker,
)


class GitHubWorkflowError(RuntimeError):
    """Base error for GitHub workflow operations."""


class GitHubAPIError(GitHubWorkflowError):
    """GitHub returned an unexpected HTTP response."""

    def __init__(self, method: str, url: str, status_code: int):
        endpoint = urlsplit(url)
        path = endpoint.path
        if endpoint.query:
            path = f"{path}?{endpoint.query}"
        super().__init__(f"GitHub {method} {path} returned HTTP {status_code}")
        self.status_code = status_code


class GitHubTransportError(GitHubWorkflowError):
    """GitHub could not be reached."""

    def __init__(self, method: str, url: str):
        endpoint = urlsplit(url)
        super().__init__(
            f"GitHub {method} {endpoint.path} failed before an HTTP response"
        )


class WorkflowPreconditionError(GitHubWorkflowError):
    """Current GitHub state does not permit the requested transition."""


class ReconciliationError(GitHubWorkflowError):
    """GitHub did not reach the requested state."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    json_body: object | None = None
    headers: tuple[tuple[str, str], ...] = ()

    def header(self, name: str) -> str | None:
        normalized = name.casefold()
        return next(
            (value for key, value in self.headers if key.casefold() == normalized),
            None,
        )


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class PullRequestSnapshot:
    number: int
    node_id: str
    url: str
    title: str
    body: str
    base_ref: str
    head_ref: str
    head_sha: str
    labels: tuple[str, ...]
    draft: bool
    state: Literal["open", "closed"]
    merged: bool
    mergeable: bool | None
    merge_commit_sha: str | None


@dataclass(frozen=True, slots=True)
class MutationResult:
    changed: bool
    resource_url: str
    state: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class _IssueComment:
    id: int
    body: str
    url: str
    author: str


@dataclass(frozen=True, slots=True)
class _ResultComment:
    comment: _IssueComment
    result: ExperimentResult


class _UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse:
        data = (
            None
            if json_body is None
            else json.dumps(
                json_body,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        github_request = request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(github_request, timeout=30) as response:
                return HttpResponse(
                    status_code=response.status,
                    json_body=_decode_json(response.read()),
                    headers=tuple(response.headers.items()),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                json_body=_decode_json(error.read()),
                headers=tuple(error.headers.items()) if error.headers else (),
            )
        except (URLError, TimeoutError) as error:
            raise GitHubTransportError(method, url) from error


def _decode_json(body: bytes) -> object | None:
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


class GitHubWorkflow:
    """Small desired-state client for Senpai pull-request transitions."""

    __slots__ = (
        "_api_url",
        "_repo",
        "_token",
        "_transport",
        "_trusted_actor",
    )

    def __init__(
        self,
        repo: str,
        token: SecretStr,
        *,
        transport: HttpTransport | None = None,
        api_url: str = "https://api.github.com",
        trusted_actor: str | None = None,
    ):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        if not isinstance(token, SecretStr):
            raise TypeError("token must be a SecretStr")
        if not token.get_secret_value().strip():
            raise ValueError("token must not be empty")
        if trusted_actor is not None and not trusted_actor.strip():
            raise ValueError("trusted actor must not be empty")

        self._repo = repo
        self._token = token
        self._transport = transport or _UrllibTransport()
        self._api_url = api_url.rstrip("/")
        self._trusted_actor = trusted_actor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(repo={self._repo!r}, api_url={self._api_url!r})"

    def __getstate__(self) -> None:
        raise TypeError("GitHubWorkflow cannot be serialized")

    def pull_request(self, number: int) -> PullRequestSnapshot:
        number = _positive_number(number)
        response = self._request(
            "GET",
            f"/repos/{self._repo}/pulls/{number}",
            expected_statuses={200},
        )
        data = _object(response.json_body, "pull request")
        head = _object(data.get("head"), "pull request head")
        base = _object(data.get("base"), "pull request base")
        labels = _labels(data.get("labels"))
        state_value = data.get("state")
        if state_value not in ("open", "closed"):
            raise ReconciliationError("GitHub returned an invalid pull request state")
        mergeable = data.get("mergeable")
        if mergeable is not None and not isinstance(mergeable, bool):
            raise ReconciliationError("GitHub returned invalid mergeability")
        merge_commit_sha = data.get("merge_commit_sha")
        if merge_commit_sha is not None and not isinstance(merge_commit_sha, str):
            raise ReconciliationError("GitHub returned an invalid merge commit SHA")

        snapshot = PullRequestSnapshot(
            number=_integer(data.get("number"), "pull request number"),
            node_id=_string(data.get("node_id"), "pull request node ID"),
            url=_string(data.get("html_url"), "pull request URL"),
            title=_string(data.get("title"), "pull request title"),
            body=_string(
                data.get("body") or "",
                "pull request body",
                allow_empty=True,
            ),
            base_ref=_string(base.get("ref"), "pull request base ref"),
            head_ref=_string(head.get("ref"), "pull request head ref"),
            head_sha=_string(head.get("sha"), "pull request head SHA"),
            labels=labels,
            draft=_boolean(data.get("draft"), "pull request draft state"),
            state=cast(Literal["open", "closed"], state_value),
            merged=_boolean(data.get("merged"), "pull request merged state"),
            mergeable=mergeable,
            merge_commit_sha=merge_commit_sha,
        )
        if snapshot.number != number:
            raise ReconciliationError("GitHub returned the wrong pull request")
        return snapshot

    def create_assignment(
        self,
        assignment: AssignmentRecord,
        *,
        title: str,
        body: str,
    ) -> MutationResult:
        """Create or reconcile one typed draft assignment PR."""

        if assignment.repo != self._repo:
            raise WorkflowPreconditionError(
                "assignment repository does not match the GitHub workflow"
            )
        title = title.strip()
        body = body.strip()
        if not title or not body:
            raise ValueError("assignment title and body must not be empty")
        marker_body = _marker_body(
            render_assignment_marker(assignment),
            body,
        )

        matches = self._assignment_pull_requests(assignment)
        if len(matches) > 1:
            raise ReconciliationError(
                "GitHub contains multiple PRs for the assignment branch"
            )
        active = self._active_student_assignment_numbers(assignment.student)
        current_number = matches[0].number if matches else None
        conflicts = tuple(number for number in active if number != current_number)
        if conflicts:
            raise WorkflowPreconditionError(
                f"student:{assignment.student} already has active assignment "
                f"PR(s): {', '.join(f'#{number}' for number in conflicts)}"
            )

        created = not matches
        if created:
            self._mutate(
                "POST",
                f"/repos/{self._repo}/pulls",
                json_body={
                    "title": title,
                    "body": marker_body,
                    "head": assignment.head_ref,
                    "base": assignment.base_ref,
                    "draft": True,
                },
                expected_statuses={201},
            )
            matches = self._assignment_pull_requests(assignment)
            if len(matches) != 1:
                raise ReconciliationError(
                    "GitHub did not create exactly one assignment PR"
                )

        current = matches[0]
        _require_open(current)
        _require_assignment_snapshot(current, assignment)
        content_changed = current.title != title or current.body != marker_body
        if content_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{current.number}",
                json_body={"title": title, "body": marker_body},
                expected_statuses={200},
            )
            current = self.pull_request(current.number)
            _require_assignment_snapshot(current, assignment)

        draft_changed = self._set_draft(current, draft=True)
        routing_labels = {
            assignment.base_ref,
            f"student:{assignment.student}",
            "status:wip",
        }
        remove = {
            label
            for label in current.labels
            if label.startswith(("student:", "status:")) and label not in routing_labels
        }
        labels_changed, desired_labels = self._set_labels(
            current.number,
            current,
            add=routing_labels,
            remove=remove,
        )
        after = self.pull_request(current.number)
        _require_assignment_snapshot(after, assignment)
        if not after.draft:
            raise ReconciliationError("assignment pull request is not draft")
        if after.title != title or after.body != marker_body:
            raise ReconciliationError(
                "assignment pull request content did not converge"
            )
        _require_exact_labels(after, desired_labels)
        return MutationResult(
            changed=created or content_changed or draft_changed or labels_changed,
            resource_url=after.url,
            state="assignment_created",
            version=after.head_sha,
        )

    def reconcile_labels(
        self,
        number: int,
        *,
        assignment_id: str,
        add: set[str],
        remove: set[str],
        expected_head_sha: str,
    ) -> MutationResult:
        before = self.pull_request(number)
        _require_head(before, expected_head_sha)
        _require_assignment_identity(
            before,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        changed, desired = self._set_labels(
            number,
            before,
            add=add,
            remove=remove,
        )
        if not changed:
            return MutationResult(
                changed=False,
                resource_url=before.url,
                state="labels_reconciled",
                version=before.head_sha,
            )

        after = self.pull_request(number)
        _require_head(after, expected_head_sha)
        _require_assignment_identity(
            after,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        if after.labels != desired:
            raise ReconciliationError("GitHub did not reach the requested label set")
        return MutationResult(
            changed=True,
            resource_url=after.url,
            state="labels_reconciled",
            version=after.head_sha,
        )

    def upsert_marker_comment(
        self,
        number: int,
        *,
        marker: str,
        body: str,
        expected_head_sha: str,
    ) -> MutationResult:
        _validate_marker(marker, body)
        before = self.pull_request(number)
        _require_head(before, expected_head_sha)
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=body,
        )
        after = self.pull_request(number)
        _require_head(after, expected_head_sha)
        return MutationResult(
            changed=changed,
            resource_url=verified.url or after.url,
            state="marker_comment_upserted",
            version=after.head_sha,
        )

    def respond_to_issue(
        self,
        number: int,
        *,
        human_message_id: int,
        response: str,
    ) -> MutationResult:
        """Reply once to one verified human-authored GitHub issue message."""

        number = _positive_number(number)
        human_message_id = _positive_message_id(human_message_id)
        body = response.strip()
        if not body:
            raise ValueError("response must not be empty")

        issue = self._human_issue(number)
        source_author = self._human_message_author(
            number,
            issue=issue,
            human_message_id=human_message_id,
        )
        if source_author == self._actor():
            raise WorkflowPreconditionError(
                "human message must not be authored by the authenticated actor"
            )

        marker = f"<!-- senpai-human-response:{human_message_id} -->"
        comment_body = _marker_body(marker, body)
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=comment_body,
        )
        self._human_issue(number)
        self._require_marker(number, marker, comment_body)
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="issue_response_upserted",
            version=str(human_message_id),
        )

    def request_revision(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        revision_id: str,
        comment: str,
    ) -> MutationResult:
        before = self.pull_request(number)
        _require_open(before)
        _require_head(before, expected_head_sha)
        assignment = _require_assignment_identity(
            before,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        revised_assignment = assignment.model_copy(update={"revision_id": revision_id})
        revised_body = _replace_assignment_marker(
            before.body,
            revised_assignment,
        )
        assignment_changed = revised_body != before.body
        if assignment_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{number}",
                json_body={"body": revised_body},
                expected_statuses={200},
            )
        marker = render_revision_marker(
            RevisionRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=revision_id,
                requested_head_sha=expected_head_sha,
            )
        )
        marker_body = _marker_body(marker, comment)
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=marker_body,
        )
        current = self.pull_request(number)
        _require_open(current)
        _require_head(current, expected_head_sha)
        if parse_assignment_markers(current.body) != (revised_assignment,):
            raise ReconciliationError("GitHub did not update the assignment revision")
        draft_changed = self._set_draft(current, draft=True)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:wip"},
            remove={"status:review"},
        )
        after = self.pull_request(number)
        _require_open(after)
        _require_head(after, expected_head_sha)
        if not after.draft:
            raise ReconciliationError(
                "GitHub did not convert the pull request to draft"
            )
        _require_exact_labels(after, desired_labels)
        self._require_marker(number, marker, marker_body)
        return MutationResult(
            changed=assignment_changed
            or marker_changed
            or draft_changed
            or labels_changed,
            resource_url=after.url,
            state="revision_requested",
            version=after.head_sha,
        )

    def preflight_submit_result(
        self,
        number: int,
        *,
        branch: str,
        current_head_sha: str,
        expected_result_head_sha: str,
        result: ExperimentResult,
    ) -> PullRequestSnapshot:
        """Validate an assignment/result pair before mutating its Git branch."""

        snapshot = self.pull_request(number)
        _require_open(snapshot)
        _require_head(snapshot, current_head_sha)
        if snapshot.head_ref != branch:
            raise WorkflowPreconditionError(
                f"pull request branch is {snapshot.head_ref!r}, expected {branch!r}"
            )
        _require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_result_head_sha,
        )
        _require_assignment_result(snapshot, result)
        return snapshot

    def submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult:
        before = self.pull_request(number)
        _require_open(before)
        _require_head(before, expected_head_sha)
        _require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_head_sha,
        )
        _require_assignment_result(before, result)
        result_changed, _ = self._upsert_result_comment(
            number,
            result=result,
        )
        current = self.pull_request(number)
        _require_open(current)
        _require_head(current, expected_head_sha)
        ready_changed = self._set_draft(current, draft=False)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:review"},
            remove={"status:wip"},
        )
        after = self.pull_request(number)
        _require_open(after)
        _require_head(after, expected_head_sha)
        if after.draft:
            raise ReconciliationError(
                "GitHub did not mark the pull request ready for review"
            )
        _require_exact_labels(after, desired_labels)
        self._require_result(
            number,
            assignment_id=result.assignment.assignment_id,
            expected_head_sha=expected_head_sha,
        )
        return MutationResult(
            changed=result_changed or ready_changed or labels_changed,
            resource_url=after.url,
            state="result_submitted",
            version=after.head_sha,
        )

    def close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult:
        before = self.pull_request(number)
        _require_head(before, expected_head_sha)
        _require_assignment_identity(
            before,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        if before.merged:
            raise WorkflowPreconditionError(
                "pull request must be unmerged before it can be closed"
            )
        marker_body = _marker_body(marker, reason)
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=marker_body,
        )
        current = self.pull_request(number)
        _require_head(current, expected_head_sha)
        _require_assignment_identity(
            current,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        state_changed = current.state != "closed"
        if state_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{number}",
                json_body={"state": "closed"},
                expected_statuses={200},
            )
        after = self.pull_request(number)
        _require_head(after, expected_head_sha)
        _require_assignment_identity(
            after,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        if after.state != "closed":
            raise ReconciliationError("GitHub did not close the pull request")
        self._require_marker(number, marker, marker_body)
        return MutationResult(
            changed=marker_changed or state_changed,
            resource_url=after.url,
            state="experiment_closed",
            version=after.head_sha,
        )

    def merge_experiment(
        self,
        number: int,
        *,
        expected_head_sha: str,
        assignment_id: str,
        merge_method: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MutationResult:
        if merge_method not in ("merge", "squash", "rebase"):
            raise ValueError("merge_method must be merge, squash, or rebase")
        if not assignment_id.strip():
            raise ValueError("assignment_id must not be empty")
        before = self.pull_request(number)
        _require_head(before, expected_head_sha)
        terminal_result = self._require_result(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_assignment_result(before, terminal_result)
        if before.merged:
            if before.state != "closed":
                raise ReconciliationError(
                    "GitHub returned a merged pull request that is not closed"
                )
            if not before.merge_commit_sha:
                raise ReconciliationError(
                    "GitHub returned a merged pull request without a merge SHA"
                )
            return MutationResult(
                changed=False,
                resource_url=before.url,
                state="experiment_merged",
                version=before.merge_commit_sha,
            )

        _require_open(before)
        if before.draft:
            raise WorkflowPreconditionError("cannot merge a draft pull request")
        _require_labels(before, required={"status:review"}, forbidden=set())
        blocking_labels = {
            "status:blocked",
            "status:hold",
            "status:needs-rebase",
            "status:wip",
        }.intersection(before.labels)
        if blocking_labels:
            raise WorkflowPreconditionError(
                "cannot merge with blocking label(s): "
                + ", ".join(sorted(blocking_labels))
            )
        if before.mergeable is False:
            raise WorkflowPreconditionError(
                "cannot merge a pull request with a merge conflict"
            )
        if before.mergeable is None:
            raise WorkflowPreconditionError(
                "cannot merge while GitHub mergeability is unknown"
            )

        self._mutate(
            "PUT",
            f"/repos/{self._repo}/pulls/{number}/merge",
            json_body={
                "sha": expected_head_sha,
                "merge_method": merge_method,
            },
            expected_statuses={200},
        )
        after = self.pull_request(number)
        _require_head(after, expected_head_sha)
        if not after.merged or after.state != "closed":
            raise ReconciliationError("GitHub did not merge the pull request")
        if not after.merge_commit_sha:
            raise ReconciliationError(
                "GitHub did not return the resulting merge commit SHA"
            )
        self._require_result(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        return MutationResult(
            changed=True,
            resource_url=after.url,
            state="experiment_merged",
            version=after.merge_commit_sha,
        )

    def _set_draft(
        self,
        snapshot: PullRequestSnapshot,
        *,
        draft: bool,
    ) -> bool:
        if snapshot.draft is draft:
            return False
        mutation = (
            "convertPullRequestToDraft" if draft else "markPullRequestReadyForReview"
        )
        response = self._mutate(
            "POST",
            "/graphql",
            json_body={
                "query": (
                    f"mutation($pullRequestId: ID!) {{ {mutation}("
                    "input: {pullRequestId: $pullRequestId}) { "
                    "pullRequest { id isDraft } } }"
                ),
                "variables": {"pullRequestId": snapshot.node_id},
            },
            expected_statuses={200},
        )
        if response is None:
            return True
        body = _object(response.json_body, "GraphQL response")
        if body.get("errors"):
            raise ReconciliationError(
                f"GitHub GraphQL {mutation} mutation returned errors"
            )
        data = _object(body.get("data"), "GraphQL data")
        mutation_result = _object(data.get(mutation), f"GraphQL {mutation}")
        pull_request = _object(
            mutation_result.get("pullRequest"),
            f"GraphQL {mutation} pull request",
        )
        if pull_request.get("isDraft") is not draft:
            raise ReconciliationError(
                f"GitHub GraphQL {mutation} returned the wrong draft state"
            )
        return True

    def _set_labels(
        self,
        number: int,
        snapshot: PullRequestSnapshot,
        *,
        add: set[str],
        remove: set[str],
    ) -> tuple[bool, tuple[str, ...]]:
        _validate_labels(add | remove)
        if overlap := add & remove:
            raise ValueError(
                "labels cannot be both added and removed: " + ", ".join(sorted(overlap))
            )
        desired = tuple(sorted((set(snapshot.labels) | add) - remove))
        if snapshot.labels == desired:
            return False, desired
        self._mutate(
            "PUT",
            f"/repos/{self._repo}/issues/{number}/labels",
            json_body={"labels": list(desired)},
            expected_statuses={200},
        )
        return True, desired

    def _upsert_marker_comment(
        self,
        number: int,
        *,
        marker: str,
        body: str,
    ) -> tuple[bool, _IssueComment]:
        matches = self._marker_comments(number, marker)
        if len(matches) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        if matches and matches[0].body == body:
            return False, matches[0]

        if matches:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/issues/comments/{matches[0].id}",
                json_body={"body": body},
                expected_statuses={200},
            )
        else:
            self._mutate(
                "POST",
                f"/repos/{self._repo}/issues/{number}/comments",
                json_body={"body": body},
                expected_statuses={201},
            )

        verified = self._marker_comments(number, marker)
        if len(verified) != 1 or verified[0].body != body:
            raise ReconciliationError(
                "GitHub did not reach the requested marker comment"
            )
        return True, verified[0]

    def _upsert_result_comment(
        self,
        number: int,
        *,
        result: ExperimentResult,
    ) -> tuple[bool, _ResultComment]:
        assignment_id = result.assignment.assignment_id
        matches = self._result_comments(number, assignment_id)
        if len(matches) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple result markers for {assignment_id!r}"
            )

        body = render_result_comment(result)
        if matches and matches[0].comment.body == body:
            return False, matches[0]
        if matches:
            self._mutate(
                "PATCH",
                (f"/repos/{self._repo}/issues/comments/{matches[0].comment.id}"),
                json_body={"body": body},
                expected_statuses={200},
            )
        else:
            self._mutate(
                "POST",
                f"/repos/{self._repo}/issues/{number}/comments",
                json_body={"body": body},
                expected_statuses={201},
            )

        verified = self._result_comments(number, assignment_id)
        if (
            len(verified) != 1
            or verified[0].result != result
            or verified[0].comment.body != body
        ):
            raise ReconciliationError(
                "GitHub did not reach the requested terminal result"
            )
        return True, verified[0]

    def _result_comments(
        self,
        number: int,
        assignment_id: str,
    ) -> tuple[_ResultComment, ...]:
        matches: list[_ResultComment] = []
        trusted_actor = self._actor()
        for comment in self._comments(number):
            if comment.author != trusted_actor:
                continue
            try:
                results = parse_result_markers(comment.body)
            except ResultMarkerError as error:
                raise ReconciliationError(
                    f"GitHub contains an invalid result marker: {error}"
                ) from error
            matches.extend(
                _ResultComment(comment=comment, result=result)
                for result in results
                if result.assignment.assignment_id == assignment_id
            )
        return tuple(matches)

    def _assignment_pull_requests(
        self,
        assignment: AssignmentRecord,
    ) -> tuple[PullRequestSnapshot, ...]:
        owner = self._repo.split("/", 1)[0]
        query = urlencode(
            {
                "state": "all",
                "head": f"{owner}:{assignment.head_ref}",
                "base": assignment.base_ref,
                "per_page": 100,
            }
        )
        response = self._request(
            "GET",
            f"/repos/{self._repo}/pulls?{query}",
            expected_statuses={200},
        )
        if not isinstance(response.json_body, list):
            raise ReconciliationError(
                "GitHub returned invalid assignment PR search results"
            )
        return tuple(
            self.pull_request(
                _integer(
                    _object(item, "assignment pull request").get("number"),
                    "assignment pull request number",
                )
            )
            for item in response.json_body
        )

    def _active_student_assignment_numbers(
        self,
        student: str,
    ) -> tuple[int, ...]:
        query = urlencode(
            {
                "state": "open",
                "labels": ",".join(
                    (
                        f"student:{student}",
                        "status:wip",
                    )
                ),
                "per_page": 100,
            }
        )
        response = self._request(
            "GET",
            f"/repos/{self._repo}/issues?{query}",
            expected_statuses={200},
        )
        if not isinstance(response.json_body, list):
            raise ReconciliationError(
                "GitHub returned invalid active assignment results"
            )
        return tuple(
            _integer(
                data.get("number"),
                "active assignment number",
            )
            for item in response.json_body
            if "pull_request" in (data := _object(item, "active assignment"))
        )

    def _require_result(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> ExperimentResult:
        matches = self._result_comments(number, assignment_id)
        if not matches:
            raise WorkflowPreconditionError(
                f"schema-valid terminal result for {assignment_id!r} is missing"
            )
        if len(matches) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple result markers for {assignment_id!r}"
            )
        result = matches[0].result
        _require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_head_sha,
        )
        return result

    def _marker_comments(
        self,
        number: int,
        marker: str,
    ) -> tuple[_IssueComment, ...]:
        trusted_actor = self._actor()
        return tuple(
            comment
            for comment in self._comments(number)
            if comment.author == trusted_actor and marker in comment.body.splitlines()
        )

    def _require_marker(
        self,
        number: int,
        marker: str,
        body: str | None = None,
    ) -> _IssueComment:
        matches = self._marker_comments(number, marker)
        if not matches:
            raise WorkflowPreconditionError(
                f"terminal result or workflow marker {marker!r} is missing"
            )
        if len(matches) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        if body is not None and matches[0].body != body:
            raise ReconciliationError(
                "GitHub marker comment does not match the requested body"
            )
        return matches[0]

    def _comments(self, number: int) -> tuple[_IssueComment, ...]:
        number = _positive_number(number)
        url: str | None = f"/repos/{self._repo}/issues/{number}/comments?per_page=100"
        comments: list[_IssueComment] = []
        visited: set[str] = set()
        while url is not None:
            absolute_url = self._url(url)
            if absolute_url in visited:
                raise ReconciliationError("GitHub comment pagination contains a cycle")
            visited.add(absolute_url)
            response = self._request(
                "GET",
                absolute_url,
                expected_statuses={200},
            )
            page = response.json_body
            if not isinstance(page, list):
                raise ReconciliationError("GitHub returned invalid paginated comments")
            for raw_comment in page:
                data = _object(raw_comment, "issue comment")
                comments.append(
                    _IssueComment(
                        id=_integer(data.get("id"), "issue comment ID"),
                        body=_string(data.get("body"), "issue comment body"),
                        url=_string(
                            data.get("html_url"),
                            "issue comment URL",
                            allow_empty=True,
                        ),
                        author=_string(
                            _object(
                                data.get("user"),
                                "issue comment author",
                            ).get("login"),
                            "issue comment author login",
                        ),
                    )
                )
            url = _next_link(response.header("Link"))
            if url is not None and not url.startswith(f"{self._api_url}/"):
                raise ReconciliationError(
                    "GitHub pagination returned an unexpected origin"
                )
        return tuple(comments)

    def _human_issue(self, number: int) -> dict[str, object]:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/issues/{number}",
            expected_statuses={200},
        )
        issue = _object(response.json_body, "issue")
        if "pull_request" in issue:
            raise WorkflowPreconditionError(
                "human messages must use an issue, not a pull request"
            )
        if _string(issue.get("state"), "issue state") != "open":
            raise WorkflowPreconditionError("human issue must be open")
        if "human" not in _labels(issue.get("labels")):
            raise WorkflowPreconditionError("human issue must retain the human label")
        return issue

    def _human_message_author(
        self,
        number: int,
        *,
        issue: dict[str, object],
        human_message_id: int,
    ) -> str:
        if _integer(issue.get("id"), "issue ID") == human_message_id:
            return _string(
                _object(issue.get("user"), "issue author").get("login"),
                "issue author login",
            )
        match = next(
            (
                comment
                for comment in self._comments(number)
                if comment.id == human_message_id
            ),
            None,
        )
        if match is None:
            raise WorkflowPreconditionError(
                f"human message ID {human_message_id} is not present on issue #{number}"
            )
        return match.author

    def _actor(self) -> str:
        if self._trusted_actor is None:
            response = self._request(
                "GET",
                "/user",
                expected_statuses={200},
            )
            self._trusted_actor = _string(
                _object(response.json_body, "authenticated user").get("login"),
                "authenticated user login",
            )
        return self._trusted_actor

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: object | None = None,
        expected_statuses: set[int],
    ) -> HttpResponse:
        absolute_url = self._url(url)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        response = self._transport.request(
            method,
            absolute_url,
            headers=headers,
            json_body=json_body,
        )
        if response.status_code not in expected_statuses:
            raise GitHubAPIError(method, absolute_url, response.status_code)
        return response

    def _mutate(
        self,
        method: str,
        url: str,
        *,
        json_body: object,
        expected_statuses: set[int],
    ) -> HttpResponse | None:
        """Issue a mutation; an ambiguous transport failure is verified by caller."""

        try:
            return self._request(
                method,
                url,
                json_body=json_body,
                expected_statuses=expected_statuses,
            )
        except GitHubTransportError:
            return None

    def _url(self, value: str) -> str:
        if value.startswith(("https://", "http://")):
            return value
        return f"{self._api_url}/{value.lstrip('/')}"


def _positive_number(number: int) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("pull request number must be a positive integer")
    return number


def _positive_message_id(message_id: int) -> int:
    if (
        isinstance(message_id, bool)
        or not isinstance(message_id, int)
        or message_id <= 0
    ):
        raise ValueError("human message ID must be a positive integer")
    return message_id


def _require_head(snapshot: PullRequestSnapshot, expected_head_sha: str) -> None:
    if not expected_head_sha:
        raise ValueError("expected_head_sha must not be empty")
    if snapshot.head_sha != expected_head_sha:
        raise WorkflowPreconditionError(
            f"pull request head SHA is {snapshot.head_sha}, "
            f"expected {expected_head_sha}"
        )


def _require_result_identity(
    result: ExperimentResult,
    *,
    repo: str,
    number: int,
    expected_head_sha: str,
) -> None:
    assignment = result.assignment
    if assignment.repo != repo:
        raise WorkflowPreconditionError(
            "result repository does not match the GitHub workflow repository"
        )
    if assignment.pr_number != number:
        raise WorkflowPreconditionError(
            "result pull request number does not match the requested pull request"
        )
    if assignment.expected_head_sha != expected_head_sha:
        raise WorkflowPreconditionError(
            "result expected head SHA does not match the requested head SHA"
        )
    if result.commit_sha != expected_head_sha:
        raise WorkflowPreconditionError(
            "result commit does not match the pull request head SHA"
        )


def _require_assignment_snapshot(
    snapshot: PullRequestSnapshot,
    assignment: AssignmentRecord,
) -> None:
    if snapshot.base_ref != assignment.base_ref:
        raise WorkflowPreconditionError(
            f"pull request base is {snapshot.base_ref!r}, "
            f"expected {assignment.base_ref!r}"
        )
    if snapshot.head_ref != assignment.head_ref:
        raise WorkflowPreconditionError(
            f"pull request head is {snapshot.head_ref!r}, "
            f"expected {assignment.head_ref!r}"
        )
    _require_head(snapshot, assignment.head_sha)
    markers = parse_assignment_markers(snapshot.body)
    if markers != (assignment,):
        raise WorkflowPreconditionError(
            "pull request must contain exactly the expected assignment marker"
        )


def _require_assignment_result(
    snapshot: PullRequestSnapshot,
    result: ExperimentResult,
) -> None:
    try:
        markers = parse_assignment_markers(snapshot.body)
    except ValueError as error:
        raise WorkflowPreconditionError(
            f"pull request contains an invalid assignment marker: {error}"
        ) from error
    if len(markers) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    record = markers[0]
    assignment = result.assignment
    if (
        record.repo != assignment.repo
        or record.assignment_id != assignment.assignment_id
        or record.revision_id != assignment.revision_id
        or record.student != assignment.student
        or record.head_ref != snapshot.head_ref
        or record.base_ref != snapshot.base_ref
    ):
        raise WorkflowPreconditionError(
            "terminal result does not match the pull request assignment marker"
        )


def _require_assignment_identity(
    snapshot: PullRequestSnapshot,
    *,
    repo: str,
    assignment_id: str,
) -> AssignmentRecord:
    if not assignment_id.strip():
        raise ValueError("assignment_id must not be empty")
    try:
        markers = parse_assignment_markers(snapshot.body)
    except ValueError as error:
        raise WorkflowPreconditionError(
            f"pull request contains an invalid assignment marker: {error}"
        ) from error
    if len(markers) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    assignment = markers[0]
    if (
        assignment.repo != repo
        or assignment.assignment_id != assignment_id
        or assignment.base_ref != snapshot.base_ref
        or assignment.head_ref != snapshot.head_ref
    ):
        raise WorkflowPreconditionError(
            "pull request assignment identity does not match the requested transition"
        )
    return assignment


def _require_open(snapshot: PullRequestSnapshot) -> None:
    if snapshot.state != "open" or snapshot.merged:
        raise WorkflowPreconditionError("pull request must be open and unmerged")


def _require_labels(
    snapshot: PullRequestSnapshot,
    *,
    required: set[str],
    forbidden: set[str],
) -> None:
    labels = set(snapshot.labels)
    missing = required - labels
    if missing:
        raise WorkflowPreconditionError(
            "pull request is missing required label(s): " + ", ".join(sorted(missing))
        )
    present = forbidden & labels
    if present:
        raise ReconciliationError(
            "pull request retains forbidden label(s): " + ", ".join(sorted(present))
        )


def _require_exact_labels(
    snapshot: PullRequestSnapshot,
    desired: tuple[str, ...],
) -> None:
    if snapshot.labels != desired:
        raise ReconciliationError("GitHub did not reach the requested label set")


def _validate_labels(labels: set[str]) -> None:
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be non-empty strings")


def _validate_marker(marker: str, body: str) -> None:
    if (
        "\n" in marker
        or "\r" in marker
        or not marker.startswith("<!-- senpai-")
        or not marker.endswith("-->")
    ):
        raise ValueError("marker must be one hidden Senpai marker")
    if body.splitlines().count(marker) != 1:
        raise ValueError("comment body must contain the marker exactly once")


def _marker_body(marker: str, content: str) -> str:
    content = content.strip()
    if not content:
        raise ValueError("marker comment content must not be empty")
    body = f"{marker}\n\n{content}"
    _validate_marker(marker, body)
    return body


def _replace_assignment_marker(
    body: str,
    assignment: AssignmentRecord,
) -> str:
    replacement = render_assignment_marker(assignment)
    lines = body.splitlines()
    indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("<!-- senpai-assignment:")
    ]
    if len(indexes) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    lines[indexes[0]] = replacement
    return "\n".join(lines)


def _labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReconciliationError("GitHub returned invalid pull request labels")
    labels = []
    for item in value:
        data = _object(item, "pull request label")
        labels.append(_string(data.get("name"), "pull request label name"))
    return tuple(sorted(set(labels)))


def _next_link(value: str | None) -> str | None:
    if value is None:
        return None
    for part in value.split(","):
        segments = [segment.strip() for segment in part.split(";")]
        if 'rel="next"' not in segments[1:]:
            continue
        target = segments[0]
        if target.startswith("<") and target.endswith(">"):
            return target[1:-1]
    return None


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReconciliationError(f"GitHub returned invalid {name}")
    return cast(dict[str, object], value)


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ReconciliationError(f"GitHub returned invalid {name}")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReconciliationError(f"GitHub returned invalid {name}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ReconciliationError(f"GitHub returned invalid {name}")
    return value
