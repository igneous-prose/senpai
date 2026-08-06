"""Idempotent GitHub workflow transitions over the GitHub API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Annotated, Literal, Protocol
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit

from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
)

from senpai_agent.github_http import next_link
from senpai_agent.models import (
    AssignmentFeedbackRecord,
    AssignmentRecord,
    Contract,
    ExperimentResult,
    ResearchBaseAcceptanceRecord,
    ResultMarkerError,
    RevisionRecord,
    experiment_result_digest,
    parse_assignment_markers,
    parse_research_base_acceptance_markers,
    parse_result_markers,
    render_assignment_feedback_marker,
    render_assignment_marker,
    render_research_base_acceptance_marker,
    render_result_comment,
    render_revision_marker,
)


_TRUSTED_HUMAN_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


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


class PullHeadMismatchError(WorkflowPreconditionError):
    """GitHub's pull-request snapshot has not reached the expected head."""


class StaleAssignmentRevisionError(WorkflowPreconditionError):
    """The requested operation belongs to another assignment revision."""


class StaleResearchBaseError(WorkflowPreconditionError):
    """The result was not reviewed against the live research base."""


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
class SubmitResultPreflight:
    snapshot: PullRequestSnapshot
    assignment: AssignmentRecord


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
    author_type: str
    author_association: str


@dataclass(frozen=True, slots=True)
class _ResultComment:
    comment: _IssueComment
    result: ExperimentResult


class _GitHubResponse(Contract):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=False,
    )


_RequiredString = Annotated[StrictStr, Field(min_length=1)]
_PositiveInteger = Annotated[StrictInt, Field(gt=0)]


class _GitHubRef(_GitHubResponse):
    ref: _RequiredString


class _GitHubHead(_GitHubRef):
    sha: _RequiredString


class _GitObject(_GitHubResponse):
    sha: _RequiredString


class _GitRefResponse(_GitHubResponse):
    ref: _RequiredString
    object: _GitObject


class _GitHubLabel(_GitHubResponse):
    name: _RequiredString


class _GitHubUser(_GitHubResponse):
    login: _RequiredString


class _GitHubAuthor(_GitHubUser):
    type: _RequiredString


class _PullRequestResponse(_GitHubResponse):
    number: _PositiveInteger
    node_id: _RequiredString
    html_url: _RequiredString
    title: _RequiredString
    body: StrictStr | None
    base: _GitHubRef
    head: _GitHubHead
    labels: tuple[_GitHubLabel, ...]
    draft: StrictBool
    state: Literal["open", "closed"]
    merged: StrictBool
    mergeable: StrictBool | None
    merge_commit_sha: StrictStr | None

    def snapshot(self) -> PullRequestSnapshot:
        return PullRequestSnapshot(
            number=self.number,
            node_id=self.node_id,
            url=self.html_url,
            title=self.title,
            body=self.body or "",
            base_ref=self.base.ref,
            head_ref=self.head.ref,
            head_sha=self.head.sha,
            labels=tuple(sorted({label.name for label in self.labels})),
            draft=self.draft,
            state=self.state,
            merged=self.merged,
            mergeable=self.mergeable,
            merge_commit_sha=self.merge_commit_sha,
        )


class _IssueCommentResponse(_GitHubResponse):
    id: _PositiveInteger
    body: StrictStr
    html_url: StrictStr
    user: _GitHubAuthor
    author_association: _RequiredString

    def comment(self) -> _IssueComment:
        return _IssueComment(
            id=self.id,
            body=self.body,
            url=self.html_url,
            author=self.user.login,
            author_type=self.user.type,
            author_association=self.author_association,
        )


class _IssueResponse(_GitHubResponse):
    id: _PositiveInteger
    state: StrictStr
    labels: tuple[_GitHubLabel, ...]
    user: _GitHubAuthor
    author_association: _RequiredString
    pull_request: dict[str, object] | None = None


class _NumberedResponse(_GitHubResponse):
    number: _PositiveInteger


class _IssueSearchResponse(_NumberedResponse):
    labels: tuple[_GitHubLabel, ...]
    pull_request: dict[str, object] | None = None


class _DraftPullRequestResponse(_GitHubResponse):
    is_draft: StrictBool = Field(alias="isDraft")


class _DraftMutationResponse(_GitHubResponse):
    pull_request: _DraftPullRequestResponse = Field(alias="pullRequest")


def _validated_response[ResponseT: _GitHubResponse](
    model: type[ResponseT],
    value: object,
    name: str,
) -> ResponseT:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise ReconciliationError(f"GitHub returned invalid {name}") from error


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
        "_assignment_lifecycle_lock",
        "_repo",
        "_role",
        "_token",
        "_transport",
        "_trusted_actor",
    )

    def __init__(
        self,
        repo: str,
        token: SecretStr,
        *,
        role: Literal["advisor", "student"],
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
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        if trusted_actor is not None and not trusted_actor.strip():
            raise ValueError("trusted actor must not be empty")

        self._repo = repo
        self._role = role
        self._token = token
        self._transport = transport or _UrllibTransport()
        self._api_url = api_url.rstrip("/")
        self._trusted_actor = trusted_actor
        # OpenHands may execute sibling tool calls concurrently. Keep the
        # one-WIP precondition and its GitHub mutation in one critical section.
        self._assignment_lifecycle_lock = RLock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(repo={self._repo!r}, api_url={self._api_url!r})"

    @property
    def repo(self) -> str:
        return self._repo

    @property
    def role(self) -> Literal["advisor", "student"]:
        return self._role

    @contextmanager
    def serialized_assignment_mutation(self) -> Iterator[None]:
        """Serialize a coupled local mutation with this workflow's transitions.

        This closes races among Senpai operations sharing this workflow instance.
        GitHub's merge API has no base-SHA compare-and-swap; external writers still
        require strict branch protection or a merge queue.
        """

        with self._assignment_lifecycle_lock:
            yield

    def __getstate__(self) -> None:
        raise TypeError("GitHubWorkflow cannot be serialized")

    def pull_request(self, number: int) -> PullRequestSnapshot:
        number = _positive_number(number)
        response = self._request(
            "GET",
            f"/repos/{self._repo}/pulls/{number}",
            expected_statuses={200},
        )
        snapshot = _validated_response(
            _PullRequestResponse,
            response.json_body,
            "pull request",
        ).snapshot()
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

        with self._assignment_lifecycle_lock:
            return self._create_assignment(assignment, title=title, body=body)

    def _create_assignment(
        self,
        assignment: AssignmentRecord,
        *,
        title: str,
        body: str,
    ) -> MutationResult:
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

    def repair_assignment_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        working_state: Literal["wip", "review"],
        blockers: set[Literal["blocked", "hold", "needs-rebase"]],
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._repair_assignment_routing(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
                working_state=working_state,
                blockers=blockers,
            )

    def _repair_assignment_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        working_state: Literal["wip", "review"],
        blockers: set[Literal["blocked", "hold", "needs-rebase"]],
    ) -> MutationResult:
        if working_state not in ("wip", "review"):
            raise ValueError("working_state must be wip or review")
        invalid_blockers = blockers - {"blocked", "hold", "needs-rebase"}
        if invalid_blockers:
            raise ValueError(
                "unsupported assignment blocker(s): "
                + ", ".join(sorted(invalid_blockers))
            )
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_open(before)
        _require_current_revision(assignment, current_revision_id)
        if working_state == "review":
            result = self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_assignment_result(before, result)
        protocol_labels = {
            assignment.base_ref,
            f"student:{assignment.student}",
            f"status:{working_state}",
            *(f"status:{blocker}" for blocker in blockers),
        }
        remove = {
            label
            for label in before.labels
            if label.startswith(("student:", "status:"))
            and label not in protocol_labels
        }
        expected_draft = working_state == "wip"
        draft_changed = self._set_draft(before, draft=expected_draft)
        labels_changed, desired = self._set_labels(
            number,
            before,
            add=protocol_labels,
            remove=remove,
        )
        if not draft_changed and not labels_changed:
            return MutationResult(
                changed=False,
                resource_url=before.url,
                state="assignment_routing_repaired",
                version=before.head_sha,
            )

        after, current_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_open(after)
        _require_current_revision(current_assignment, current_revision_id)
        if after.draft is not expected_draft:
            raise ReconciliationError(
                "GitHub did not reach the requested assignment draft state"
            )
        _require_exact_labels(after, desired)
        return MutationResult(
            changed=draft_changed or labels_changed,
            resource_url=after.url,
            state="assignment_routing_repaired",
            version=after.head_sha,
        )

    def respond_to_issue(
        self,
        number: int,
        *,
        human_message_id: int,
        audience_labels: set[str],
        response: str,
    ) -> MutationResult:
        """Reply once to one verified human-authored GitHub issue message."""

        number = _positive_number(number)
        human_message_id = _positive_message_id(human_message_id)
        body = response.strip()
        if not body:
            raise ValueError("response must not be empty")
        _validate_labels(audience_labels)
        if not audience_labels:
            raise ValueError("audience_labels must not be empty")

        issue = self._human_issue(number, audience_labels=audience_labels)
        source_author = self._human_message_author(
            number,
            issue=issue,
            human_message_id=human_message_id,
        )
        if source_author.casefold() == self._actor().casefold():
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
        self._human_issue(number, audience_labels=audience_labels)
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
        current_revision_id: str,
        new_revision_id: str,
        expected_head_sha: str,
        required_base_sha: str,
        comment: str,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._request_revision(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                new_revision_id=new_revision_id,
                expected_head_sha=expected_head_sha,
                required_base_sha=required_base_sha,
                comment=comment,
            )

    def _request_revision(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        new_revision_id: str,
        expected_head_sha: str,
        required_base_sha: str,
        comment: str,
    ) -> MutationResult:
        if current_revision_id == new_revision_id:
            raise ValueError("new_revision_id must differ from current_revision_id")
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_open(before)
        if assignment.revision_id == current_revision_id:
            live_base_sha = self._branch_head_sha(assignment.base_ref)
            if live_base_sha != required_base_sha:
                raise StaleResearchBaseError(
                    f"required research base {assignment.base_ref}@"
                    f"{required_base_sha} does not match live {live_base_sha}"
                )
        elif not (
            assignment.revision_id == new_revision_id
            and assignment.base_sha == required_base_sha
        ):
            raise StaleAssignmentRevisionError(
                f"assignment revision is {assignment.revision_id!r}, expected "
                f"current {current_revision_id!r} or applied {new_revision_id!r}"
            )
        conflicts = tuple(
            active_number
            for active_number in self._active_student_assignment_numbers(
                assignment.student
            )
            if active_number != number
        )
        if conflicts:
            raise WorkflowPreconditionError(
                f"student:{assignment.student} already has active assignment "
                f"PR(s): {', '.join(f'#{active_number}' for active_number in conflicts)}"
            )
        marker = render_revision_marker(
            RevisionRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=new_revision_id,
                requested_head_sha=expected_head_sha,
            )
        )
        marker_body = _marker_body(marker, comment)
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=marker_body,
        )
        revised_assignment = assignment.model_copy(
            update={
                "revision_id": new_revision_id,
                "base_sha": required_base_sha,
            }
        )
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
        current = self._pull_at_head(number, expected_head_sha)
        _require_open(current)
        if parse_assignment_markers(current.body) != (revised_assignment,):
            raise ReconciliationError("GitHub did not update the assignment revision")
        draft_changed = self._set_draft(current, draft=True)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:wip"},
            remove={"status:review"},
        )
        after = self._pull_at_head(number, expected_head_sha)
        _require_open(after)
        if not after.draft:
            raise ReconciliationError(
                "GitHub did not convert the pull request to draft"
            )
        _require_exact_labels(after, desired_labels)
        return MutationResult(
            changed=assignment_changed
            or marker_changed
            or draft_changed
            or labels_changed,
            resource_url=after.url,
            state="revision_requested",
            version=after.head_sha,
        )

    def send_assignment_feedback(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        feedback_id: str,
        comment: str,
    ) -> MutationResult:
        """Upsert guidance for the current assignment without starting a revision."""

        with self._assignment_lifecycle_lock:
            return self._send_assignment_feedback(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                feedback_id=feedback_id,
                comment=comment,
            )

    def _send_assignment_feedback(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        feedback_id: str,
        comment: str,
    ) -> MutationResult:

        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_open(before)
        _require_current_revision(assignment, revision_id)
        _require_active_assignment_routing(before, assignment)
        feedback_id = feedback_id.strip()
        body = comment.strip()
        if not feedback_id or not body:
            raise ValueError("feedback_id and comment must not be empty")

        marker = render_assignment_feedback_marker(
            AssignmentFeedbackRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                feedback_id=feedback_id,
            )
        )
        marker_body = _marker_body(marker, body)
        existing = self._marker_comments(number, marker)
        if len(existing) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        desired_body = _role_prefixed_comment(marker_body, self._role)
        if existing and _role_prefixed_comment(
            existing[0].body,
            self._role,
        ) != desired_body:
            raise WorkflowPreconditionError(
                "feedback_id already identifies different guidance; "
                "use a new feedback_id"
            )
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=marker_body,
        )
        after, current_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_open(after)
        _require_current_revision(current_assignment, revision_id)
        _require_active_assignment_routing(after, current_assignment)
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="assignment_feedback_upserted",
            version=after.head_sha,
        )

    def accept_result_on_current_base(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        expected_current_base_sha: str,
        reason: str,
    ) -> MutationResult:
        """Durably approve one exact result against the exact live research base."""

        with self._assignment_lifecycle_lock:
            before, assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_open(before)
            _require_current_revision(assignment, current_revision_id)
            if before.draft:
                raise WorkflowPreconditionError(
                    "cannot accept a result while its pull request is draft"
                )
            _require_labels(before, required={"status:review"}, forbidden=set())
            result = self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_assignment_result(before, result)
            _require_exact_research_base(
                assignment,
                live_base_sha=self._branch_head_sha(assignment.base_ref),
                expected_current_base_sha=expected_current_base_sha,
            )
            acceptance = ResearchBaseAcceptanceRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                result_head_sha=expected_head_sha,
                result_digest=experiment_result_digest(result),
                evaluated_base_sha=assignment.base_sha,
                base_ref=assignment.base_ref,
                accepted_base_sha=expected_current_base_sha,
            )
            marker = render_research_base_acceptance_marker(acceptance)
            changed, verified = self._upsert_marker_comment(
                number,
                marker=marker,
                body=_marker_body(marker, reason),
            )

            after, current_assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_open(after)
            _require_current_revision(current_assignment, current_revision_id)
            current_result = self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_assignment_result(after, current_result)
            _require_exact_research_base(
                current_assignment,
                live_base_sha=self._branch_head_sha(current_assignment.base_ref),
                expected_current_base_sha=expected_current_base_sha,
            )
            self._require_research_base_acceptance(number, acceptance)
            return MutationResult(
                changed=changed,
                resource_url=verified.url,
                state="research_base_accepted",
                version=expected_current_base_sha,
            )

    def preflight_submit_result(
        self,
        number: int,
        *,
        branch: str,
        current_head_sha: str,
        expected_result_head_sha: str,
        result: ExperimentResult,
    ) -> SubmitResultPreflight:
        """Validate an assignment/result pair before mutating its Git branch."""

        snapshot = self._pull_at_head(number, current_head_sha)
        _require_open(snapshot)
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
        assignment = _require_assignment_result(snapshot, result)
        return SubmitResultPreflight(snapshot=snapshot, assignment=assignment)

    def submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._submit_result(
                number,
                expected_head_sha=expected_head_sha,
                result=result,
            )

    def _submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult:
        before = self._pull_at_head(number, expected_head_sha)
        _require_open(before)
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
        current = self._pull_at_head(number, expected_head_sha)
        _require_open(current)
        _require_assignment_result(current, result)
        ready_changed = self._set_draft(current, draft=False)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:review"},
            remove={"status:wip"},
        )
        after = self._pull_at_head(number, expected_head_sha)
        _require_open(after)
        try:
            _require_assignment_result(after, result)
        except StaleAssignmentRevisionError:
            self._recover_stale_submit_routing(
                number,
                assignment_id=result.assignment.assignment_id,
                expected_head_sha=expected_head_sha,
                stale_result=result,
            )
            raise
        if after.draft:
            raise ReconciliationError(
                "GitHub did not mark the pull request ready for review"
            )
        _require_exact_labels(after, desired_labels)
        return MutationResult(
            changed=result_changed or ready_changed or labels_changed,
            resource_url=after.url,
            state="result_submitted",
            version=after.head_sha,
        )

    def _recover_stale_submit_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        stale_result: ExperimentResult,
    ) -> None:
        """Undo stale review routing without clobbering newer valid evidence."""

        for _attempt in range(3):
            current, assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            _require_open(current)
            if assignment.revision_id == stale_result.assignment.revision_id:
                return
            if self._has_result_for_snapshot(current, assignment_id):
                return

            self._set_draft(current, draft=True)
            current, refreshed = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            if refreshed != assignment:
                continue
            if self._has_result_for_snapshot(current, assignment_id):
                self._restore_current_result_review(
                    current,
                    assignment_id=assignment_id,
                    expected_head_sha=expected_head_sha,
                )
                return

            self._set_labels(
                number,
                current,
                add={"status:wip"},
                remove={"status:review"},
            )
            after, verified = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            if verified != assignment:
                continue
            if self._has_result_for_snapshot(after, assignment_id):
                self._restore_current_result_review(
                    after,
                    assignment_id=assignment_id,
                    expected_head_sha=expected_head_sha,
                )
                return
            if (
                after.draft
                and "status:wip" in after.labels
                and "status:review" not in after.labels
            ):
                return
            raise ReconciliationError(
                "GitHub did not restore the current assignment revision to WIP"
            )
        raise ReconciliationError(
            "assignment revision kept changing during stale-submit recovery"
        )

    def _has_result_for_snapshot(
        self,
        snapshot: PullRequestSnapshot,
        assignment_id: str,
    ) -> bool:
        for match in self._result_comments(snapshot.number, assignment_id):
            try:
                _require_result_identity(
                    match.result,
                    repo=self._repo,
                    number=snapshot.number,
                    expected_head_sha=snapshot.head_sha,
                )
                _require_assignment_result(snapshot, match.result)
            except WorkflowPreconditionError:
                continue
            return True
        return False

    def _restore_current_result_review(
        self,
        snapshot: PullRequestSnapshot,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> None:
        """Keep a concurrently submitted current-revision result reviewable."""

        self._set_draft(snapshot, draft=False)
        current, assignment = self._assigned_pull_at_head(
            snapshot.number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        if not self._has_result_for_snapshot(current, assignment_id):
            raise ReconciliationError(
                "current-revision result disappeared during stale-submit recovery"
            )
        self._set_labels(
            snapshot.number,
            current,
            add={"status:review"},
            remove={"status:wip"},
        )
        after, verified = self._assigned_pull_at_head(
            snapshot.number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        if verified != assignment or not self._has_result_for_snapshot(
            after, assignment_id
        ):
            raise ReconciliationError(
                "current-revision result changed during stale-submit recovery"
            )

    def close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._close_experiment(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
                marker=marker,
                reason=reason,
            )

    def _close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult:
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_current_revision(assignment, current_revision_id)
        _require_unmerged(before)
        state_changed = before.state != "closed"
        if state_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{number}",
                json_body={"state": "closed"},
                expected_statuses={200},
            )
        closed, closed_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_current_revision(closed_assignment, current_revision_id)
        _require_unmerged(closed)
        if closed.state != "closed":
            raise ReconciliationError("GitHub did not close the pull request")
        marker_body = _marker_body(marker, reason)
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=marker_body,
        )
        after, after_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        _require_current_revision(after_assignment, current_revision_id)
        _require_unmerged(after)
        if after.state != "closed":
            raise ReconciliationError("pull request reopened during reconciliation")
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
        current_revision_id: str,
        expected_current_base_sha: str,
        merge_method: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._merge_experiment(
                number,
                expected_head_sha=expected_head_sha,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_current_base_sha=expected_current_base_sha,
                merge_method=merge_method,
            )

    def _merge_experiment(
        self,
        number: int,
        *,
        expected_head_sha: str,
        assignment_id: str,
        current_revision_id: str,
        expected_current_base_sha: str,
        merge_method: Literal["merge", "squash", "rebase"],
    ) -> MutationResult:
        if merge_method not in ("merge", "squash", "rebase"):
            raise ValueError("merge_method must be merge, squash, or rebase")
        if not assignment_id.strip():
            raise ValueError("assignment_id must not be empty")
        before = self._pull_at_head(number, expected_head_sha)
        terminal_result = self._require_result(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        assignment = _require_assignment_result(before, terminal_result)
        _require_current_revision(assignment, current_revision_id)
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

        _require_exact_research_base(
            assignment,
            live_base_sha=self._branch_head_sha(assignment.base_ref),
            expected_current_base_sha=expected_current_base_sha,
        )
        if assignment.base_sha != expected_current_base_sha:
            self._require_research_base_acceptance(
                number,
                ResearchBaseAcceptanceRecord(
                    repo=self._repo,
                    pr_number=number,
                    assignment_id=assignment.assignment_id,
                    revision_id=assignment.revision_id,
                    result_head_sha=expected_head_sha,
                    result_digest=experiment_result_digest(terminal_result),
                    evaluated_base_sha=assignment.base_sha,
                    base_ref=assignment.base_ref,
                    accepted_base_sha=expected_current_base_sha,
                ),
            )

        _require_exact_research_base(
            assignment,
            live_base_sha=self._branch_head_sha(assignment.base_ref),
            expected_current_base_sha=expected_current_base_sha,
        )

        _require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="immediately before merge",
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
        _require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="immediately after merge",
        )
        after = self._pull_at_head(number, expected_head_sha)
        if not after.merged or after.state != "closed":
            raise ReconciliationError("GitHub did not merge the pull request")
        if not after.merge_commit_sha:
            raise ReconciliationError(
                "GitHub did not return the resulting merge commit SHA"
            )
        _require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="final merge reconciliation",
        )
        return MutationResult(
            changed=True,
            resource_url=after.url,
            state="experiment_merged",
            version=after.merge_commit_sha,
        )

    def _branch_head_sha(self, branch: str) -> str:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/git/ref/heads/{quote(branch, safe='')}",
            expected_statuses={200},
        )
        git_ref = _validated_response(
            _GitRefResponse,
            response.json_body,
            "git reference",
        )
        expected_ref = f"refs/heads/{branch}"
        if git_ref.ref != expected_ref:
            raise ReconciliationError(
                f"GitHub returned git reference {git_ref.ref!r}, "
                f"expected {expected_ref!r}"
            )
        return git_ref.object.sha

    def _pull_at_head(
        self,
        number: int,
        expected_head_sha: str,
    ) -> PullRequestSnapshot:
        snapshot = self.pull_request(number)
        _require_head(snapshot, expected_head_sha)
        return snapshot

    def _assigned_pull_at_head(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> tuple[PullRequestSnapshot, AssignmentRecord]:
        snapshot = self._pull_at_head(number, expected_head_sha)
        assignment = _require_assignment_identity(
            snapshot,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        return snapshot, assignment

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
        if not isinstance(response.json_body, dict):
            raise ReconciliationError("GitHub returned invalid GraphQL response")
        if response.json_body.get("errors"):
            raise ReconciliationError(
                f"GitHub GraphQL {mutation} mutation returned errors"
            )
        data = response.json_body.get("data")
        mutation_payload = data.get(mutation) if isinstance(data, dict) else None
        mutation_result = _validated_response(
            _DraftMutationResponse,
            mutation_payload,
            f"GraphQL {mutation} result",
        )
        if mutation_result.pull_request.is_draft is not draft:
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
        return self._upsert_comment(
            number,
            body=body,
            matches=lambda: self._marker_comments(number, marker),
            subject=f"comments for marker {marker!r}",
            desired_state="marker comment",
        )

    def _upsert_result_comment(
        self,
        number: int,
        *,
        result: ExperimentResult,
    ) -> tuple[bool, _IssueComment]:
        assignment_id = result.assignment.assignment_id
        body = _role_prefixed_comment(render_result_comment(result), self._role)
        existing = self._result_comments(number, assignment_id)
        distinct = _distinct_results(existing)
        if len(distinct) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple distinct result markers for "
                f"{assignment_id!r}"
            )
        if distinct:
            published = distinct[0]
            if experiment_result_digest(published) == experiment_result_digest(result):
                comments = {match.comment.id: match.comment for match in existing}
                changed = False
                for comment in comments.values():
                    if comment.body == body:
                        continue
                    self._mutate(
                        "PATCH",
                        f"/repos/{self._repo}/issues/comments/{comment.id}",
                        json_body={"body": body},
                        expected_statuses={200},
                    )
                    changed = True
                if not changed:
                    return False, existing[0].comment
                verified = self._result_comments(number, assignment_id)
                if not verified or any(
                    match.comment.body != body for match in verified
                ):
                    raise ReconciliationError(
                        "GitHub did not normalize the terminal result comment"
                    )
                return True, verified[0].comment
            if (
                published.assignment.revision_id == result.assignment.revision_id
                and published.assignment.expected_head_sha
                == result.assignment.expected_head_sha
            ):
                raise WorkflowPreconditionError(
                    "a published terminal result is immutable at the same "
                    "assignment revision and head; use a new revision or head"
                )
            comments = {match.comment.id: match.comment for match in existing}
            for comment in comments.values():
                self._mutate(
                    "PATCH",
                    f"/repos/{self._repo}/issues/comments/{comment.id}",
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
        verified_distinct = _distinct_results(verified)
        if (
            len(verified_distinct) != 1
            or experiment_result_digest(verified_distinct[0])
            != experiment_result_digest(result)
        ):
            raise ReconciliationError(
                "GitHub did not reach the requested terminal result"
            )
        return True, verified[0].comment

    def _upsert_comment(
        self,
        number: int,
        *,
        body: str,
        matches: Callable[[], tuple[_IssueComment, ...]],
        subject: str,
        desired_state: str,
    ) -> tuple[bool, _IssueComment]:
        body = _role_prefixed_comment(body, self._role)
        existing = matches()
        if len(existing) > 1:
            raise ReconciliationError(f"GitHub contains multiple {subject}")
        if existing and existing[0].body == body:
            return False, existing[0]
        if existing:
            method = "PATCH"
            path = f"/repos/{self._repo}/issues/comments/{existing[0].id}"
            expected_statuses = {200}
        else:
            method = "POST"
            path = f"/repos/{self._repo}/issues/{number}/comments"
            expected_statuses = {201}
        self._mutate(
            method,
            path,
            json_body={"body": body},
            expected_statuses=expected_statuses,
        )
        verified = matches()
        if len(verified) != 1 or verified[0].body != body:
            raise ReconciliationError(
                f"GitHub did not reach the requested {desired_state}"
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
            if comment.author.casefold() != trusted_actor.casefold():
                continue
            for line in comment.body.splitlines():
                try:
                    results = parse_result_markers(line)
                except ResultMarkerError:
                    continue
                matches.extend(
                    _ResultComment(comment=comment, result=result)
                    for result in results
                    if result.assignment.assignment_id == assignment_id
                )
        return tuple(matches)

    def _research_base_acceptances(
        self,
        number: int,
    ) -> tuple[ResearchBaseAcceptanceRecord, ...]:
        trusted_actor = self._actor()
        acceptances: list[ResearchBaseAcceptanceRecord] = []
        for comment in self._comments(number):
            if comment.author.casefold() != trusted_actor.casefold():
                continue
            try:
                acceptances.extend(
                    parse_research_base_acceptance_markers(comment.body)
                )
            except ValueError:
                continue
        return tuple(acceptances)

    def _require_research_base_acceptance(
        self,
        number: int,
        expected: ResearchBaseAcceptanceRecord,
    ) -> None:
        matches = tuple(
            acceptance
            for acceptance in self._research_base_acceptances(number)
            if acceptance == expected
        )
        if not matches:
            raise StaleResearchBaseError(
                "the current result has no durable acceptance for research base "
                f"{expected.base_ref}@{expected.accepted_base_sha}"
            )

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
                _validated_response(
                    _NumberedResponse,
                    item,
                    "assignment pull request",
                ).number
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
                "labels": f"student:{student}",
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
        issues = tuple(
            _validated_response(
                _IssueSearchResponse,
                item,
                "active assignment",
            )
            for item in response.json_body
        )
        return tuple(
            issue.number
            for issue in issues
            if issue.pull_request is not None
            and "status:wip" in {label.name for label in issue.labels}
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
        distinct = _distinct_results(matches)
        if len(distinct) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple distinct result markers for "
                f"{assignment_id!r}"
            )
        result = distinct[0]
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
            if comment.author.casefold() == trusted_actor.casefold()
            and marker in comment.body.splitlines()
        )

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
                comments.append(
                    _validated_response(
                        _IssueCommentResponse,
                        raw_comment,
                        "issue comment",
                    ).comment()
                )
            url = next_link(response.header("Link"))
            if url is not None and not url.startswith(f"{self._api_url}/"):
                raise ReconciliationError(
                    "GitHub pagination returned an unexpected origin"
                )
        return tuple(comments)

    def _human_issue(
        self,
        number: int,
        *,
        audience_labels: set[str],
    ) -> _IssueResponse:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/issues/{number}",
            expected_statuses={200},
        )
        issue = _validated_response(_IssueResponse, response.json_body, "issue")
        if issue.pull_request is not None:
            raise WorkflowPreconditionError(
                "human messages must use an issue, not a pull request"
            )
        if issue.state != "open":
            raise WorkflowPreconditionError("human issue must be open")
        labels = {label.name for label in issue.labels}
        if "human" not in labels:
            raise WorkflowPreconditionError("human issue must retain the human label")
        if not labels.intersection(audience_labels):
            raise WorkflowPreconditionError(
                "human issue must retain a team or current-role audience label"
            )
        return issue

    def _human_message_author(
        self,
        number: int,
        *,
        issue: _IssueResponse,
        human_message_id: int,
    ) -> str:
        if issue.id == human_message_id:
            _require_trusted_human_author(
                login=issue.user.login,
                author_type=issue.user.type,
                association=issue.author_association,
            )
            return issue.user.login
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
        _require_trusted_human_author(
            login=match.author,
            author_type=match.author_type,
            association=match.author_association,
        )
        return match.author

    def _actor(self) -> str:
        if self._trusted_actor is None:
            response = self._request(
                "GET",
                "/user",
                expected_statuses={200},
            )
            self._trusted_actor = _validated_response(
                _GitHubUser,
                response.json_body,
                "authenticated user",
            ).login
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


def _require_trusted_human_author(
    *,
    login: str,
    author_type: str,
    association: str,
) -> None:
    if author_type != "User" or association not in _TRUSTED_HUMAN_ASSOCIATIONS:
        raise WorkflowPreconditionError(
            f"human message author {login!r} must be an OWNER, MEMBER, or "
            "COLLABORATOR User"
        )


def _distinct_results(
    matches: tuple[_ResultComment, ...],
) -> tuple[ExperimentResult, ...]:
    distinct: dict[str, ExperimentResult] = {}
    for match in matches:
        distinct.setdefault(experiment_result_digest(match.result), match.result)
    return tuple(distinct.values())


def _require_same_result(
    expected: ExperimentResult,
    current: ExperimentResult,
    *,
    phase: str,
) -> None:
    if experiment_result_digest(expected) != experiment_result_digest(current):
        raise ReconciliationError(f"terminal result changed {phase}")


def _require_head(snapshot: PullRequestSnapshot, expected_head_sha: str) -> None:
    if not expected_head_sha:
        raise ValueError("expected_head_sha must not be empty")
    if snapshot.head_sha != expected_head_sha:
        raise PullHeadMismatchError(
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
) -> AssignmentRecord:
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
    mismatches = [
        name
        for name, current, proposed in (
            ("repo", record.repo, assignment.repo),
            ("assignment_id", record.assignment_id, assignment.assignment_id),
            ("revision_id", record.revision_id, assignment.revision_id),
            ("student", record.student, assignment.student),
            ("head_ref", record.head_ref, snapshot.head_ref),
            ("base_ref", record.base_ref, snapshot.base_ref),
        )
        if current != proposed
    ]
    if mismatches:
        error_type = (
            StaleAssignmentRevisionError
            if record.revision_id != assignment.revision_id
            else WorkflowPreconditionError
        )
        raise error_type(
            "terminal result assignment mismatch "
            f"({', '.join(mismatches)}). Current PR #{snapshot.number} marker: "
            f"revision={record.revision_id!r}, student={record.student!r}, "
            f"head={record.head_sha!r}; result: "
            f"revision={assignment.revision_id!r}, "
            f"student={assignment.student!r}. Refresh PR #{snapshot.number} and "
            "rebuild the result from its current assignment marker before retrying."
        )
    return record


def _require_exact_research_base(
    assignment: AssignmentRecord,
    *,
    live_base_sha: str,
    expected_current_base_sha: str,
) -> None:
    if not expected_current_base_sha.strip():
        raise ValueError("expected_current_base_sha must not be empty")
    if live_base_sha != expected_current_base_sha:
        raise StaleResearchBaseError(
            f"expected research base {assignment.base_ref}@"
            f"{expected_current_base_sha}, but live base is {live_base_sha}"
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


def _require_unmerged(snapshot: PullRequestSnapshot) -> None:
    if snapshot.merged:
        raise WorkflowPreconditionError(
            "pull request must be unmerged before it can be closed"
        )


def _require_current_revision(
    assignment: AssignmentRecord,
    revision_id: str,
) -> None:
    if assignment.revision_id != revision_id:
        raise StaleAssignmentRevisionError(
            f"assignment revision is {assignment.revision_id!r}, "
            f"expected {revision_id!r}"
        )


def _require_active_assignment_routing(
    snapshot: PullRequestSnapshot,
    assignment: AssignmentRecord,
) -> None:
    labels = set(snapshot.labels)
    student_labels = {label for label in labels if label.startswith("student:")}
    if student_labels != {f"student:{assignment.student}"}:
        raise WorkflowPreconditionError(
            "pull request must retain exactly its assigned student label"
        )
    if "status:wip" not in labels or "status:review" in labels:
        raise WorkflowPreconditionError(
            "pull request must have status:wip as its only active assignment status"
        )


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


_ROLE_COMMENT_PREFIX = re.compile(r"^(?:ADVISOR|STUDENT(?: [^:\s]+)?):[ \t]*")


def _role_prefixed_comment(
    body: str,
    role: Literal["advisor", "student"],
) -> str:
    marker, separator, content = body.partition("\n\n")
    if not (separator and marker.startswith("<!-- senpai-")):
        marker, separator, content = "", "", body
    content = _ROLE_COMMENT_PREFIX.sub("", content)
    return f"{marker}{separator}{role.upper()}: {content}"


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
