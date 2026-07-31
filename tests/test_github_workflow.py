import pickle
from dataclasses import FrozenInstanceError
from typing import ClassVar, cast
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr

from senpai_agent.git_workflow import PushResult
from senpai_agent.github_workflow import (
    GitHubAPIError,
    GitHubTransportError,
    GitHubWorkflow,
    HttpResponse,
    PullRequestSnapshot,
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import (
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
    RevisionRecord,
    WandbRunRef,
    render_assignment_marker,
    render_result_comment,
    render_revision_marker,
)
from senpai_agent.tools import (
    GitHubTransitionAction,
    GitHubTransitionTool,
    SubmitResultTransition,
)

REPO = "acme/widgets"
API_URL = "https://api.github.test"
HEAD_SHA = "a" * 40
ASSIGNMENT_ID = "assignment-7"


def pull_request(
    *,
    labels: set[str] | None = None,
    draft: bool = False,
    state: str = "open",
    merged: bool = False,
    mergeable: bool | None = True,
    title: str = "Try lower learning rate",
    body: str | None = None,
    base_ref: str = "schmidhuber",
    head_ref: str = "student-one/lower-lr",
    head_sha: str = HEAD_SHA,
) -> dict[str, object]:
    if body is None:
        body = render_assignment_marker(
            AssignmentRecord(
                repo=REPO,
                assignment_id=ASSIGNMENT_ID,
                revision_id="revision-1",
                student="student-one",
                base_ref=base_ref,
                base_sha="b" * 40,
                head_ref=head_ref,
                head_sha=HEAD_SHA,
            )
        )
    return {
        "number": 7,
        "node_id": "PR_node_7",
        "html_url": f"https://github.com/{REPO}/pull/7",
        "head_sha": head_sha,
        "labels": set(labels or {"student:one", "status:wip"}),
        "draft": draft,
        "state": state,
        "merged": merged,
        "mergeable": mergeable,
        "merge_commit_sha": "merge-sha" if merged else None,
        "title": title,
        "body": body,
        "base_ref": base_ref,
        "head_ref": head_ref,
    }


def comment(
    comment_id: int,
    body: str,
    *,
    author: str = "senpai-bot",
) -> dict[str, object]:
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": author},
        "html_url": (f"https://github.com/{REPO}/pull/7#issuecomment-{comment_id}"),
    }


def human_issue(
    *,
    issue_id: int = 700,
    state: str = "open",
    labels: set[str] | None = None,
    author: str = "human-researcher",
    pull_request_url: str | None = None,
) -> dict[str, object]:
    issue: dict[str, object] = {
        "id": issue_id,
        "number": 7,
        "html_url": f"https://github.com/{REPO}/issues/7",
        "state": state,
        "body": "Please investigate the new result.",
        "labels": [
            {"name": label}
            for label in sorted(labels if labels is not None else {"human", "team"})
        ],
        "user": {"login": author},
    }
    if pull_request_url is not None:
        issue["pull_request"] = {"url": pull_request_url}
    return issue


def experiment_result(
    *,
    commit_sha: str = HEAD_SHA,
    repo: str = REPO,
    pr_number: int = 7,
    expected_head_sha: str = HEAD_SHA,
) -> ExperimentResult:
    return ExperimentResult(
        assignment=AssignmentKey(
            repo=repo,
            pr_number=pr_number,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            expected_head_sha=expected_head_sha,
            student="student-one",
        ),
        status=ResultStatus.SUCCEEDED,
        hypothesis="The candidate improves the primary metric.",
        summary="Terminal result with complete W&B evidence.",
        runs=(
            WandbRunRef(
                run_id="run-123",
                url="https://wandb.ai/acme/project/runs/run-123",
                state="finished",
            ),
        ),
        primary_metric=MetricComparison(
            name="val/loss",
            direction="minimize",
            baseline=0.42,
            candidate=0.38,
            delta=-0.04,
        ),
        commit_sha=commit_sha,
    )


class FakeGitHub:
    def __init__(
        self,
        pr: dict[str, object],
        *,
        comments: list[dict[str, object]] | None = None,
        issue: dict[str, object] | None = None,
        comment_page_size: int = 100,
        ignore_label_put: bool = False,
    ):
        self.pr = pr
        self.comments = list(comments or [])
        self.issue = issue
        self.comment_page_size = comment_page_size
        self.ignore_label_put = ignore_label_put
        self.next_heads: list[str] = []
        self.requests: list[tuple[str, str, object | None, dict[str, str]]] = []

    @property
    def mutations(self) -> list[tuple[str, str, object | None]]:
        return [
            (method, path, body)
            for method, path, body, _headers in self.requests
            if method != "GET"
        ]

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse:
        parsed = urlsplit(url)
        path = parsed.path
        self.requests.append((method, path, json_body, dict(headers)))

        pull_path = f"/repos/{REPO}/pulls/7"
        pulls_path = f"/repos/{REPO}/pulls"
        issues_path = f"/repos/{REPO}/issues"
        comments_path = f"/repos/{REPO}/issues/7/comments"
        labels_path = f"/repos/{REPO}/issues/7/labels"
        issue_path = f"/repos/{REPO}/issues/7"

        if method == "GET" and path == pull_path:
            if self.next_heads:
                self.pr["head_sha"] = self.next_heads.pop(0)
            return HttpResponse(200, self._pull_payload())

        if method == "GET" and path == issue_path and self.issue is not None:
            return HttpResponse(200, self.issue)

        if method == "GET" and path == pulls_path:
            return HttpResponse(
                200,
                [self._pull_payload()]
                if self.pr.get("number")
                and self.pr["head_ref"]
                == (parse_qs(parsed.query).get("head", [":"])[0].split(":", 1)[-1])
                else [],
            )

        if method == "GET" and path == issues_path:
            labels = set(parse_qs(parsed.query).get("labels", [""])[0].split(","))
            return HttpResponse(
                200,
                (
                    [
                        {
                            "number": self.pr["number"],
                            "pull_request": {"url": self.pr["html_url"]},
                        }
                    ]
                    if labels.issubset(cast(set[str], self.pr["labels"]))
                    else []
                ),
            )

        if method == "POST" and path == pulls_path:
            payload = cast(dict[str, object], json_body)
            self.pr.update(
                {
                    "number": 7,
                    "title": payload["title"],
                    "body": payload["body"],
                    "base_ref": payload["base"],
                    "head_ref": payload["head"],
                    "draft": payload["draft"],
                    "state": "open",
                }
            )
            return HttpResponse(201, self._pull_payload())

        if method == "GET" and path == comments_path:
            page = int(parse_qs(parsed.query).get("page", ["1"])[0])
            start = (page - 1) * self.comment_page_size
            end = start + self.comment_page_size
            headers_out: tuple[tuple[str, str], ...] = ()
            if end < len(self.comments):
                next_url = f"{API_URL}{comments_path}?per_page=100&page={page + 1}"
                headers_out = (("Link", f'<{next_url}>; rel="next"'),)
            return HttpResponse(200, self.comments[start:end], headers_out)

        if method == "POST" and path == comments_path:
            body = cast(dict[str, str], json_body)["body"]
            created = comment(
                max((int(item["id"]) for item in self.comments), default=0) + 1,
                body,
            )
            self.comments.append(created)
            return HttpResponse(201, created)

        comment_prefix = f"/repos/{REPO}/issues/comments/"
        if method == "PATCH" and path.startswith(comment_prefix):
            comment_id = int(path.removeprefix(comment_prefix))
            body = cast(dict[str, str], json_body)["body"]
            existing = next(item for item in self.comments if item["id"] == comment_id)
            existing["body"] = body
            return HttpResponse(200, existing)

        if method == "PUT" and path == labels_path:
            labels = set(cast(dict[str, list[str]], json_body)["labels"])
            if not self.ignore_label_put:
                self.pr["labels"] = labels
            return HttpResponse(
                200,
                [{"name": label} for label in sorted(labels)],
            )

        if method == "POST" and path == "/graphql":
            request = cast(dict[str, object], json_body)
            query = cast(str, request["query"])
            if "convertPullRequestToDraft" in query:
                self.pr["draft"] = True
                field = "convertPullRequestToDraft"
            elif "markPullRequestReadyForReview" in query:
                self.pr["draft"] = False
                field = "markPullRequestReadyForReview"
            else:
                raise AssertionError(f"Unexpected GraphQL mutation: {query}")
            return HttpResponse(
                200,
                {
                    "data": {
                        field: {
                            "pullRequest": {
                                "id": self.pr["node_id"],
                                "isDraft": self.pr["draft"],
                            }
                        }
                    }
                },
            )

        if method == "PATCH" and path == pull_path:
            update = cast(dict[str, object], json_body)
            if "state" in update:
                self.pr["state"] = update["state"]
            for field in ("title", "body"):
                if field in update:
                    self.pr[field] = update[field]
            return HttpResponse(200, self._pull_payload())

        if method == "PUT" and path == f"{pull_path}/merge":
            request = cast(dict[str, str], json_body)
            assert request["sha"] == self.pr["head_sha"]
            self.pr["state"] = "closed"
            self.pr["merged"] = True
            self.pr["merge_commit_sha"] = "merge-sha"
            return HttpResponse(
                200,
                {
                    "merged": True,
                    "sha": "merge-sha",
                    "message": "Pull Request successfully merged",
                },
            )

        raise AssertionError(f"Unexpected request: {method} {url} {json_body!r}")

    def _pull_payload(self) -> dict[str, object]:
        return {
            "number": self.pr["number"],
            "node_id": self.pr["node_id"],
            "html_url": self.pr["html_url"],
            "base": {"ref": self.pr["base_ref"]},
            "head": {
                "sha": self.pr["head_sha"],
                "ref": self.pr["head_ref"],
            },
            "title": self.pr["title"],
            "body": self.pr["body"],
            "labels": [
                {"name": label} for label in sorted(cast(set[str], self.pr["labels"]))
            ],
            "draft": self.pr["draft"],
            "state": self.pr["state"],
            "merged": self.pr["merged"],
            "mergeable": self.pr["mergeable"],
            "merge_commit_sha": self.pr["merge_commit_sha"],
        }


class AmbiguousMutationGitHub(FakeGitHub):
    def __init__(self, *args, fail_method: str, fail_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_method = fail_method
        self.fail_path = fail_path
        self.failed = False

    def request(self, method, url, *, headers, json_body=None):
        response = super().request(
            method,
            url,
            headers=headers,
            json_body=json_body,
        )
        if (
            not self.failed
            and method == self.fail_method
            and urlsplit(url).path == self.fail_path
        ):
            self.failed = True
            raise GitHubTransportError(method, url)
        return response


def workflow(
    fake: FakeGitHub,
) -> GitHubWorkflow:
    return GitHubWorkflow(
        REPO,
        SecretStr("github-secret"),
        transport=fake,
        api_url=API_URL,
        trusted_actor="senpai-bot",
    )


def test_pull_request_returns_an_immutable_compact_snapshot_without_secret_leakage():
    fake = FakeGitHub(pull_request(labels={"status:wip", "student:one"}))
    client = workflow(fake)

    snapshot = client.pull_request(7)

    assert snapshot == PullRequestSnapshot(
        number=7,
        node_id="PR_node_7",
        url=f"https://github.com/{REPO}/pull/7",
        head_sha=HEAD_SHA,
        base_ref="schmidhuber",
        head_ref="student-one/lower-lr",
        title="Try lower learning rate",
        body=cast(str, fake.pr["body"]),
        labels=("status:wip", "student:one"),
        draft=False,
        state="open",
        merged=False,
        mergeable=True,
        merge_commit_sha=None,
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.state = "closed"  # type: ignore[misc]
    assert "github-secret" not in repr(client)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(client)
    assert fake.requests[0][3]["Authorization"] == "Bearer github-secret"


def test_respond_to_human_issue_is_verified_and_idempotent():
    fake = FakeGitHub(pull_request(), issue=human_issue())
    client = workflow(fake)

    first = client.respond_to_issue(
        7,
        human_message_id=700,
        response="ADVISOR: I will investigate this now.",
    )
    second = client.respond_to_issue(
        7,
        human_message_id=700,
        response="ADVISOR: I will investigate this now.",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.state == "issue_response_upserted"
    assert first.resource_url.endswith("#issuecomment-1")
    assert len(fake.comments) == 1
    assert fake.comments[0]["body"] == (
        "<!-- senpai-human-response:700 -->\n\nADVISOR: I will investigate this now."
    )


def test_respond_to_human_issue_accepts_a_specific_human_comment():
    fake = FakeGitHub(
        pull_request(),
        issue=human_issue(),
        comments=[comment(42, "Please also compare memory use.", author="ada")],
    )

    result = workflow(fake).respond_to_issue(
        7,
        human_message_id=42,
        response="STUDENT fern: I included memory in the comparison.",
    )

    assert result.changed is True
    assert len(fake.comments) == 2
    assert "<!-- senpai-human-response:42 -->" in cast(
        str,
        fake.comments[-1]["body"],
    )


@pytest.mark.parametrize(
    ("issue", "comments", "message_id", "match"),
    [
        (
            human_issue(state="closed"),
            [],
            700,
            "must be open",
        ),
        (
            human_issue(labels={"team"}),
            [],
            700,
            "human",
        ),
        (
            human_issue(
                pull_request_url=f"https://api.github.test/repos/{REPO}/pulls/7"
            ),
            [],
            700,
            "pull request",
        ),
        (
            human_issue(author="senpai-bot"),
            [],
            700,
            "authenticated actor",
        ),
        (
            human_issue(),
            [comment(42, "Already answered.", author="senpai-bot")],
            42,
            "authenticated actor",
        ),
        (
            human_issue(),
            [],
            999,
            "not present",
        ),
    ],
)
def test_respond_to_human_issue_rejects_unsafe_sources_before_mutation(
    issue,
    comments,
    message_id,
    match,
):
    fake = FakeGitHub(
        pull_request(),
        issue=issue,
        comments=comments,
    )

    with pytest.raises(WorkflowPreconditionError, match=match):
        workflow(fake).respond_to_issue(
            7,
            human_message_id=message_id,
            response="ADVISOR: bounded response",
        )

    assert fake.mutations == []


def test_create_assignment_reconciles_one_draft_pr_and_replays():
    assignment = AssignmentRecord(
        repo=REPO,
        assignment_id=ASSIGNMENT_ID,
        revision_id="revision-1",
        student="student-one",
        base_ref="schmidhuber",
        base_sha="b" * 40,
        head_ref="student-one/lower-lr",
        head_sha=HEAD_SHA,
    )
    fake = FakeGitHub(
        pull_request(
            labels=set(),
            title="",
            body="",
        )
    )
    fake.pr["number"] = 0
    client = workflow(fake)

    first = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )
    second = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )

    expected_body = (
        f"{render_assignment_marker(assignment)}\n\n"
        "Run the bounded learning-rate experiment."
    )
    assert first.changed is True
    assert second.changed is False
    assert fake.pr["body"] == expected_body
    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {
        "schmidhuber",
        "student:student-one",
        "status:wip",
    }


def test_create_assignment_rejects_a_same_student_wip_on_another_base():
    assignment = AssignmentRecord(
        repo=REPO,
        assignment_id="assignment-8",
        revision_id="revision-1",
        student="student-one",
        base_ref="schmidhuber",
        base_sha="b" * 40,
        head_ref="student-one/new-candidate",
        head_sha="c" * 40,
    )
    fake = FakeGitHub(
        pull_request(
            labels={"other-base", "student:student-one", "status:wip"},
            base_ref="other-base",
            head_ref="student-one/other-candidate",
        )
    )

    with pytest.raises(WorkflowPreconditionError, match="already has active"):
        workflow(fake).create_assignment(
            assignment,
            title="Try another candidate",
            body="Run one bounded comparison.",
        )

    assert fake.mutations == []


def test_client_rejects_ambiguous_configuration_before_any_request():
    with pytest.raises(ValueError, match="owner/name"):
        GitHubWorkflow("widgets", SecretStr("token"))
    with pytest.raises(TypeError, match="SecretStr"):
        GitHubWorkflow(REPO, cast(SecretStr, "raw-token"))


def test_reconcile_labels_puts_one_exact_desired_set_and_replay_is_a_noop():
    fake = FakeGitHub(pull_request(labels={"student:one", "status:wip", "keep"}))
    client = workflow(fake)

    first = client.reconcile_labels(
        7,
        assignment_id=ASSIGNMENT_ID,
        add={"status:review"},
        remove={"status:wip"},
        expected_head_sha=HEAD_SHA,
    )
    mutations_after_first = list(fake.mutations)
    second = client.reconcile_labels(
        7,
        assignment_id=ASSIGNMENT_ID,
        add={"status:review"},
        remove={"status:wip"},
        expected_head_sha=HEAD_SHA,
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["labels"] == {"student:one", "status:review", "keep"}
    assert mutations_after_first == [
        (
            "PUT",
            f"/repos/{REPO}/issues/7/labels",
            {"labels": ["keep", "status:review", "student:one"]},
        )
    ]
    assert fake.mutations == mutations_after_first


def test_reconcile_labels_fails_stale_head_before_mutation_and_verifies_the_result():
    stale = FakeGitHub(pull_request())
    with pytest.raises(WorkflowPreconditionError, match="head SHA"):
        workflow(stale).reconcile_labels(
            7,
            assignment_id=ASSIGNMENT_ID,
            add={"status:review"},
            remove={"status:wip"},
            expected_head_sha="b" * 40,
        )
    assert stale.mutations == []

    ignored = FakeGitHub(pull_request(), ignore_label_put=True)
    with pytest.raises(ReconciliationError, match="label set"):
        workflow(ignored).reconcile_labels(
            7,
            assignment_id=ASSIGNMENT_ID,
            add={"status:review"},
            remove={"status:wip"},
            expected_head_sha=HEAD_SHA,
        )
    assert len(ignored.mutations) == 1


def test_reconcile_labels_rejects_the_wrong_assignment_before_mutation():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        workflow(fake).reconcile_labels(
            7,
            assignment_id="other-assignment",
            add={"status:review"},
            remove={"status:wip"},
            expected_head_sha=HEAD_SHA,
        )

    assert fake.mutations == []


def test_request_revision_reconciles_comment_draft_and_labels_once():
    marker = render_revision_marker(
        RevisionRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-2",
            requested_head_sha=HEAD_SHA,
        )
    )
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False)
    )
    client = workflow(fake)

    first = client.request_revision(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        revision_id="revision-2",
        comment="Run the requested ablation.",
    )
    mutations_after_first = list(fake.mutations)
    requests_after_first = len(fake.requests)
    second = client.request_revision(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        revision_id="revision-2",
        comment="Run the requested ablation.",
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {"student:one", "status:wip"}
    assert fake.comments[0]["body"] == (f"{marker}\n\nRun the requested ablation.")
    assert [mutation[0] for mutation in mutations_after_first] == [
        "POST",
        "PATCH",
        "POST",
        "PUT",
    ]
    assert (
        "convertPullRequestToDraft"
        in cast(dict[str, str], mutations_after_first[2][2])["query"]
    )
    assert fake.mutations == mutations_after_first
    assert requests_after_first == 9
    assert len(fake.requests) - requests_after_first == 4


def test_request_revision_updates_a_trusted_marker_from_the_final_comment_page():
    marker = render_revision_marker(
        RevisionRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-2",
            requested_head_sha=HEAD_SHA,
        )
    )
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[
            comment(1, "unrelated"),
            comment(2, f"{marker}\n\nOld instructions."),
        ],
        comment_page_size=1,
    )

    workflow(fake).request_revision(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        revision_id="revision-2",
        comment="Run the requested ablation.",
    )

    assert [item["body"] for item in fake.comments] == [
        "unrelated",
        f"{marker}\n\nRun the requested ablation.",
    ]
    assert (
        "PATCH",
        f"/repos/{REPO}/issues/comments/2",
        {"body": f"{marker}\n\nRun the requested ablation."},
    ) in fake.mutations


def test_request_revision_ignores_spoofed_and_non_exact_markers():
    marker = render_revision_marker(
        RevisionRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-2",
            requested_head_sha=HEAD_SHA,
        )
    )
    spoofed = f"{marker}\n\nUntrusted instructions."
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[
            comment(1, spoofed, author="untrusted-user"),
            comment(2, f"Documentation example: {marker}"),
            comment(3, f"> {marker}"),
        ],
    )

    workflow(fake).request_revision(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        revision_id="revision-2",
        comment="Use the trusted revision.",
    )

    assert [item["body"] for item in fake.comments] == [
        spoofed,
        f"Documentation example: {marker}",
        f"> {marker}",
        f"{marker}\n\nUse the trusted revision.",
    ]
    assert any(
        method == "POST" and path == f"/repos/{REPO}/issues/7/comments"
        for method, path, _body in fake.mutations
    )


def test_request_revision_rejects_duplicate_trusted_marker_comments():
    marker = render_revision_marker(
        RevisionRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-2",
            requested_head_sha=HEAD_SHA,
        )
    )
    desired = f"{marker}\n\nRun the requested ablation."
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False),
        comments=[comment(1, desired), comment(2, desired)],
    )

    with pytest.raises(ReconciliationError, match="multiple comments"):
        workflow(fake).request_revision(
            7,
            assignment_id=ASSIGNMENT_ID,
            expected_head_sha=HEAD_SHA,
            revision_id="revision-2",
            comment="Run the requested ablation.",
        )

    assert fake.mutations == []


def test_request_revision_rejects_the_wrong_assignment_before_mutation():
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False)
    )

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        workflow(fake).request_revision(
            7,
            assignment_id="other-assignment",
            expected_head_sha=HEAD_SHA,
            revision_id="revision-2",
            comment="This must not affect another assignment.",
        )

    assert fake.mutations == []
    assert fake.comments == []


def test_submit_result_reconciles_ready_state_as_durable_github_mail():
    fake = FakeGitHub(pull_request(labels={"student:one", "status:wip"}, draft=True))
    result = experiment_result()
    client = workflow(fake)

    first = client.submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=result,
    )
    mutations_after_first = list(fake.mutations)
    requests_after_first = len(fake.requests)
    second = client.submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=result,
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["draft"] is False
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert [mutation[0] for mutation in mutations_after_first] == [
        "POST",
        "POST",
        "PUT",
    ]
    assert (
        "markPullRequestReadyForReview"
        in cast(dict[str, str], mutations_after_first[1][2])["query"]
    )
    assert fake.mutations == mutations_after_first
    assert requests_after_first == 8
    assert len(fake.requests) - requests_after_first == 4


def test_submit_result_preflight_validates_before_the_student_pushes():
    final_head = "c" * 40
    fake = FakeGitHub(pull_request(head_sha=HEAD_SHA, draft=True))
    client = workflow(fake)
    result = experiment_result(
        commit_sha=final_head,
        expected_head_sha=final_head,
    )

    snapshot = client.preflight_submit_result(
        7,
        branch="student-one/lower-lr",
        current_head_sha=HEAD_SHA,
        expected_result_head_sha=final_head,
        result=result,
    )

    assert snapshot.head_sha == HEAD_SHA
    assert fake.mutations == []


def test_submit_result_preflight_rejects_wrong_assignment_without_mutation():
    fake = FakeGitHub(pull_request(draft=True))
    result = experiment_result().model_copy(
        update={
            "assignment": experiment_result().assignment.model_copy(
                update={"assignment_id": "other-assignment"}
            )
        }
    )

    with pytest.raises(WorkflowPreconditionError) as raised:
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/lower-lr",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=result,
        )

    message = str(raised.value)
    assert "assignment mismatch (assignment_id)" in message
    assert (
        "Current PR #7 marker: revision='revision-1', student='student-one', "
        f"head='{HEAD_SHA}'"
        in message
    )
    assert "result: revision='revision-1', student='student-one'" in message
    assert "Refresh PR #7" in message
    assert fake.mutations == []


def test_submit_result_preflight_identifies_a_stale_assignment_revision():
    current = AssignmentRecord(
        repo=REPO,
        assignment_id=ASSIGNMENT_ID,
        revision_id="revision-2",
        student="student-one",
        base_ref="schmidhuber",
        base_sha="b" * 40,
        head_ref="student-one/lower-lr",
        head_sha=HEAD_SHA,
    )
    fake = FakeGitHub(
        pull_request(body=render_assignment_marker(current), draft=True)
    )

    with pytest.raises(StaleAssignmentRevisionError) as raised:
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/lower-lr",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=experiment_result(),
        )

    message = str(raised.value)
    assert "assignment mismatch (revision_id)" in message
    assert "Current PR #7 marker: revision='revision-2'" in message
    assert "result: revision='revision-1', student='student-one'" in message
    assert "Refresh PR #7" in message
    assert fake.mutations == []


def test_submit_result_preflight_rejects_a_foreign_branch_before_mutation():
    fake = FakeGitHub(pull_request(draft=True))

    with pytest.raises(WorkflowPreconditionError, match="branch"):
        workflow(fake).preflight_submit_result(
            7,
            branch="student-one/unrelated",
            current_head_sha=HEAD_SHA,
            expected_result_head_sha=HEAD_SHA,
            result=experiment_result(),
        )

    assert fake.mutations == []


def test_submit_result_waits_for_the_pushed_head_to_reach_github(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    final_head = "c" * 40
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True)
    )
    sleeps: list[float] = []
    pushes: list[str] = []

    def push(_workspace, **kwargs):
        pushes.append(kwargs["expected_local_sha"])
        fake.next_heads = [HEAD_SHA, HEAD_SHA, kwargs["expected_local_sha"]]
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    monkeypatch.setattr("senpai_agent.tools.push_assignment_branch", push)
    monkeypatch.setattr("senpai_agent.tools.time.sleep", sleeps.append)
    transition = GitHubTransitionTool.create(
        workflow=workflow(fake),
        role="student",
        workspace=tmp_path,
    )[0]

    observation = transition(
        GitHubTransitionAction(
            transition=SubmitResultTransition(
                operation="submit_result",
                pr_number=7,
                branch="student-one/lower-lr",
                expected_remote_sha=HEAD_SHA,
                expected_head_sha=final_head,
                result=experiment_result(
                    commit_sha=final_head,
                    expected_head_sha=final_head,
                ),
            )
        )
    )

    assert observation.state == "result_submitted"
    assert pushes == [final_head]
    assert sleeps == [0.5, 1.0]
    assert fake.pr["head_sha"] == final_head
    assert fake.pr["labels"] == {"student:one", "status:review"}
    assert fake.pr["draft"] is False
    assert len(fake.comments) == 1
    assert "<!-- senpai-result:v1 " in str(fake.comments[0]["body"])


def test_submit_result_recovers_in_call_from_an_ambiguous_comment_write():
    fake = AmbiguousMutationGitHub(
        pull_request(labels={"student:one", "status:wip"}, draft=True),
        fail_method="POST",
        fail_path=f"/repos/{REPO}/issues/7/comments",
    )

    result = workflow(fake).submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=experiment_result(),
    )

    assert result.state == "result_submitted"
    assert fake.failed is True
    assert fake.pr["labels"] == {"student:one", "status:review"}


def test_submit_result_keeps_initial_assignment_head_as_durable_identity():
    assignment = AssignmentRecord(
        repo=REPO,
        assignment_id=ASSIGNMENT_ID,
        revision_id="revision-1",
        student="student-one",
        base_ref="schmidhuber",
        base_sha="b" * 40,
        head_ref="student-one/lower-lr",
        head_sha="c" * 40,
    )
    fake = FakeGitHub(
        pull_request(
            labels={"student:one", "status:wip"},
            draft=True,
            body=render_assignment_marker(assignment),
        )
    )

    submitted = workflow(fake).submit_result(
        7,
        expected_head_sha=HEAD_SHA,
        result=experiment_result(),
    )

    assert submitted.state == "result_submitted"


@pytest.mark.parametrize(
    "result",
    [
        experiment_result(commit_sha="b" * 40),
        experiment_result(repo="other/widgets"),
        experiment_result(pr_number=8),
        experiment_result(expected_head_sha="b" * 40),
    ],
)
def test_submit_result_rejects_result_that_does_not_match_remote_assignment(
    result: ExperimentResult,
):
    fake = FakeGitHub(pull_request(labels={"student:one", "status:wip"}, draft=True))

    with pytest.raises(WorkflowPreconditionError, match="result"):
        workflow(fake).submit_result(
            7,
            expected_head_sha=HEAD_SHA,
            result=result,
        )

    assert fake.mutations == []
    assert fake.comments == []


def test_close_experiment_upserts_reason_closes_and_replays_without_mutation():
    marker = "<!-- senpai-disposition:v1 dead-end-7 -->"
    fake = FakeGitHub(pull_request(labels={"status:review"}))
    client = workflow(fake)

    first = client.close_experiment(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        marker=marker,
        reason="The hypothesis was falsified.",
    )
    mutations_after_first = list(fake.mutations)
    requests_after_first = len(fake.requests)
    second = client.close_experiment(
        7,
        assignment_id=ASSIGNMENT_ID,
        expected_head_sha=HEAD_SHA,
        marker=marker,
        reason="The hypothesis was falsified.",
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["state"] == "closed"
    assert [mutation[0] for mutation in mutations_after_first] == [
        "POST",
        "PATCH",
    ]
    assert fake.mutations == mutations_after_first
    assert requests_after_first == 7
    assert len(fake.requests) - requests_after_first == 4


def test_close_experiment_rejects_an_already_merged_pull_request():
    marker = "<!-- senpai-disposition:v1 dead-end-7 -->"
    fake = FakeGitHub(
        pull_request(
            labels={"status:review"},
            state="closed",
            merged=True,
        )
    )

    with pytest.raises(WorkflowPreconditionError, match="unmerged"):
        workflow(fake).close_experiment(
            7,
            assignment_id=ASSIGNMENT_ID,
            expected_head_sha=HEAD_SHA,
            marker=marker,
            reason="This must not overwrite a merged winner.",
        )

    assert fake.mutations == []
    assert fake.comments == []


def test_close_experiment_rejects_the_wrong_assignment_before_mutation():
    marker = "<!-- senpai-disposition:v1 dead-end-7 -->"
    fake = FakeGitHub(pull_request(labels={"status:review"}))

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        workflow(fake).close_experiment(
            7,
            assignment_id="other-assignment",
            expected_head_sha=HEAD_SHA,
            marker=marker,
            reason="This PR belongs to another assignment.",
        )

    assert fake.mutations == []
    assert fake.comments == []


def test_merge_experiment_sends_expected_head_and_replay_verifies_existing_merge():
    result = experiment_result()
    fake = FakeGitHub(
        pull_request(labels={"status:review"}, draft=False, mergeable=True),
        comments=[comment(1, render_result_comment(result))],
    )
    client = workflow(fake)

    first = client.merge_experiment(
        7,
        expected_head_sha=HEAD_SHA,
        assignment_id=ASSIGNMENT_ID,
    )
    mutations_after_first = list(fake.mutations)
    second = client.merge_experiment(
        7,
        expected_head_sha=HEAD_SHA,
        assignment_id=ASSIGNMENT_ID,
    )

    assert first.changed is True
    assert first.version == "merge-sha"
    assert second.changed is False
    assert fake.pr["state"] == "closed"
    assert fake.pr["merged"] is True
    assert mutations_after_first == [
        (
            "PUT",
            f"/repos/{REPO}/pulls/7/merge",
            {"sha": HEAD_SHA, "merge_method": "squash"},
        )
    ]
    assert fake.mutations == mutations_after_first


@pytest.mark.parametrize(
    ("pr_overrides", "has_result", "expected_head_sha", "message"),
    [
        ({}, True, "b" * 40, "head SHA"),
        ({"draft": True}, True, HEAD_SHA, "draft"),
        (
            {"labels": {"status:review", "status:hold"}},
            True,
            HEAD_SHA,
            "blocking label",
        ),
        ({"labels": {"student:one"}}, True, HEAD_SHA, "status:review"),
        ({"mergeable": False}, True, HEAD_SHA, "merge conflict"),
        ({}, False, HEAD_SHA, "terminal result"),
    ],
)
def test_merge_experiment_rejects_every_preflight_without_mutation(
    pr_overrides: dict[str, object],
    has_result: bool,
    expected_head_sha: str,
    message: str,
):
    pr = pull_request(labels={"status:review"})
    pr.update(pr_overrides)
    fake = FakeGitHub(
        pr,
        comments=(
            [comment(1, render_result_comment(experiment_result()))]
            if has_result
            else []
        ),
    )

    with pytest.raises(WorkflowPreconditionError, match=message):
        workflow(fake).merge_experiment(
            7,
            expected_head_sha=expected_head_sha,
            assignment_id=ASSIGNMENT_ID,
        )

    assert fake.mutations == []


def test_merge_requires_one_schema_valid_result_for_the_current_head():
    stale = FakeGitHub(
        pull_request(labels={"status:review"}),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result(commit_sha="b" * 40)),
            )
        ],
    )
    with pytest.raises(WorkflowPreconditionError, match="result commit"):
        workflow(stale).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
        )
    assert stale.mutations == []

    opaque = FakeGitHub(
        pull_request(labels={"status:review"}),
        comments=[comment(1, f"prose mentions {ASSIGNMENT_ID}")],
    )
    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        workflow(opaque).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
        )
    assert opaque.mutations == []


def test_merge_rejects_malformed_or_duplicate_result_markers():
    malformed = FakeGitHub(
        pull_request(labels={"status:review"}),
        comments=[
            comment(
                1,
                '<!-- senpai-result:v2 {"assignment_id":"assignment-7"} -->',
            )
        ],
    )
    with pytest.raises(ReconciliationError, match="result marker"):
        workflow(malformed).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
        )
    assert malformed.mutations == []

    body = render_result_comment(experiment_result())
    duplicate = FakeGitHub(
        pull_request(labels={"status:review"}),
        comments=[comment(1, body), comment(2, body)],
    )
    with pytest.raises(ReconciliationError, match="multiple"):
        workflow(duplicate).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
        )
    assert duplicate.mutations == []


def test_merge_ignores_terminal_results_written_by_an_untrusted_author():
    fake = FakeGitHub(
        pull_request(labels={"status:review"}),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result()),
                author="untrusted-user",
            )
        ],
    )

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        workflow(fake).merge_experiment(
            7,
            expected_head_sha=HEAD_SHA,
            assignment_id=ASSIGNMENT_ID,
        )


def test_api_errors_and_results_never_expose_the_github_token():
    class FailingTransport:
        def request(self, method, url, *, headers, json_body=None):
            return HttpResponse(503, {"message": "unavailable"})

    client = GitHubWorkflow(
        REPO,
        SecretStr("never-show-this"),
        transport=FailingTransport(),
        api_url=API_URL,
    )

    with pytest.raises(GitHubAPIError) as raised:
        client.pull_request(7)

    assert "never-show-this" not in str(raised.value)
    assert "never-show-this" not in repr(raised.value)
    assert "never-show-this" not in repr(client)


def test_default_transport_maps_invalid_json_and_network_failures_to_typed_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    class InvalidJSONResponse:
        status = 200
        headers: ClassVar[dict[str, str]] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"<html>not JSON</html>"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: InvalidJSONResponse(),
    )
    invalid_json_client = GitHubWorkflow(
        REPO,
        SecretStr("never-show-this"),
        api_url=API_URL,
    )
    with pytest.raises(ReconciliationError, match="pull request"):
        invalid_json_client.pull_request(7)

    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    offline_client = GitHubWorkflow(
        REPO,
        SecretStr("never-show-this"),
        api_url=API_URL,
    )
    with pytest.raises(GitHubTransportError) as raised:
        offline_client.pull_request(7)

    assert "never-show-this" not in str(raised.value)
    assert "never-show-this" not in repr(raised.value)
