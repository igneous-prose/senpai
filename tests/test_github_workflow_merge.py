import pytest

from senpai_agent.github_workflow import (
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleBaselineError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker, render_result_comment
from github_workflow_support import (
    ASSIGNMENT_ID,
    HEAD_SHA,
    REPO,
    AmbiguousMutationGitHub,
    FakeGitHub,
    assignment_record,
    comment,
    experiment_result,
    pull_request,
    workflow,
)


def mergeable_pull(**overrides):
    pr = pull_request(labels={"status:review"}, draft=False, mergeable=True)
    pr.update(overrides)
    return pr


def result_comment():
    return comment(1, render_result_comment(experiment_result()))


def merge_experiment(
    client,
    *,
    expected_head_sha: str = HEAD_SHA,
    accepted_base_sha: str | None = None,
):
    return client.merge_experiment(
        7,
        expected_head_sha=expected_head_sha,
        assignment_id=ASSIGNMENT_ID,
        accepted_base_sha=accepted_base_sha,
    )


def test_merge_sends_the_expected_head_and_replays_without_baseline_reads():
    fake = FakeGitHub(mergeable_pull(), comments=[result_comment()])
    client = workflow(fake)

    first = merge_experiment(client)
    mutations_after_first = list(fake.mutations)
    base_reads_after_first = sum(
        method == "GET" and "/git/ref/heads/" in path
        for method, path, _body, _headers in fake.requests
    )
    second = merge_experiment(client)

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
    assert sum(
        method == "GET" and "/git/ref/heads/" in path
        for method, path, _body, _headers in fake.requests
    ) == base_reads_after_first


def test_merge_recovers_when_the_success_response_is_lost():
    fake = AmbiguousMutationGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        fail_method="PUT",
        fail_path=f"/repos/{REPO}/pulls/7/merge",
    )

    merged = merge_experiment(workflow(fake))

    assert fake.failed is True
    assert merged.changed is True
    assert merged.version == "merge-sha"
    assert fake.pr["merged"] is True


@pytest.mark.parametrize(
    ("accepted_base_sha", "message"),
    [(None, "advanced"), ("d" * 40, "does not match")],
    ids=("unacknowledged", "wrong-acknowledgement"),
)
def test_merge_rejects_baseline_drift_without_exact_live_acceptance(
    accepted_base_sha,
    message,
):
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )

    with pytest.raises(StaleBaselineError, match=message):
        merge_experiment(
            workflow(fake),
            accepted_base_sha=accepted_base_sha,
        )

    assert fake.mutations == []


def test_merge_accepts_baseline_drift_only_at_the_exact_live_sha():
    current_base_sha = "c" * 40
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[result_comment()],
        branch_heads={"schmidhuber": current_base_sha},
    )

    result = merge_experiment(
        workflow(fake),
        accepted_base_sha=current_base_sha,
    )

    assert result.state == "experiment_merged"
    assert fake.pr["merged"] is True


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
        ({"mergeable": None}, True, HEAD_SHA, "unknown"),
        ({"state": "closed"}, True, HEAD_SHA, "open"),
        ({}, False, HEAD_SHA, "terminal result"),
    ],
    ids=(
        "stale-head",
        "draft",
        "blocking-label",
        "missing-review-label",
        "merge-conflict",
        "unknown-mergeability",
        "closed-unmerged",
        "missing-result",
    ),
)
def test_merge_rejects_unsafe_state_or_missing_evidence_before_writing(
    pr_overrides,
    has_result,
    expected_head_sha,
    message,
):
    fake = FakeGitHub(
        mergeable_pull(**pr_overrides),
        comments=[result_comment()] if has_result else [],
    )

    with pytest.raises(WorkflowPreconditionError, match=message):
        merge_experiment(
            workflow(fake),
            expected_head_sha=expected_head_sha,
        )

    assert fake.mutations == []


def test_merge_rejects_a_result_for_an_older_assignment_revision():
    current_assignment = assignment_record(revision_id="revision-2")
    fake = FakeGitHub(
        mergeable_pull(body=render_assignment_marker(current_assignment)),
        comments=[result_comment()],
    )

    with pytest.raises(StaleAssignmentRevisionError, match="revision_id"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_rejects_a_result_for_an_older_head():
    stale = experiment_result(commit_sha="b" * 40)
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[comment(1, render_result_comment(stale))],
    )

    with pytest.raises(WorkflowPreconditionError, match="result commit"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_does_not_treat_assignment_prose_as_terminal_evidence():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[comment(1, f"prose mentions {ASSIGNMENT_ID}")],
    )

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


@pytest.mark.parametrize(
    ("comments", "match"),
    [
        (
            [comment(1, '<!-- senpai-result:v2 {"assignment_id":"assignment-7"} -->')],
            "result marker",
        ),
        (
            [result_comment(), comment(2, render_result_comment(experiment_result()))],
            "multiple",
        ),
    ],
    ids=("malformed", "duplicate"),
)
def test_merge_fails_closed_on_invalid_trusted_result_markers(comments, match):
    fake = FakeGitHub(mergeable_pull(), comments=comments)

    with pytest.raises(ReconciliationError, match=match):
        merge_experiment(workflow(fake))

    assert fake.mutations == []


def test_merge_ignores_a_result_marker_from_an_untrusted_author():
    fake = FakeGitHub(
        mergeable_pull(),
        comments=[
            comment(
                1,
                render_result_comment(experiment_result()),
                author="untrusted-user",
            )
        ],
    )

    with pytest.raises(WorkflowPreconditionError, match="terminal result"):
        merge_experiment(workflow(fake))

    assert fake.mutations == []
