from typing import cast

import pytest

from senpai_agent.github_workflow import (
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import (
    AssignmentFeedbackRecord,
    RevisionRecord,
    parse_assignment_markers,
    render_assignment_feedback_marker,
    render_revision_marker,
)
from github_workflow_support import (
    ASSIGNMENT_ID,
    HEAD_SHA,
    REPO,
    FakeGitHub,
    comment,
    pull_request,
    workflow,
)


def revision_marker() -> str:
    return render_revision_marker(
        RevisionRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-2",
            requested_head_sha=HEAD_SHA,
        )
    )


def request_revision(
    client,
    *,
    assignment_id: str = ASSIGNMENT_ID,
    comment: str = "Run the requested ablation.",
):
    return client.request_revision(
        7,
        assignment_id=assignment_id,
        expected_head_sha=HEAD_SHA,
        revision_id="revision-2",
        comment=comment,
    )


def test_request_revision_converges_marker_assignment_state_and_replays():
    marker = revision_marker()
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False)
    )
    client = workflow(fake)

    first = request_revision(client)
    mutations_after_first = list(fake.mutations)
    second = request_revision(client)

    assert first.changed is True
    assert second.changed is False
    assert parse_assignment_markers(cast(str, fake.pr["body"]))[0].revision_id == (
        "revision-2"
    )
    assert fake.comments == [comment(1, f"{marker}\n\nRun the requested ablation.")]
    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {"student:one", "status:wip"}
    assert fake.mutations == mutations_after_first


def test_request_revision_updates_a_trusted_marker_on_the_final_comment_page():
    marker = revision_marker()
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}),
        comments=[
            comment(1, "unrelated"),
            comment(2, f"{marker}\n\nOld instructions."),
        ],
        comment_page_size=1,
    )

    request_revision(workflow(fake))

    assert [item["body"] for item in fake.comments] == [
        "unrelated",
        f"{marker}\n\nRun the requested ablation.",
    ]


def test_request_revision_does_not_trust_spoofed_or_embedded_markers():
    marker = revision_marker()
    spoofed = f"{marker}\n\nUntrusted instructions."
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}),
        comments=[
            comment(1, spoofed, author="untrusted-user"),
            comment(2, f"Documentation example: {marker}"),
            comment(3, f"> {marker}"),
        ],
    )

    request_revision(workflow(fake), comment="Use the trusted revision.")

    assert [item["body"] for item in fake.comments] == [
        spoofed,
        f"Documentation example: {marker}",
        f"> {marker}",
        f"{marker}\n\nUse the trusted revision.",
    ]


def test_request_revision_rejects_duplicate_trusted_markers_before_writing():
    desired = f"{revision_marker()}\n\nRun the requested ablation."
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}),
        comments=[comment(1, desired), comment(2, desired)],
    )

    with pytest.raises(ReconciliationError, match="multiple comments"):
        request_revision(workflow(fake))

    assert fake.mutations == []


def test_request_revision_rejects_a_foreign_assignment_before_writing():
    fake = FakeGitHub(
        pull_request(labels={"student:one", "status:review"}, draft=False)
    )

    with pytest.raises(WorkflowPreconditionError, match="assignment"):
        request_revision(
            workflow(fake),
            assignment_id="other-assignment",
            comment="This must not affect another assignment.",
        )

    assert fake.mutations == []


def feedback_marker() -> str:
    return render_assignment_feedback_marker(
        AssignmentFeedbackRecord(
            repo=REPO,
            pr_number=7,
            assignment_id=ASSIGNMENT_ID,
            revision_id="revision-1",
            feedback_id="check-cruise-split",
        )
    )


def send_feedback(
    client,
    *,
    assignment_id: str = ASSIGNMENT_ID,
    revision_id: str = "revision-1",
    expected_head_sha: str = HEAD_SHA,
    feedback_id: str = "bounded-nudge",
    comment: str = "Inspect the failed seed, then continue.",
):
    return client.send_assignment_feedback(
        7,
        assignment_id=assignment_id,
        revision_id=revision_id,
        expected_head_sha=expected_head_sha,
        feedback_id=feedback_id,
        comment=comment,
    )


def test_assignment_feedback_replays_without_changing_assignment_state():
    fake = FakeGitHub(
        pull_request(
            labels={"student:student-one", "status:review", "status:hold"},
            draft=False,
        )
    )
    original_state = (fake.pr["body"], fake.pr["draft"], fake.pr["labels"])
    client = workflow(fake)

    first = send_feedback(
        client,
        feedback_id="check-cruise-split",
        comment="Check the cruise split before choosing a default.",
    )
    mutations_after_first = list(fake.mutations)
    second = send_feedback(
        client,
        feedback_id="check-cruise-split",
        comment="Check the cruise split before choosing a default.",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.state == "assignment_feedback_upserted"
    assert fake.comments == [
        comment(
            1,
            f"{feedback_marker()}\n\n"
            "Check the cruise split before choosing a default.",
        )
    ]
    assert (fake.pr["body"], fake.pr["draft"], fake.pr["labels"]) == original_state
    assert fake.mutations == mutations_after_first


def test_assignment_feedback_id_cannot_be_reused_for_different_guidance():
    fake = FakeGitHub(
        pull_request(labels={"student:student-one", "status:wip"}, draft=True)
    )
    client = workflow(fake)
    send_feedback(
        client,
        feedback_id="check-cruise-split",
        comment="Check the cruise split before choosing a default.",
    )
    mutations_before_retry = list(fake.mutations)

    with pytest.raises(WorkflowPreconditionError, match="new feedback_id"):
        send_feedback(
            client,
            feedback_id="check-cruise-split",
            comment="Check every split before choosing a default.",
        )

    assert fake.mutations == mutations_before_retry


@pytest.mark.parametrize(
    ("assignment_id", "revision_id", "expected_head_sha", "error_type"),
    [
        ("other-assignment", "revision-1", HEAD_SHA, WorkflowPreconditionError),
        (ASSIGNMENT_ID, "revision-2", HEAD_SHA, StaleAssignmentRevisionError),
        (ASSIGNMENT_ID, "revision-1", "b" * 40, PullHeadMismatchError),
    ],
    ids=("assignment", "revision", "head"),
)
def test_assignment_feedback_rejects_stale_identity_before_writing(
    assignment_id,
    revision_id,
    expected_head_sha,
    error_type,
):
    fake = FakeGitHub(
        pull_request(labels={"student:student-one", "status:wip"}, draft=True)
    )

    with pytest.raises(error_type):
        send_feedback(
            workflow(fake),
            assignment_id=assignment_id,
            revision_id=revision_id,
            expected_head_sha=expected_head_sha,
        )

    assert fake.mutations == []


@pytest.mark.parametrize(
    "labels",
    [
        {"student:someone-else", "status:wip"},
        {"student:student-one"},
        {"student:student-one", "status:wip", "status:review"},
    ],
    ids=("wrong-student", "missing-status", "ambiguous-status"),
)
def test_assignment_feedback_requires_unambiguous_active_routing(labels):
    fake = FakeGitHub(pull_request(labels=labels))

    with pytest.raises(WorkflowPreconditionError):
        send_feedback(workflow(fake))

    assert fake.mutations == []
