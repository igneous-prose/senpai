import pytest

from senpai_agent.github_workflow import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker
from github_workflow_support import (
    ASSIGNMENT_ID,
    HEAD_SHA,
    FakeGitHub,
    assignment_record,
    pull_request,
    workflow,
)


def test_create_assignment_converges_one_draft_pull_request_and_replays():
    assignment = assignment_record()
    fake = FakeGitHub(pull_request(labels=set(), title="", body=""))
    fake.pr["number"] = 0
    client = workflow(fake)

    first = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )
    mutations_after_first = list(fake.mutations)
    second = client.create_assignment(
        assignment,
        title="Try lower learning rate",
        body="Run the bounded learning-rate experiment.",
    )

    assert first.changed is True
    assert second.changed is False
    assert fake.pr["body"] == (
        f"{render_assignment_marker(assignment)}\n\n"
        "Run the bounded learning-rate experiment."
    )
    assert fake.pr["draft"] is True
    assert fake.pr["labels"] == {
        "schmidhuber",
        "student:student-one",
        "status:wip",
    }
    assert fake.mutations == mutations_after_first


def test_create_assignment_does_not_repurpose_a_foreign_pull_request():
    foreign_body = render_assignment_marker(
        assignment_record(assignment_id="someone-elses-assignment")
    )
    fake = FakeGitHub(pull_request(body=foreign_body))

    with pytest.raises(WorkflowPreconditionError, match="assignment marker"):
        workflow(fake).create_assignment(
            assignment_record(),
            title="Try lower learning rate",
            body="Run the bounded learning-rate experiment.",
        )

    assert fake.mutations == []


@pytest.mark.parametrize("status", ["status:wip", "status:review"])
def test_create_assignment_rejects_another_active_pull_for_the_student(status):
    assignment = assignment_record(
        assignment_id="assignment-8",
        head_ref="student-one/new-candidate",
        head_sha="c" * 40,
    )
    fake = FakeGitHub(
        pull_request(
            labels={"other-base", "student:student-one", status},
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


def test_reconcile_labels_sets_the_exact_union_and_replays_without_writes():
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
    assert fake.mutations == mutations_after_first


def test_reconcile_labels_rejects_a_stale_head_before_writing():
    fake = FakeGitHub(pull_request())

    with pytest.raises(WorkflowPreconditionError, match="head SHA"):
        workflow(fake).reconcile_labels(
            7,
            assignment_id=ASSIGNMENT_ID,
            add={"status:review"},
            remove={"status:wip"},
            expected_head_sha="b" * 40,
        )

    assert fake.mutations == []


def test_reconcile_labels_fails_if_github_does_not_apply_the_write():
    fake = FakeGitHub(pull_request(), ignore_label_put=True)

    with pytest.raises(ReconciliationError, match="label set"):
        workflow(fake).reconcile_labels(
            7,
            assignment_id=ASSIGNMENT_ID,
            add={"status:review"},
            remove={"status:wip"},
            expected_head_sha=HEAD_SHA,
        )

    assert len(fake.mutations) == 1


def test_reconcile_labels_rejects_a_foreign_assignment_before_writing():
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
