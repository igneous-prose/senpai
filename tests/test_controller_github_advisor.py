import pytest
from pydantic import SecretStr

from senpai_agent.github_mailbox import GitHubMailbox
from senpai_agent.models import AssignmentRecord, render_assignment_marker


def pull(
    *,
    labels,
    number=17,
    body="",
    head_sha=None,
    updated_at="2099-07-29T18:00:00Z",
):
    return {
        "number": number,
        "title": "Try bounded change",
        "html_url": f"https://github.test/acme/widgets/pull/{number}",
        "updated_at": updated_at,
        "body": body,
        "head": {
            "ref": f"student/candidate-{number}",
            "sha": head_sha or str(number % 10) * 40,
        },
        "labels": [{"name": label} for label in labels],
    }


def mailbox(monkeypatch, pulls, *, students=()):
    value = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        students=students,
    )
    monkeypatch.setattr(value, "_pulls", lambda: list(pulls))
    monkeypatch.setattr(value, "_issues", list)
    return value


def assignment(*, base_sha="b" * 40):
    return AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-17",
        revision_id="revision-2",
        student="student-1",
        base_ref="research",
        base_sha=base_sha,
        head_ref="student/candidate-17",
        head_sha="7" * 40,
    )


def test_review_label_wakes_the_advisor_and_releases_the_student_slot(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
            )
        ],
        students=("student-1", "student-2"),
    )

    events = advisor.poll()

    assert [event.kind for event in events] == [
        "review_ready",
        "idle_student",
        "idle_student",
    ]
    assert events[0].payload["number"] == 17
    assert events[1].payload == {"student": "student-1"}
    assert events[2].payload == {"student": "student-2"}


@pytest.mark.parametrize(
    ("labels", "updated_at", "reason"),
    [
        (("student:student-1", "status:blocked"), "2099-01-01T00:00:00Z", "blocked"),
        (
            ("student:student-1", "status:needs-rebase"),
            "2099-01-01T00:00:00Z",
            "needs_rebase",
        ),
        (("status:review",), "2099-01-01T00:00:00Z", "missing_student"),
        (
            ("student:student-1", "student:student-2", "status:review"),
            "2099-01-01T00:00:00Z",
            "multiple_students",
        ),
        (
            ("student:student-1", "status:wip"),
            "2020-01-01T00:00:00Z",
            "stale_wip",
        ),
    ],
)
def test_advisor_action_reports_each_unsafe_assignment_state(
    monkeypatch,
    labels,
    updated_at,
    reason,
):
    advisor = mailbox(
        monkeypatch,
        [pull(labels=("research", *labels), updated_at=updated_at)],
    )

    actions = [event for event in advisor.poll() if event.kind == "advisor_action"]

    assert len(actions) == 1
    assert actions[0].payload["reasons"] == [reason]


def test_duplicate_assignments_report_every_pr_for_the_student(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(labels=("student:student-1", "status:wip"), number=17),
            pull(labels=("student:student-1", "status:wip"), number=18),
        ],
        students=("student-1",),
    )

    duplicate = next(
        event for event in advisor.poll() if event.kind == "duplicate_assignment"
    )

    assert duplicate.dedupe_key == "duplicate_assignment:student-1:17,18"
    assert duplicate.payload["pull_requests"] == [17, 18]


def test_baseline_advance_uses_the_fresh_live_branch_head_on_each_poll(
    monkeypatch,
):
    assigned_sha = "b" * 40
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:wip"),
                body=render_assignment_marker(assignment(base_sha=assigned_sha)),
                head_sha="7" * 40,
            )
        ],
        students=("student-1",),
    )
    current_sha = ["c" * 40]
    ref_reads = []

    def get_ref(path):
        ref_reads.append(path)
        return {"object": {"sha": current_sha[0]}}

    monkeypatch.setattr(advisor._github, "get", get_ref)

    first = next(
        event for event in advisor.poll() if event.kind == "baseline_advanced"
    )
    current_sha[0] = "d" * 40
    second = next(
        event for event in advisor.poll() if event.kind == "baseline_advanced"
    )

    assert first.dedupe_key == f"baseline_advanced:17:{assigned_sha}:{'c' * 40}"
    assert first.payload["assigned_base_sha"] == assigned_sha
    assert first.payload["current_base_sha"] == "c" * 40
    assert first.payload["compare_url"] == (
        f"https://github.test/acme/widgets/compare/{assigned_sha}...{'c' * 40}"
    )
    assert second.payload["current_base_sha"] == "d" * 40
    assert ref_reads == [
        "/repos/acme/widgets/git/ref/heads/research",
        "/repos/acme/widgets/git/ref/heads/research",
    ]


def test_current_assignment_baseline_does_not_emit_a_false_advance(monkeypatch):
    current_sha = "b" * 40
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:wip"),
                body=render_assignment_marker(assignment(base_sha=current_sha)),
                head_sha="7" * 40,
            )
        ],
        students=("student-1",),
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": current_sha}},
    )

    assert "baseline_advanced" not in {event.kind for event in advisor.poll()}


def test_baseline_ref_failure_does_not_suppress_other_advisor_events(
    monkeypatch,
    capsys,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
            )
        ],
        students=("student-1", "student-2"),
    )

    def invalid_ref(_path):
        raise TypeError("invalid ref response")

    monkeypatch.setattr(advisor._github, "get", invalid_ref)

    events = advisor.poll()

    assert {event.kind for event in events} == {"review_ready", "idle_student"}
    assert "SENPAI_BASELINE_WATCH_ERROR TypeError" in capsys.readouterr().err
