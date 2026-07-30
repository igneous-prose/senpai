from pathlib import Path
from uuid import UUID

from pydantic import SecretStr

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.controller import (
    AssignmentConversationRegistry,
    Controller,
    ControllerEvent,
    ConversationLedger,
    GitHubMailbox,
    LocalStudentMailbox,
    StudentConversationSelector,
    SystemContextLedger,
    TurnResult,
    _full_prompt,
)
from senpai_agent.models import AssignmentRecord, render_assignment_marker
from senpai_agent.supervisor import ProgressLease, WorkerLease


class Mailbox:
    def __init__(self, polls):
        self.polls = list(polls)
        self.calls = 0

    def poll(self):
        self.calls += 1
        return self.polls.pop(0) if self.polls else ()

    def acknowledge(self, _dedupe_keys):
        pass


class Turns:
    def __init__(self):
        self.calls = []

    def run(
        self,
        prompt,
        *,
        conversation_id,
        continue_session,
        event_keys,
    ):
        self.calls.append((prompt, conversation_id, continue_session, event_keys))
        return TurnResult(exit_code=0)


class SequencedTurns(Turns):
    def __init__(self, exit_codes):
        super().__init__()
        self.exit_codes = iter(exit_codes)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        return TurnResult(
            exit_code=next(self.exit_codes),
            delivered_event_keys=result.delivered_event_keys,
        )


class RaisingTurns(Turns):
    def __init__(self):
        super().__init__()
        self.fail = True

    def run(self, *args, **kwargs):
        if self.fail:
            self.fail = False
            raise RuntimeError("SDK transport failed")
        return super().run(*args, **kwargs)


def test_assignment_conversation_is_reused_for_monitor_wake(tmp_path: Path):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")

    first = registry.for_assignment("assignment-1", "revision-2")
    wake = registry.for_assignment("assignment-1", "revision-2")
    next_revision = registry.for_assignment("assignment-1", "revision-3")

    assert isinstance(first, UUID)
    assert wake == first
    assert next_revision != first


def test_late_student_child_result_wakes_its_exact_parent(tmp_path: Path):
    first_parent = UUID("00000000-0000-0000-0000-000000000011")
    second_parent = UUID("00000000-0000-0000-0000-000000000012")
    store_path = tmp_path / "student-events.sqlite3"
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind="agent_result",
                dedupe_key="agent_result:first",
                payload={"parent_conversation_id": str(first_parent)},
            )
        )
        store.enqueue(
            AdvisorEvent(
                kind="agent_result",
                dedupe_key="agent_result:second",
                payload={"parent_conversation_id": str(second_parent)},
            )
        )

    events = LocalStudentMailbox(store_path).poll()
    selected = StudentConversationSelector(
        AssignmentConversationRegistry(tmp_path / "students.json")
    )(events)

    assert len(events) == 1
    assert events[0].payload["count"] == 1
    assert events[0].payload["conversation_id"] == str(first_parent)
    assert selected == first_parent


def test_controller_builds_first_turn_from_program_and_role_task(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    instructions = workspace / "instructions"
    instructions.mkdir(parents=True)
    (workspace / "program.md").write_text("Minimize test error.")
    (instructions / "prompt-student.md").write_text(
        "Work as $STUDENT_NAME on $ADVISOR_BRANCH."
    )
    env = {
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "GH_REPO": "acme/widgets",
        "ADVISOR_BRANCH": "research",
        "WANDB_ENTITY": "acme",
        "WANDB_PROJECT": "cfd",
        "STUDENT_NAME": "fern",
    }

    prompt = _full_prompt("student", env)

    assert "Minimize test error." in prompt
    assert "Work as fern on research." in prompt
    assert "Role: student; repository: acme/widgets" in prompt


def test_advisor_repolls_immediately_after_a_turn_when_github_changed():
    review = ControllerEvent(
        kind="review_ready",
        dedupe_key="review_ready:17:abc",
        payload={"pr": 17, "head_sha": "abc"},
    )
    mailbox = Mailbox(
        [
            (
                ControllerEvent(
                    kind="idle_student",
                    dedupe_key="idle:student-1",
                    payload={"student": "student-1"},
                ),
            ),
            (review,),
            (),
        ]
    )
    turns = Turns()
    controller = Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 2
    assert "idle_student" in turns.calls[0][0]
    assert "Current time (UTC):" in turns.calls[0][0]
    assert "review_ready" in turns.calls[1][0]
    assert mailbox.calls >= 3
    assert turns.calls[1][2] is True


def test_no_github_work_means_no_model_turn():
    turns = Turns()
    controller = Controller(
        role="student",
        mailbox=Mailbox([(), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000002"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert turns.calls == []


def test_controller_retries_an_unacknowledged_event_after_turn_failure():
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    turns = SequencedTurns([1, 0])
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (event,), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000003"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 2
    assert turns.calls[1][2] is True


def test_controller_retries_an_unacknowledged_event_after_sdk_exception():
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    turns = RaisingTurns()
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (event,), ()]),
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000005"),
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=2)

    assert len(turns.calls) == 1
    assert turns.calls[0][2] is True


def test_controller_continues_a_conversation_recorded_before_restart(
    tmp_path: Path,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000004")
    ledger = ConversationLedger(tmp_path / "conversations.json")
    ledger.mark_started(conversation_id)
    turns = Turns()
    event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="training:finished",
        payload={"conversation_id": str(conversation_id)},
    )
    controller = Controller(
        role="student",
        mailbox=Mailbox([(event,), ()]),
        turns=turns,
        conversation_id=conversation_id,
        full_prompt="programme",
        conversation_ledger=ConversationLedger(tmp_path / "conversations.json"),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert turns.calls[0][2] is True
    assert "programme" not in turns.calls[0][0]


def test_controller_publishes_a_hard_deadline_for_its_current_phase(
    tmp_path: Path,
):
    lease_path = tmp_path / "controller-lease.json"
    observed_leases = []

    class LeaseReadingTurns(Turns):
        def run(self, *args, **kwargs):
            observed_leases.append(WorkerLease.read(lease_path))
            return super().run(*args, **kwargs)

    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:abc",
        payload={"number": 17},
    )
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,)]),
        turns=LeaseReadingTurns(),
        conversation_id=UUID("00000000-0000-0000-0000-000000000048"),
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        operation_timeout_seconds=123,
        turn_timeout_seconds=456,
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    lease = observed_leases[0]
    assert lease.phase == "openhands-turn"
    assert lease.pid > 0


def test_changed_system_context_is_injected_once_without_rotating_uuid(
    tmp_path: Path,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000006")
    conversations = ConversationLedger(tmp_path / "conversations.json")
    conversations.mark_started(conversation_id)
    system_contexts = SystemContextLedger(tmp_path / "system-contexts.json")
    system_contexts.mark(conversation_id, "old harness and role")
    event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:17:new-role",
        payload={"number": 17},
    )
    next_event = ControllerEvent(
        kind="review_ready",
        dedupe_key="review:18:new-role",
        payload={"number": 18},
    )
    turns = Turns()
    controller = Controller(
        role="advisor",
        mailbox=Mailbox([(event,), (next_event,), ()]),
        turns=turns,
        conversation_id=conversation_id,
        full_prompt="programme",
        conversation_ledger=conversations,
        system_context="current harness and role",
        system_context_ledger=system_contexts,
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert "# Updated Senpai system context" in turns.calls[0][0]
    assert "current harness and role" in turns.calls[0][0]
    assert "# Updated Senpai system context" not in turns.calls[1][0]
    assert system_contexts.is_current(
        conversation_id,
        "current harness and role",
    )


def pull(*, labels, body=""):
    return {
        "number": 17,
        "title": "Try bounded change",
        "html_url": "https://github.test/acme/widgets/pull/17",
        "updated_at": "2026-07-29T18:00:00Z",
        "body": body,
        "head": {"ref": "student/candidate", "sha": "a" * 40},
        "labels": [{"name": label} for label in labels],
    }


def test_github_mailbox_uses_pr_labels_as_the_cross_node_protocol(
    monkeypatch,
):
    advisor = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        students=("student-1", "student-2"),
    )
    monkeypatch.setattr(
        advisor,
        "_pulls",
        lambda: [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    "status:review",
                )
            )
        ],
    )
    monkeypatch.setattr(advisor, "_issues", list)

    events = advisor.poll()

    assert {event.kind for event in events} == {
        "review_ready",
        "idle_student",
    }
    assert (
        next(event for event in events if event.kind == "review_ready").payload[
            "number"
        ]
        == 17
    )
    assert next(event for event in events if event.kind == "idle_student").payload == {
        "student": "student-2"
    }


def test_student_assignment_event_carries_durable_assignment_identity(
    monkeypatch,
):
    assignment = AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-17",
        revision_id="revision-2",
        student="student-1",
        base_ref="research",
        base_sha="b" * 40,
        head_ref="student/candidate",
        head_sha="a" * 40,
    )
    student = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="student",
        advisor_branch="research",
        student_name="student-1",
    )
    monkeypatch.setattr(
        student,
        "_pulls",
        lambda: [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    "status:wip",
                ),
                body=render_assignment_marker(assignment),
            )
        ],
    )
    monkeypatch.setattr(student, "_issues", list)

    event = student.poll()[0]

    assert event.kind == "student_assignment"
    assert event.payload["assignment_id"] == "assignment-17"
    assert event.payload["revision_id"] == "revision-2"


def test_human_issue_event_tracks_the_exact_latest_human_message(
    monkeypatch,
):
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        trusted_actor="senpai-bot",
    )
    issue = {
        "id": 700,
        "number": 23,
        "title": "Change direction",
        "html_url": "https://github.test/acme/widgets/issues/23",
        "updated_at": "2026-07-29T18:10:00Z",
        "created_at": "2026-07-29T18:00:00Z",
        "body": "Start with the cheaper baseline.",
        "user": {"login": "ada"},
        "labels": [{"name": "human"}, {"name": "team"}],
    }
    monkeypatch.setattr(mailbox, "_pulls", list)
    monkeypatch.setattr(mailbox, "_issues", lambda: [issue])
    monkeypatch.setattr(
        mailbox,
        "_issue_comments",
        lambda _issue: [
            {
                "id": 701,
                "body": "ADVISOR: acknowledged",
                "created_at": "2026-07-29T18:05:00Z",
                "user": {"login": "senpai-bot"},
            },
            {
                "id": 702,
                "body": "Also compare memory.",
                "created_at": "2026-07-29T18:10:00Z",
                "user": {"login": "ada"},
            },
        ],
    )

    event = mailbox.poll()[0]

    assert event.kind == "human_issue"
    assert event.dedupe_key == "human_issue:23:702"
    assert event.payload["human_message_id"] == 702
    assert event.payload["author"] == "ada"
    assert event.payload["message"] == "Also compare memory."


def test_human_issue_polling_can_be_disabled_for_an_isolated_launch(
    monkeypatch,
):
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        trusted_actor="senpai-bot",
        human_issues_enabled=False,
    )
    monkeypatch.setattr(mailbox, "_pulls", list)
    monkeypatch.setattr(
        mailbox,
        "_issues",
        lambda: [
            {
                "id": 700,
                "number": 23,
                "title": "Out of scope",
                "html_url": "https://github.test/acme/widgets/issues/23",
                "updated_at": "2026-07-29T18:10:00Z",
                "created_at": "2026-07-29T18:00:00Z",
                "body": "Do not deliver this.",
                "user": {"login": "ada"},
                "labels": [{"name": "human"}, {"name": "team"}],
            }
        ],
    )

    assert mailbox.poll() == ()
