import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from senpai_agent.advisor import AdvisorEvent, AdvisorEventPump, AdvisorEventStore
from senpai_agent.controller import OpenHandsTurnRunner
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.state import AssignmentConversationRegistry


@dataclass(frozen=True)
class Config:
    role: str
    state_dir: Path
    conversation_id: UUID


class Mailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events


def feedback_event(revision_id="revision-2"):
    return ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key=f"student_pr_feedback:issue_comment:17:{revision_id}",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": revision_id,
            "message": f"Feedback for {revision_id}.",
        },
    )


def test_running_student_receives_only_feedback_bound_to_its_conversation(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    current = feedback_event("revision-2")
    other_revision = feedback_event("revision-3")
    messages = []

    def run_openhands(_prompt, config):
        class Conversation:
            def send_message(self, message):
                messages.append(message)

        with AdvisorEventStore(
            state_dir / "student-events.sqlite3"
        ) as store, AdvisorEventPump(
            store,
            Conversation(),
            poll_interval=0.001,
            parent_conversation_id=str(config.conversation_id),
        ):
            deadline = time.monotonic() + 1
            while not messages and time.monotonic() < deadline:
                time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        github_mailbox=Mailbox((current, other_revision)),
        active_poll_interval_seconds=0.001,
    ).run(
        "current student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert len(messages) == 1
    assert "Feedback for revision-2." in messages[0]
    assert "Feedback for revision-3." not in messages[0]
    assert str(conversation_id) in messages[0]
    assert result.delivered_event_keys == frozenset({current.dedupe_key})
    with AdvisorEventStore(state_dir / "student-events.sqlite3") as store:
        assert store.pending() == []


def test_observed_feedback_is_not_reported_delivered_until_the_event_pump_sends_it(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    feedback = feedback_event()
    store_path = state_dir / "student-events.sqlite3"

    def run_openhands(_prompt, _config):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with AdvisorEventStore(store_path) as store:
                if store.pending_count():
                    break
            time.sleep(0.001)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        github_mailbox=Mailbox((feedback,)),
        active_poll_interval_seconds=0.001,
    ).run(
        "student turn",
        conversation_id=conversation_id,
        event_keys=frozenset(),
    )

    assert result.delivered_event_keys == frozenset()
    with AdvisorEventStore(store_path) as store:
        assert [event.dedupe_key for event in store.pending()] == [
            feedback.dedupe_key
        ]


def test_prompt_delivery_suppresses_a_late_duplicate_watcher_event(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    registry = AssignmentConversationRegistry(
        state_dir / "student-conversations.json"
    )
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    feedback = feedback_event()
    store_path = state_dir / "student-events.sqlite3"
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind=feedback.kind,
                dedupe_key=feedback.dedupe_key,
                payload={
                    **feedback.payload,
                    "parent_conversation_id": str(conversation_id),
                },
            )
        )
    messages = []

    def run_openhands(_prompt, config):
        class Conversation:
            def send_message(self, message):
                messages.append(message)

        with AdvisorEventStore(store_path) as store, AdvisorEventPump(
            store,
            Conversation(),
            poll_interval=0.001,
            parent_conversation_id=str(config.conversation_id),
        ):
            time.sleep(0.01)
        return 0

    monkeypatch.setattr("senpai_agent.openhands_runner.run_openhands", run_openhands)

    result = OpenHandsTurnRunner(
        Config("student", state_dir, conversation_id),
        github_mailbox=Mailbox((feedback,)),
        active_poll_interval_seconds=0.001,
    ).run(
        feedback.to_prompt(),
        conversation_id=conversation_id,
        event_keys=frozenset({feedback.dedupe_key}),
    )

    assert messages == []
    assert result.delivered_event_keys == frozenset()
    with AdvisorEventStore(store_path) as store:
        assert store.pending() == []
