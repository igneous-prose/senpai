from pathlib import Path
from uuid import UUID

import pytest

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.inbox import PersistentInbox
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    SlotAvailabilityMailbox,
)
from senpai_agent.monitor import (
    MonitorEvaluation,
    MonitorMailbox,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
)
from senpai_agent.training import TrainingState


class StaticMailbox:
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self):
        return self.events

    def acknowledge(self, _dedupe_keys):
        return


def slot_event(student: str = "Fern") -> ControllerEvent:
    return ControllerEvent(
        kind="student_slot_available",
        dedupe_key=f"student_slot_available:{student}",
        payload={"student": student},
    )


def test_composite_mailbox_preserves_healthy_events_when_a_peer_fails(capsys):
    event = ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:healthy",
        payload={"assignment_id": "healthy"},
    )

    class BrokenMailbox:
        def poll(self):
            raise RuntimeError("monitor backend unavailable")

        def acknowledge(self, _dedupe_keys):
            return

    mailbox = CompositeMailbox(BrokenMailbox(), StaticMailbox((event,)))

    assert mailbox.poll() == (event,)
    assert "SENPAI_MAILBOX_ERROR RuntimeError" in capsys.readouterr().err


def test_reserved_slot_retracts_unseen_inbox_and_staging_events(tmp_path: Path):
    conversation_id = UUID(int=123)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    store_path = tmp_path / "advisor-events.sqlite3"
    event = slot_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    with AdvisorEventStore(store_path) as store:
        store.enqueue(
            AdvisorEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload=event.payload,
            )
        )
        store.acknowledge(event.dedupe_key)

    mailbox = SlotAvailabilityMailbox(
        StaticMailbox(()),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=store_path,
    )

    assert mailbox.poll() == ()
    assert inbox.pending_count(conversation_id) == 0
    with AdvisorEventStore(store_path) as store:
        assert store.enqueue(
            AdvisorEvent(
                kind=event.kind,
                dedupe_key=event.dedupe_key,
                payload=event.payload,
            )
        ) is True


def test_available_slot_preserves_its_queued_event(tmp_path: Path):
    conversation_id = UUID(int=124)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    event = slot_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    mailbox = SlotAvailabilityMailbox(
        StaticMailbox((event,)),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=tmp_path / "advisor-events.sqlite3",
    )

    assert mailbox.poll() == (event,)
    assert inbox.pending_count(conversation_id) == 1


def test_snapshot_retracts_a_removed_student_slot(tmp_path: Path):
    conversation_id = UUID(int=125)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    store_path = tmp_path / "advisor-events.sqlite3"
    stale = slot_event("Old-name")
    retired = ControllerEvent(
        kind="idle_student",
        dedupe_key="idle_student:Older-name",
        payload={"student": "Older-name"},
    )
    current = slot_event("New-name")
    inbox.enqueue(conversation_id, stale.dedupe_key, stale.to_prompt())
    inbox.enqueue(conversation_id, retired.dedupe_key, retired.to_prompt())
    with AdvisorEventStore(store_path) as store:
        for event in (stale, retired):
            store.enqueue(
                AdvisorEvent(
                    kind=event.kind,
                    dedupe_key=event.dedupe_key,
                    payload=event.payload,
                )
            )
            store.acknowledge(event.dedupe_key)

    mailbox = SlotAvailabilityMailbox(
        StaticMailbox((current,)),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=store_path,
    )

    assert mailbox.poll() == (current,)
    assert inbox.pending_count(conversation_id) == 0
    with AdvisorEventStore(store_path) as store:
        for event in (stale, retired):
            assert store.enqueue(
                AdvisorEvent(
                    kind=event.kind,
                    dedupe_key=event.dedupe_key,
                    payload=event.payload,
                )
            ) is True


def test_failed_slot_poll_does_not_retract_queued_availability(tmp_path: Path):
    class BrokenMailbox:
        def poll(self):
            raise RuntimeError("GitHub unavailable")

        def acknowledge(self, _dedupe_keys):
            return

    conversation_id = UUID(int=125)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    event = slot_event()
    inbox.enqueue(conversation_id, event.dedupe_key, event.to_prompt())
    mailbox = SlotAvailabilityMailbox(
        BrokenMailbox(),
        inbox=inbox,
        conversation_id=conversation_id,
        event_store_path=tmp_path / "advisor-events.sqlite3",
    )

    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        mailbox.poll()
    assert inbox.pending_count(conversation_id) == 1


def test_monitor_mailbox_routes_and_acknowledges_each_signal_independently(
    tmp_path: Path,
):
    first_id = UUID("00000000-0000-0000-0000-000000000086")
    second_id = UUID("00000000-0000-0000-0000-000000000087")
    first = MonitorSignal(
        kind="training_status",
        dedupe_key="training:first:failed",
        training_id="training-first",
        state=TrainingState.FAILED,
        detail="first training failed",
    )
    second = MonitorSignal(
        kind="training_status",
        dedupe_key="training:second:finished",
        training_id="training-second",
        state=TrainingState.FINISHED,
        detail="second training finished",
    )

    class Engine:
        def poll(self):
            return ()

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        for signal, conversation_id in ((first, first_id), (second, second_id)):
            spec = TrainingMonitorSpec(
                training_id=signal.training_id,
                conversation_id=conversation_id,
            )
            store.register(spec)
            store.record_poll(spec, MonitorEvaluation(signals=(signal,)), None)
        mailbox = MonitorMailbox(Engine(), store)

        events = mailbox.poll()

        assert {
            event.dedupe_key: event.payload["conversation_id"] for event in events
        } == {
            first.dedupe_key: str(first_id),
            second.dedupe_key: str(second_id),
        }
        mailbox.acknowledge((first.dedupe_key,))
        assert [event.dedupe_key for event in mailbox.poll()] == [second.dedupe_key]
