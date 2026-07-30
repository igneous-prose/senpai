import time
from pathlib import Path

import pytest

from senpai_agent import advisor
from senpai_agent.advisor import (
    AdvisorEvent,
    AdvisorEventPump,
    AdvisorEventStore,
    advisor_conversation_id,
    advisor_main,
    deliver_pending_events,
    merge_system_instructions,
)


def test_harness_and_role_are_one_stable_system_suffix(tmp_path: Path):
    harness = tmp_path / "SENPAI-HARNESS.md"
    role = tmp_path / "SENPAI-ADVISOR.md"
    harness.write_text("# Harness\n\nUse typed Senpai tools.\n")
    role.write_text("# Advisor\n\nDirect the research programme.\n")

    first = merge_system_instructions(harness, role)
    second = merge_system_instructions(harness, role)

    assert first == second
    assert first == (
        "# Senpai harness\n\n"
        "# Harness\n\nUse typed Senpai tools.\n\n"
        "# Senpai role\n\n"
        "# Advisor\n\nDirect the research programme.\n"
    )


def test_advisor_uuid_is_stable_and_senpai_does_not_prune_conversations(
    tmp_path: Path,
):
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    for number in range(401):
        (conversations / f"closed-{number:03d}").mkdir()

    first = advisor_conversation_id(tmp_path)
    second = advisor_conversation_id(tmp_path)

    assert first == second
    assert len(list(conversations.iterdir())) == 401


def test_event_store_deduplicates_and_survives_reopen(tmp_path: Path):
    database = tmp_path / "advisor-events.sqlite3"
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:3467:abc123",
        payload={"pr": 3467, "head_sha": "abc123"},
    )

    with AdvisorEventStore(database) as store:
        assert store.enqueue(event) is True
        assert store.pending_count() == 1
        assert store.enqueue(event) is False

    with AdvisorEventStore(database) as reopened:
        pending = reopened.pending()
        assert pending == [event]
        reopened.acknowledge(event.dedupe_key)
        assert reopened.pending_count() == 0
        assert reopened.pending() == []


def test_event_message_contains_current_time_and_structured_payload(tmp_path: Path):
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:17:abc",
        payload={
            "pr": 17,
            "head_sha": "abc",
        },
    )

    rendered = event.to_user_message()

    assert rendered.startswith("# Senpai event: review_ready\n\n")
    assert '"pr": 17' in rendered
    assert '"head_sha": "abc"' in rendered
    assert "Observed at (UTC):" in rendered


def test_advisor_events_are_local_only_and_have_no_network_server():
    assert not hasattr(advisor, "notify_advisor")
    assert not hasattr(advisor, "AdvisorEventServer")
    assert not hasattr(advisor, "accept_advisor_event")


def test_deliver_pending_events_acknowledges_only_messages_sent(tmp_path: Path):
    first = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:11:ddd",
        payload={"pr": 11},
    )
    second = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:task-1",
        payload={"task_id": "task-1"},
    )

    class Conversation:
        def __init__(self):
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            if self.messages:
                raise RuntimeError("conversation unavailable")
            self.messages.append(message)

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(first)
        store.enqueue(second)
        conversation = Conversation()

        with pytest.raises(RuntimeError, match="conversation unavailable"):
            deliver_pending_events(store, conversation)

        assert conversation.messages == [first.to_user_message()]
        assert store.pending() == [second]


def test_event_pump_injects_new_events_while_conversation_is_running(
    tmp_path: Path,
):
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:12:eee",
        payload={"pr": 12},
    )

    class Conversation:
        def __init__(self):
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            self.messages.append(message)

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        conversation = Conversation()
        with AdvisorEventPump(store, conversation, poll_interval=0.01):
            store.enqueue(event)
            deadline = time.monotonic() + 1
            while not conversation.messages and time.monotonic() < deadline:
                time.sleep(0.01)

        assert conversation.messages == [event.to_user_message()]
        assert store.pending() == []


def test_event_pump_routes_child_results_to_their_parent_conversation(
    tmp_path: Path,
):
    first_parent = "00000000-0000-0000-0000-000000000001"
    second_parent = "00000000-0000-0000-0000-000000000002"
    first = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:first",
        payload={"parent_conversation_id": first_parent},
    )
    second = AdvisorEvent(
        kind="agent_result",
        dedupe_key="agent_result:second",
        payload={"parent_conversation_id": second_parent},
    )

    class Conversation:
        def __init__(self):
            self.messages: list[str] = []

        def send_message(self, message: str) -> None:
            self.messages.append(message)

    with AdvisorEventStore(tmp_path / "student-events.sqlite3") as store:
        store.enqueue(first)
        store.enqueue(second)
        conversation = Conversation()
        with AdvisorEventPump(
            store,
            conversation,
            poll_interval=0.01,
            parent_conversation_id=first_parent,
        ):
            deadline = time.monotonic() + 1
            while not conversation.messages and time.monotonic() < deadline:
                time.sleep(0.01)

        assert conversation.messages == [first.to_user_message()]
        assert store.pending() == [second]


def test_event_pump_surfaces_delivery_failure_and_leaves_event_pending(
    tmp_path: Path,
):
    event = AdvisorEvent(
        kind="review_ready",
        dedupe_key="review_ready:13:fff",
        payload={"pr": 13},
    )

    class Conversation:
        def send_message(self, _message: str) -> None:
            raise RuntimeError("conversation rejected event")

    with AdvisorEventStore(tmp_path / "events.sqlite3") as store:
        store.enqueue(event)
        with (
            pytest.raises(RuntimeError, match="conversation rejected event"),
            AdvisorEventPump(store, Conversation(), poll_interval=0.01),
        ):
            deadline = time.monotonic() + 1
            while store.pending() and time.monotonic() < deadline:
                time.sleep(0.01)

        assert store.pending() == [event]


def test_advisor_cli_only_reports_the_local_pending_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    with AdvisorEventStore(tmp_path / "advisor-events.sqlite3") as store:
        store.enqueue(
            AdvisorEvent(
                kind="review_ready",
                dedupe_key="review_ready:1:a",
                payload={"pr": 1},
            )
        )

    assert (
        advisor_main(
            [
                "pending-count",
                "--state-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "1"
