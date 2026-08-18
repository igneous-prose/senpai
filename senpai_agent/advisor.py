import argparse
import threading
import uuid
from collections.abc import Callable, Sequence, Set
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState

from senpai_agent.inbox import PersistentInbox
from senpai_agent.local_events import LocalEventStore


def advisor_conversation_id(
    state_dir: Path,
    explicit_id: str | None = None,
) -> uuid.UUID:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "advisor-conversation-id"
    if explicit_id is not None:
        conversation_id = uuid.UUID(explicit_id)
        path.write_text(f"{conversation_id}\n")
        return conversation_id
    if path.exists():
        return uuid.UUID(path.read_text().strip())

    conversation_id = uuid.uuid4()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{conversation_id}\n")
    temporary.replace(path)
    return conversation_id


class MessageConversation(Protocol):
    def send_message(self, message: str) -> None: ...


def _deliver_pending_events(
    store: LocalEventStore,
    conversation: MessageConversation,
    *,
    record_delivery: Callable[[str], None],
    already_delivered: Set[str] = frozenset(),
    parent_conversation_id: str | None = None,
) -> int:
    delivered = 0
    pending = store.pending()
    if parent_conversation_id is not None:
        pending = [
            event
            for event in pending
            if event.payload.get("parent_conversation_id")
            == parent_conversation_id
        ]
    for event in pending:
        if event.dedupe_key in already_delivered:
            continue
        conversation.send_message(event.to_user_message())
        record_delivery(event.dedupe_key)
        delivered += 1
    return delivered


def deliver_pending_events(
    store: LocalEventStore,
    conversation: MessageConversation,
    *,
    parent_conversation_id: str | None = None,
) -> int:
    return _deliver_pending_events(
        store,
        conversation,
        record_delivery=store.acknowledge,
        parent_conversation_id=parent_conversation_id,
    )


class AdvisorEventPump:
    def __init__(
        self,
        store: LocalEventStore,
        conversation: MessageConversation,
        *,
        poll_interval: float = 0.5,
        parent_conversation_id: str | None = None,
        inbox: PersistentInbox | None = None,
        conversation_id: str | uuid.UUID | None = None,
    ):
        self._store = store
        self._conversation = conversation
        self._poll_interval = poll_interval
        self._parent_conversation_id = parent_conversation_id
        self._inbox = inbox
        if inbox is not None:
            if conversation_id is None:
                raise ValueError("inbox event pump requires a conversation ID")
            self._conversation_id = str(uuid.UUID(str(conversation_id)))
        else:
            self._conversation_id = str(
                conversation_id or getattr(conversation, "id", "local")
            )
        self._delivered_event_keys: set[str] = set()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="senpai-agent-event-pump",
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._store.pending_count():
                    self._deliver_if_safe()
                self._stop.wait(self._poll_interval)
        except BaseException as error:  # noqa: BLE001
            self._error = error
            self._stop.set()

    def _deliver_if_safe(self) -> int:
        if self._inbox is not None:
            return self._transfer_to_inbox()
        state = getattr(self._conversation, "state", None)
        if state is None:
            # Lightweight message adapters have no tool-action state to guard.
            return _deliver_pending_events(
                self._store,
                self._conversation,
                record_delivery=self._delivered_event_keys.add,
                already_delivered=self._delivered_event_keys,
                parent_conversation_id=self._parent_conversation_id,
            )
        with state:
            if ConversationState.get_unmatched_actions(state.active_branch()):
                return 0
            return _deliver_pending_events(
                self._store,
                self._conversation,
                record_delivery=self._delivered_event_keys.add,
                already_delivered=self._delivered_event_keys,
                parent_conversation_id=self._parent_conversation_id,
            )

    def _transfer_to_inbox(self) -> int:
        pending = self._store.pending()
        if self._parent_conversation_id is not None:
            pending = [
                event
                for event in pending
                if event.payload.get("parent_conversation_id")
                == self._parent_conversation_id
            ]
        for event in pending:
            self._inbox.enqueue(
                self._conversation_id,
                event.dedupe_key,
                event.to_inbox_message(),
            )
            self._store.acknowledge(event.dedupe_key)
        return len(pending)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        # Failed turns may abandon their active branch; replay those events instead.
        if self._inbox is None and exc_type is None and self._error is None:
            state = getattr(self._conversation, "state", None)
            if (
                state is None
                or state.execution_status == ConversationExecutionStatus.FINISHED
            ):
                for key in sorted(self._delivered_event_keys):
                    self._store.acknowledge(key)
        if exc_type is None and self._error is not None:
            raise self._error


def advisor_main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Inspect local Senpai advisor state")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pending = subparsers.add_parser("pending-count")
    pending.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "pending-count":
        with LocalEventStore(
            args.state_dir.expanduser().resolve() / "advisor-events.sqlite3"
        ) as store:
            print(store.pending_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(advisor_main())
