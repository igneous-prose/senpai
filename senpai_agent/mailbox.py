"""Controller event values and local mailbox composition."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from senpai_agent.advisor import AdvisorEventStore


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    kind: str
    dedupe_key: str
    payload: dict[str, object]

    def to_prompt(self) -> str:
        return (
            f"## {self.kind}\n\n"
            f"{json.dumps(self.payload, sort_keys=True, separators=(',', ':'))}"
        )


class Mailbox(Protocol):
    def poll(self) -> Sequence[ControllerEvent]: ...

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None: ...


class CompositeMailbox:
    """Merge independent mailboxes without letting one failure suppress another."""

    def __init__(self, *mailboxes: Mailbox):
        self.mailboxes = mailboxes

    def poll(self) -> tuple[ControllerEvent, ...]:
        by_key: dict[str, ControllerEvent] = {}
        for mailbox in self.mailboxes:
            try:
                events = mailbox.poll()
            except Exception as error:  # noqa: BLE001
                print(
                    f"SENPAI_MAILBOX_ERROR {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            for event in events:
                by_key.setdefault(event.dedupe_key, event)
        return tuple(by_key.values())

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for mailbox in self.mailboxes:
            mailbox.acknowledge(dedupe_keys)


class LocalAdvisorMailbox:
    """Wake an idle advisor so its event pump can drain local child results."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def poll(self) -> tuple[ControllerEvent, ...]:
        with AdvisorEventStore(self.store_path) as store:
            pending = store.pending()
        if not pending:
            return ()
        identity = "|".join(event.dedupe_key for event in pending)
        return (
            ControllerEvent(
                kind="local_events_pending",
                dedupe_key=f"local_events:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}",
                payload={
                    "count": len(pending),
                    "kinds": sorted({event.kind for event in pending}),
                    "delivery": (
                        "The OpenHands event pump will inject these events at "
                        "the next safe conversation boundary."
                    ),
                },
            ),
        )

    def acknowledge(self, _dedupe_keys: Sequence[str]) -> None:
        return


class LocalStudentMailbox:
    """Wake the student conversation that dispatched a finished child."""

    def __init__(self, store_path: Path):
        self.store_path = store_path

    def poll(self) -> tuple[ControllerEvent, ...]:
        with AdvisorEventStore(self.store_path) as store:
            pending = store.pending()
        if not pending:
            return ()
        parent_id = pending[0].payload.get("parent_conversation_id")
        if not isinstance(parent_id, str):
            raise RuntimeError("student child event has no parent conversation")
        matching = [
            event
            for event in pending
            if event.payload.get("parent_conversation_id") == parent_id
        ]
        identity = "|".join(event.dedupe_key for event in matching)
        return (
            ControllerEvent(
                kind="local_events_pending",
                dedupe_key=f"local_events:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}",
                payload={
                    "conversation_id": parent_id,
                    "count": len(matching),
                    "kinds": sorted({event.kind for event in matching}),
                    "delivery": (
                        "The OpenHands event pump will inject these events at "
                        "the next safe conversation boundary."
                    ),
                },
            ),
        )

    def acknowledge(self, _dedupe_keys: Sequence[str]) -> None:
        return
