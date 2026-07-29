"""Portable Senpai control loop with GitHub as its only remote mailbox."""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from base64 import b64decode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from string import Template
from types import TracebackType
from typing import Literal, Protocol, Self
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from uuid import UUID

from pydantic import SecretStr

from senpai_agent.advisor import (
    AdvisorEvent,
    AdvisorEventStore,
    compose_system_instructions,
)
from senpai_agent.models import parse_assignment_markers
from senpai_agent.monitor import (
    MonitorDecision,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorEngine,
    WandbMetricSource,
)
from senpai_agent.supervisor import LEASE_ENV, ProgressLease


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


@dataclass(frozen=True, slots=True)
class TurnResult:
    exit_code: int
    delivered_event_keys: frozenset[str] = frozenset()


class TurnRunner(Protocol):
    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        continue_session: bool,
        event_keys: frozenset[str],
    ) -> TurnResult: ...


class AssignmentConversationRegistry:
    """Persist one OpenHands conversation UUID per assignment revision."""

    def __init__(self, path: Path):
        self.path = path

    def for_assignment(self, assignment_id: str, revision_id: str) -> UUID:
        key = f"{assignment_id}:{revision_id}"
        values = self._read()
        if key not in values:
            values[key] = str(uuid.uuid4())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(values, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return UUID(values[key])

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise RuntimeError(f"invalid conversation registry: {self.path}")
        return value


class ConversationLedger:
    """Remember which durable conversation IDs have already received a turn."""

    def __init__(self, path: Path):
        self.path = path

    def has_started(self, conversation_id: UUID) -> bool:
        return str(conversation_id) in self._read()

    def mark_started(self, conversation_id: UUID) -> None:
        values = self._read()
        value = str(conversation_id)
        if value in values:
            return
        values.add(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sorted(values), indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise RuntimeError(f"invalid conversation ledger: {self.path}")
        return set(value)


class SystemContextLedger:
    """Track the merged harness/role revision seen by each conversation."""

    def __init__(self, path: Path):
        self.path = path

    def is_current(self, conversation_id: UUID, context: str) -> bool:
        return self._read().get(str(conversation_id)) == self._digest(context)

    def mark(self, conversation_id: UUID, context: str) -> None:
        values = self._read()
        values[str(conversation_id)] = self._digest(context)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(values, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _digest(context: str) -> str:
        return sha256(context.encode()).hexdigest()

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise RuntimeError(f"invalid system context ledger: {self.path}")
        return value


class GitHubMailbox:
    """Read level-triggered Senpai work from GitHub PRs and Issues."""

    def __init__(
        self,
        *,
        repo: str,
        token: SecretStr,
        role: Literal["advisor", "student"],
        advisor_branch: str,
        students: Sequence[str] = (),
        student_name: str | None = None,
        stale_wip_seconds: int = 7200,
        api_url: str = "https://api.github.com",
        trusted_actor: str | None = None,
        human_issues_enabled: bool = True,
    ):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        if role == "student" and not student_name:
            raise ValueError("student mailbox requires student_name")
        self.repo = repo
        self.token = token
        self.role = role
        self.advisor_branch = advisor_branch
        self.students = tuple(student for student in students if student)
        self.student_name = student_name
        self.stale_wip_seconds = stale_wip_seconds
        self.api_url = api_url.rstrip("/")
        self.trusted_actor = trusted_actor
        self.human_issues_enabled = human_issues_enabled

    def poll(self) -> tuple[ControllerEvent, ...]:
        pulls = self._pulls()
        issues = self._issues() if self.human_issues_enabled else ()
        if self.role == "advisor":
            return self._advisor_events(pulls, issues)
        return self._student_events(pulls, issues)

    def acknowledge(self, _dedupe_keys: Sequence[str]) -> None:
        # GitHub state is acknowledged only by a typed state transition.
        return

    def _advisor_events(
        self,
        pulls: Sequence[dict[str, object]],
        issues: Sequence[dict[str, object]],
    ) -> tuple[ControllerEvent, ...]:
        events: list[ControllerEvent] = []
        active_by_student: dict[str, list[int]] = {
            student: [] for student in self.students
        }
        now = datetime.now(UTC)
        for pull in pulls:
            labels = _label_names(pull)
            number = int(pull["number"])
            head = _object(pull["head"])
            head_sha = str(head["sha"])
            students = sorted(
                label.removeprefix("student:")
                for label in labels
                if label.startswith("student:")
            )
            for student in students:
                active_by_student.setdefault(student, []).append(number)
            payload = _pull_payload(pull)
            if "status:review" in labels:
                events.append(
                    ControllerEvent(
                        kind="review_ready",
                        dedupe_key=f"review_ready:{number}:{head_sha}",
                        payload=payload,
                    )
                )
            reasons: list[str] = []
            if "status:blocked" in labels:
                reasons.append("blocked")
            if "status:needs-rebase" in labels:
                reasons.append("needs_rebase")
            if not students:
                reasons.append("missing_student")
            if len(students) > 1:
                reasons.append("multiple_students")
            if "status:wip" in labels:
                updated = _github_datetime(str(pull["updated_at"]))
                if (now - updated).total_seconds() >= self.stale_wip_seconds:
                    reasons.append("stale_wip")
            if reasons:
                events.append(
                    ControllerEvent(
                        kind="advisor_action",
                        dedupe_key=(
                            f"advisor_action:{number}:{head_sha}:{','.join(reasons)}"
                        ),
                        payload={**payload, "reasons": reasons},
                    )
                )

        for student, numbers in active_by_student.items():
            if not numbers:
                events.append(
                    ControllerEvent(
                        kind="idle_student",
                        dedupe_key=f"idle_student:{student}",
                        payload={"student": student},
                    )
                )
            elif len(numbers) > 1:
                events.append(
                    ControllerEvent(
                        kind="duplicate_assignment",
                        dedupe_key=(
                            f"duplicate_assignment:{student}:"
                            f"{','.join(map(str, sorted(numbers)))}"
                        ),
                        payload={
                            "student": student,
                            "pull_requests": sorted(numbers),
                        },
                    )
                )
        events.extend(self._human_issue_events(issues))
        return tuple(events)

    def _student_events(
        self,
        pulls: Sequence[dict[str, object]],
        issues: Sequence[dict[str, object]],
    ) -> tuple[ControllerEvent, ...]:
        assert self.student_name is not None
        assignment_label = f"student:{self.student_name}"
        assigned = [
            pull
            for pull in pulls
            if assignment_label in _label_names(pull)
            and "status:wip" in _label_names(pull)
        ]
        if len(assigned) > 1:
            numbers = sorted(int(pull["number"]) for pull in assigned)
            return (
                ControllerEvent(
                    kind="duplicate_assignment",
                    dedupe_key=(
                        f"duplicate_assignment:{self.student_name}:"
                        f"{','.join(map(str, numbers))}"
                    ),
                    payload={
                        "student": self.student_name,
                        "pull_requests": numbers,
                    },
                ),
            )

        events: list[ControllerEvent] = []
        if assigned:
            pull = assigned[0]
            body = str(pull.get("body") or "")
            markers = parse_assignment_markers(body)
            if len(markers) != 1:
                raise RuntimeError(
                    f"assigned PR #{pull['number']} must contain one assignment marker"
                )
            assignment = markers[0]
            payload = {
                **_pull_payload(pull),
                "assignment_id": assignment.assignment_id,
                "revision_id": assignment.revision_id,
                "base_ref": assignment.base_ref,
            }
            events.append(
                ControllerEvent(
                    kind="student_assignment",
                    dedupe_key=(
                        f"student_assignment:{assignment.assignment_id}:"
                        f"{assignment.revision_id}"
                    ),
                    payload=payload,
                )
            )
        events.extend(self._human_issue_events(issues))
        return tuple(events)

    def _human_issue_events(
        self,
        issues: Sequence[dict[str, object]],
    ) -> list[ControllerEvent]:
        role_labels = {"team"}
        if self.role == "advisor":
            role_labels.add(self.advisor_branch)
        else:
            assert self.student_name is not None
            role_labels.add(f"student:{self.student_name}")
        events = []
        for issue in issues:
            labels = _label_names(issue)
            if "human" not in labels or not role_labels & labels:
                continue
            actor = self._actor()
            messages = [
                {
                    "id": int(issue["id"]),
                    "author": str(_object(issue["user"])["login"]),
                    "body": str(issue.get("body") or ""),
                    "created_at": str(issue["created_at"]),
                },
                *[
                    {
                        "id": int(comment["id"]),
                        "author": str(_object(comment["user"])["login"]),
                        "body": str(comment.get("body") or ""),
                        "created_at": str(comment["created_at"]),
                    }
                    for comment in self._issue_comments(issue)
                ],
            ]
            human_messages = [
                message for message in messages if message["author"] != actor
            ]
            if not human_messages:
                continue
            latest = max(
                human_messages,
                key=lambda message: (
                    _github_datetime(str(message["created_at"])),
                    int(message["id"]),
                ),
            )
            number = int(issue["number"])
            events.append(
                ControllerEvent(
                    kind="human_issue",
                    dedupe_key=f"human_issue:{number}:{latest['id']}",
                    payload={
                        "number": number,
                        "title": str(issue["title"]),
                        "url": str(issue["html_url"]),
                        "human_message_id": int(latest["id"]),
                        "author": str(latest["author"]),
                        "message": _bounded_text(
                            str(latest["body"]),
                            limit=12_000,
                        ),
                        "created_at": str(latest["created_at"]),
                    },
                )
            )
        return events

    def _actor(self) -> str:
        if self.trusted_actor is None:
            actor = self._get_object("/user")
            self.trusted_actor = str(actor["login"])
        return self.trusted_actor

    def _issue_comments(
        self,
        issue: Mapping[str, object],
    ) -> list[dict[str, object]]:
        comments_url = issue.get("comments_url")
        if not comments_url:
            return []
        return self._objects(str(comments_url))

    def _pulls(self) -> list[dict[str, object]]:
        query = urlencode(
            {
                "state": "open",
                "base": self.advisor_branch,
                "per_page": 100,
            }
        )
        return self._objects(f"/repos/{self.repo}/pulls?{query}")

    def _issues(self) -> list[dict[str, object]]:
        query = urlencode(
            {
                "state": "open",
                "labels": "human",
                "per_page": 100,
            }
        )
        return [
            issue
            for issue in self._objects(f"/repos/{self.repo}/issues?{query}")
            if "pull_request" not in issue
        ]

    def _objects(self, path: str) -> list[dict[str, object]]:
        objects: list[dict[str, object]] = []
        url: str | None = (
            path
            if path.startswith(("https://", "http://"))
            else f"{self.api_url}/{path.lstrip('/')}"
        )
        while url is not None:
            github_request = request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": (f"Bearer {self.token.get_secret_value()}"),
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            try:
                with request.urlopen(github_request, timeout=30) as response:
                    payload = json.loads(response.read())
                    if not isinstance(payload, list):
                        raise TypeError("GitHub mailbox returned invalid JSON")
                    objects.extend(_object(item) for item in payload)
                    url = _next_link(response.headers.get("Link"))
            except HTTPError as error:
                raise RuntimeError(
                    f"GitHub mailbox GET failed with HTTP {error.code}"
                ) from error
            except (URLError, TimeoutError) as error:
                raise RuntimeError("GitHub mailbox is unreachable") from error
        return objects

    def _get_object(self, path: str) -> dict[str, object]:
        url = f"{self.api_url}/{path.lstrip('/')}"
        github_request = request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token.get_secret_value()}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with request.urlopen(github_request, timeout=30) as response:
                return _object(json.loads(response.read()))
        except HTTPError as error:
            raise RuntimeError(
                f"GitHub mailbox GET failed with HTTP {error.code}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError("GitHub mailbox is unreachable") from error


def _pull_payload(pull: Mapping[str, object]) -> dict[str, object]:
    head = _object(pull["head"])
    return {
        "number": int(pull["number"]),
        "title": str(pull["title"]),
        "url": str(pull["html_url"]),
        "head_ref": str(head["ref"]),
        "head_sha": str(head["sha"]),
        "labels": sorted(_label_names(pull)),
        "updated_at": str(pull["updated_at"]),
    }


def _label_names(value: Mapping[str, object]) -> set[str]:
    labels = value.get("labels")
    if not isinstance(labels, list):
        raise TypeError("GitHub mailbox item has invalid labels")
    return {str(_object(label)["name"]) for label in labels}


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("GitHub mailbox returned an invalid object")
    return value


def _github_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _bounded_text(value: str, *, limit: int) -> str:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value
    return encoded[-limit:].decode(errors="ignore")


def _next_link(value: str | None) -> str | None:
    if value is None:
        return None
    for part in value.split(","):
        sections = [section.strip() for section in part.split(";")]
        if 'rel="next"' in sections[1:]:
            target = sections[0]
            if target.startswith("<") and target.endswith(">"):
                return target[1:-1]
    return None


class MonitorTriage(Protocol):
    def decide(
        self, signal: MonitorSignal, conversation_id: UUID
    ) -> MonitorDecision: ...


class MonitorMailbox:
    """Turn durable monitor signals into sparse student wake events."""

    def __init__(
        self,
        engine: TrainingMonitorEngine,
        store: MonitorStore,
        triage: MonitorTriage,
    ):
        self.engine = engine
        self.store = store
        self.triage = triage

    def poll(self) -> tuple[ControllerEvent, ...]:
        self.engine.poll()
        events = []
        for monitor_signal in self.store.pending_signals():
            spec = self.store.spec(monitor_signal.training_id)
            decision = self.store.decision(monitor_signal.dedupe_key)
            if decision is None:
                try:
                    decision = self.triage.decide(
                        monitor_signal,
                        spec.conversation_id,
                    )
                except Exception as error:  # noqa: BLE001
                    decision = MonitorDecision(
                        wake_main=True,
                        summary=monitor_signal.detail,
                        reason=(
                            "Monitor triage failed; waking the student "
                            f"conservatively ({type(error).__name__})."
                        ),
                    )
                self.store.record_decision(monitor_signal.dedupe_key, decision)
            if not decision.wake_main and not monitor_signal.hard_failure:
                self.store.acknowledge(monitor_signal.dedupe_key)
                continue
            events.append(
                ControllerEvent(
                    kind="training_monitor",
                    dedupe_key=monitor_signal.dedupe_key,
                    payload={
                        "conversation_id": str(spec.conversation_id),
                        "training_id": monitor_signal.training_id,
                        "summary": decision.summary,
                        "reason": decision.reason,
                        "signal": monitor_signal.model_dump(mode="json"),
                    },
                )
            )
        return tuple(events)

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for key in dedupe_keys:
            self.store.acknowledge(key)


class CompositeMailbox:
    def __init__(self, *mailboxes: Mailbox):
        self.mailboxes = mailboxes

    def poll(self) -> tuple[ControllerEvent, ...]:
        by_key: dict[str, ControllerEvent] = {}
        for mailbox in self.mailboxes:
            for event in mailbox.poll():
                by_key.setdefault(event.dedupe_key, event)
        return tuple(by_key.values())

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        for mailbox in self.mailboxes:
            mailbox.acknowledge(dedupe_keys)


class LocalAdvisorMailbox:
    """Wake an idle advisor so its SDK event pump can drain local child results."""

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


class OpenHandsMonitorTriage:
    """Run one context-free generic child for an actionable monitor signal."""

    def __init__(self, child_config: object, timeout_seconds: float = 300):
        self.child_config = child_config
        self.timeout_seconds = timeout_seconds

    def decide(
        self,
        signal: MonitorSignal,
        conversation_id: UUID,
    ) -> MonitorDecision:
        from senpai_agent.child_process import OpenHandsChildProcess
        from senpai_agent.tools import AgentDispatchRequest

        request_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            signal.dedupe_key,
        )
        request = AgentDispatchRequest(
            task_id=str(request_id),
            parent_conversation_id=str(conversation_id),
            parent_context=(),
        )
        child = OpenHandsChildProcess(self.child_config, request)
        task = (
            "Decide whether this training event warrants waking the main student "
            "conversation. Return only JSON matching "
            '{"wake_main":true,"summary":"...","reason":"..."}. '
            "Wake for failures, timeouts, cancelled jobs, acceptance-gate "
            "crossings, regressions, or stalls that need intervention. A clean "
            "finish should wake when the student must inspect or submit results.\n\n"
            f"{signal.to_prompt()}"
        )
        return _parse_monitor_decision(child.run(task, self.timeout_seconds))


def _parse_monitor_decision(value: str) -> MonitorDecision:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    try:
        return MonitorDecision.model_validate_json(stripped)
    except ValueError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("monitor triage returned no JSON object")
        return MonitorDecision.model_validate_json(stripped[start : end + 1])


class ActiveGitHubWatcher:
    """Feed new GitHub state into a running advisor at SDK-safe boundaries."""

    def __init__(
        self,
        mailbox: GitHubMailbox,
        store_path: Path,
        *,
        known_keys: frozenset[str],
        poll_interval_seconds: float = 30,
    ):
        self.mailbox = mailbox
        self.store_path = store_path
        self.known_keys = set(known_keys)
        self.poll_interval_seconds = poll_interval_seconds
        self.observed_keys: set[str] = set()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="senpai-github-watcher",
        )

    def _run(self) -> None:
        try:
            with AdvisorEventStore(self.store_path) as store:
                while not self.stop.wait(self.poll_interval_seconds):
                    events = self.mailbox.poll()
                    current = {event.dedupe_key for event in events}
                    for event in events:
                        if event.dedupe_key in self.known_keys:
                            continue
                        store.enqueue(
                            AdvisorEvent(
                                kind=event.kind,
                                dedupe_key=event.dedupe_key,
                                payload=event.payload,
                            )
                        )
                        self.observed_keys.add(event.dedupe_key)
                    self.known_keys = current
        except BaseException as error:  # noqa: BLE001
            self.error = error
            self.stop.set()

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.stop.set()
        self.thread.join()
        if exc_type is None and self.error is not None:
            print(
                "SENPAI_GITHUB_WATCHER_ERROR "
                f"{type(self.error).__name__}: {self.error}",
                file=sys.stderr,
                flush=True,
            )


class OpenHandsTurnRunner:
    def __init__(
        self,
        config: object,
        *,
        github_mailbox: GitHubMailbox | None = None,
        active_poll_interval_seconds: float = 30,
    ):
        self.config = config
        self.github_mailbox = github_mailbox
        self.active_poll_interval_seconds = active_poll_interval_seconds

    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        continue_session: bool,
        event_keys: frozenset[str],
    ) -> TurnResult:
        from senpai_agent.openhands_runner import run_openhands

        config = replace(
            self.config,
            conversation_id=conversation_id,
            continue_session=continue_session,
        )
        if config.role != "advisor" or self.github_mailbox is None:
            return TurnResult(exit_code=run_openhands(prompt, config))

        store_path = config.state_dir / "advisor-events.sqlite3"
        with ActiveGitHubWatcher(
            self.github_mailbox,
            store_path,
            known_keys=event_keys,
            poll_interval_seconds=self.active_poll_interval_seconds,
        ) as watcher:
            exit_code = run_openhands(prompt, config)
        with AdvisorEventStore(store_path) as store:
            delivered = store.acknowledged(tuple(watcher.observed_keys))
        return TurnResult(
            exit_code=exit_code,
            delivered_event_keys=frozenset(delivered),
        )


class StudentConversationSelector:
    def __init__(self, registry: AssignmentConversationRegistry):
        self.registry = registry

    def __call__(self, events: Sequence[ControllerEvent]) -> UUID:
        monitor_ids = {
            UUID(str(event.payload["conversation_id"]))
            for event in events
            if event.kind == "training_monitor"
        }
        if monitor_ids:
            if len(monitor_ids) != 1:
                raise RuntimeError("monitor events target multiple conversations")
            return monitor_ids.pop()
        assignments = [event for event in events if event.kind == "student_assignment"]
        if assignments:
            assignment = assignments[0]
            return self.registry.for_assignment(
                str(assignment.payload["assignment_id"]),
                str(assignment.payload["revision_id"]),
            )
        issues = [event for event in events if event.kind == "human_issue"]
        if issues:
            return self.registry.for_assignment(
                f"human-issue-{issues[0].payload['number']}",
                "thread",
            )
        return self.registry.for_assignment("student-control", "current")


class StudentWorkspaceReconciler:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def __call__(self, events: Sequence[ControllerEvent]) -> None:
        assignments = [event for event in events if event.kind == "student_assignment"]
        if not assignments:
            return
        head_ref = str(assignments[0].payload["head_ref"])
        subprocess.run(
            ["git", "fetch", "origin", head_ref],
            cwd=self.workspace,
            check=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "checkout", "-B", head_ref, "FETCH_HEAD"],
            cwd=self.workspace,
            check=True,
            timeout=300,
        )


class Controller:
    """Poll, reconcile, run one turn, and immediately verify GitHub again."""

    def __init__(
        self,
        *,
        role: Literal["advisor", "student"],
        mailbox: Mailbox,
        turns: TurnRunner,
        conversation_id: UUID,
        full_prompt: str,
        conversation_ledger: ConversationLedger | None = None,
        system_context: str = "",
        system_context_ledger: SystemContextLedger | None = None,
        conversation_for_events: (
            Callable[[Sequence[ControllerEvent]], UUID] | None
        ) = None,
        reconcile: Callable[[Sequence[ControllerEvent]], None] | None = None,
        progress: ProgressLease | None = None,
        operation_timeout_seconds: float = 300,
        turn_timeout_seconds: float = 3660,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 600,
        jitter_seconds: float = 120,
    ):
        if poll_interval_seconds < 0 or jitter_seconds < 0:
            raise ValueError("poll and jitter intervals must not be negative")
        if operation_timeout_seconds <= 0 or turn_timeout_seconds <= 0:
            raise ValueError("controller phase timeouts must be positive")
        self.role = role
        self.mailbox = mailbox
        self.turns = turns
        self.conversation_id = conversation_id
        self.conversation_for_events = conversation_for_events
        self.reconcile = reconcile
        self.progress = progress
        self.operation_timeout_seconds = operation_timeout_seconds
        self.turn_timeout_seconds = turn_timeout_seconds
        self.full_prompt = full_prompt.strip()
        self.conversation_ledger = conversation_ledger
        self.system_context = system_context.strip()
        self.system_context_ledger = system_context_ledger
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.jitter_seconds = jitter_seconds
        self._started: set[UUID] = set()
        self._visible: set[str] = set()

    def run(self, *, max_cycles: int | None = None) -> None:
        cycles = 0
        poll_failures = 0
        turn_failures = 0
        while max_cycles is None or cycles < max_cycles:
            try:
                self._publish_progress("poll")
                events = self._new_events(self.mailbox.poll())
                poll_failures = 0
            except Exception as error:  # noqa: BLE001
                poll_failures += 1
                cycles += 1
                print(
                    f"SENPAI_POLL_ERROR {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                delay = min(
                    self.poll_interval_seconds,
                    2 ** min(poll_failures, 8),
                )
                self._sleep("poll-backoff", delay)
                continue
            turn_failed = False
            while events:
                try:
                    if self.reconcile is not None:
                        self._publish_progress("reconcile")
                        self.reconcile(events)
                    conversation_id = (
                        self.conversation_for_events(events)
                        if self.conversation_for_events is not None
                        else self.conversation_id
                    )
                    continuing = self._has_started(conversation_id)
                    refresh_system_context = (
                        continuing
                        and self.system_context_ledger is not None
                        and not self.system_context_ledger.is_current(
                            conversation_id,
                            self.system_context,
                        )
                    )
                    prompt = self._prompt(
                        events,
                        continuing=continuing,
                        refresh_system_context=refresh_system_context,
                    )
                    self._mark_started(conversation_id)
                    self._publish_progress(
                        "openhands-turn",
                        self.turn_timeout_seconds,
                    )
                    result = self.turns.run(
                        prompt,
                        conversation_id=conversation_id,
                        continue_session=continuing,
                        event_keys=frozenset(event.dedupe_key for event in events),
                    )
                    if self.system_context_ledger is not None:
                        self.system_context_ledger.mark(
                            conversation_id,
                            self.system_context,
                        )
                except Exception as error:  # noqa: BLE001
                    turn_failures += 1
                    self._visible.difference_update(
                        event.dedupe_key for event in events
                    )
                    print(
                        f"SENPAI_TURN_EXCEPTION {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    delay = min(
                        self.poll_interval_seconds,
                        2 ** min(turn_failures, 8),
                    )
                    self._sleep("turn-backoff", delay)
                    turn_failed = True
                    break
                if result.exit_code == 0:
                    turn_failures = 0
                    self.mailbox.acknowledge(
                        tuple(event.dedupe_key for event in events)
                    )
                else:
                    turn_failures += 1
                    self._visible.difference_update(
                        event.dedupe_key for event in events
                    )
                    print(
                        "SENPAI_TURN_ERROR "
                        f"exit_code={result.exit_code} "
                        f"conversation_id={conversation_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    delay = min(
                        self.poll_interval_seconds,
                        2 ** min(turn_failures, 8),
                    )
                    self._sleep("turn-backoff", delay)
                    turn_failed = True
                    break
                self._visible.update(result.delivered_event_keys)
                # Post-turn reconciliation avoids waiting one heartbeat for work
                # that appeared while OpenHands was reasoning.
                try:
                    self._publish_progress("poll")
                    events = self._new_events(self.mailbox.poll())
                except Exception as error:  # noqa: BLE001
                    print(
                        f"SENPAI_POST_TURN_POLL_ERROR {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    events = ()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
            if turn_failed:
                continue
            self._sleep(
                "sleep",
                self.poll_interval_seconds + random.uniform(0, self.jitter_seconds),
            )

    def _publish_progress(
        self,
        phase: str,
        timeout_seconds: float | None = None,
    ) -> None:
        if self.progress is not None:
            self.progress.update(
                phase,
                timeout_seconds or self.operation_timeout_seconds,
            )

    def _sleep(self, phase: str, seconds: float) -> None:
        self._publish_progress(
            phase,
            max(seconds + self.operation_timeout_seconds, 1),
        )
        self.sleep(seconds)

    def _has_started(self, conversation_id: UUID) -> bool:
        return conversation_id in self._started or (
            self.conversation_ledger is not None
            and self.conversation_ledger.has_started(conversation_id)
        )

    def _mark_started(self, conversation_id: UUID) -> None:
        self._started.add(conversation_id)
        if self.conversation_ledger is not None:
            self.conversation_ledger.mark_started(conversation_id)

    def _new_events(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ControllerEvent, ...]:
        current = {event.dedupe_key for event in events}
        new = tuple(event for event in events if event.dedupe_key not in self._visible)
        self._visible = current
        return new

    def _prompt(
        self,
        events: Sequence[ControllerEvent],
        *,
        continuing: bool,
        refresh_system_context: bool = False,
    ) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_prompt = "\n\n".join(event.to_prompt() for event in events)
        if not continuing:
            return (
                f"{self.full_prompt}\n\nCurrent time (UTC): {now}\n\n"
                f"# Current GitHub state\n\n{event_prompt}"
            )
        prompt = (
            f"Continue the {self.role} loop. Current time (UTC): {now}. "
            "GitHub now contains the following actionable state:\n\n"
            f"{event_prompt}"
        )
        if refresh_system_context:
            prompt += (
                "\n\n# Updated Senpai system context\n\n"
                "The deployed harness or role charter changed since this "
                "conversation last ran. Treat the following as the current "
                "Senpai operating context:\n\n"
                f"{self.system_context}"
            )
        return prompt


def _full_prompt(role: Literal["advisor", "student"], env: Mapping[str, str]) -> str:
    workspace = Path(env["SENPAI_OPENHANDS_WORKSPACE"]).resolve()
    instructions = workspace / "instructions" / f"prompt-{role}.md"
    program = workspace / "program.md"
    prompt = (
        "# Research programme\n\n"
        f"{program.read_text(encoding='utf-8').strip()}\n\n"
        f"# {role.title()} task\n\n"
        f"{Template(instructions.read_text(encoding='utf-8')).safe_substitute(env).strip()}"
    )
    encoded_extra = env.get("EXTRA_INSTRUCTIONS_B64")
    if encoded_extra:
        extra = b64decode(encoded_extra, validate=True).decode()
        prompt += f"\n\n# Additional launch instructions\n\n{extra.strip()}"
    identity = (
        f"Role: {role}; repository: {env['GH_REPO']}; "
        f"advisor branch: {env['ADVISOR_BRANCH']}; "
        f"W&B: {env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}."
    )
    if role == "advisor":
        identity += f" Students: {env.get('STUDENT_NAMES', '')}."
    else:
        identity += f" Student: {env['STUDENT_NAME']}."
    return f"{prompt}\n\n# Runtime identity\n\n{identity}"


def _role_interval(
    env: Mapping[str, str],
    role: Literal["advisor", "student"],
    suffix: str,
    default: float,
) -> float:
    role_key = f"SENPAI_{role.upper()}_{suffix}"
    shared_key = f"SENPAI_{suffix}"
    return float(env.get(role_key, env.get(shared_key, str(default))))


def controller_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    import argparse

    progress = ProgressLease(Path(env[LEASE_ENV])) if env.get(LEASE_ENV) else None
    if progress is not None:
        progress.update("startup", 300)

    from senpai_agent.openhands_runner import (
        child_process_config,
        parse_runner_args,
        read_role_instructions,
        resolve_config,
    )
    from senpai_agent.tools import (
        close_training_runtimes,
        training_runtime,
    )
    from senpai_agent.weave_monitoring import finish_weave_monitoring

    parser = argparse.ArgumentParser(
        description="Run the portable Senpai GitHub/OpenHands controller."
    )
    parser.add_argument("role", choices=("advisor", "student"))
    args = parser.parse_args(argv)
    role = args.role
    if env.get("SENPAI_ROLE") != role:
        raise RuntimeError(f"SENPAI_ROLE must be {role}")
    human_issues = env.get("SENPAI_ENABLE_HUMAN_ISSUES", "true").lower()
    if human_issues not in {"true", "false"}:
        raise RuntimeError("SENPAI_ENABLE_HUMAN_ISSUES must be true or false")

    max_turns = int(env.get("SENPAI_OPENHANDS_MAX_TURNS", "100000"))
    runner_config = resolve_config(
        parse_runner_args(["--max-turns", str(max_turns)]),
        env,
    )
    os.environ.pop(runner_config.api_key_env, None)
    github_mailbox = GitHubMailbox(
        repo=runner_config.github_repo,
        token=runner_config.github_token,
        role=role,
        advisor_branch=env["ADVISOR_BRANCH"],
        students=tuple(
            student.strip()
            for student in env.get("STUDENT_NAMES", "").split(",")
            if student.strip()
        ),
        student_name=env.get("STUDENT_NAME"),
        stale_wip_seconds=int(env.get("SENPAI_STALE_WIP_SECONDS", "7200")),
        trusted_actor=runner_config.github_trusted_actor,
        human_issues_enabled=human_issues == "true",
    )
    mailbox: Mailbox = github_mailbox
    conversation_selector = None
    reconcile = None

    if role == "advisor":
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalAdvisorMailbox(runner_config.state_dir / "advisor-events.sqlite3"),
        )
    else:
        training, monitor_store = training_runtime(
            runner_config.workspace,
            runner_config.state_dir / "training",
            max_timeout_seconds=runner_config.training_max_timeout_seconds,
        )
        metrics = WandbMetricSource(
            env["WANDB_ENTITY"],
            env["WANDB_PROJECT"],
        )
        triage = OpenHandsMonitorTriage(
            replace(
                child_process_config(runner_config),
                enable_browser=False,
            ),
            timeout_seconds=float(
                env.get("SENPAI_MONITOR_TRIAGE_TIMEOUT_SECONDS", "300")
            ),
        )
        mailbox = CompositeMailbox(
            github_mailbox,
            MonitorMailbox(
                TrainingMonitorEngine(monitor_store, training, metrics),
                monitor_store,
                triage,
            ),
        )
        registry = AssignmentConversationRegistry(
            runner_config.state_dir / "student-conversations.json"
        )
        conversation_selector = StudentConversationSelector(registry)
        reconcile = StudentWorkspaceReconciler(runner_config.workspace)

    turns = OpenHandsTurnRunner(
        runner_config,
        github_mailbox=github_mailbox if role == "advisor" else None,
        active_poll_interval_seconds=float(
            env.get("SENPAI_ACTIVE_GITHUB_POLL_INTERVAL_S", "30")
        ),
    )
    controller = Controller(
        role=role,
        mailbox=mailbox,
        turns=turns,
        conversation_id=runner_config.conversation_id,
        conversation_ledger=ConversationLedger(
            runner_config.state_dir / "started-conversations.json"
        ),
        system_context=compose_system_instructions(
            read_role_instructions(runner_config.harness_file),
            read_role_instructions(runner_config.role_file),
        ),
        system_context_ledger=SystemContextLedger(
            runner_config.state_dir / "system-context-revisions.json"
        ),
        conversation_for_events=conversation_selector,
        reconcile=reconcile,
        progress=progress,
        operation_timeout_seconds=float(
            env.get("SENPAI_CONTROLLER_OPERATION_TIMEOUT_SECONDS", "300")
        ),
        turn_timeout_seconds=runner_config.timeout_seconds + 60,
        full_prompt=_full_prompt(role, env),
        poll_interval_seconds=_role_interval(
            env,
            role,
            "POLL_INTERVAL_S",
            600,
        ),
        jitter_seconds=_role_interval(
            env,
            role,
            "POLL_JITTER_S",
            120,
        ),
    )

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        controller.run()
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        close_training_runtimes()
        finish_weave_monitoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(controller_main())
