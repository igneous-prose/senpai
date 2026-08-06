"""Level-triggered GitHub events for advisor and student controllers."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from urllib.parse import quote, urlencode

from pydantic import SecretStr

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.github_http import GitHubReader, GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentRecord,
    parse_assignment_feedback_markers,
    parse_assignment_markers,
)


_TRUSTED_HUMAN_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_FEEDBACK_KEY_PREFIX = "student_pr_feedback:"
_FEEDBACK_EXCERPT_BYTES = 4_000
_DEFAULT_FEEDBACK_BATCH_EVENTS = 8
_DEFAULT_FEEDBACK_BATCH_BYTES = 32_000


@dataclass(frozen=True, slots=True)
class _FeedbackBinding:
    assignment_id: str
    revision_id: str
    acknowledged: bool = False


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
        feedback_path: Path | None = None,
        feedback_batch_events: int = _DEFAULT_FEEDBACK_BATCH_EVENTS,
        feedback_batch_bytes: int = _DEFAULT_FEEDBACK_BATCH_BYTES,
    ):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        if role == "student" and not student_name:
            raise ValueError("student mailbox requires student_name")
        if feedback_batch_events <= 0 or feedback_batch_bytes <= 0:
            raise ValueError("feedback batch limits must be positive")
        self.repo = repo
        self.role = role
        self.advisor_branch = advisor_branch
        self.students = tuple(student for student in students if student)
        self.student_name = student_name
        self.stale_wip_seconds = stale_wip_seconds
        self.human_issues_enabled = human_issues_enabled
        self.feedback_path = feedback_path
        self.feedback_batch_events = feedback_batch_events
        self.feedback_batch_bytes = feedback_batch_bytes
        self._memory_feedback: dict[str, _FeedbackBinding] = {}
        self._github = GitHubReader(
            token,
            api_url=api_url,
            trusted_actor=trusted_actor,
        )

    def poll(self) -> tuple[ControllerEvent, ...]:
        pulls = self._pulls()
        issues = self._issues() if self.human_issues_enabled else ()
        if self.role == "advisor":
            return self._advisor_events(pulls, issues)
        return self._student_events(pulls, issues)

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        """Mark persisted feedback delivered after a successful controller turn."""
        feedback_keys = {
            key for key in dedupe_keys if key.startswith(_FEEDBACK_KEY_PREFIX)
        }
        if not feedback_keys:
            return
        ledger = self._read_feedback_ledger()
        missing = feedback_keys - ledger.keys()
        if missing:
            raise RuntimeError(
                "cannot acknowledge unseen student PR feedback: "
                f"{', '.join(sorted(missing))}"
            )
        changed = False
        for key in feedback_keys:
            binding = ledger[key]
            if binding.acknowledged:
                continue
            ledger[key] = _FeedbackBinding(
                assignment_id=binding.assignment_id,
                revision_id=binding.revision_id,
                acknowledged=True,
            )
            changed = True
        if changed:
            self._write_feedback_ledger(ledger)

    def _advisor_events(
        self,
        pulls: Sequence[dict[str, object]],
        issues: Sequence[dict[str, object]],
    ) -> tuple[ControllerEvent, ...]:
        events: list[ControllerEvent] = []
        active_assignments: list[tuple[dict[str, object], AssignmentRecord]] = []
        active_by_student: dict[str, list[int]] = {
            student: [] for student in self.students
        }
        now = datetime.now(UTC)
        for pull in pulls:
            labels = _label_names(pull)
            number = int(pull["number"])
            head_sha = str(_object(pull["head"])["sha"])
            students = sorted(
                label.removeprefix("student:")
                for label in labels
                if label.startswith("student:")
            )
            if "status:wip" in labels:
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
            if {"status:wip", "status:review"} & labels:
                try:
                    assignments = parse_assignment_markers(
                        str(pull.get("body") or "")
                    )
                except ValueError:
                    assignments = []
                if len(assignments) == 1:
                    active_assignments.append((pull, assignments[0]))
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
        if active_assignments:
            try:
                current_base_sha = self._advisor_head_sha()
            except (GitHubReadError, TypeError) as error:
                print(
                    "SENPAI_BASELINE_WATCH_ERROR "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                for pull, assignment in active_assignments:
                    if assignment.base_sha == current_base_sha:
                        continue
                    number = int(pull["number"])
                    events.append(
                        ControllerEvent(
                            kind="baseline_advanced",
                            dedupe_key=(
                                f"baseline_advanced:{number}:"
                                f"{assignment.base_sha}:{current_base_sha}"
                            ),
                            payload={
                                **_pull_payload(pull),
                                "assignment_id": assignment.assignment_id,
                                "revision_id": assignment.revision_id,
                                "student": assignment.student,
                                "base_ref": assignment.base_ref,
                                "assigned_base_sha": assignment.base_sha,
                                "current_base_sha": current_base_sha,
                                "compare_url": (
                                    f"{str(pull['html_url']).rsplit('/pull/', 1)[0]}"
                                    f"/compare/{assignment.base_sha}..."
                                    f"{current_base_sha}"
                                ),
                            },
                        )
                    )
        events.extend(self._human_issue_events(issues))
        return tuple(events)

    def _advisor_head_sha(self) -> str:
        ref = self._github.get(
            f"/repos/{self.repo}/git/ref/heads/"
            f"{quote(self.advisor_branch, safe='')}"
        )
        if not isinstance(ref, dict):
            raise TypeError("GitHub advisor branch ref is not an object")
        target = ref.get("object")
        if not isinstance(target, dict) or not isinstance(target.get("sha"), str):
            raise TypeError("GitHub advisor branch ref has no target SHA")
        return target["sha"]

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
                *self._human_issue_events(issues),
            )

        events: list[ControllerEvent] = []
        if assigned:
            pull = assigned[0]
            try:
                markers = parse_assignment_markers(str(pull.get("body") or ""))
                if len(markers) != 1:
                    raise ValueError(
                        "assigned PR must contain exactly one Senpai assignment marker"
                    )
            except ValueError as error:
                number = int(pull["number"])
                head_sha = str(_object(pull["head"])["sha"])
                events.append(
                    ControllerEvent(
                        kind="malformed_assignment",
                        dedupe_key=f"malformed_assignment:{number}:{head_sha}",
                        payload={
                            **_pull_payload(pull),
                            "error": f"Assigned PR #{number}: {error}",
                        },
                    )
                )
            else:
                assignment = markers[0]
                feedback = self._student_pr_feedback_events(pull, assignment)
                prior_revision_pending = any(
                    event.payload["assignment_id"] != assignment.assignment_id
                    or event.payload["revision_id"] != assignment.revision_id
                    for event in feedback
                )
                if (
                    "status:wip" in _label_names(pull)
                    and not prior_revision_pending
                ):
                    events.append(
                        ControllerEvent(
                            kind="student_assignment",
                            dedupe_key=(
                                f"student_assignment:{assignment.assignment_id}:"
                                f"{assignment.revision_id}"
                            ),
                            payload={
                                **_pull_payload(pull),
                                "assignment_id": assignment.assignment_id,
                                "revision_id": assignment.revision_id,
                                "base_ref": assignment.base_ref,
                            },
                        )
                    )
                events.extend(feedback)
        events.extend(self._human_issue_events(issues))
        return tuple(events)

    def _student_pr_feedback_events(
        self,
        pull: Mapping[str, object],
        assignment: AssignmentRecord,
    ) -> list[ControllerEvent]:
        number = int(pull["number"])
        sources: list[tuple[str, str]] = []
        if comments_url := pull.get("comments_url"):
            sources.append(("issue_comment", f"{comments_url}?per_page=100"))
        if pull_url := pull.get("url"):
            sources.append(
                ("review", f"{str(pull_url).rstrip('/')}/reviews?per_page=100")
            )
        if comments_url := pull.get("review_comments_url"):
            sources.append(("inline_comment", f"{comments_url}?per_page=100"))
        if not sources:
            return []

        actor = self._github.actor()
        feedback_by_surface = {
            surface: self._github.objects(url) for surface, url in sources
        }
        submitted_review_ids = {
            int(item["id"])
            for item in feedback_by_surface.get("review", ())
            if item.get("submitted_at") is not None
        }
        events: list[ControllerEvent] = []
        for surface, _url in sources:
            for item in feedback_by_surface[surface]:
                if surface == "review" and item.get("submitted_at") is None:
                    continue
                if (
                    surface == "inline_comment"
                    and int(item["pull_request_review_id"])
                    not in submitted_review_ids
                ):
                    continue
                trusted = _trusted_feedback(
                    item,
                    actor=actor,
                    repo=self.repo,
                    pr_number=number,
                    assignment=assignment,
                )
                if trusted is None:
                    continue
                binding, feedback_body = trusted
                feedback_id = int(item["id"])
                created_at = str(
                    item["submitted_at"] if surface == "review" else item["created_at"]
                )
                message, message_truncated = _feedback_excerpt(
                    feedback_body,
                    limit=_FEEDBACK_EXCERPT_BYTES,
                )
                payload: dict[str, object] = {
                    "number": number,
                    "pr_url": str(pull["html_url"]),
                    "feedback_url": str(item["html_url"]),
                    "feedback_id": feedback_id,
                    "feedback_type": surface,
                    "assignment_id": binding.assignment_id,
                    "revision_id": binding.revision_id,
                    "author": str(_object(item["user"])["login"]),
                    "author_association": str(item["author_association"]),
                    "message": message,
                    "created_at": created_at,
                }
                if message_truncated:
                    payload.update(
                        message_truncated=True,
                        full_message_instruction=(
                            "Open feedback_url to read the omitted text."
                        ),
                    )
                if surface == "review":
                    payload["state"] = str(item["state"])
                elif surface == "inline_comment":
                    payload["path"] = str(item["path"])
                    line = item.get("line") or item.get("original_line")
                    if line is not None:
                        payload["line"] = int(line)
                events.append(
                    ControllerEvent(
                        kind="student_pr_feedback",
                        dedupe_key=(
                            f"student_pr_feedback:{surface}:{number}:{feedback_id}"
                        ),
                        payload=payload,
                    )
                )
        events.sort(
            key=lambda event: (
                _github_datetime(str(event.payload["created_at"])),
                str(event.payload["feedback_type"]),
                int(event.payload["feedback_id"]),
            )
        )
        ledger = self._read_feedback_ledger()
        ledger_changed = False
        bound_events: list[ControllerEvent] = []
        for event in events:
            binding = ledger.get(event.dedupe_key)
            if binding is None:
                binding = _FeedbackBinding(
                    assignment_id=str(event.payload["assignment_id"]),
                    revision_id=str(event.payload["revision_id"]),
                )
                ledger[event.dedupe_key] = binding
                ledger_changed = True
            bound_events.append(
                ControllerEvent(
                    kind=event.kind,
                    dedupe_key=event.dedupe_key,
                    payload={
                        **event.payload,
                        "assignment_id": binding.assignment_id,
                        "revision_id": binding.revision_id,
                    },
                )
            )
        if ledger_changed:
            self._write_feedback_ledger(ledger)
        pending = [
            event
            for event in bound_events
            if not ledger[event.dedupe_key].acknowledged
        ]
        prior_revision = [
            event
            for event in pending
            if event.payload["assignment_id"] != assignment.assignment_id
            or event.payload["revision_id"] != assignment.revision_id
        ]
        return self._feedback_batch(prior_revision or pending)

    def _feedback_batch(
        self,
        events: Iterable[ControllerEvent],
    ) -> list[ControllerEvent]:
        selected: list[ControllerEvent] = []
        prompt_bytes = 0
        for event in events:
            if len(selected) >= self.feedback_batch_events:
                break
            event_bytes = len(event.to_prompt().encode())
            separator_bytes = 2 if selected else 0
            if event_bytes > self.feedback_batch_bytes:
                if selected:
                    break
                raise RuntimeError(
                    f"student PR feedback event exceeds "
                    f"{self.feedback_batch_bytes} prompt bytes"
                )
            if (
                prompt_bytes + separator_bytes + event_bytes
                > self.feedback_batch_bytes
            ):
                break
            selected.append(event)
            prompt_bytes += separator_bytes + event_bytes
        return selected

    def _read_feedback_ledger(self) -> dict[str, _FeedbackBinding]:
        if self.feedback_path is None:
            return dict(self._memory_feedback)
        if not self.feedback_path.exists():
            return {}
        value = json.loads(self.feedback_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(
                f"invalid student PR feedback ledger: {self.feedback_path}"
            )
        ledger: dict[str, _FeedbackBinding] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key.startswith(_FEEDBACK_KEY_PREFIX)
                or not isinstance(item, dict)
                or not isinstance(item.get("assignment_id"), str)
                or not isinstance(item.get("revision_id"), str)
                or not isinstance(item.get("acknowledged"), bool)
            ):
                raise RuntimeError(
                    f"invalid student PR feedback ledger: {self.feedback_path}"
                )
            ledger[key] = _FeedbackBinding(
                assignment_id=item["assignment_id"],
                revision_id=item["revision_id"],
                acknowledged=item["acknowledged"],
            )
        return ledger

    def _write_feedback_ledger(
        self,
        ledger: Mapping[str, _FeedbackBinding],
    ) -> None:
        if self.feedback_path is None:
            self._memory_feedback = dict(ledger)
            return
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.feedback_path.with_suffix(
            f"{self.feedback_path.suffix}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    key: {
                        "assignment_id": binding.assignment_id,
                        "revision_id": binding.revision_id,
                        "acknowledged": binding.acknowledged,
                    }
                    for key, binding in sorted(ledger.items())
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.feedback_path)

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
            actor = self._github.actor()
            messages = []
            for item in (issue, *self._issue_comments(issue)):
                user = _object(item["user"])
                if user.get("type") != "User":
                    continue
                messages.append(
                    {
                        "id": int(item["id"]),
                        "author": str(user["login"]),
                        "body": str(item.get("body") or ""),
                        "created_at": str(item["created_at"]),
                    }
                )
            human_messages = [
                message
                for message in messages
                if str(message["author"]).casefold() != actor.casefold()
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

    def _issue_comments(
        self,
        issue: Mapping[str, object],
    ) -> list[dict[str, object]]:
        comments_url = issue.get("comments_url")
        return self._github.objects(str(comments_url)) if comments_url else []

    def _pulls(self) -> list[dict[str, object]]:
        query = urlencode(
            {
                "state": "open",
                "base": self.advisor_branch,
                "per_page": 100,
            }
        )
        return self._github.objects(f"/repos/{self.repo}/pulls?{query}")

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
            for issue in self._github.objects(f"/repos/{self.repo}/issues?{query}")
            if "pull_request" not in issue
        ]


class ActiveGitHubWatcher:
    """Feed new GitHub state into a running agent at SDK-safe boundaries."""

    def __init__(
        self,
        mailbox: GitHubMailbox,
        store_path: Path,
        *,
        known_keys: frozenset[str],
        poll_interval_seconds: float = 30,
        map_event: Callable[[ControllerEvent], AdvisorEvent | None] | None = None,
    ):
        self.mailbox = mailbox
        self.store_path = store_path
        self.known_keys = set(known_keys)
        self.poll_interval_seconds = poll_interval_seconds
        self.map_event = map_event or _advisor_event
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
                        local_event = self.map_event(event)
                        if local_event is None:
                            continue
                        store.enqueue(local_event)
                        self.observed_keys.add(local_event.dedupe_key)
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


def _advisor_event(event: ControllerEvent) -> AdvisorEvent:
    return AdvisorEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload=event.payload,
    )


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


def _trusted_feedback(
    item: Mapping[str, object],
    *,
    actor: str,
    repo: str,
    pr_number: int,
    assignment: AssignmentRecord,
) -> tuple[_FeedbackBinding, str] | None:
    user = item.get("user")
    if not isinstance(user, dict):
        return None
    body = str(item.get("body") or "")
    current = _FeedbackBinding(
        assignment_id=assignment.assignment_id,
        revision_id=assignment.revision_id,
    )
    same_actor = str(user.get("login") or "").casefold() == actor.casefold()
    trusted_human = (
        user.get("type") == "User"
        and item.get("author_association") in _TRUSTED_HUMAN_ASSOCIATIONS
    )
    if not same_actor:
        if not trusted_human:
            return None
        return current, body
    protocol_markers = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("<!-- senpai-")
        and line.strip().endswith(" -->")
    ]
    if not protocol_markers:
        return (current, body) if trusted_human else None
    try:
        records = parse_assignment_feedback_markers(body)
    except ValueError:
        return None
    if len(protocol_markers) != 1 or len(records) != 1:
        return None
    record = records[0]
    if (
        record.repo != repo
        or record.pr_number != pr_number
        or record.assignment_id != assignment.assignment_id
    ):
        return None
    content = "\n".join(
        line for line in body.splitlines() if line.strip() != protocol_markers[0]
    ).strip()
    if not content:
        return None
    return (
        _FeedbackBinding(
            assignment_id=record.assignment_id,
            revision_id=record.revision_id,
        ),
        content,
    )


def _github_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _bounded_text(value: str, *, limit: int) -> str:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value
    return encoded[-limit:].decode(errors="ignore")


def _feedback_excerpt(value: str, *, limit: int) -> tuple[str, bool]:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value, False
    marker = "\n\n[... middle omitted; open feedback_url for full text ...]\n\n".encode()
    content_bytes = limit - len(marker)
    head_bytes = 3 * content_bytes // 4
    excerpt = (
        encoded[:head_bytes].decode(errors="ignore")
        + marker.decode()
        + encoded[-(content_bytes - head_bytes) :].decode(errors="ignore")
    )
    return excerpt, True
