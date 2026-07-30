"""Level-triggered GitHub events for advisor and student controllers."""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from urllib.parse import urlencode

from pydantic import SecretStr

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.github_http import GitHubReader
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import parse_assignment_markers


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
        self.role = role
        self.advisor_branch = advisor_branch
        self.students = tuple(student for student in students if student)
        self.student_name = student_name
        self.stale_wip_seconds = stale_wip_seconds
        self.human_issues_enabled = human_issues_enabled
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
            head_sha = str(_object(pull["head"])["sha"])
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
            actor = self._github.actor()
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
