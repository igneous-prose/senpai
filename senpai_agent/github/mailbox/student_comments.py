"""Trusted student assignment comments delivered to advisor controllers."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from senpai_agent.github.http import GitHubReadError
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentRecord,
    authoritative_marker_line,
    parse_assignment_comment_markers,
    render_assignment_comment_marker,
)

from .values import (
    FEEDBACK_EXCERPT_BYTES,
    bounded_text,
    github_datetime,
    object_value,
    payload_digest,
)

if TYPE_CHECKING:
    from .core import GitHubMailbox


def student_assignment_comment_events(
    mailbox: GitHubMailbox,
    assignments: Sequence[tuple[Mapping[str, object], AssignmentRecord]],
) -> list[ControllerEvent]:
    """Wake the advisor for trusted typed messages from assigned students."""

    assignments = tuple(
        (pull, assignment)
        for pull, assignment in assignments
        if pull.get("comments_url")
    )
    if not assignments:
        return []
    try:
        actor = mailbox._github.actor()
    except (GitHubReadError, TypeError) as error:
        _report_read_error(f"actor {type(error).__name__}: {error}")
        return []

    events_by_key: dict[str, ControllerEvent] = {}
    for pull, assignment in assignments:
        number = int(pull["number"])
        try:
            comments = mailbox._github.objects(
                f"{pull['comments_url']}?per_page=100"
            )
        except (GitHubReadError, TypeError) as error:
            _report_read_error(f"pr={number} {type(error).__name__}: {error}")
            continue
        for comment in comments:
            event = _comment_event(
                mailbox,
                pull,
                assignment,
                comment,
                actor=actor,
            )
            if event is None:
                continue
            previous = events_by_key.get(event.dedupe_key)
            if previous is None or int(event.payload["github_comment_id"]) < int(
                previous.payload["github_comment_id"]
            ):
                events_by_key[event.dedupe_key] = event

    return sorted(
        events_by_key.values(),
        key=lambda event: (
            github_datetime(str(event.payload["created_at"])),
            int(event.payload["github_comment_id"]),
        ),
    )


def _comment_event(
    mailbox: GitHubMailbox,
    pull: Mapping[str, object],
    assignment: AssignmentRecord,
    comment: Mapping[str, object],
    *,
    actor: str,
) -> ControllerEvent | None:
    try:
        author = str(object_value(comment["user"])["login"])
        body = str(comment.get("body") or "")
        records = parse_assignment_comment_markers(body)
        github_comment_id = int(comment["id"])
        comment_url = str(comment["html_url"])
        created_at = str(comment["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if author.casefold() != actor.casefold() or len(records) != 1:
        return None
    record = records[0]
    number = int(pull["number"])
    if (
        record.repo != mailbox.repo
        or record.pr_number != number
        or record.assignment_id != assignment.assignment_id
        or record.revision_id != assignment.revision_id
        or record.student != assignment.student
        or authoritative_marker_line(body)
        != render_assignment_comment_marker(record)
    ):
        return None
    message = "\n".join(body.splitlines()[1:]).strip()
    if not message:
        return None
    payload = {
        "number": number,
        "pr_url": str(pull["html_url"]),
        "comment_url": comment_url,
        "github_comment_id": github_comment_id,
        "comment_id": record.comment_id,
        "assignment_id": record.assignment_id,
        "revision_id": record.revision_id,
        "student": record.student,
        "message": bounded_text(message, limit=FEEDBACK_EXCERPT_BYTES),
        "created_at": created_at,
    }
    semantic_payload = {
        "number": number,
        "assignment_id": record.assignment_id,
        "revision_id": record.revision_id,
        "student": record.student,
        "comment_id": record.comment_id,
        "message": message,
    }
    return ControllerEvent(
        kind="student_assignment_comment",
        dedupe_key=(
            "student_assignment_comment:v1:" + payload_digest(semantic_payload)
        ),
        payload=payload,
    )


def _report_read_error(message: str) -> None:
    print(
        f"SENPAI_STUDENT_COMMENT_READ_ERROR {message}",
        file=sys.stderr,
        flush=True,
    )
