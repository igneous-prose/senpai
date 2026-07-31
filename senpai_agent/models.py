"""Typed terminal experiment results and their GitHub marker format."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


_NonEmptyString = Annotated[str, Field(min_length=1)]


class AssignmentKey(Contract):
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    expected_head_sha: _NonEmptyString
    student: _NonEmptyString


class AssignmentRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    student: _NonEmptyString
    base_ref: _NonEmptyString
    base_sha: _NonEmptyString
    head_ref: _NonEmptyString
    head_sha: _NonEmptyString


class RevisionRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    revision_id: _NonEmptyString
    requested_head_sha: _NonEmptyString


class DispositionRecord(Contract):
    schema_version: Literal[1] = 1
    repo: _NonEmptyString
    pr_number: int = Field(gt=0)
    assignment_id: _NonEmptyString
    head_sha: _NonEmptyString


class ResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    CANCELLED = "cancelled"


class WandbRunRef(Contract):
    run_id: _NonEmptyString
    url: _NonEmptyString
    state: Literal["finished", "failed", "crashed", "killed"]


class MetricComparison(Contract):
    name: _NonEmptyString
    direction: Literal["minimize", "maximize"]
    baseline: float | None = None
    candidate: float
    delta: float | None = None


class ExperimentResult(Contract):
    schema_version: Literal[1] = 1
    assignment: AssignmentKey
    status: ResultStatus
    hypothesis: _NonEmptyString
    summary: Annotated[str, Field(min_length=1, max_length=4_000)]
    runs: tuple[WandbRunRef, ...]
    primary_metric: MetricComparison | None = None
    commit_sha: _NonEmptyString


class ResultMarkerError(ValueError):
    """A result marker line is malformed, unsupported, or schema-invalid."""


_RESULT_PREFIX = "<!-- senpai-result:"
_RESULT_MARKER = re.compile(
    r"<!-- senpai-result:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)
_ASSIGNMENT_PREFIX = "<!-- senpai-assignment:"
_ASSIGNMENT_MARKER = re.compile(
    r"<!-- senpai-assignment:v(?P<version>[0-9]+) "
    r"(?P<payload>\{.*\}) -->"
)


def _marker_payload(value: Contract) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace(">", r"\u003e")


def render_assignment_marker(assignment: AssignmentRecord) -> str:
    return f"<!-- senpai-assignment:v1 {_marker_payload(assignment)} -->"


def render_revision_marker(revision: RevisionRecord) -> str:
    return f"<!-- senpai-revision:v1 {_marker_payload(revision)} -->"


def render_disposition_marker(disposition: DispositionRecord) -> str:
    return f"<!-- senpai-disposition:v1 {_marker_payload(disposition)} -->"


def parse_assignment_markers(body: str) -> tuple[AssignmentRecord, ...]:
    assignments: list[AssignmentRecord] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        if not line.startswith(_ASSIGNMENT_PREFIX):
            continue
        marker = _ASSIGNMENT_MARKER.fullmatch(line)
        if marker is None or marker.group("version") != "1":
            raise ValueError(
                f"malformed or unsupported Senpai assignment marker on line "
                f"{line_number}"
            )
        try:
            assignments.append(
                AssignmentRecord.model_validate_json(marker.group("payload"))
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid Senpai assignment marker on line {line_number}"
            ) from error
    return tuple(assignments)


def render_result_marker(result: ExperimentResult) -> str:
    return f"<!-- senpai-result:v1 {_marker_payload(result)} -->"


def render_result_comment(result: ExperimentResult) -> str:
    lines = [
        render_result_marker(result),
        "",
        f"Status: {result.status.value}",
        f"Commit: `{result.commit_sha}`",
        "",
        result.summary,
    ]
    if result.runs:
        lines.extend(
            [
                "",
                "W&B runs:",
                *(f"- {run.url}" for run in result.runs),
            ]
        )
    return "\n".join(lines)


def parse_result_markers(comment_body: str) -> tuple[ExperimentResult, ...]:
    results: list[ExperimentResult] = []
    for line_number, line in enumerate(comment_body.splitlines(), start=1):
        if not line.startswith(_RESULT_PREFIX):
            continue

        marker = _RESULT_MARKER.fullmatch(line)
        if marker is None:
            raise ResultMarkerError(
                f"malformed senpai result marker on line {line_number}"
            )

        version = marker.group("version")
        if version != "1":
            raise ResultMarkerError(
                f"unknown senpai result marker version v{version} on line {line_number}"
            )

        try:
            payload = json.loads(marker.group("payload"))
        except json.JSONDecodeError as error:
            raise ResultMarkerError(
                f"invalid JSON in senpai result marker on line {line_number}"
            ) from error

        try:
            results.append(ExperimentResult.model_validate(payload))
        except ValidationError as error:
            raise ResultMarkerError(
                f"invalid senpai result payload on line {line_number}: {error}"
            ) from error

    return tuple(results)
