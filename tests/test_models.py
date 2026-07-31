# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import pytest
from pydantic import ValidationError

from senpai_agent.models import (
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    MetricComparison,
    ResultMarkerError,
    ResultStatus,
    WandbRunRef,
    parse_assignment_markers,
    parse_result_markers,
    render_assignment_marker,
    render_result_comment,
    render_result_marker,
)

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40


def assignment() -> AssignmentRecord:
    return AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-7",
        revision_id="revision-2",
        student="student-one",
        base_ref="schmidhuber",
        base_sha=BASE_SHA,
        head_ref="student-one/lower-lr",
        head_sha=HEAD_SHA,
    )


def result(
    *,
    assignment_id: str = "assignment-7",
    status: ResultStatus = ResultStatus.SUCCEEDED,
    summary: str = "The candidate improved validation loss.",
    runs: tuple[WandbRunRef, ...] | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=7,
            assignment_id=assignment_id,
            revision_id="revision-2",
            expected_head_sha=HEAD_SHA,
            student="student-one",
        ),
        status=status,
        hypothesis="Lowering the learning rate improves generalization.",
        summary=summary,
        runs=(
            runs
            if runs is not None
            else (
                WandbRunRef(
                    run_id="run-123",
                    url="https://wandb.ai/acme/widgets/runs/run-123",
                    state="finished",
                ),
            )
        ),
        primary_metric=MetricComparison(
            name="validation/loss",
            direction="minimize",
            baseline=0.42,
            candidate=0.38,
            delta=-0.04,
        ),
        commit_sha=HEAD_SHA,
    )


def test_contracts_are_immutable_and_reject_unknown_fields():
    assignment = result().assignment

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssignmentKey(
            repo="acme/widgets",
            pr_number=7,
            assignment_id="assignment-7",
            revision_id="revision-2",
            expected_head_sha=HEAD_SHA,
            student="student-one",
            unexpected=True,
        )
    with pytest.raises(ValidationError, match="frozen"):
        assignment.student = "student-two"  # type: ignore[misc]


def test_assignment_marker_is_canonical_and_round_trips():
    marker = render_assignment_marker(assignment())

    assert marker.startswith("<!-- senpai-assignment:v1 ")
    assert "\n" not in marker
    assert parse_assignment_markers(marker) == (assignment(),)


@pytest.mark.parametrize(
    "line",
    [
        "<!-- senpai-assignment:v2 {} -->",
        "<!-- senpai-assignment:v1 not-json -->",
        "<!-- senpai-assignment:v1 {} --> trailing",
    ],
)
def test_assignment_parser_rejects_malformed_marker_lines(line: str):
    with pytest.raises(ValueError, match="assignment marker"):
        parse_assignment_markers(line)


@pytest.mark.parametrize(
    "field",
    [
        "repo",
        "assignment_id",
        "revision_id",
        "expected_head_sha",
        "student",
    ],
)
@pytest.mark.parametrize("empty", ["", " \t\n"])
def test_assignment_key_rejects_empty_identifiers(field: str, empty: str):
    values = {
        "repo": "acme/widgets",
        "pr_number": 7,
        "assignment_id": "assignment-7",
        "revision_id": "revision-2",
        "expected_head_sha": HEAD_SHA,
        "student": "student-one",
    }
    values[field] = empty

    with pytest.raises(ValidationError, match="at least 1 character"):
        AssignmentKey.model_validate(values)


@pytest.mark.parametrize("pr_number", [0, -1])
def test_assignment_key_requires_a_positive_pull_request_number(pr_number: int):
    with pytest.raises(ValidationError, match="greater than 0"):
        AssignmentKey(
            repo="acme/widgets",
            pr_number=pr_number,
            assignment_id="assignment-7",
            revision_id="revision-2",
            expected_head_sha=HEAD_SHA,
            student="student-one",
        )


@pytest.mark.parametrize(
    "status",
    [
        ResultStatus.SUCCEEDED,
        ResultStatus.FAILED,
        ResultStatus.INCONCLUSIVE,
        ResultStatus.CANCELLED,
    ],
)
def test_result_statuses_are_exact_terminal_values(status: ResultStatus):
    assert result(status=status).status is status

    with pytest.raises(ValidationError):
        ExperimentResult.model_validate({**result().model_dump(), "status": "running"})


@pytest.mark.parametrize("state", ["finished", "failed", "crashed", "killed"])
def test_wandb_run_states_are_exact_terminal_values(state: str):
    run = WandbRunRef(
        run_id="run-123",
        url="https://wandb.ai/acme/widgets/runs/run-123",
        state=state,  # type: ignore[arg-type]
    )

    assert run.state == state


def test_non_terminal_wandb_state_is_rejected():
    with pytest.raises(ValidationError):
        WandbRunRef(
            run_id="run-123",
            url="https://wandb.ai/acme/widgets/runs/run-123",
            state="running",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["run_id", "url"])
@pytest.mark.parametrize("empty", ["", " \t\n"])
def test_wandb_run_rejects_empty_identity_fields(field: str, empty: str):
    values = {
        "run_id": "run-123",
        "url": "https://wandb.ai/acme/widgets/runs/run-123",
        "state": "finished",
    }
    values[field] = empty

    with pytest.raises(ValidationError, match="at least 1 character"):
        WandbRunRef.model_validate(values)


@pytest.mark.parametrize("empty", ["", " \t\n"])
def test_metric_name_must_not_be_empty(empty: str):
    with pytest.raises(ValidationError, match="at least 1 character"):
        MetricComparison(
            name=empty,
            direction="minimize",
            baseline=0.42,
            candidate=0.38,
            delta=-0.04,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline", float("nan")),
        ("candidate", float("inf")),
        ("delta", float("-inf")),
    ],
)
def test_metric_comparison_rejects_non_finite_numbers(
    field: str,
    value: float,
):
    values: dict[str, object] = {
        "name": "validation/loss",
        "direction": "minimize",
        "baseline": 0.42,
        "candidate": 0.38,
        "delta": -0.04,
    }
    values[field] = value

    with pytest.raises(ValidationError, match="finite number"):
        MetricComparison.model_validate(values)


def test_result_summary_is_bounded_to_four_thousand_characters():
    assert len(result(summary="x" * 4_000).summary) == 4_000

    with pytest.raises(ValidationError, match="at most 4000 characters"):
        result(summary="x" * 4_001)


@pytest.mark.parametrize("field", ["hypothesis", "summary", "commit_sha"])
@pytest.mark.parametrize("empty", ["", " \t\n"])
def test_result_rejects_empty_semantic_text(field: str, empty: str):
    values = result().model_dump()
    values[field] = empty

    with pytest.raises(ValidationError, match="at least 1 character"):
        ExperimentResult.model_validate(values)


def test_result_schema_version_is_exact_and_primary_metric_can_be_none():
    values = result().model_dump()
    values["primary_metric"] = None
    assert ExperimentResult.model_validate(values).primary_metric is None

    values["schema_version"] = 2
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(values)


def test_result_schema_documents_the_result_commit_identity():
    assignment_schema = AssignmentKey.model_json_schema()["properties"]
    result_schema = ExperimentResult.model_json_schema()["properties"]

    assert "not the current remote branch SHA" in assignment_schema[
        "expected_head_sha"
    ]["description"]
    assert "assignment.expected_head_sha must equal commit_sha" in result_schema[
        "assignment"
    ]["description"]
    assert "Must equal assignment.expected_head_sha" in result_schema["commit_sha"][
        "description"
    ]


def test_marker_is_one_canonical_compact_sorted_json_line():
    marker = render_result_marker(result())

    assert marker == (
        '<!-- senpai-result:v1 {"assignment":{"assignment_id":"assignment-7",'
        '"expected_head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"pr_number":7,"repo":"acme/widgets","revision_id":"revision-2",'
        '"student":"student-one"},"commit_sha":'
        '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","hypothesis":'
        '"Lowering the learning rate improves generalization.",'
        '"primary_metric":{"baseline":0.42,"candidate":0.38,"delta":-0.04,'
        '"direction":"minimize","name":"validation/loss"},'
        '"runs":[{"run_id":"run-123","state":"finished","url":'
        '"https://wandb.ai/acme/widgets/runs/run-123"}],'
        '"schema_version":1,"status":"succeeded","summary":'
        '"The candidate improved validation loss."} -->'
    )
    assert "\n" not in marker


def test_marker_round_trips_the_typed_result():
    original = result()

    assert parse_result_markers(render_result_marker(original)) == (original,)


def test_marker_escapes_html_comment_terminators_and_round_trips_text():
    original = result(summary="The literal --> remains valid result text.")
    marker = render_result_marker(original)

    assert marker.count("-->") == 1
    assert r"The literal --\u003e remains valid result text." in marker
    assert parse_result_markers(marker) == (original,)


def test_visible_result_comment_is_deterministic_and_compact():
    runs = (
        WandbRunRef(
            run_id="run-123",
            url="https://wandb.ai/acme/widgets/runs/run-123",
            state="finished",
        ),
        WandbRunRef(
            run_id="run-456",
            url="https://wandb.ai/acme/widgets/runs/run-456",
            state="failed",
        ),
    )
    experiment_result = result(runs=runs)

    assert render_result_comment(experiment_result) == (
        f"{render_result_marker(experiment_result)}\n\n"
        "Status: succeeded\n"
        f"Commit: `{HEAD_SHA}`\n\n"
        "The candidate improved validation loss.\n\n"
        "W&B runs:\n"
        "- https://wandb.ai/acme/widgets/runs/run-123\n"
        "- https://wandb.ai/acme/widgets/runs/run-456"
    )
    assert render_result_comment(experiment_result) == render_result_comment(
        experiment_result
    )


def test_parser_returns_markers_in_line_order_and_preserves_duplicates():
    first = result()
    second = result(
        assignment_id="assignment-8",
        status=ResultStatus.INCONCLUSIVE,
        summary="The evidence was mixed.",
    )
    body = "\n".join(
        [
            "Visible introduction.",
            render_result_marker(first),
            render_result_marker(second),
            render_result_marker(first),
        ]
    )

    assert parse_result_markers(body) == (first, second, first)


def test_parser_ignores_quoted_indented_and_inline_marker_substrings():
    marker = render_result_marker(result())
    body = "\n".join(
        [
            f"Read this example: {marker}",
            f"> {marker}",
            f" {marker}",
            "`<!-- senpai-result:v1 {} -->`",
            "ordinary prose",
        ]
    )

    assert parse_result_markers(body) == ()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("<!-- senpai-result:v2 {} -->", "unknown senpai result marker version"),
        ("<!-- senpai-result:v1 not-json -->", "malformed senpai result marker"),
        ("<!-- senpai-result:v1 {} --> trailing", "malformed senpai result marker"),
        ("<!-- senpai-result:v1 {invalid} -->", "invalid JSON"),
        (
            '<!-- senpai-result:v1 {"schema_version":1} -->',
            "invalid senpai result payload",
        ),
    ],
)
def test_parser_rejects_every_malformed_marker_line_clearly(
    line: str,
    message: str,
):
    with pytest.raises(ResultMarkerError, match=message):
        parse_result_markers(line)
