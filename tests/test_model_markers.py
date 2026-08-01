# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json

import pytest

from model_test_support import HEAD_SHA, assignment, result
from senpai_agent.models import (
    ResultMarkerError,
    ResultStatus,
    WandbRunRef,
    parse_assignment_markers,
    parse_result_markers,
    render_assignment_marker,
    render_result_comment,
    render_result_marker,
)


def test_assignment_marker_round_trips_as_one_line():
    original = assignment()
    marker = render_assignment_marker(original)

    assert marker.startswith("<!-- senpai-assignment:v1 {")
    assert "\n" not in marker
    assert parse_assignment_markers(marker) == (original,)


@pytest.mark.parametrize(
    "line",
    [
        "<!-- senpai-assignment:v2 {} -->",
        "<!-- senpai-assignment:v1 not-json -->",
        "<!-- senpai-assignment:v1 {} --> trailing",
        "<!-- senpai-assignment:v1 {} -->",
    ],
)
def test_assignment_parser_rejects_malformed_or_invalid_markers(line: str):
    with pytest.raises(ValueError, match="assignment marker"):
        parse_assignment_markers(line)


def test_assignment_parser_ignores_marker_examples_in_prose():
    marker = render_assignment_marker(assignment())

    assert parse_assignment_markers(f"example: {marker}\n> {marker}") == ()


def test_result_marker_is_canonical_and_round_trips():
    original = result()
    marker = render_result_marker(original)
    payload = marker.removeprefix("<!-- senpai-result:v1 ").removesuffix(" -->")

    assert "\n" not in marker
    assert payload == json.dumps(
        json.loads(payload), sort_keys=True, separators=(",", ":")
    )
    assert parse_result_markers(marker) == (original,)


def test_result_marker_escapes_comment_terminators_without_changing_text():
    original = result(summary="The literal --> remains valid result text.")
    marker = render_result_marker(original)

    assert marker.count("-->") == 1
    assert r"The literal --\u003e remains valid result text." in marker
    assert parse_result_markers(marker) == (original,)


def test_visible_result_comment_contains_status_commit_summary_and_runs():
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

    assert render_result_comment(experiment_result).splitlines() == [
        render_result_marker(experiment_result),
        "",
        "Status: succeeded",
        f"Commit: `{HEAD_SHA}`",
        "",
        "The candidate improved validation loss.",
        "",
        "W&B runs:",
        "- https://wandb.ai/acme/widgets/runs/run-123",
        "- https://wandb.ai/acme/widgets/runs/run-456",
    ]


def test_result_parser_preserves_marker_order_and_duplicates():
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


def test_result_parser_ignores_marker_examples_in_prose_or_quotes():
    marker = render_result_marker(result())
    body = "\n".join(
        [
            f"Read this example: {marker}",
            f"> {marker}",
            f" {marker}",
            "`<!-- senpai-result:v1 {} -->`",
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
def test_result_parser_reports_malformed_marker_categories(
    line: str,
    message: str,
):
    with pytest.raises(ResultMarkerError, match=message):
        parse_result_markers(line)


def test_result_parser_rejects_an_unknown_payload_schema_version():
    marker = render_result_marker(result()).replace(
        '"schema_version":1', '"schema_version":2'
    )

    with pytest.raises(ResultMarkerError, match="invalid senpai result payload"):
        parse_result_markers(marker)
