# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import pytest
from pydantic import ValidationError

from model_test_support import HEAD_SHA, result
from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    MetricComparison,
    WandbRunRef,
)


@pytest.mark.parametrize(
    "field",
    ["repo", "assignment_id", "revision_id", "expected_head_sha", "student"],
)
def test_assignment_identity_rejects_blank_fields(field: str):
    values = {
        "repo": "acme/widgets",
        "pr_number": 7,
        "assignment_id": "assignment-7",
        "revision_id": "revision-2",
        "expected_head_sha": HEAD_SHA,
        "student": "student-one",
    }
    values[field] = " \t\n"

    with pytest.raises(ValidationError, match="at least 1 character"):
        AssignmentKey.model_validate(values)


def test_assignment_identity_requires_a_positive_pull_request_number():
    values = result().assignment.model_dump()
    values["pr_number"] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        AssignmentKey.model_validate(values)


def test_result_rejects_a_nonterminal_status():
    values = result().model_dump()
    values["status"] = "running"

    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(values)


def test_wandb_reference_rejects_an_active_run():
    with pytest.raises(ValidationError):
        WandbRunRef(
            run_id="run-123",
            url="https://wandb.ai/acme/widgets/runs/run-123",
            state="running",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["run_id", "url"])
def test_wandb_reference_requires_identity(field: str):
    values = {
        "run_id": "run-123",
        "url": "https://wandb.ai/acme/widgets/runs/run-123",
        "state": "finished",
    }
    values[field] = " "

    with pytest.raises(ValidationError, match="at least 1 character"):
        WandbRunRef.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline", float("nan")),
        ("candidate", float("inf")),
        ("delta", float("-inf")),
    ],
)
def test_metric_comparison_rejects_non_finite_numbers(field: str, value: float):
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


def test_result_summary_has_a_four_thousand_character_limit():
    assert len(result(summary="x" * 4_000).summary) == 4_000

    with pytest.raises(ValidationError, match="at most 4000 characters"):
        result(summary="x" * 4_001)


@pytest.mark.parametrize("field", ["hypothesis", "summary", "commit_sha"])
def test_result_requires_its_claim_and_commit_identity(field: str):
    values = result().model_dump()
    values[field] = " "

    with pytest.raises(ValidationError, match="at least 1 character"):
        ExperimentResult.model_validate(values)
