# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from senpai_agent.models import (
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    MetricComparison,
    ResultStatus,
    WandbRunRef,
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
