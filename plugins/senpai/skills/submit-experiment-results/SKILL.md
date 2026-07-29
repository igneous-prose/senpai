---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: submit-experiment-results
description: >
  Commit and submit a terminal experiment result for advisor review through the
  typed Senpai GitHub transition.
argument-hint: "<pr-number> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Submit experiment results

First make the target worktree clean by committing only the assigned change.
Collect the current local commit SHA and the current remote assignment-branch
SHA. Build the strict `ExperimentResult` required by the `github_transition`
schema:

- the assignment repository, PR, assignment ID, revision ID, student, and
  current expected head SHA;
- terminal status, hypothesis, and bounded summary;
- every W&B run ID, URL, and terminal state;
- the primary metric comparison; and
- the same local commit SHA.

Call `github_transition` with `operation="submit_result"`, the PR number,
branch, previous remote SHA, current head SHA, and typed result.

That single transition lease-pushes the clean assignment branch, verifies the
new PR head, upserts the authenticated structured result, marks the PR ready,
and reconciles `status:review`. That label is the durable advisor notification.
Do not run `git push`, edit labels, write result markers, or call `gh pr ready`
yourself.

If any run is still active or could change the conclusion, keep the assignment
in progress and register it with `monitor_training`; do not submit a terminal
result.
