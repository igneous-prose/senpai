---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: merge-winner
description: >
  Verify and merge one winning experiment, then record the new baseline.
argument-hint: "<pr-number> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Merge a winner

Retrieve the complete PR with `get_prs` and validate its code, assignment,
authenticated terminal result, W&B evidence, metric direction, and current
remote head SHA. Then call `github_transition` with
`operation="merge_experiment"`, the PR number, exact head SHA, assignment ID,
and the desired merge method.

The transition refuses drafts, missing or foreign results, stale heads,
blocking labels, unknown mergeability, and conflicts. It verifies the merged
state and is safe to replay. Do not call `gh pr merge`.

It also compares the assignment marker's base SHA with the live base-branch
Git ref immediately before merging. When that baseline has advanced, reassess
the result against the new winner. Request a rerun if the conclusion is no
longer supported. If the existing evidence is still decisive, retry with
`accepted_base_sha` set to the exact live SHA reported by the
`baseline_advanced` event. Never invent that value merely to bypass the guard.

After a successful merge, update the target-prescribed baseline/research log
with the PR, metrics, run IDs and links, reproduction command, and conclusion.
Commit that advisor-owned update. Publish it only through the typed branch-push
operation advertised by `github_transition`; never use raw `git push`.

Merge multiple winners strongest-first and refresh the advisor baseline between
each decision.
