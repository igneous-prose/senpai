---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: assign-experiment
description: >
  Create a typed assignment branch and draft PR for one student. Use when the
  advisor has a concrete hypothesis and an idle student.
argument-hint: "<student-name> <hypothesis-slug> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Assign an experiment

Read the current baseline and `program.md`, then write one complete assignment:

- a falsifiable hypothesis and mechanism;
- exact files and changes in scope;
- baseline metrics and W&B evidence;
- commands, run limits, metrics, and stopping conditions; and
- one student, one branch, and one experiment.

Fetch the advisor branch and record its exact remote SHA. Call
`github_transition` with `operation="create_assignment"` and:

- `repo`: `$GH_REPO`;
- stable `assignment_id` and initial `revision_id`;
- `student`;
- `base_branch`: `$ADVISOR_BRANCH`;
- `expected_base_sha`: the fetched remote advisor SHA;
- `head_branch`: `<student>/<hypothesis-slug>`;
- a concise title; and
- the complete assignment body.

The transition creates an empty assignment commit without changing the advisor
worktree, pushes the branch with a lease, creates or reconciles one draft PR,
adds the exact routing labels, embeds the typed assignment marker, verifies the
result, and rejects a second active assignment for the student.

Do not create the branch, PR, labels, or assignment marker through shell
commands. If the transition reports stale state, refresh the named SHA and
re-evaluate rather than bypassing the precondition.
