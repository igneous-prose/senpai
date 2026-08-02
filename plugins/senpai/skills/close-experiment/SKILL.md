---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: close-experiment
description: >
  Record and close a reviewed experiment that should not merge, including a
  control, clean negative, invalid candidate, or bounded dead end.
argument-hint: "<pr-number> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Close an experiment

Retrieve the complete PR with `get_prs` and verify its assignment, current head
SHA, terminal structured result, W&B evidence, and reason not to merge. Record
the conclusion in the target-prescribed research log when appropriate, commit
that advisor-owned update, and publish it through `push_branch`.

Close the PR only through this exact typed transition shape:

```json
{
  "transition": {
    "operation": "close_experiment",
    "repo": "owner/repo",
    "pr_number": 123,
    "expected_head_sha": "CURRENT_PR_HEAD_SHA",
    "assignment_id": "assignment-id",
    "reason": "Concise durable scientific disposition."
  }
}
```

Use a concrete reason that distinguishes a useful negative from an invalid or
incomplete run. Do not edit labels, write disposition markers, close the PR
with `gh`, or omit the head/assignment preconditions.
