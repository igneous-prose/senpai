---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: merge-winner
description: >
  Review and disposition one terminal experiment: merge a reproducible winner,
  close a useful negative or dead end, or request the missing evidence.
argument-hint: "<pr-number> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Review a terminal experiment

Retrieve the complete PR with `get_prs`. Verify its assignment, current head
SHA, terminal structured result, W&B evidence, metric direction, and scientific
conclusion. Then choose one disposition:

- merge a terminal, reproducible improvement;
- close a terminal control, clean negative, invalid candidate, or bounded dead
  end with a durable reason; or
- request a revision when the evidence is incomplete or another bounded run is
  required.

## Merge a winner

```json
{
  "transition": {
    "operation": "merge_experiment",
    "pr_number": 123,
    "expected_head_sha": "CURRENT_PR_HEAD_SHA",
    "assignment_id": "assignment-id",
    "merge_method": "squash"
  }
}
```

The transition refuses drafts, missing or foreign results, stale heads,
blocking labels, unknown mergeability, and conflicts. It compares the
assignment's base SHA with the live base branch immediately before merging. If
the baseline advanced, reassess the result. Request a rerun when the conclusion
no longer holds; otherwise retry with the event's exact `current_base_sha` as
`accepted_base_sha`. Never invent that value or call `gh pr merge`.

## Close a non-winner

```json
{
  "transition": {
    "operation": "close_experiment",
    "pr_number": 123,
    "expected_head_sha": "CURRENT_PR_HEAD_SHA",
    "assignment_id": "assignment-id",
    "reason": "Concise durable scientific disposition."
  }
}
```

Distinguish a useful negative from an invalid or incomplete run. Do not edit
labels, write disposition markers, or close the PR with `gh`.

## Request a revision

```json
{
  "transition": {
    "operation": "request_revision",
    "pr_number": 123,
    "assignment_id": "assignment-id",
    "expected_head_sha": "CURRENT_PR_HEAD_SHA",
    "revision_id": "new-revision-id",
    "comment": "Exact missing evidence and the bounded next run required."
  }
}
```

Use a new stable revision ID. State one concrete change or experiment and its
acceptance evidence; do not close an experiment that can still answer the
assigned question with one bounded correction.

## Record the outcome

Update the target-prescribed baseline or research log with the PR, metrics, run
IDs and links, reproduction command, and conclusion. Commit that advisor-owned
change and publish it only through:

```json
{
  "transition": {
    "operation": "push_branch",
    "branch": "advisor-branch",
    "expected_remote_sha": "REMOTE_SHA_BEFORE_PUSH",
    "expected_head_sha": "LOCAL_COMMIT_SHA"
  }
}
```

Review multiple candidates strongest-first and refresh the baseline between
each decision.
