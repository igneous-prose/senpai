---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: maintain-research-state
description: >
  Maintain and publish the advisor-owned current research state, durable
  research-ideas syntheses, and dataset analysis. Use when evidence or human
  guidance changes the research focus or direction, durable research ideas
  should be recorded, or dataset understanding changes.
---

# Maintain research state

Read the `program.md` identified in the system prompt. It is authoritative for
target-specific objectives, metrics, constraints, protected files, and any
baseline or research-log format.

Maintain these advisor-owned artifacts:

- Keep `research/CURRENT_RESEARCH_STATE.md` as a living, pruned view of the
  current high-level focus, themes, hypotheses, experiments, and potential next
  directions. Do not use it as an archive or experiment-by-experiment log. Use
  this schema:

  ```markdown
  # SENPAI Research State
  - <current date and time>
  - <most recent research direction from human researcher team>
  - <current research focus and themes>
  - <list of potential next research directions and themes>
  ```

- Optionally record durable research or round-planning synthesis in
  `research/RESEARCH_IDEAS_<YYYY-MM-DD_HH:MM>.md`.
- Save rigorous dataset analysis and durable future dataset insights in
  `research/DATASET_ANALYSIS.md`.

Commit only advisor-owned changes. First call `sync_git_refs` for the configured
advisor branch. Reconcile the local branch with the returned
SHA at `refs/remotes/origin/<advisor-branch>` without discarding either remote
merges or local work. Do not publish a local commit that omits the returned
remote SHA. Then call `publish_advisor_branch` with that remote SHA as the lease:

```json
{
  "remote_branch_sha_before_push": "REMOTE_SHA_BEFORE_PUSH",
  "local_commit_sha": "LOCAL_COMMIT_SHA"
}
```

Do not publish the advisor branch with `git push`.
