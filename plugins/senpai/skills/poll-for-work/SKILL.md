---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: poll-for-work
description: >
  Poll for assigned experiment PRs for a student. Use this skill to: check for assignments, poll for work, see if
  there's a PR assigned to me. Triggers for: "any work for me?", "check for assignments", "poll for PRs".
argument-hint: "<student-name>"
context: fork
model: claude-sonnet-4-6
effort: high
---

# poll-for-work

Interpret the assignment event supplied by the Senpai controller. This is a
one-shot check: return the result to the parent agent immediately.

Assignments are Github labels, not GitHub assignees:

- The current advisor branch is also a routing label. A PR is assigned to you only when it has the following labels: `$ADVISOR_BRANCH` (current advisor branch), your name label like so: `student:$0`, and this status label: `status:wip`.
- Never create a polling loop or reconstruct assignment routing with `gh`.
- The controller owns GitHub polling, waiting, and conversation re-entry.

## Arguments

- **$0** — Your student name (e.g. `fern`)

## Steps

1. **Read the current `student_assignment` event.** It contains the PR number,
   title, branch, assignment ID, and revision ID. If no such event is present,
   return `NO_WORK`.

2. **For the assigned PR:**
   - Note the PR number, title, and branch name
   - Call `get_prs` with `numbers=[<number>]` and the default
     `max_inline_prs=5`. This returns the complete PR body, issue comments,
     reviews, and inline review comments through the bounded read interface.
     Check that Markdown for an advisor revision request.
   - Do not reconstruct PR history with `gh` or shell helpers.

3. **Return a summary:**

If work is available:
```
WORK_AVAILABLE: PR <pr-number> "<pr-title>" on branch <branch-name>
```

If the PR has revision comments from the advisor, include that:
```
WORK_AVAILABLE (REVISION): PR <pr-number> "<pr-title>" on branch <branch-name> — advisor requests: "<advisor-comment>"
```

If no assignment event is present:
```
NO_WORK
```

Keep the response short — the parent agent just needs to know whether to start working or keep waiting.
