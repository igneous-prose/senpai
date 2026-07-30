# Senpai OpenHands runtime contract

Status: implemented on the OpenHands rewrite branch.

## Objective

Senpai is a small deterministic Python control plane around OpenHands.
OpenHands owns research judgment, code changes, evidence interpretation, and
bounded delegation. Python owns operations that should not depend on an LLM
composing fragile tool calls:

- GitHub polling, workflow transitions, and verification;
- assignment branch publication;
- training process supervision and W&B metric monitoring;
- conversation selection and durable local events;
- command policy and stop checks; and
- cadence, retry, deadlines, and shutdown.

The rewrite preserves the advisor/student research workflow while reducing raw
data copied through model history and removing Claude Code runtime
dependencies.

## Invariants

1. Agent commits and PRs land in the target repository, never in this runner.
2. GitHub and W&B are the durable research records.
3. GitHub PRs and Issues are the only cross-node protocol.
4. An LLM does not poll, sleep, tail logs, supervise processes, or assemble a
   multi-call GitHub transaction.
5. GitHub mutations are typed, preconditioned, convergent on replay, and
   verified against remote state.
6. The advisor uses one durable conversation UUID. A student uses one UUID per
   assignment revision, and monitor wakes continue it.
7. Conversation and generated artifact state cannot fall back into the target
   checkout.
8. Senpai does not prune conversation history.
9. Only the student image carries CUDA, PyTorch, and the training stack.
10. Secrets are passed at narrow executor boundaries and redacted before
    monitored content is attached.
11. Hivemind is disabled, not redesigned, in this change.

## Control loop and remote protocol

```text
entrypoint
  clone/configure
  exec python -m senpai_agent.supervisor advisor|student

Python supervisor
  start one controller worker process group
  restart crashes with bounded exponential backoff
  TERM/KILL a worker whose current phase lease expires

Python controller worker
  poll GitHub + local durable monitor/event state
  reconcile the target checkout
  start one bounded OpenHands turn
  verify durable state
  sleep/backoff/jitter
```

The worker publishes an atomic lease containing its PID, current phase, and
hard deadline. The supervisor is independent of OpenHands and Kubernetes.
Kubernetes liveness and Docker health checks inspect the same lease, while the
supervisor provides the same recovery on a plain host.

The core controller imports no Kubernetes API and needs no Service, port, DNS
record, ServiceAccount, RBAC, cross-node token, or tailnet.

GitHub state is level-triggered:

- `status:wip` plus exactly one `student:<name>` label is an assignment;
- `status:review` is a durable advisor wake;
- `status:blocked`, `status:needs-rebase`, missing or duplicate student labels,
  stale WIP, and duplicate assignments are advisor-action events; and
- an open Issue labeled `human` plus `team`, the advisor branch, or one student
  label is a human message.

Human Issue events use the exact latest human-authored body/comment ID as their
dedupe key and `human_message_id`. An agent reply updates the Issue but does not
create a new wake for its own comment. `respond_to_issue` verifies the exact
human message before writing an idempotent response.
Launches with human-Issue handling disabled skip that GitHub query entirely.

While an advisor OpenHands turn is running, `ActiveGitHubWatcher` polls the same
GitHub state and enqueues newly visible events in the local advisor event
store. OpenHands 1.39 supports concurrent `send_message`; `AdvisorEventPump`
injects at its state lock boundary without cancelling unrelated work.

Generic child results use a local SQLite WAL event store because parent and
child run on the same advisor instance. That is not an inter-node protocol.

The only SQLite databases are `advisor-events.sqlite3`, for unacknowledged
advisor watcher/child events, and `training/monitors.sqlite3`, for student
monitor policy, samples, signals, and triage decisions. OpenHands conversation
history is a separate file-backed per-UUID event log.

## State and conversations

Advisor state:

```text
/var/lib/senpai/<research-tag>/advisor/openhands_state/
├── advisor-conversation-id
├── controller-lease.json
├── advisor-events.sqlite3
├── started-conversations.json
├── system-context-revisions.json
├── github/
└── conversations managed by OpenHands
```

The advisor UUID is created once and reused. Its conversation may cover several
ideas and monitoring threads concurrently.

Student state:

```text
/var/lib/senpai/openhands_state/
├── controller-lease.json
├── student-conversations.json
├── started-conversations.json
├── system-context-revisions.json
├── training/
│   ├── <training-id>.json
│   ├── <training-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<training-id>.json
├── github/
└── conversations managed by OpenHands
```

`student-conversations.json` maps one `(assignment_id, revision_id)` to one
UUID. `started-conversations.json` preserves correct continuation semantics if
the Python controller restarts. A `training_monitor` event carries its original
conversation UUID and therefore resumes, rather than replaces, the student
conversation.

OpenHands stores base state and individual events beneath that UUID. A killed
worker resumes from the last persisted event. An in-flight response or tool
call without a durable event is retried from the preceding event.

Student state is ephemeral by default. Losing it is acceptable after the
assignment ends because the PR, branch, typed result, W&B runs, and Weave trace
are durable. The advisor state is persisted by the deployment.

No default path may be relative to the current workspace. Senpai removes only
its generated PR Markdown artifacts after 24 hours. It does not delete
OpenHands conversations or impose a retention count.

## Prompt and progressive disclosure

The model receives:

1. OpenHands' native base system prompt and tool schemas.
2. One stable system suffix assembled from:
   - `system_instructions/SENPAI-HARNESS.md`; and
   - the rendered advisor or student role charter.
3. Applicable target `AGENTS.md` and compatible `CLAUDE.md` project context.
4. A compact skill catalog whose bodies are loaded only when invoked.
5. User turns containing `program.md`, target role instructions, current state,
   and current UTC time.

Harness and role remain separate source documents because they have different
owners, but are merged into one system suffix so the agent knows both the
OpenHands operating contract and its Senpai role. The complete role is not
periodically duplicated in user messages; OpenHands includes the system suffix
on every inference. A persisted merged-context hash detects a changed deployed
harness or role and injects the current text once into the same conversation
UUID. Current time is rendered for every controller wake.

File-based subagents are discovered from `.agents/agents`. Skill bodies are not
concatenated into agent definitions. Skill model/effort frontmatter remains
intact pending native OpenHands support for skill-declared child configuration.

## Prompt caching

The pinned SDK fork is
[`morganmcg1/software-agent-sdk`](https://github.com/morganmcg1/software-agent-sdk)
at commit `da7d76fe3d0b0f5b169ff47c5617a8ecf38a004c`, based on OpenHands SDK
1.39.1.

`prompt_cache_configuration()` sets:

- Anthropic: `prompt_cache_ttl="1h"`;
- GPT-5.6: one explicit cache breakpoint on the stable system block,
  `prompt_cache_options.mode="explicit"`, and a 30-minute TTL;
- older compatible OpenAI models: `prompt_cache_retention="24h"`; and
- other providers: no provider-specific cache option.

The fork emits an Anthropic cache-control `ttl` only when explicit Anthropic
caching is active. Its tests prove the five-minute wire form remains unchanged,
the one-hour TTL is forwarded, and OpenAI retention continues to work without
receiving an Anthropic TTL parameter.

Direct `openai/*` models use a stored Responses API chain. The active branch's
latest `resp_*` ID is recovered from the durable OpenHands event log after
every process restart, passed as `previous_response_id`, and paired only with
inputs created after that response. System instructions and tools remain
explicit on every request.

Senpai sets `reasoning_context="all_turns"` and `reasoning_summary="auto"` so
supported models can reuse server-side private reasoning and return the most
detailed available summary. The default effort is `xhigh`; GPT-5.6 still
accepts an explicit `max` override. Automatic OpenAI compaction starts at
200,000 rendered tokens. The OpenHands condenser is disabled for that provider
chain, but its complete local event log remains durable and is used to recover
the latest response ID after restart. Other providers retain the high-quality
OpenHands condenser.

## Typed tools

### `search_conversation_history`

This read-only tool performs a case-insensitive fixed-text search over the
complete active OpenHands event branch. It searches model-visible message text
and tool calls, then returns at most 20 bounded snippets newest first. It does
not expose raw storage, abandoned branches, or automatically reinsert the whole
history into the model context. This preserves the usefulness of a durable
transcript without undoing provider compaction or creating a large token spike.

### `get_prs`

One function accepts explicit numbers, an inclusive creation-date range, and/or
a GitHub search expression. Every selected PR contains its full body, all issue
comments, all submitted reviews, and all inline review comments across
pagination.

`max_inline_prs` defaults to five. At or below the limit, Markdown is returned
in context. Above it, the same Markdown is written to one deterministically
named mode-0600 artifact outside the target checkout, and the model receives a
compact manifest and path. Raising the inline limit above five warns about
context pollution. There is no duplicate JSON artifact and no hidden
summarizing subagent.

### `github_transition`

One discriminated tool owns:

- `create_assignment`;
- `push_branch`;
- `reconcile_labels`;
- `request_revision`;
- `respond_to_issue`;
- `submit_result`;
- `close_experiment`; and
- `merge_experiment`.

Student publication happens only inside `submit_result`, which validates
repository, PR, assignment, revision, student, current remote head, and
proposed result head before it can push. Assignment identity is required for
revision, label, close, and merge transitions. Marker comments are trusted only
when authored by the authenticated token actor.

Assignment creation checks the remote base SHA, creates an isolated empty
assignment commit with `git commit-tree`, publishes with force-with-lease,
refuses a second active assignment for the student, creates or reconciles one
draft PR, embeds a typed assignment marker, and verifies routing state.

Student submission requires a clean assignment branch, lease-pushes the local
commit, upserts the typed result, marks the PR ready, reconciles
`status:review`, and verifies all postconditions. The label itself is the
cross-node notification.

Definitive HTTP failures fail clearly. An ambiguous transport failure after a
mutation is resolved by reading and verifying desired state before any retry.

### `dispatch_agent`

```text
dispatch_agent(task: str, include_context: bool = false)
  -> {task_id, status}
```

The advisor has one generic asynchronous delegation primitive. With
`include_context=false`, the fresh child receives only the normal merged system
prompt and the task. With `include_context=true`, it also receives a snapshot of
the complete model-visible parent history, including progressively disclosed
skill content.

The child runs in a separate process group with a hard ten-minute runtime,
reports `agent_result` or `agent_error` to the local event store, closes, and
disappears. It is not a hard-coded reviewer. When `review_ready` arrives during
other advisor work, the role policy asks the advisor to dispatch a generic
full-context PR review and continue unrelated work.

Advisor child agents receive neither training tools nor recursive dispatch.

### Training and monitoring

Students receive:

```text
run_training(spec: TrainingSpec) -> TrainingResult
get_training_status(training_id: str) -> TrainingResult
monitor_training(
  training_id,
  metric=None,
  direction=None,
  gates=(),
  poll_interval_seconds=60,
  stale_after_seconds=600,
  notify_on_status={finished,failed,timed_out,cancelled},
) -> MonitorTrainingObservation
```

`TrainingSupervisor` owns one process group, the configured timeout ceiling,
TERM/KILL cleanup, restart identity checks using PID/PGID/create-time, a bounded
8 KiB error tail, streamed 64 KiB log parsing, persisted state, and discovered
W&B run IDs. Run IDs are persisted while training is still running so metric
monitoring can begin immediately.

The controller polls only monitors that are due. It fetches one latest selected
metric value from W&B, evaluates deterministic threshold/change/staleness and
terminal-state rules, and persists deduplicated compact signals. Ordinary
polls use no LLM tokens.

A fresh no-context generic child triages each actionable signal. Hard failures
always wake even if triage fails. A no-wake decision acknowledges the signal.
A wake decision is persisted with the signal, so retrying an unacknowledged
student turn does not pay for the triage child again. The triage child omits
browser tools. A wake event carries the original student conversation UUID and
a compact summary.

The Stop hook permits a running job only after its durable monitor marker
exists, allowing the student turn to end while the controller supervises it.
The advisor and advisor children never receive training tools.

## Hooks, deadlines, and shutdown

The native plugin declares OpenHands `PreToolUse`, `Stop`, and `SessionEnd`
hooks. Hooks give early model-visible feedback. `senpai_terminal` evaluates the
same pure policy in-process and fails closed if policy evaluation fails.

Denied patterns include raw GitHub mutations, raw `git push`, direct training
launches, sleeps, polling loops, `watch`, and `tail -f`, including nested shell
and `env` wrappers.

Every OpenHands turn has a controller-configured hard deadline. The deadline
interrupts the conversation, produces a non-success result, and leaves durable
events unacknowledged. The controller then retries with bounded exponential
backoff. Controller termination interrupts and closes the current conversation,
cancels active supervised training, closes local stores, and flushes Weave
before the controller exits. Standalone and child runners flush Weave at runner
exit.

## Secrets and Weave

The GitHub write token is resolved before tool initialization and held in a
typed in-process vault. It is absent from model-facing tool schemas and terminal
secrets. Generic child processes receive it through a private mode-0600
one-use file, consume and unlink that file, then re-vault the token.

Git operations use a temporary askpass helper rather than a persistent
credential store. The runner repository cannot push, and a target pre-push hook
enforces role/branch rules.

Weave content capture applies a longest-first transform over all configured
API keys, tokens, passwords, secrets, credentials, and the selected custom
model credential before content is sent. The pinned `weave-openhands`
integration is initialized before OpenHands imports. Each conversation run is
an agent trace with child LLM and tool spans, all carrying the durable
OpenHands conversation ID.

## Images and launch acceptance

Two images are built from the same exact source commit:

- advisor: Python/OpenHands, GitHub CLI, and Chromium; no PyTorch, CUDA, or
  Kubernetes tooling;
- student: the CUDA/PyTorch stack plus the same OpenHands and Chromium runtime.

Both build Chromium and run a browser smoke test. The student image validates
CUDA architecture support. The launcher accepts only matching full source-SHA
tags or immutable digests and checks out that exact revision.

Launch preflight verifies:

- target-repository push and branch access;
- the Anthropic key;
- the Exa key with one `type="instant"`, publication-category, one-result
  search; and
- the W&B key with a minimal viewer query.

Exa is a progressive skill/script integration, not an always-connected MCP
server.

The Kubernetes launcher creates one Secret, ConfigMaps, and Deployments. It
creates no Service or RBAC. Docker and local hosts need no shared network for
Senpai communication.

Hivemind startup remains commented with a clear note. Cluster cutoff still
waits for readiness/deadline and deletes launch resources; all conversation
harvest/archive code is removed.

## Removed code

Removed:

- Claude Code and its image install;
- `.claude/` runtime resources;
- Claude-named and OpenHands shell watchdog/supervisor loops;
- the Exa MCP configuration;
- the HTTP advisor service, bearer token, port, probes, and Kubernetes RBAC;
- shell GitHub polling and pod-process inspection;
- cutoff conversation harvesting;
- obsolete tool-role instructions; and
- full skill-body inlining for subagents.

Retained intentionally:

- Agent skills and their model/effort metadata under `.agents`;
- OpenHands Browser, task tracker, Think, and the high-quality default
  condenser for providers not using stored OpenAI Responses continuation;
- the pinned `weave-openhands` agent, LLM, and tool tracing integration; and
- only a small bootstrap shell path for clone, identity, and Git push guards.

## Acceptance

The change is acceptable when:

- unit and local integration tests pass;
- shell scripts pass `bash -n`;
- manifests render matching immutable source revisions without Service/RBAC;
- browser smoke succeeds in both image builds;
- no operational prompt advertises a missing tool or service;
- no runtime role requires Claude Code semantics;
- secrets do not appear in serialized tool specs or captured content;
- monitor wakes resume the original student UUID; and
- a live credential preflight plus GitHub read-only smoke succeeds before
  production rollout.
