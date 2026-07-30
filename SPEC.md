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
child run on the same advisor or student instance. That is not an inter-node
protocol.

The only SQLite databases are `advisor-events.sqlite3`, for unacknowledged
advisor watcher/child events; `student-events.sqlite3`, for unacknowledged
student child events; and `training/monitors.sqlite3`, for student monitor
policy, samples, and deduplicated actionable signals. OpenHands conversation
history is a separate file-backed per-UUID event log.

## State and conversations

Advisor state:

```text
/var/lib/senpai/<research-tag>/advisor/openhands_state/
├── advisor-conversation-id
├── controller-lease.json
├── advisor-events.sqlite3
├── conversation-state.json
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
├── student-events.sqlite3
├── conversation-state.json
├── training/
│   ├── <training-id>.json
│   ├── <training-id>.log
│   ├── monitors.sqlite3
│   └── monitors/<training-id>.json
├── github/
└── conversations managed by OpenHands
```

`student-conversations.json` maps one `(assignment_id, revision_id)` to one
UUID. `conversation-state.json` records, per UUID, both successful initial
instruction delivery and the digest of the delivered merged system context.
The controller replaces this one document atomically after a successful turn,
so a restart cannot observe those two facts at different revisions. A
`training_monitor` event carries its original conversation UUID and therefore
resumes, rather than replaces, the student conversation.

When `conversation-state.json` does not yet exist, startup atomically migrates
the previous `started-conversations.json` and
`system-context-revisions.json` files. A conversation caught between those
legacy files' two writes resumes without replaying its initial brief and
receives the current system context once.

OpenHands stores base state and individual events beneath that UUID. A killed
worker resumes from the last persisted event. An in-flight response or tool
call without a durable event is retried from the preceding event.

The controller marks a conversation's initial instructions delivered and
records its current system-context digest in the same atomic update, only after
the OpenHands turn succeeds. A crash or nonzero first turn therefore retries
the complete programme and assignment prompt instead of incorrectly
continuing from instructions that were never delivered.

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
concatenated into agent definitions. The pinned OpenHands fork applies each
agent definition's `reasoning_effort` override after resolving its inherited
LLM or stored model profile.

## Prompt caching

The pinned SDK fork is
[`morganmcg1/software-agent-sdk`](https://github.com/morganmcg1/software-agent-sdk)
at commit `6822ab324b7c207dce55fe25ab927dab5d874c2b`, based on OpenHands SDK
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
receiving an Anthropic TTL parameter. Laminar is an optional SDK extra and is
not part of Senpai's locked runtime; Weave is the agent observability
integration.

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
the latest response ID after restart.

Direct Anthropic models use native server-side compaction with a 200,000-input-
token trigger. OpenHands persists the returned typed compaction block in the
normal event log and replays it first in each later request, including after a
process restart. The local condenser is disabled for these conversations.
Other providers retain the high-quality OpenHands condenser.

The complete durable transcript remains available as plain event JSON under
`$SENPAI_OPENHANDS_STATE_DIR/$SENPAI_CONVERSATION_ID/events/`. The harness
directs the model to use `rg` and bounded reads because the directory can be
large. No dedicated history-search tool duplicates shell capabilities. A
dispatched child receives `$SENPAI_PARENT_CONVERSATION_HISTORY_DIR`, allowing a
main advisor or student to delegate broad history recovery without copying the
full parent context.

## Typed tools

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

### `delegate_agent`

```text
delegate_agent(
  task: str,
  agent: general-purpose | explore | search | bash-runner = general-purpose,
  model: smart | fast = smart,
  background: bool = false,
  include_context: bool = false,
  search_mode: general-web | research-publications | null = null,
) -> {task_id, status, result?}
```

This is the only model-facing subagent mechanism. Every call launches the
selected Markdown-defined agent in its own process group and fresh OpenHands
conversation. The parent can emit eight independent calls in one response;
OpenHands' parallel tool executor and the delegation semaphore both cap active
children at eight.

`background=false` waits and returns the compact report inline.
`background=true` returns a task ID immediately, then reports `agent_result` or
`agent_error` through the local durable event store. No child-specific runtime
deadline is imposed by default; parent conversation and controller supervision
still provide interruption and process recovery boundaries.

`model=fast` selects `SENPAI_OPENHANDS_FAST_MODEL` (default
`anthropic/claude-haiku-4-5` for the default Anthropic stack) for mechanical
search, command execution, and extraction. Without an explicit fast model, a
non-Anthropic stack uses its smart model rather than sending that provider's
API key elsewhere. `model=smart` selects the main configured model for review,
literature research, subtle synthesis, or ambiguous failure diagnosis. The
file-defined agent may override reasoning effort independently.

`explore` searches code, data, PR artifacts, and durable history and returns
concise conclusions with paths and line numbers. `search` requires exactly one
mode: `general-web` uses Exa's general index with agent-oriented defaults,
while `research-publications` uses Exa's publication index and primary papers.
`general-purpose` handles mixed investigation, editing, tests, and typed Senpai
operations. `bash-runner` has only the terminal and runs tests, builds, linters,
formatters, dependency commands, Git inspection, or system checks. It normally
uses the fast model and returns counts and actionable failures rather than raw
command output.

With `include_context=false`, the child receives the merged system prompt and
task and may search the parent's durable history path. With
`include_context=true`, it also receives the complete model-visible parent
history, including progressively disclosed skill content.

Each child receives only the tools and progressively disclosed skills declared
by its Markdown definition. Bash Runner is terminal-only. General-purpose and
Explore children can also delegate foreground work; Search and Bash Runner
cannot. Children receive neither GitHub credentials nor GitHub read/write
tools; the parent prepares any large PR Markdown artifact and owns every typed
GitHub operation. They do not receive training tools. Background nesting is
rejected because a child may exit before a grandchild's durable result can be
consumed.

When `review_ready` arrives during other advisor work, the role policy asks the
advisor to launch a smart, full-context, background general-purpose review and
continue unrelated work. Every background result records its parent
conversation UUID, allowing the controller to resume the exact student
conversation when a result arrives after its turn.

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

Metric samples reject NaN and infinities. A failure in one monitor's training
status or W&B lookup advances that monitor's schedule and emits one
deduplicated `monitor_error` hard signal; it cannot block other monitors,
GitHub events, child results, or an already-pending hard-failure wake. Repeating
`monitor_training` with a changed policy replaces the stored policy and resets
its derived samples and signals to match the new marker.

Every persisted actionable signal directly creates a compact
`training_monitor` wake for the signal's original student conversation UUID.
No intermediate LLM call gates these events: registering the monitor policy is
the student's request to resume when one of its conditions emits a signal. The
signal remains pending until that exact conversation successfully handles it.

Controller events are partitioned by their exact conversation UUID before a
turn. Each partition is acknowledged only after its own successful turn, so a
child result for one assignment cannot consume or permanently block a training
event for another.

The Stop hook permits a running job only after its durable monitor marker
exists, allowing the student turn to end while the controller supervises it.
The advisor and advisor children never receive training tools.

## Hooks, deadlines, and shutdown

The native plugin declares OpenHands `PreToolUse`, `Stop`, and `SessionEnd`
hooks. Its pre-tool hook covers both `senpai_terminal` and the raw `terminal`
used by file-defined children, so delegation cannot bypass workflow or training
boundaries. Hooks give early model-visible feedback. `senpai_terminal` also
evaluates the same pure policy in-process and fails closed if policy evaluation
fails.

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

The entrypoint uses the GitHub write token only for bootstrap, writes it to a
private mode-0600 file under the pod-local `/tmp`, removes the askpass helper,
clears all raw token environment variables, and execs the supervisor. The
supervisor consumes and unlinks that bootstrap file into typed in-process
memory. Before each controller restart it creates a one-shot inherited pipe;
the worker reads and closes that pipe before tool initialization. No raw token
is written to conversation/dataset storage. The long-lived PID 1 environment,
model-facing tool schemas, and agent terminal contain no GitHub token.

Generic child processes receive no GitHub token and no GitHub tools. Main-role
GitHub operations remain typed and lease/state guarded. Terminal and hook
policies are behavioral guardrails, not a credential-containment boundary.

Git operations use a temporary askpass helper rather than a persistent
credential store. The runner repository cannot push, and a target pre-push hook
enforces the exact role/branch matrix. Images run as an unprivileged user, and
the Kubernetes containers drop every Linux capability, disallow privilege
escalation, and use the runtime-default seccomp profile.

Weave content capture applies a longest-first transform over all configured
API keys, tokens, passwords, secrets, credentials, and the selected custom
model credential before content is sent. The pinned `weave-openhands`
integration is initialized before OpenHands imports. Each conversation run is
an agent trace with child LLM and tool spans, all carrying the durable
OpenHands conversation ID.

## Images and launch acceptance

Three images are built from the same exact source commit:

- advisor: Python/OpenHands, GitHub CLI, and Chromium; no PyTorch, CUDA, or
  Kubernetes tooling;
- student: the CUDA/PyTorch stack plus the same OpenHands and Chromium runtime;
- cutoff: a minimal shell/Python runtime with one checksum-verified, pinned
  `kubectl`.

Advisor and student build Chromium and run a browser smoke test. The student
image validates CUDA architecture support. The launcher and cutoff arming
script accept only matching full source-SHA tags or immutable digests and check
out that exact revision.

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

Hivemind startup remains commented with a clear note. The Python controller
waits for the optional cluster start gate while continuously refreshing a
`start-gate` lease; readiness therefore cannot deadlock gated launch. Cluster
launch and cutoff CLIs accept a gate only when it is an absolute normalized
file path beneath their shared PVC mount. Cluster cutoff arms as soon as all
expected resources are Ready or when its bounded readiness window expires,
whichever comes first, and opens the optional start gate in either case. One
missing or crash-looping pod therefore cannot prevent the runtime budget from
starting. At the persisted deadline it deletes launch resources; all
conversation harvest/archive code is removed.

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
  condenser for providers not using stored OpenAI Responses continuation or
  Anthropic native compaction;
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
- monitor wakes resume the original student UUID;
- cutoff arming completes after a bounded readiness window even when a pod
  never becomes Ready; and
- a live credential preflight plus GitHub read-only smoke succeeds before
  production rollout.
