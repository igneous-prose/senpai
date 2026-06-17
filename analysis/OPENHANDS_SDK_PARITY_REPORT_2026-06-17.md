# OpenHands SDK Parity Report - 2026-06-17

## Scope

Migrated the shared Senpai headless agent invocation seam from Claude Code-only
to a runtime switch:

- `SENPAI_AGENT_RUNTIME=claude` keeps the existing Claude Code path.
- `SENPAI_AGENT_RUNTIME=openhands` routes the same stdin prompt, max-turn
budget, timeout wrapper, logs, and continuation flag into the OpenHands SDK.

The Kubernetes advisor/student shell loops still own polling, assignment
routing, git checkout, watchdogs, and idle sleeps.

## OpenHands References Reviewed

- [OpenHands SDK overview](https://docs.openhands.dev/sdk)
- [SDK getting started](https://docs.openhands.dev/sdk/getting-started)
- [SDK architecture overview](https://docs.openhands.dev/sdk/arch/overview)
- [Skills guide](https://docs.openhands.dev/sdk/guides/skill)
- [Task tool/subagent guide](https://docs.openhands.dev/sdk/guides/task-tool-set)
- Local source clone: `/tmp/software-agent-sdk`, especially `LLM`,
  `AgentDefinition`, `agent_definition_to_factory`, default tools, skills, and
  subagent registry behavior.

## Implemented Runtime Coverage

- Local OpenHands `LLM` + `Agent` + `Conversation` runner.
- Default model: `anthropic/claude-opus-4-8`.
- API key source: `SENPAI_OPENHANDS_API_KEY_ENV`, tested with
  `ANTHROPIC_API_KEY2`.
- Prompt passed over stdin, not argv.
- `-c` continuation support via persisted OpenHands conversation UUID.
- Nearest parent `CLAUDE.md` loaded as role instruction suffix.
- Senpai `.claude/skills`, plugin skills, `.agents/skills`, and
  `.openhands/skills` loaded into agent context.
- Claude `.claude/agents/*.md` bridged into OpenHands registered subagents,
  including `senpai:*` skill-name normalization.
- Native `.agents/agents` and `.openhands/agents` directories are also
  discovered.
- Default OpenHands tools: terminal, file editor, task tracker, task/subagent.
- Direct named-agent mode via `--agent`, used to compare `researcher-agent`
  against Claude Code's `--agent researcher-agent`.
- Reasoning effort is pinned to the strongest available setting on each side:
  OpenHands `reasoning_effort=xhigh`; Claude Code comparison runs use
  `--effort max`.

## Deterministic Tests

Command:

```bash
PYTHONPATH=/Users/mmcguire/ML/senpai PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run --with pytest pytest \
  tests/test_agent_runtime_dispatch.py \
  tests/test_openhands_runner_config.py -q
```

Result: `14 passed`.

These tests cover runtime dispatch, stdin prompt safety, max-turn forwarding,
unknown-runtime failure, API key resolution, parent `CLAUDE.md` discovery,
conversation continuation UUIDs, skill namespace normalization, OpenHands tool
name normalization for bridged subagents, and packaging import smoke.

## Live Agentic Tests

Command:

```bash
PYTHONPATH=/Users/mmcguire/ML/senpai \
  uv run python tools/run_openhands_parity_trials.py \
  --runs 10 --workers 10 --timeout 420 \
  --output-dir /tmp/senpai-openhands-parity-20260617
```

Model: `anthropic/claude-opus-4-8`

Full JSON report:

```text
/tmp/senpai-openhands-parity-20260617/report.json
```

| Task | What It Exercises | Passed | Total | Pass Rate | Avg Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `advisor_triage` | idle student detection, review/blocked PR classification, assignment labels | 10 | 10 | 100% | 35.81 |
| `code_patch` | surgical code edit and local check execution | 10 | 10 | 100% | 18.07 |
| `context_contract` | program/instruction reading and exact structured output | 10 | 10 | 100% | 24.13 |
| `result_marker` | training-log parsing and `SENPAI-RESULT` JSON construction | 10 | 10 | 100% | 30.33 |
| `skill_subagent` | `.claude` skill loading, researcher subagent bridge, task-tool delegation | 10 | 10 | 100% | 97.30 |

Overall: 50/50 live trials passed.

## Qualitative Researcher-Agent Tests

Command:

```bash
PYTHONPATH=/Users/mmcguire/ML/senpai \
  uv run python tools/run_research_quality_trials.py \
  --runs 10 --workers 10 \
  --output-dir /tmp/senpai-research-quality-20260617 \
  --timeout 360
```

Models and runtime settings:

- OpenHands: `anthropic/claude-opus-4-8`, `reasoning_effort=xhigh`,
  direct `--agent researcher-agent`.
- Claude Code headless comparison: `claude-opus-4-8`, `--effort max`,
  direct `--agent researcher-agent`, `--bare`, `--no-session-persistence`,
  and a bounded tool setup intended to avoid Claude Code auto-memory/session
  overhead. Trace review later showed this direct agent exposed only `Read`.
- Both runtimes used read-only GitHub PR context fetched from
  `morganmcg1/gemma-challenge-senpai`; agents were instructed not to mutate
  GitHub and to write only `answer.json` in each temporary workspace.

Full JSON report:

```text
/tmp/senpai-research-quality-20260617/report.json
```

| Task | OpenHands Passed | Claude Code Passed | OpenHands Avg Score | Claude Avg Score | OpenHands Avg Seconds | Claude Avg Seconds | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `post_result_next_steps` | 10/10 | 10/10 | 24.0 | 24.0 | 127.88 | 120.26 | Quality tied; Claude 7.62s faster |
| `researcher_axis_synthesis` | 10/10 | 10/10 | 24.0 | 24.0 | 283.81 | 216.54 | Quality tied; Claude 67.27s faster |
| `direction_selection` | 9/10 | 5/10 | 21.6 | 12.0 | 311.82 | 324.75 | OpenHands +4 passes, +9.6 avg score, 12.93s faster |

Completed-answer quality:

| Task | OpenHands Completed Answer Score | Claude Completed Answer Score | Interpretation |
| --- | ---: | ---: | --- |
| `post_result_next_steps` | 24/24 on 10/10 | 24/24 on 10/10 | Both runtimes make the right advisor decision when scoped to one result. |
| `researcher_axis_synthesis` | 24/24 on 10/10 | 24/24 on 10/10 | Both runtimes synthesize the closed framework/FlashInfer axes well. |
| `direction_selection` | 24/24 on 9/9 completed | 24/24 on 5/5 completed | Claude's weakness was completion reliability, not quality after it finished. |

Qualitative judgment:

- Completed OpenHands answers scored 24/24 across every qualitative task.
- Completed Claude Code answers also scored 24/24; the difference was
  reliability under the 360s cap, not weaker reasoning in completed outputs.
- OpenHands had one timeout in the hardest broad direction-selection prompt.
  Claude Code had five timeouts on that same prompt.
- On the qualitative researcher-agent surface, OpenHands does not show a
  diminished-intelligence pattern. It matched answer quality and was more
  reliable than the Claude Code headless comparison on the broadest decision
  prompt under the same timeout.

## Why Claude Code Was Worse On Direction Selection

The `direction_selection` prompt was the hardest qualitative case: it included
five recent PR contexts and asked the agent to choose one next research
direction, explain why it dominated dead axes, propose a falsifier, and describe
what the advisor should assign next. This widened the search space from "judge
this result" to "reconstruct the whole research state and pick the next branch
of the program."

Trace evidence points to a runtime and artifact-reliability issue, not a weaker
completed answer:

- Successful Claude direction runs were very large single-turn completions:
  original pass traces reported roughly `20.7k` to `25.7k` output tokens and
  finished in about `260s` to `308s`. That leaves little headroom under a
  `360s` timeout.
- Failed Claude direction runs did not produce bad JSON. They produced no
  final JSON at all. In the original run, three failures reached only init or
  thinking, one reached thinking only, and one wrote a partial analysis, tried
  to `Read` the run directory, hit `EISDIR`, then entered compaction before the
  timeout.
- Claude Code exposed only `Read` to the direct `researcher-agent` in this
  `--bare` comparison, despite the harness attempting to allow read/write/web
  tools. Successful Claude runs therefore printed JSON instead of writing
  `answer.json`; failures could not satisfy the artifact contract before
  timeout.
- OpenHands had a more artifact-first trajectory. In successful direction runs
  it used the SDK/file-editor path to create `answer.json`, validate it, and
  then summarize. The one OpenHands failure was a first-response timeout, not a
  late compaction or tool/artifact loop.
- I reran the Claude `direction_selection` slice after changing the harness to
  pass tool names as separate CLI arguments. Claude Code still exposed only
  `Read`, and the rerun passed `2/10` under the same `360s` cap. I do not treat
  that rerun as the primary parity table because it changed the command shape,
  but it supports the diagnosis that direct Claude Code researcher-agent mode is
  brittle and latency-limited on this broad prompt.

The practical conclusion is that Claude Code was not "dumber" on direction
selection. Its completed answers were excellent. It was worse because the
direct headless researcher-agent path was less reliable under a bounded eval:
large one-shot answers, no dependable `answer.json` write path, and occasional
compaction/tool detours consumed the timeout. OpenHands was better at turning
the same reasoning into the required artifact before the clock expired.

## Parity Judgment

The OpenHands runtime has feature parity for the core Senpai SDK tasks covered
by the current headless Claude Code seam:

- read target research context;
- perform scoped code edits and run checks;
- parse metrics and produce terminal result markers;
- triage advisor state and construct assignments;
- invoke migrated skill/subagent workflows.
- run the qualitative `researcher-agent` directly and produce research
  direction decisions at Claude Code-quality or better under the tested budget.

Known non-parity surface: Claude Code-specific telemetry plugins such as the
Weave Claude Code plugin and Hivemind session ingestion are not byte-for-byte
equivalent. The OpenHands runner emits compact structured `OPENHANDS_*` log
lines and uses SDK conversation persistence instead. The broadest qualitative
research task also exposed a timeout/reliability surface that should remain in
CI/evals: use timeout failures as runtime reliability failures even when
completed answers are excellent.
