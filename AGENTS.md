<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai - Development Context

Development of a problem-agnostic autonomous ML research loop for target ML
problem repositories. The current research programs are often CFD surrogate
experiments, but the runner should stay target-repo agnostic.

## User Clarifications

### Interviewing the developer about how to do a task:
When asked for a large piece of work that seems vague, consequential, or full
of hidden tradeoffs, ask the user detailed clarifying questions about the real
implementation choices: technical design, workflow, UX, risks, validation,
operations, and tradeoffs. Prefer non-obvious questions that expose constraints
or intent. When the answers change durable project behavior, write the learnings
to README.md or SPEC.md as appropriate.


## Coding guidelines and philosophy

- You should generate code that is simple and readable. Avoid unnecessary abstractions and complexity. This is a research codebase, so maintainability and clarity matter.
- Avoid overly defensive coding. No need for lots of `try`/`except` patterns, fallbacks, or backups. Prefer code that fails clearly when something is wrong so it can be fixed.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to the repository's Python 3.13 runtime.

## Key docs

- `README.md` - operator-facing overview, launch examples, and problem-package layout.
- `SPEC.md` - target architecture and rewrite contract for the senpai orchestration loop.
- `senpai.yaml` - launch defaults, including the target repo, target branch, advisor branch, and `problem_dir`.
- `$PROBLEM_DIR/program.md` - authoritative target research context, goals, metrics, training constraints, and file boundaries. With the default config this is `target/program.md` after the target repo is cloned.
- `$PROBLEM_DIR/instructions/prompt-advisor.md` - target-specific advisor prompt.
- `$PROBLEM_DIR/instructions/prompt-student.md` - target-specific student prompt.
- `system_instructions/SENPAI-HARNESS.md` - shared OpenHands harness contract.
- `system_instructions/ADVISOR.md` - advisor role workflow.
- `system_instructions/STUDENT.md` - student role workflow.

## Architecture

- **Runner repo** - this repo. Owns orchestration, Kubernetes launch, role instructions, GitHub helpers, W&B integration, and operational docs.
- **Target repo** - cloned into `$PROBLEM_DIR` from `target_repo_url`. Owns the data code, training code, evaluation code, `program.md`, target prompts, and experiment branches. Agent commits and PRs land in the target repo, not in the runner repo.
- **Advisor pod** - lightweight, no GPU, keeps one durable OpenHands
  conversation and uses typed control-plane tools for GitHub and generic
  child-agent dispatch.
- **Student pods** - heavy GPU workers, use one OpenHands conversation per
  assignment revision, implement one assigned PR, run supervised training, and
  resume the same conversation for actionable monitor events.
- **Cross-node communication** - GitHub PR labels and human-tagged Issues only;
  Senpai requires no RPC service or cluster-specific network setup.
- **GitHub Issues** - human-to-agent communication channel. Agents poll for and respond to these alongside their normal PR workflow.
- **W&B** - canonical experiment metrics store for training runs, comparisons, and merge decisions.

## k8s layout

- `k8s/advisor-deployment.yaml` / `k8s/student-deployment.yaml` — pod specs
- `k8s/entrypoint-advisor.sh` / `k8s/entrypoint-student.sh` — startup scripts
- `k8s/launch.py` — helper to template and apply deployments

## system_instructions/

The OpenHands base prompt is extended with a stable merged suffix from the
shared harness file and one rendered role file:

- `system_instructions/SENPAI-HARNESS.md`
- `system_instructions/ADVISOR.md` or
  `system_instructions/STUDENT.md`

Target `AGENTS.md`, compatible `CLAUDE.md`, and skills are loaded through
OpenHands project context and progressive disclosure. The checked-in root
`CLAUDE.md` is only a compact pointer to this development context; neither root
file is a pod role instruction.
