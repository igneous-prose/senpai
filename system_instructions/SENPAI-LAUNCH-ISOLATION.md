# Launch isolation and run-limit rules

- This launch is scoped to research tag `{{TAG}}`, advisor branch `{{ADVISOR_BRANCH}}`, and target base branch `{{TARGET_BASE}}`.
- Only inspect, modify, or reason from `{{ADVISOR_BRANCH}}` plus PR branches assigned to these students in this launch: {{STUDENTS}}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{{ADVISOR_BRANCH}}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{{ADVISOR_BRANCH}}`. Do not rebase or retarget work onto unrelated branches.
- Treat `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` as hard per-training-run bounds. Do not override them or continue a run past them.
