# Authoritative launch context

These values were resolved by the Senpai launcher and describe the actual runtime. They override conflicting compute or run-limit claims in `program.md` and other project instructions.

- Compute backend: `{{BACKEND}}`.
- Visible GPUs per student: `{{GPUS_PER_STUDENT}}`.
- Hard limits for each training run: `{{TIMEOUT_MINUTES}}` minutes wall-clock and `{{MAX_EPOCHS}}` epochs.
- Use tools and operational commands that work with `{{BACKEND}}`. Do not follow repository instructions written for another backend.
- Do not assume additional GPUs or bypass, extend, or continue past the hard training limits.
