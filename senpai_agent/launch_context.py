# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Render the launch-time role context stored in system_instructions/."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header

INSTRUCTIONS_ROOT = Path(__file__).resolve().parent.parent / "system_instructions"
RUNTIME_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-LAUNCH-RUNTIME.md"
ISOLATION_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-LAUNCH-ISOLATION.md"
OPERATOR_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-OPERATOR-INSTRUCTIONS.md"
PLACEHOLDER = re.compile(r"{{([A-Z_][A-Z0-9_]*)}}")
ROLE_TEMPLATE_VALUES = {
    "advisor": (
        "GH_REPO",
        "ADVISOR_BRANCH",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "STUDENT_NAMES",
    ),
    "student": (
        "GH_REPO",
        "ADVISOR_BRANCH",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "STUDENT_NAME",
    ),
}


def _render(path: Path, values: Mapping[str, str]) -> str:
    template = read_agent_markdown(path)
    missing = sorted(set(PLACEHOLDER.findall(template)) - values.keys())
    if missing:
        raise ValueError(f"Missing {path.name} values: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], template).strip()


def render_role_prompt(
    path: Path,
    role: Literal["advisor", "student"],
    env: Mapping[str, str],
) -> str:
    """Render one role charter from explicitly allowlisted non-secret values."""

    values = {
        "ROLE": role,
        **{
            name: env[name]
            for name in ROLE_TEMPLATE_VALUES[role]
            if env.get(name)
        },
    }
    return _render(path, values)


def render_launch_context(
    *,
    backend: str,
    gpus_per_student: int,
    timeout_minutes: float,
    max_epochs: int,
    tag: str,
    advisor_branch: str,
    target_base: str,
    students: list[str],
    extra_instructions: str = "",
) -> str:
    """Render backend facts, isolation, and optional operator instructions."""

    sections = [
        _render(
            RUNTIME_TEMPLATE,
            {
                "BACKEND": backend,
                "GPUS_PER_STUDENT": str(gpus_per_student),
                "TIMEOUT_MINUTES": f"{timeout_minutes:g}",
                "MAX_EPOCHS": str(max_epochs),
            },
        ),
        _render(
            ISOLATION_TEMPLATE,
            {
                "TAG": tag,
                "ADVISOR_BRANCH": advisor_branch,
                "TARGET_BASE": target_base or "<default>",
                "STUDENTS": ", ".join(students),
            },
        ),
    ]
    if extra_instructions:
        path = Path(extra_instructions)
        text = (
            read_agent_markdown(path)
            if path.exists()
            else strip_spdx_header(extra_instructions)
        )
        sections.append(_render(OPERATOR_TEMPLATE, {"EXTRA_INSTRUCTIONS": text}))
    return "\n\n".join(sections)
