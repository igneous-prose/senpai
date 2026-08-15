# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Render and transport launch-time instructions."""

import binascii
import re
from base64 import b64decode
from pathlib import Path

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header

INSTRUCTIONS_ROOT = Path(__file__).resolve().parent.parent / "system_instructions"
LAUNCH_CONTEXT_TEMPLATE = INSTRUCTIONS_ROOT / "SENPAI-LAUNCH-CONTEXT.md"
LAUNCH_CONTEXT_ENV = "SENPAI_LAUNCH_CONTEXT_B64"
PLACEHOLDER = re.compile(r"{{([A-Z_]+)}}")


def _render(path: Path, values: dict[str, str]) -> str:
    template = read_agent_markdown(path)
    missing = sorted(set(PLACEHOLDER.findall(template)) - values.keys())
    if missing:
        raise ValueError(f"Missing {path.name} values: {', '.join(missing)}")
    for key, value in values.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template.strip()


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
) -> str:
    """Render authoritative runtime and isolation rules."""

    return _render(
        LAUNCH_CONTEXT_TEMPLATE,
        {
            "BACKEND": backend,
            "GPUS_PER_STUDENT": str(gpus_per_student),
            "TIMEOUT_MINUTES": f"{timeout_minutes:g}",
            "MAX_EPOCHS": str(max_epochs),
            "TAG": tag,
            "ADVISOR_BRANCH": advisor_branch,
            "TARGET_BASE": target_base or "<default>",
            "STUDENTS": ", ".join(students),
        },
    )


def load_operator_instructions(value: str) -> str:
    """Load optional human guidance without granting it system authority."""

    if not value:
        return ""
    path = Path(value)
    text = read_agent_markdown(path) if path.exists() else strip_spdx_header(value)
    return text.strip()


def decode_launch_context(encoded: str) -> str:
    """Decode the mandatory authoritative context at process startup."""

    if not encoded:
        raise ValueError(f"{LAUNCH_CONTEXT_ENV} is required")
    try:
        context = b64decode(encoded, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"{LAUNCH_CONTEXT_ENV} must be valid base64-encoded UTF-8"
        ) from error
    if not context.strip():
        raise ValueError(f"{LAUNCH_CONTEXT_ENV} must not be empty")
    return context.strip()
