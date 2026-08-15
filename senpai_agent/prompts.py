# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Load the model-facing instruction templates assembled by Python."""

from __future__ import annotations

import re
from pathlib import Path

from senpai_agent.agent_markdown import read_agent_markdown


PROMPTS_PATH = Path(__file__).with_name("PROMPTS.md")
_SECTION_HEADING = re.compile(r"^## ([A-Z][A-Z0-9_]*_PROMPT)\s*$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"{{([A-Z][A-Z0-9_]*)}}")
_PROMPT_NAMES = (
    "CONTEXT_RECOVERY_PROMPT",
    "INITIAL_CONTROLLER_PROMPT",
    "CONTINUATION_CONTROLLER_PROMPT",
    "SYSTEM_CONTEXT_REFRESH_PROMPT",
    "ADDITIONAL_LAUNCH_INSTRUCTIONS_PROMPT",
    "ADVISOR_RUNTIME_IDENTITY_PROMPT",
    "STUDENT_RUNTIME_IDENTITY_PROMPT",
    "CURRENT_LAUNCH_CONTEXT_PROMPT",
    "SENPAI_SYSTEM_INSTRUCTIONS_PROMPT",
    "PROGRAM_SYSTEM_PROMPT",
    "DELEGATED_SEARCH_MODE_PROMPT",
    "DELEGATED_TASK_PROMPT",
    "DELEGATED_TASK_WITH_CONTEXT_PROMPT",
    "RECOVERED_ACTION_PROMPT",
    "ADVISOR_EVENT_PROMPT",
    "EVENT_PROMPT",
    "WORKSPACE_DIVERGENCE_PROMPT",
    "TRUNCATED_FEEDBACK_PROMPT",
    "MONITOR_TRAINING_STARTED_PROMPT",
    "AWAIT_AGENTS_SATISFIED_PROMPT",
    "AWAIT_AGENTS_TIMEOUT_PROMPT",
    "DELEGATED_TASK_FINISHED_PROMPT",
    "DELEGATED_TASK_BACKGROUND_PROMPT",
    "DELEGATE_AGENT_DEPRECATION_PROMPT",
)


def _parse_prompt_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in sections:
            raise RuntimeError(f"duplicate prompt section: {name}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        if not content:
            raise RuntimeError(f"empty prompt section: {name}")
        sections[name] = content
    return sections


def _load_prompts() -> dict[str, str]:
    sections = _parse_prompt_sections(read_agent_markdown(PROMPTS_PATH))
    expected = set(_PROMPT_NAMES)
    missing = sorted(expected - sections.keys())
    unexpected = sorted(sections.keys() - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise RuntimeError(f"invalid {PROMPTS_PATH.name}: {'; '.join(details)}")
    return sections


def render_prompt(template: str, /, **values: str) -> str:
    """Render one prompt without interpreting placeholders in inserted values."""

    placeholders = set(_PLACEHOLDER.findall(template))
    missing = sorted(placeholders - values.keys())
    unexpected = sorted(values.keys() - placeholders)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(f"invalid prompt values: {'; '.join(details)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


_PROMPTS = _load_prompts()

CONTEXT_RECOVERY_PROMPT = _PROMPTS["CONTEXT_RECOVERY_PROMPT"]
INITIAL_CONTROLLER_PROMPT = _PROMPTS["INITIAL_CONTROLLER_PROMPT"]
CONTINUATION_CONTROLLER_PROMPT = _PROMPTS["CONTINUATION_CONTROLLER_PROMPT"]
SYSTEM_CONTEXT_REFRESH_PROMPT = _PROMPTS["SYSTEM_CONTEXT_REFRESH_PROMPT"]
ADDITIONAL_LAUNCH_INSTRUCTIONS_PROMPT = _PROMPTS[
    "ADDITIONAL_LAUNCH_INSTRUCTIONS_PROMPT"
]
ADVISOR_RUNTIME_IDENTITY_PROMPT = _PROMPTS["ADVISOR_RUNTIME_IDENTITY_PROMPT"]
STUDENT_RUNTIME_IDENTITY_PROMPT = _PROMPTS["STUDENT_RUNTIME_IDENTITY_PROMPT"]
CURRENT_LAUNCH_CONTEXT_PROMPT = _PROMPTS["CURRENT_LAUNCH_CONTEXT_PROMPT"]
SENPAI_SYSTEM_INSTRUCTIONS_PROMPT = _PROMPTS[
    "SENPAI_SYSTEM_INSTRUCTIONS_PROMPT"
]
PROGRAM_SYSTEM_PROMPT = _PROMPTS["PROGRAM_SYSTEM_PROMPT"]
DELEGATED_SEARCH_MODE_PROMPT = _PROMPTS["DELEGATED_SEARCH_MODE_PROMPT"]
DELEGATED_TASK_PROMPT = _PROMPTS["DELEGATED_TASK_PROMPT"]
DELEGATED_TASK_WITH_CONTEXT_PROMPT = _PROMPTS[
    "DELEGATED_TASK_WITH_CONTEXT_PROMPT"
]
RECOVERED_ACTION_PROMPT = _PROMPTS["RECOVERED_ACTION_PROMPT"]
ADVISOR_EVENT_PROMPT = _PROMPTS["ADVISOR_EVENT_PROMPT"]
EVENT_PROMPT = _PROMPTS["EVENT_PROMPT"]
WORKSPACE_DIVERGENCE_PROMPT = _PROMPTS["WORKSPACE_DIVERGENCE_PROMPT"]
TRUNCATED_FEEDBACK_PROMPT = _PROMPTS["TRUNCATED_FEEDBACK_PROMPT"]
MONITOR_TRAINING_STARTED_PROMPT = _PROMPTS["MONITOR_TRAINING_STARTED_PROMPT"]
AWAIT_AGENTS_SATISFIED_PROMPT = _PROMPTS["AWAIT_AGENTS_SATISFIED_PROMPT"]
AWAIT_AGENTS_TIMEOUT_PROMPT = _PROMPTS["AWAIT_AGENTS_TIMEOUT_PROMPT"]
DELEGATED_TASK_FINISHED_PROMPT = _PROMPTS["DELEGATED_TASK_FINISHED_PROMPT"]
DELEGATED_TASK_BACKGROUND_PROMPT = _PROMPTS["DELEGATED_TASK_BACKGROUND_PROMPT"]
DELEGATE_AGENT_DEPRECATION_PROMPT = _PROMPTS[
    "DELEGATE_AGENT_DEPRECATION_PROMPT"
]

__all__ = (
    *_PROMPT_NAMES,
    "PROMPTS_PATH",
    "render_prompt",
)
