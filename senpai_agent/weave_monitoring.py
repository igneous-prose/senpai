"""Initialize W&B Weave tracing before OpenHands enters the process."""

from __future__ import annotations

import os
from collections.abc import Mapping

from weave_openhands import finish as weave_finish
from weave_openhands import init as weave_init


_initialized = False
_project_name: str | None = None


def weave_project_name(env: Mapping[str, str]) -> str | None:
    entity = env.get("WANDB_ENTITY")
    project = env.get("WANDB_PROJECT")
    if not entity and not project:
        return None
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY and WANDB_PROJECT must be set together")
    return f"{entity}/{project}"


def weave_agent_name(env: Mapping[str, str]) -> str:
    role = env.get("SENPAI_ROLE", "senpai")
    student_name = env.get("STUDENT_NAME")
    if role == "student" and student_name:
        return f"student-{student_name}"
    return role


def initialize_weave_monitoring(
    env: Mapping[str, str] = os.environ,
) -> str | None:
    global _initialized, _project_name
    if _initialized:
        return _project_name

    project_name = weave_project_name(env)
    if project_name is None:
        return None

    weave_init(
        project_name,
        agent_name=weave_agent_name(env),
        capture_content=True,
    )
    _initialized = True
    _project_name = project_name
    return project_name


def finish_weave_monitoring() -> None:
    global _initialized, _project_name
    if not _initialized:
        return
    try:
        weave_finish()
    finally:
        _initialized = False
        _project_name = None
