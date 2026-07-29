"""Initialize W&B Weave tracing before OpenHands enters the process."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from weave_openhands import finish as weave_finish
from weave_openhands import init as weave_init

_initialized = False
_project_name: str | None = None
TRACE_SECRET_ENV_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "WANDB_API_KEY",
    "EXA_API_KEY",
    "ANTHROPIC_API_KEY",
)


def _is_secret_env(name: str) -> bool:
    return name in TRACE_SECRET_ENV_NAMES or name.endswith(
        ("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET", "_CREDENTIAL")
    )


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


def secret_redactor(env: Mapping[str, str]) -> Callable[[str], str]:
    configured_model_key = env.get("SENPAI_OPENHANDS_API_KEY_ENV")
    secret_values = sorted(
        {
            value
            for name, value in env.items()
            if value
            and (
                _is_secret_env(name)
                or (configured_model_key is not None and name == configured_model_key)
            )
        },
        key=len,
        reverse=True,
    )

    def redact(content: str) -> str:
        for value in secret_values:
            content = content.replace(value, "<secret-hidden>")
        return content

    return redact


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
        content_transform=secret_redactor(env),
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
