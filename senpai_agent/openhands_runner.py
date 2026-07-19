"""Run Senpai loops on the OpenHands SDK.

This module is the OpenHands equivalent of the Claude Code headless invocation
in ``k8s/run-senpai-claude.sh``. The shell loop still owns polling, checkout,
watchdogs, assignment routing, and logs; this file owns only a single agent run.
"""

# OpenHands imports intentionally follow Weave initialization below.
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from senpai_agent.weave_monitoring import (
    finish_weave_monitoring,
    initialize_weave_monitoring,
)

WEAVE_PROJECT = initialize_weave_monitoring()

from openhands.sdk import Agent, AgentContext, Conversation, LLM, load_skills_from_dir
from openhands.sdk.plugin import PluginSource
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    register_agent_if_absent,
)
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.preset.default import (
    get_default_condenser,
    get_default_tools,
    register_builtins_agents,
)
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from pydantic import SecretStr
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import flag

DEFAULT_MODEL = "anthropic/claude-opus-4-8"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_REASONING_EFFORT = "xhigh"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra", "none")
SENPAI_CONTINUATION_FILE = "current_conversation_id"
FAILING_STATUSES = {"error", "stuck"}
EVENT_TEXT_LIMIT = 20000


@dataclass(frozen=True)
class RunnerArgs:
    max_turns: int = field(alias="--max-turns")
    continue_session: bool = field(
        default=False, alias=["-c", "--continue"], action="store_true"
    )
    model: str | None = field(default=None, alias="--model")
    api_key_env: str | None = field(default=None, alias="--api-key-env")
    reasoning_effort: str | None = field(
        default=None,
        alias="--reasoning-effort",
        choices=REASONING_EFFORTS,
    )
    workspace: str | None = field(default=None, alias="--workspace")
    state_dir: str | None = field(default=None, alias="--state-dir")
    conversation_id: str | None = field(default=None, alias="--conversation-id")
    role_file: str | None = field(default=None, alias="--role-file")
    plugin_dir: str | None = field(default=None, alias="--plugin-dir")
    agent: str | None = field(default=None, alias="--agent")
    enable_browser: bool = flag(
        default=True,
        alias="--browser",
        negative_option="--no-browser",
    )


@dataclass(frozen=True)
class RunnerConfig:
    max_turns: int
    model: str
    api_key_env: str
    api_key: str
    reasoning_effort: str
    workspace: Path
    state_dir: Path
    conversation_id: uuid.UUID
    continue_session: bool
    enable_browser: bool
    agent_name: str | None
    role_file: Path
    plugin_dir: Path
    skill_dirs: tuple[Path, ...]
    agent_dirs: tuple[Path, ...]


def parse_runner_args(argv: Sequence[str] | None = None) -> RunnerArgs:
    parser = ArgumentParser(description="Run a Senpai OpenHands agent.")
    parser.add_arguments(RunnerArgs, dest="args")
    return parser.parse_args(argv).args


def openhands_reasoning_effort(reasoning_effort: str) -> str:
    if reasoning_effort in {"max", "ultra"}:
        return "xhigh"
    return reasoning_effort


def env_value(
    parsed_value: str | None,
    env: Mapping[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    return parsed_value if parsed_value is not None else env.get(key, default)


def resolve_api_key(env: Mapping[str, str], key_env: str) -> str:
    value = env.get(key_env)
    if not value:
        raise RuntimeError(f"{key_env} is required for the OpenHands runtime")
    return value


def default_state_dir(workspace: Path, env: Mapping[str, str]) -> Path:
    if env.get("LOGDIR"):
        return Path(env["LOGDIR"]) / "openhands_state"
    return workspace / ".senpai" / "openhands"


def find_role_file(workspace: Path, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"OpenHands role file does not exist: {path}")
        return path

    for current in (workspace, *workspace.parents):
        candidate = current / "CLAUDE.md"
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "OpenHands role instructions were not found; set SENPAI_OPENHANDS_ROLE_FILE"
    )


def read_role_instructions(path: Path) -> str:
    instructions = path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise RuntimeError(f"OpenHands role file is empty: {path}")
    return instructions


def resolve_plugin_dir(explicit: str | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else Path(__file__).resolve().parents[1] / "plugins" / "senpai"
    )
    manifest = path / ".plugin" / "plugin.json"
    if not path.is_dir() or not manifest.is_file():
        raise RuntimeError(f"Senpai OpenHands plugin does not exist: {path}")
    return path


def candidate_skill_dirs(workspace: Path, env: Mapping[str, str]) -> tuple[Path, ...]:
    paths = [
        Path.home() / ".claude" / "skills",
        Path.home() / ".agents" / "skills",
        Path.home() / ".openhands" / "skills",
        workspace / ".claude" / "skills",
        workspace / ".agents" / "skills",
        workspace / ".openhands" / "skills",
    ]
    if env.get("SENPAI_PLUGIN"):
        paths.append(Path(env["SENPAI_PLUGIN"]) / "skills")
        paths.append(Path(env["SENPAI_PLUGIN"]).parent.parent / ".claude" / "skills")
    for current in workspace.parents:
        paths.extend(
            [
                current / ".claude" / "skills",
                current / ".agents" / "skills",
                current / ".openhands" / "skills",
            ]
        )
    return existing_unique_paths(paths)


def candidate_agent_dirs(workspace: Path, env: Mapping[str, str]) -> tuple[Path, ...]:
    paths = [
        Path.home() / ".claude" / "agents",
        Path.home() / ".agents" / "agents",
        Path.home() / ".openhands" / "agents",
        workspace / ".claude" / "agents",
        workspace / ".agents" / "agents",
        workspace / ".openhands" / "agents",
    ]
    if env.get("SENPAI_PLUGIN"):
        paths.append(Path(env["SENPAI_PLUGIN"]).parent.parent / ".claude" / "agents")
    for current in workspace.parents:
        paths.extend(
            [
                current / ".claude" / "agents",
                current / ".agents" / "agents",
                current / ".openhands" / "agents",
            ]
        )
    return existing_unique_paths(paths)


def existing_unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        existing.append(path)
    return tuple(existing)


def fresh_conversation_id() -> uuid.UUID:
    return uuid.uuid4()


def select_conversation_id(
    state_dir: Path,
    *,
    continue_session: bool,
    explicit_id: str | None = None,
) -> uuid.UUID:
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / SENPAI_CONTINUATION_FILE
    if explicit_id:
        conversation_id = uuid.UUID(explicit_id)
        marker.write_text(str(conversation_id), encoding="utf-8")
        return conversation_id
    if continue_session and marker.exists():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous:
            conversation_id = uuid.UUID(previous)
            marker.write_text(str(conversation_id), encoding="utf-8")
            return conversation_id
    conversation_id = fresh_conversation_id()
    marker.write_text(str(conversation_id), encoding="utf-8")
    return conversation_id


def normalize_skill_name(name: str) -> str:
    return name.split(":", 1)[1] if ":" in name else name


def normalize_skill_names(names: Sequence[str]) -> list[str]:
    return [normalize_skill_name(name) for name in names]


def openhands_tool_aliases(enable_browser: bool) -> dict[str, str]:
    aliases = {
        "TerminalTool": TerminalTool.name,
        "terminal": TerminalTool.name,
        "FileEditorTool": FileEditorTool.name,
        "file_editor": FileEditorTool.name,
        "str_replace_editor": FileEditorTool.name,
        "TaskTrackerTool": TaskTrackerTool.name,
        "task_tracker": TaskTrackerTool.name,
    }
    if enable_browser:
        aliases.update(
            {
                "BrowserToolSet": BrowserToolSet.name,
                "browser": BrowserToolSet.name,
                "browser_tool_set": BrowserToolSet.name,
            }
        )
    return aliases


def normalize_tool_names(names: Sequence[str], *, enable_browser: bool) -> list[str]:
    aliases = openhands_tool_aliases(enable_browser)
    return [aliases.get(name, name) for name in names]


def default_subagent_tools(*, enable_browser: bool) -> list[str]:
    names = [TerminalTool.name, FileEditorTool.name, TaskTrackerTool.name]
    if enable_browser:
        names.append(BrowserToolSet.name)
    return names


def resolve_config(
    args: RunnerArgs,
    env: Mapping[str, str] = os.environ,
) -> RunnerConfig:
    workspace_arg = (
        env_value(args.workspace, env, "SENPAI_OPENHANDS_WORKSPACE", os.getcwd())
        or os.getcwd()
    )
    workspace = Path(workspace_arg).expanduser().resolve()
    if not workspace.exists():
        raise RuntimeError(f"OpenHands workspace does not exist: {workspace}")

    state_dir_arg = env_value(args.state_dir, env, "SENPAI_OPENHANDS_STATE_DIR")
    state_dir = (
        Path(state_dir_arg).expanduser().resolve()
        if state_dir_arg
        else default_state_dir(workspace, env)
    )
    api_key_env = (
        env_value(
            args.api_key_env, env, "SENPAI_OPENHANDS_API_KEY_ENV", DEFAULT_API_KEY_ENV
        )
        or DEFAULT_API_KEY_ENV
    )
    return RunnerConfig(
        max_turns=args.max_turns,
        model=env_value(args.model, env, "SENPAI_OPENHANDS_MODEL", DEFAULT_MODEL)
        or DEFAULT_MODEL,
        api_key_env=api_key_env,
        api_key=resolve_api_key(env, api_key_env),
        reasoning_effort=env_value(
            args.reasoning_effort,
            env,
            "SENPAI_OPENHANDS_REASONING_EFFORT",
            DEFAULT_REASONING_EFFORT,
        )
        or DEFAULT_REASONING_EFFORT,
        workspace=workspace,
        state_dir=state_dir,
        conversation_id=select_conversation_id(
            state_dir,
            continue_session=args.continue_session,
            explicit_id=env_value(
                args.conversation_id, env, "SENPAI_OPENHANDS_CONVERSATION_ID"
            ),
        ),
        continue_session=args.continue_session,
        enable_browser=args.enable_browser,
        agent_name=env_value(args.agent, env, "SENPAI_OPENHANDS_AGENT"),
        role_file=find_role_file(
            workspace,
            env_value(args.role_file, env, "SENPAI_OPENHANDS_ROLE_FILE"),
        ),
        plugin_dir=resolve_plugin_dir(
            env_value(args.plugin_dir, env, "SENPAI_PLUGIN"),
        ),
        skill_dirs=candidate_skill_dirs(workspace, env),
        agent_dirs=candidate_agent_dirs(workspace, env),
    )


def load_agent_definition(
    path: Path,
    skills_by_name: Mapping[str, object],
    *,
    enable_browser: bool,
):
    definition = AgentDefinition.load(path)
    skill_sections = []
    for name in normalize_skill_names(definition.skills):
        skill = skills_by_name.get(name)
        if not skill:
            continue
        content = getattr(skill, "content", "")
        if content:
            skill_sections.append(f"## {name}\n\n{content}")

    system_prompt = definition.system_prompt
    if skill_sections:
        system_prompt = f"{system_prompt}\n\n# Referenced skills\n\n" + "\n\n".join(
            skill_sections
        )

    updates = {
        "skills": [],
        "system_prompt": system_prompt,
        "permission_mode": definition.permission_mode or "never_confirm",
    }
    if definition.model in {"opus", "sonnet", "haiku"}:
        updates["model"] = "inherit"
    updates["tools"] = (
        normalize_tool_names(definition.tools, enable_browser=enable_browser)
        if definition.tools
        else default_subagent_tools(enable_browser=enable_browser)
    )
    return definition.model_copy(update=updates)


def load_named_agent_definition(
    name: str,
    agent_dirs: Sequence[Path],
    skills: Sequence[object],
    *,
    enable_browser: bool,
):
    skills_by_name = {getattr(skill, "name"): skill for skill in skills}
    for agent_dir in agent_dirs:
        for path in sorted(agent_dir.glob("*.md")):
            definition = AgentDefinition.load(path)
            if definition.name == name or path.stem == name:
                return load_agent_definition(
                    path, skills_by_name, enable_browser=enable_browser
                )
    raise RuntimeError(f"OpenHands agent not found: {name}")


def append_role_instructions(
    definition: object, role_instructions: str | None
) -> object:
    if not role_instructions:
        return definition
    prompt = getattr(definition, "system_prompt", "")
    return definition.model_copy(
        update={
            "system_prompt": (
                f"{prompt}\n\n# Senpai role instructions\n\n{role_instructions}"
                if prompt
                else role_instructions
            )
        }
    )


def load_skills(skill_dirs: Sequence[Path]):
    skills_by_name = {}
    for skill_dir in skill_dirs:
        repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(skill_dir)
        skills_by_name.update(repo_skills)
        skills_by_name.update(knowledge_skills)
        skills_by_name.update(agent_skills)
    return list(skills_by_name.values())


def register_subagents(
    agent_dirs: Sequence[Path],
    skills: Sequence[object],
    *,
    enable_browser: bool,
) -> list[str]:
    skills_by_name = {getattr(skill, "name"): skill for skill in skills}
    registered = register_builtins_agents(enable_browser=enable_browser)
    for agent_dir in agent_dirs:
        for path in sorted(agent_dir.glob("*.md")):
            definition = load_agent_definition(
                path,
                skills_by_name,
                enable_browser=enable_browser,
            )
            if register_agent_if_absent(
                name=definition.name,
                factory_func=agent_definition_to_factory(definition),
                description=definition,
            ):
                registered.append(definition.name)
    return registered


def build_main_agent_context(skills: Sequence[object], role_instructions: str):
    return AgentContext(
        skills=list(skills),
        system_message_suffix=role_instructions,
        load_public_skills=False,
        load_user_skills=True,
        load_project_skills=True,
    )


def event_summary(event: object) -> dict[str, object]:
    summary: dict[str, object] = {"event": event.__class__.__name__}
    for attr in ("source", "tool_name", "action", "status"):
        value = getattr(event, attr, None)
        if value is not None:
            summary[attr] = str(value)

    message = getattr(event, "llm_message", None)
    if getattr(event, "source", None) == "agent" and message is not None:
        text_parts = [
            getattr(part, "text", "")
            for part in getattr(message, "content", [])
            if getattr(part, "text", "")
        ]
        text = "\n".join(text_parts).strip()
        if text:
            summary["text"] = text[-EVENT_TEXT_LIMIT:]
    return summary


def print_event(event: object) -> None:
    print(
        "OPENHANDS_EVENT " + json.dumps(event_summary(event), sort_keys=True),
        flush=True,
    )


def run_openhands(prompt: str, config: RunnerConfig) -> int:
    skills = load_skills(config.skill_dirs)
    role_instructions = read_role_instructions(config.role_file)
    direct_agent_definition = None
    if config.agent_name:
        direct_agent_definition = append_role_instructions(
            load_named_agent_definition(
                config.agent_name,
                config.agent_dirs,
                skills,
                enable_browser=config.enable_browser,
            ),
            role_instructions,
        )
        subagents = [getattr(direct_agent_definition, "name", config.agent_name)]
    else:
        subagents = register_subagents(
            config.agent_dirs,
            skills,
            enable_browser=config.enable_browser,
        )

    print(
        "OPENHANDS_RUN "
        + json.dumps(
            {
                "workspace": str(config.workspace),
                "state_dir": str(config.state_dir),
                "conversation_id": str(config.conversation_id),
                "continue": config.continue_session,
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "openhands_reasoning_effort": openhands_reasoning_effort(
                    config.reasoning_effort
                ),
                "agent": config.agent_name,
                "enable_browser": config.enable_browser,
                "role_file": str(config.role_file) if config.role_file else None,
                "plugin_dir": str(config.plugin_dir),
                "skill_dirs": [str(path) for path in config.skill_dirs],
                "skills": [skill.name for skill in skills],
                "agent_dirs": [str(path) for path in config.agent_dirs],
                "subagents": subagents,
                "weave_project": WEAVE_PROJECT,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    llm = LLM(
        model=config.model,
        api_key=SecretStr(config.api_key),
        reasoning_effort=openhands_reasoning_effort(config.reasoning_effort),
        usage_id="senpai",
    )
    get_default_tools(enable_browser=config.enable_browser, enable_sub_agents=True)
    if direct_agent_definition:
        agent = agent_definition_to_factory(
            direct_agent_definition, work_dir=config.workspace
        )(llm)
        agent = agent.model_copy(
            update={
                "agent_context": agent.agent_context.model_copy(
                    update={
                        "load_user_skills": True,
                        "load_project_skills": True,
                    }
                )
            }
        )
    else:
        agent = Agent(
            llm=llm,
            tools=get_default_tools(
                enable_browser=config.enable_browser,
                enable_sub_agents=True,
            ),
            agent_context=build_main_agent_context(skills, role_instructions),
            system_prompt_kwargs={"cli_mode": True},
            condenser=get_default_condenser(
                llm.model_copy(update={"usage_id": "senpai-condenser"})
            ),
        )
    conversation = Conversation(
        agent=agent,
        workspace=config.workspace,
        plugins=[PluginSource(source=str(config.plugin_dir))],
        persistence_dir=config.state_dir,
        conversation_id=config.conversation_id,
        callbacks=[print_event],
        max_iteration_per_run=config.max_turns,
        visualizer=None,
        tags={"runtime": "senpai-openhands"},
    )
    conversation.send_message(prompt)
    conversation.run()

    status = str(conversation.state.execution_status.value)
    print(
        "OPENHANDS_RESULT "
        + json.dumps(
            {"conversation_id": str(conversation.id), "status": status}, sort_keys=True
        ),
        flush=True,
    )
    return 1 if status in FAILING_STATUSES else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_runner_args(argv)
    prompt = sys.stdin.read()
    if not prompt:
        raise RuntimeError("OpenHands runner requires a prompt on stdin")
    try:
        config = resolve_config(args)
        return run_openhands(prompt, config)
    finally:
        finish_weave_monitoring()


if __name__ == "__main__":
    raise SystemExit(main())
