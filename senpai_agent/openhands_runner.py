"""Run one bounded Senpai OpenHands turn for the Python controller."""

# OpenHands imports intentionally follow Weave initialization below.

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from senpai_agent.advisor import (
    AdvisorEventPump,
    AdvisorEventStore,
    advisor_conversation_id,
    compose_system_instructions,
)
from senpai_agent.delegation import (
    MAX_PARALLEL_AGENTS,
    DelegationConfig,
    configure_delegation,
)
from senpai_agent.weave_monitoring import (
    finish_weave_monitoring,
    initialize_weave_monitoring,
)

WEAVE_PROJECT = initialize_weave_monitoring()

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import TextContent
from openhands.sdk.plugin import PluginSource
from openhands.sdk.subagent import (
    AgentDefinition,
    agent_definition_to_factory,
    discover_agents,
    register_file_agents,
)
from openhands.tools.preset.default import (
    get_default_condenser,
    get_default_tools,
)
from pydantic import SecretStr
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import flag

from senpai_agent.tools import (
    clear_github_credentials,
    configure_github_credentials,
    register_senpai_tools,
)

DEFAULT_MODEL = "anthropic/claude-opus-4-8"
DEFAULT_FAST_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_FAST_REASONING_EFFORT = "low"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra", "none")
SENPAI_CONTINUATION_FILE = "current_conversation_id"
COMMAND_SECRET_ENV_NAMES = (
    "WANDB_API_KEY",
    "EXA_API_KEY",
)
GITHUB_SECRET_ENV_NAMES = ("GITHUB_TOKEN", "GH_TOKEN")
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
    harness_file: str | None = field(default=None, alias="--harness-file")
    plugin_dir: str | None = field(default=None, alias="--plugin-dir")
    agent: str | None = field(default=None, alias="--agent")
    enable_browser: bool = flag(
        default=True,
        alias="--browser",
        negative_option="--no-browser",
    )
    child: bool = flag(default=False, alias="--child")


@dataclass(frozen=True)
class RunnerConfig:
    max_turns: int
    model: str
    api_key_env: str
    api_key: SecretStr
    github_repo: str
    github_token: SecretStr
    github_trusted_actor: str | None
    command_secrets: Mapping[str, str]
    reasoning_effort: str
    smart_model: str
    fast_model: str
    fast_reasoning_effort: str
    workspace: Path
    state_dir: Path
    conversation_id: uuid.UUID
    continue_session: bool
    role: str
    enable_browser: bool
    agent_name: str | None
    harness_file: Path
    role_file: Path
    plugin_dir: Path
    training_max_timeout_seconds: int = 1800
    timeout_seconds: float = 3600
    child: bool = False


def parse_runner_args(argv: Sequence[str] | None = None) -> RunnerArgs:
    parser = ArgumentParser(description="Run a Senpai OpenHands agent.")
    parser.add_arguments(RunnerArgs, dest="args")
    return parser.parse_args(argv).args


def openhands_reasoning_effort(reasoning_effort: str, model: str) -> str:
    provider, _, model_name = model.lower().partition("/")
    if reasoning_effort in {"max", "ultra"} and (
        provider != "openai" or not model_name.startswith("gpt-5.6")
    ):
        return "xhigh"
    return "max" if reasoning_effort == "ultra" else reasoning_effort


def env_value(
    parsed_value: str | None,
    env: Mapping[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    return parsed_value if parsed_value is not None else env.get(key, default)


def resolve_api_key(env: Mapping[str, str], key_env: str) -> SecretStr:
    value = env.get(key_env)
    if not value:
        raise RuntimeError(f"{key_env} is required for the OpenHands runtime")
    return SecretStr(value)


def command_secrets(env: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value for name in COMMAND_SECRET_ENV_NAMES if (value := env.get(name))
    }


def github_token(env: Mapping[str, str]) -> SecretStr:
    token_file = env.get("SENPAI_GITHUB_TOKEN_FILE")
    if token_file:
        path = Path(token_file)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RuntimeError(
                "SENPAI_GITHUB_TOKEN_FILE must be a private regular file"
            )
        try:
            value = path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)
        value = value.strip()
        if not value:
            raise RuntimeError("SENPAI_GITHUB_TOKEN_FILE is empty")
        return SecretStr(value)

    value = next(
        (env[name] for name in GITHUB_SECRET_ENV_NAMES if env.get(name)),
        None,
    )
    if value is None:
        raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required")
    return SecretStr(value.strip())


def github_repo(env: Mapping[str, str]) -> str:
    value = env.get("GH_REPO", "")
    if len(value.split("/")) != 2 or not all(value.split("/")):
        raise RuntimeError("GH_REPO must use owner/name form")
    return value


def find_role_file(explicit: str | None) -> Path:
    if not explicit:
        raise RuntimeError(
            "OpenHands role instructions are required; set SENPAI_OPENHANDS_ROLE_FILE"
        )
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"OpenHands role file does not exist: {path}")
    return path


def find_harness_file(explicit: str | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit
        else Path(__file__).resolve().parents[1]
        / "system_instructions"
        / "SENPAI-HARNESS.md"
    )
    if not path.is_file():
        raise RuntimeError(f"OpenHands harness file does not exist: {path}")
    return path


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
    if not state_dir_arg:
        raise RuntimeError(
            "OpenHands state directory is required; set SENPAI_OPENHANDS_STATE_DIR"
        )
    state_dir = Path(state_dir_arg).expanduser().resolve()
    if state_dir == workspace or state_dir.is_relative_to(workspace):
        raise RuntimeError(
            "OpenHands state directory must be outside the target workspace"
        )
    api_key_env = (
        env_value(
            args.api_key_env, env, "SENPAI_OPENHANDS_API_KEY_ENV", DEFAULT_API_KEY_ENV
        )
        or DEFAULT_API_KEY_ENV
    )
    role = env.get("SENPAI_ROLE", "")
    if role not in {"advisor", "student"}:
        raise RuntimeError("SENPAI_ROLE must be advisor or student")
    try:
        training_max_timeout_seconds = round(
            float(env.get("SENPAI_TIMEOUT_MINUTES", "30")) * 60
        )
    except ValueError as error:
        raise RuntimeError("SENPAI_TIMEOUT_MINUTES must be numeric") from error
    if training_max_timeout_seconds <= 0:
        raise RuntimeError("SENPAI_TIMEOUT_MINUTES must be positive")
    try:
        timeout_seconds = float(env.get("SENPAI_OPENHANDS_TIMEOUT_SECONDS", "3600"))
    except ValueError as error:
        raise RuntimeError(
            "SENPAI_OPENHANDS_TIMEOUT_SECONDS must be numeric"
        ) from error
    if timeout_seconds <= 0:
        raise RuntimeError("SENPAI_OPENHANDS_TIMEOUT_SECONDS must be positive")
    model = env_value(args.model, env, "SENPAI_OPENHANDS_MODEL", DEFAULT_MODEL)
    if not model:
        model = DEFAULT_MODEL
    smart_model = env.get("SENPAI_OPENHANDS_SMART_MODEL") or (
        env.get("SENPAI_OPENHANDS_MODEL") if args.child else model
    )
    if not smart_model:
        smart_model = DEFAULT_MODEL
    fast_model = env.get("SENPAI_OPENHANDS_FAST_MODEL") or (
        DEFAULT_FAST_MODEL
        if smart_model.partition("/")[0].lower() == "anthropic"
        else smart_model
    )
    return RunnerConfig(
        max_turns=args.max_turns,
        model=model,
        api_key_env=api_key_env,
        api_key=resolve_api_key(env, api_key_env),
        github_repo=github_repo(env),
        github_token=github_token(env),
        github_trusted_actor=env.get("SENPAI_GITHUB_ACTOR"),
        command_secrets=command_secrets(env),
        reasoning_effort=env_value(
            args.reasoning_effort,
            env,
            "SENPAI_OPENHANDS_REASONING_EFFORT",
            DEFAULT_REASONING_EFFORT,
        )
        or DEFAULT_REASONING_EFFORT,
        smart_model=smart_model,
        fast_model=fast_model,
        fast_reasoning_effort=env.get(
            "SENPAI_OPENHANDS_FAST_REASONING_EFFORT",
            DEFAULT_FAST_REASONING_EFFORT,
        ),
        workspace=workspace,
        state_dir=state_dir,
        conversation_id=(
            advisor_conversation_id(
                state_dir,
                env_value(
                    args.conversation_id,
                    env,
                    "SENPAI_OPENHANDS_CONVERSATION_ID",
                ),
            )
            if role == "advisor"
            else select_conversation_id(
                state_dir,
                continue_session=args.continue_session,
                explicit_id=env_value(
                    args.conversation_id,
                    env,
                    "SENPAI_OPENHANDS_CONVERSATION_ID",
                ),
            )
        ),
        continue_session=args.continue_session,
        role=role,
        enable_browser=args.enable_browser,
        agent_name=env_value(args.agent, env, "SENPAI_OPENHANDS_AGENT"),
        harness_file=find_harness_file(
            env_value(
                args.harness_file,
                env,
                "SENPAI_OPENHANDS_HARNESS_FILE",
            )
        ),
        role_file=find_role_file(
            env_value(args.role_file, env, "SENPAI_OPENHANDS_ROLE_FILE"),
        ),
        plugin_dir=resolve_plugin_dir(
            env_value(args.plugin_dir, env, "SENPAI_PLUGIN"),
        ),
        training_max_timeout_seconds=training_max_timeout_seconds,
        timeout_seconds=timeout_seconds,
        child=args.child,
    )


def find_named_agent(
    name: str,
    definitions: Sequence[AgentDefinition],
) -> AgentDefinition:
    for definition in definitions:
        if definition.name == name:
            return definition
    raise RuntimeError(f"OpenHands agent not found: {name}")


def with_role_and_project_context(
    agent: Agent,
    harness_instructions: str,
    role_instructions: str,
) -> Agent:
    context = agent.agent_context or AgentContext()
    role_suffix = compose_system_instructions(
        harness_instructions,
        role_instructions,
    )
    system_suffix = (
        f"{context.system_message_suffix}\n\n{role_suffix}"
        if context.system_message_suffix
        else role_suffix
    )
    return agent.model_copy(
        update={
            "agent_context": context.model_copy(
                update={
                    "system_message_suffix": system_suffix,
                    "current_datetime": None,
                    "load_project_skills": True,
                }
            )
        }
    )


def build_main_agent_context(
    harness_instructions: str,
    role_instructions: str,
) -> AgentContext:
    return AgentContext(
        system_message_suffix=compose_system_instructions(
            harness_instructions,
            role_instructions,
        ),
        current_datetime=None,
        load_public_skills=False,
        load_user_skills=True,
        load_project_skills=True,
    )


def prompt_cache_configuration(model: str) -> dict[str, object]:
    provider, _, model_name = model.lower().partition("/")
    if provider == "anthropic" and "prompt_cache_ttl" in LLM.model_fields:
        return {"prompt_cache_ttl": "1h"}
    if provider == "openai":
        if model_name.startswith("gpt-5.6"):
            return {
                "prompt_cache_retention": None,
                "responses_prompt_cache_breakpoint": True,
                "litellm_extra_body": {
                    "prompt_cache_options": {
                        "mode": "explicit",
                        "ttl": "30m",
                    }
                },
            }
        return {"prompt_cache_retention": "24h"}
    return {}


def conversation_prompt_cache_key(config: RunnerConfig) -> str | None:
    if config.model.split("/", 1)[0].lower() != "openai":
        return None
    agent_kind = config.agent_name or ("child" if config.child else "main")
    return f"senpai:{config.role}:{agent_kind}"


def openai_responses_configuration(model: str) -> dict[str, str | bool | int]:
    if model.split("/", 1)[0].lower() != "openai":
        return {}
    return {
        "api_mode": "responses",
        # OpenAI defines "auto" as the most detailed summarizer available.
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": True,
        "responses_use_previous_response_id": True,
        "responses_compact_threshold": 200_000,
    }


def anthropic_compaction_configuration(model: str) -> dict[str, int]:
    if model.split("/", 1)[0].lower() != "anthropic":
        return {}
    return {"anthropic_compact_threshold": 200_000}


def local_event_db_path(config: RunnerConfig) -> Path:
    return config.state_dir / f"{config.role}-events.sqlite3"


def build_main_tools(config: RunnerConfig) -> list[Tool]:
    """Keep native reasoning tools while replacing unsafe control boundaries."""

    register_senpai_tools()
    tools = [
        tool
        for tool in get_default_tools(
            enable_browser=config.enable_browser,
            enable_sub_agents=False,
        )
        if tool.name != "terminal"
    ]
    tools.extend(
        (
            Tool(name="senpai_terminal", params={"role": config.role}),
            Tool(
                name="get_prs",
                params={"state_dir": str(config.state_dir / "github")},
            ),
            Tool(name="github_transition", params={"role": config.role}),
        )
    )
    tools.append(
        Tool(
            name="delegate_agent",
            params={"event_db_path": str(local_event_db_path(config))},
        )
    )
    if config.role == "student" and not config.child:
        training_params: dict[str, str | int] = {
            "state_dir": str(config.state_dir / "training"),
            "max_timeout_seconds": config.training_max_timeout_seconds,
        }
        tools.append(Tool(name="senpai_training", params=training_params))
    return tools


def delegation_config(config: RunnerConfig) -> DelegationConfig:
    return DelegationConfig(
        python_executable=Path(sys.executable),
        workspace=config.workspace,
        state_dir=config.state_dir,
        smart_model=config.smart_model,
        fast_model=config.fast_model,
        api_key_env=config.api_key_env,
        api_key=config.api_key.get_secret_value(),
        github_repo=config.github_repo,
        github_token=config.github_token.get_secret_value(),
        github_trusted_actor=config.github_trusted_actor,
        smart_reasoning_effort=config.reasoning_effort,
        fast_reasoning_effort=config.fast_reasoning_effort,
        role_file=config.role_file,
        harness_file=config.harness_file,
        plugin_dir=config.plugin_dir,
        enable_browser=config.enable_browser,
        command_secrets=config.command_secrets,
        role=config.role,
        background_allowed=not config.child,
    )


@contextmanager
def graceful_interrupts(conversation: object) -> Iterator[None]:
    def interrupt(signum: int, _frame: object) -> None:
        print(f"OPENHANDS_INTERRUPT signal={signum}", file=sys.stderr, flush=True)
        conversation.interrupt()
        raise SystemExit(128 + signum)

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


@contextmanager
def turn_deadline(
    conversation: object,
    timeout_seconds: float,
) -> Iterator[None]:
    def interrupt() -> None:
        print(
            f"OPENHANDS_TIMEOUT seconds={timeout_seconds:g}",
            file=sys.stderr,
            flush=True,
        )
        conversation.interrupt()

    timer = threading.Timer(timeout_seconds, interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()


def event_summary(event: object) -> dict[str, object]:
    summary: dict[str, object] = {"event": event.__class__.__name__}
    for attr in ("source", "tool_name", "action", "status"):
        value = getattr(event, attr, None)
        if value is not None:
            summary[attr] = _bounded_event_text(value)
    thought = getattr(event, "thought", None)
    if thought:
        summary["thought"] = _bounded_event_text(thought)

    message = getattr(event, "llm_message", None)
    if getattr(event, "source", None) == "agent" and message is not None:
        text_parts = [
            getattr(part, "text", "")
            for part in getattr(message, "content", [])
            if getattr(part, "text", "")
        ]
        text = "\n".join(text_parts).strip()
        if text:
            summary["text"] = _bounded_event_text(text)
    return summary


def _bounded_event_text(value: object) -> str:
    text = str(value)
    encoded = text.encode()
    if len(encoded) <= EVENT_TEXT_LIMIT:
        return text
    return encoded[-EVENT_TEXT_LIMIT:].decode(errors="ignore")


def print_event(event: object) -> None:
    print(
        "OPENHANDS_EVENT " + json.dumps(event_summary(event), sort_keys=True),
        flush=True,
    )


def final_agent_result(conversation: object) -> str:
    for event in reversed(conversation.state.view.events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            text = "".join(
                content.text
                for content in event.to_llm_message().content
                if isinstance(content, TextContent)
            ).strip()
            if text:
                return text
        if isinstance(event, ActionEvent):
            message = getattr(event.action, "message", None)
            if isinstance(message, str) and message.strip():
                return message.strip()
    raise RuntimeError("child finished without a model-visible result")


def run_openhands(prompt: str, config: RunnerConfig) -> int:
    harness_instructions = read_role_instructions(config.harness_file)
    role_instructions = read_role_instructions(config.role_file)
    register_senpai_tools()
    file_agents = discover_agents(config.workspace)
    register_file_agents(config.workspace)
    available_agents = [definition.name for definition in file_agents]
    os.environ["SENPAI_CONVERSATION_ID"] = config.conversation_id.hex

    print(
        "OPENHANDS_RUN "
        + json.dumps(
            {
                "workspace": str(config.workspace),
                "state_dir": str(config.state_dir),
                "conversation_id": str(config.conversation_id),
                "continue": config.continue_session,
                "role": config.role,
                "model": config.model,
                "smart_model": config.smart_model,
                "fast_model": config.fast_model,
                "prompt_cache": (
                    prompt_cache_configuration(config.model)
                    or {"provider_default": True}
                ),
                "reasoning_effort": config.reasoning_effort,
                "openhands_reasoning_effort": openhands_reasoning_effort(
                    config.reasoning_effort, config.model
                ),
                "agent": config.agent_name,
                "enable_browser": config.enable_browser,
                "role_file": str(config.role_file) if config.role_file else None,
                "plugin_dir": str(config.plugin_dir),
                "available_agents": available_agents,
                "weave_project": WEAVE_PROJECT,
                "child": config.child,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    configure_github_credentials(
        config.github_repo,
        config.github_token,
        trusted_actor=config.github_trusted_actor,
    )
    configure_delegation(delegation_config(config))
    for name in (
        *GITHUB_SECRET_ENV_NAMES,
        "SENPAI_GITHUB_TOKEN_FILE",
    ):
        os.environ.pop(name, None)
    conversation = None
    try:
        llm = LLM(
            model=config.model,
            api_key=config.api_key,
            reasoning_effort=openhands_reasoning_effort(
                config.reasoning_effort, config.model
            ),
            usage_id="senpai",
            **prompt_cache_configuration(config.model),
            **openai_responses_configuration(config.model),
            **anthropic_compaction_configuration(config.model),
        )
        if config.agent_name:
            definition = find_named_agent(config.agent_name, file_agents)
            agent = with_role_and_project_context(
                agent_definition_to_factory(
                    definition,
                    work_dir=config.workspace,
                )(llm),
                harness_instructions,
                role_instructions,
            )
            agent = agent.model_copy(
                update={"tool_concurrency_limit": MAX_PARALLEL_AGENTS}
            )
            if (
                llm.responses_use_previous_response_id
                or llm.uses_anthropic_compaction()
            ):
                agent = agent.model_copy(update={"condenser": None})
        else:
            condenser = (
                None
                if (
                    llm.responses_use_previous_response_id
                    or llm.uses_anthropic_compaction()
                )
                else get_default_condenser(
                    llm.model_copy(update={"usage_id": "senpai-condenser"})
                )
            )
            agent = Agent(
                llm=llm,
                tools=build_main_tools(config),
                agent_context=build_main_agent_context(
                    harness_instructions,
                    role_instructions,
                ),
                system_prompt_kwargs={"cli_mode": True},
                condenser=condenser,
                tool_concurrency_limit=MAX_PARALLEL_AGENTS,
            )
        conversation = Conversation(
            agent=agent,
            workspace=config.workspace,
            plugins=[PluginSource(source=str(config.plugin_dir))],
            persistence_dir=config.state_dir,
            conversation_id=config.conversation_id,
            callbacks=[] if config.child else [print_event],
            max_iteration_per_run=config.max_turns,
            visualizer=None,
            secrets=dict(config.command_secrets),
            tags={"runtime": "senpai-openhands"},
            delete_on_close=config.child,
            prompt_cache_key=conversation_prompt_cache_key(config),
        )
        try:
            # send_message performs OpenHands' lazy tool initialization.
            conversation.send_message(prompt)
        finally:
            clear_github_credentials()
            configure_delegation(None)
        with (
            graceful_interrupts(conversation),
            turn_deadline(
                conversation,
                config.timeout_seconds,
            ),
        ):
            if not config.child:
                with (
                    AdvisorEventStore(local_event_db_path(config)) as event_store,
                    AdvisorEventPump(
                        event_store,
                        conversation,
                        parent_conversation_id=(
                            str(config.conversation_id)
                            if config.role == "student"
                            else None
                        ),
                    ),
                ):
                    conversation.run()
            else:
                conversation.run()
        status = conversation.state.execution_status
        child_result = (
            final_agent_result(conversation)
            if config.child and status == ConversationExecutionStatus.FINISHED
            else None
        )
    finally:
        clear_github_credentials()
        configure_delegation(None)
        if conversation is not None:
            conversation.close()

    print(
        "OPENHANDS_RESULT "
        + json.dumps(
            {
                "conversation_id": str(conversation.id),
                "status": status.value,
                **({"result": child_result} if config.child else {}),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == ConversationExecutionStatus.FINISHED else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_runner_args(argv)
        prompt = sys.stdin.read()
        if not prompt:
            raise RuntimeError("OpenHands runner requires a prompt on stdin")
        config = resolve_config(args)
        os.environ.pop(config.api_key_env, None)
        return run_openhands(prompt, config)
    finally:
        finish_weave_monitoring()


if __name__ == "__main__":
    raise SystemExit(main())
