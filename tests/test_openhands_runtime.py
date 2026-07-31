import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import uuid
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm import get_optional_params
from openhands.sdk import Agent, LLM, LocalConversation, load_project_skills
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition, agent_definition_to_factory
from openhands.tools.preset.default import register_default_tools
from pydantic import SecretStr

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    EVENT_TEXT_LIMIT,
    RunnerConfig,
    anthropic_compaction_configuration,
    build_main_agent_context,
    build_main_tools,
    conversation_prompt_cache_key,
    event_summary,
    find_role_file,
    graceful_interrupts,
    main,
    openai_responses_configuration,
    openhands_reasoning_effort,
    parse_runner_args,
    prompt_cache_configuration,
    read_role_instructions,
    resolve_config,
    resolve_plugin_dir,
    run_openhands,
)
from senpai_agent.tools import register_senpai_tools

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "senpai"
BASH_RUNNER_AGENT = REPO_ROOT / ".agents" / "agents" / "bash-runner.md"
EXPLORE_AGENT = REPO_ROOT / ".agents" / "agents" / "explore.md"
SEARCH_AGENT = REPO_ROOT / ".agents" / "agents" / "search.md"


def test_openhands_fork_revision_is_consistent_across_install_paths():
    paths = (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / ".github" / "workflows" / "test.yaml",
    )
    pins = []
    for path in paths:
        matches = {
            match
            for line in path.read_text(encoding="utf-8").splitlines()
            if "morganmcg1/software-agent-sdk" in line
            for match in re.findall(r"[0-9a-f]{40}", line)
        }
        assert len(matches) == 1, f"expected one OpenHands fork revision in {path}"
        pins.extend(matches)
    assert len(set(pins)) == 1


def runtime_config(tmp_path: Path, **updates) -> RunnerConfig:
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    values = {
        "max_turns": 1,
        "model": "anthropic/claude-opus-4-8",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key": SecretStr("test-key"),
        "github_repo": "acme/widgets",
        "github_token": SecretStr("github-key"),
        "github_trusted_actor": None,
        "command_secrets": {"WANDB_API_KEY": "wandb-key"},
        "reasoning_effort": "xhigh",
        "smart_model": "anthropic/claude-opus-4-8",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_reasoning_effort": "low",
        "workspace": tmp_path,
        "state_dir": tmp_path / "state",
        "conversation_id": uuid.uuid4(),
        "role": "advisor",
        "enable_browser": False,
        "agent_name": None,
        "harness_file": harness_file,
        "role_file": role_file,
        "plugin_dir": PLUGIN_DIR,
    }
    values.update(updates)
    return RunnerConfig(**values)


@pytest.mark.parametrize(
    ("effort", "model", "expected"),
    [
        ("max", "anthropic/claude-opus-4-8", "xhigh"),
        ("ultra", "openai/gpt-5.6", "max"),
        ("max", "openai/gpt-5.6-sol", "max"),
        ("max", "openai/gpt-5.4", "xhigh"),
        ("high", "openai/gpt-5.6", "high"),
    ],
)
def test_reasoning_effort_uses_highest_supported_value(effort, model, expected):
    args = parse_runner_args(["--max-turns", "1", "--reasoning-effort", effort])

    assert args.reasoning_effort == effort
    assert openhands_reasoning_effort(args.reasoning_effort, model) == expected


def test_default_reasoning_effort_is_xhigh():
    assert runner.DEFAULT_REASONING_EFFORT == "xhigh"


def test_anthropic_xhigh_enables_adaptive_thinking():
    options = get_optional_params(
        model="claude-opus-4-8",
        custom_llm_provider="anthropic",
        reasoning_effort="xhigh",
    )

    assert options["thinking"] == {"type": "adaptive"}
    assert options["output_config"] == {"effort": "xhigh"}


def test_browser_is_enabled_by_default_and_can_be_disabled():
    default_args = parse_runner_args(["--max-turns", "1"])
    disabled_args = parse_runner_args(["--max-turns", "1", "--no-browser"])

    assert default_args.enable_browser is True
    assert disabled_args.enable_browser is False


def test_child_mode_keeps_one_foreground_delegation_path(tmp_path):
    args = parse_runner_args(["--max-turns", "1", "--child"])
    config = runtime_config(tmp_path, child=True)

    assert args.child is True
    names = {tool.name for tool in build_main_tools(config)}
    assert "task_tool_set" not in names
    assert "delegate_agent" in names
    assert "senpai_training" not in names


def test_explicit_role_file_cannot_be_replaced_by_target_instructions(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("target instructions", encoding="utf-8")
    role_file = tmp_path / "CLAUDE.md"
    role_file.write_text("student role", encoding="utf-8")

    selected = find_role_file(str(role_file))

    assert selected == role_file
    assert read_role_instructions(selected) == "student role"


def test_explicit_role_file_must_exist(tmp_path):
    with pytest.raises(RuntimeError, match="role file does not exist"):
        find_role_file(str(tmp_path / "missing.md"))


def test_target_claude_file_is_never_used_as_the_senpai_role(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("target instructions", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SENPAI_OPENHANDS_ROLE_FILE"):
        find_role_file(None)


def test_openhands_loads_target_instructions_and_keeps_skills_progressive(tmp_path):
    (tmp_path / "AGENTS.md").write_text("target agents", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("target compatibility", encoding="utf-8")
    skill_dir = tmp_path / ".agents" / "skills" / "target-analysis"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: target-analysis\n"
        "description: Analyze the target when requested.\n"
        "---\n\n"
        "Large progressively disclosed instructions.\n",
        encoding="utf-8",
    )

    skills = {skill.name: skill for skill in load_project_skills(tmp_path)}

    assert skills["agents"].content == "target agents"
    assert skills["claude"].content == "target compatibility"
    assert skills["target-analysis"].is_agentskills_format is True


def test_role_is_system_context_loaded_before_project_skills():
    context = build_main_agent_context("harness instructions", "advisor role")

    assert context.system_message_suffix == (
        "# Senpai harness\n\nharness instructions\n\n# Senpai role\n\nadvisor role\n"
    )
    assert context.current_datetime is None
    assert context.load_user_skills is True
    assert context.load_project_skills is True


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "anthropic/claude-opus-4-8",
            (
                {"prompt_cache_ttl": "1h"}
                if "prompt_cache_ttl" in LLM.model_fields
                else {}
            ),
        ),
        ("openai/gpt-5.4", {"prompt_cache_retention": "24h"}),
        (
            "openai/gpt-5.6",
            {
                "prompt_cache_retention": None,
                "responses_prompt_cache_breakpoint": True,
                "litellm_extra_body": {
                    "prompt_cache_options": {
                        "mode": "explicit",
                        "ttl": "30m",
                    },
                },
            },
        ),
        ("gemini/gemini-3-pro", {}),
    ],
)
def test_provider_specific_prompt_cache_configuration(model, expected):
    assert prompt_cache_configuration(model) == expected


def test_openai_uses_stored_responses_with_provider_compaction():
    configuration = openai_responses_configuration("openai/gpt-5.4")
    llm = LLM(
        model="openai/gpt-5.4",
        api_key=SecretStr("test-key"),
        reasoning_effort="xhigh",
        **configuration,
    )

    assert configuration == {
        "api_mode": "responses",
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": True,
        "responses_use_previous_response_id": True,
        "responses_compact_threshold": 200_000,
    }
    assert llm.uses_responses_api() is True

    call_kwargs = llm._prepare_responses_params(
        messages=[
            Message(role="user", content=[TextContent(text="Investigate the task")])
        ],
        tools=[],
        include=None,
        store=None,
        add_security_risk_prediction=False,
        kwargs={},
    )[3]

    assert call_kwargs["store"] is True
    assert call_kwargs["reasoning"] == {
        "effort": "xhigh",
        "summary": "auto",
        "context": "all_turns",
    }
    assert call_kwargs["context_management"] == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]
    assert "include" not in call_kwargs


def test_anthropic_uses_native_provider_compaction() -> None:
    configuration = anthropic_compaction_configuration("anthropic/claude-opus-4-8")
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
        **configuration,
    )

    assert configuration == {"anthropic_compact_threshold": 200_000}
    assert llm.uses_anthropic_compaction() is True
    assert anthropic_compaction_configuration("openai/gpt-5.6") == {}


def test_gpt56_marks_only_the_stable_system_cache_boundary():
    llm = LLM(
        model="openai/gpt-5.6",
        api_key=SecretStr("test-key"),
        **prompt_cache_configuration("openai/gpt-5.6"),
        **openai_responses_configuration("openai/gpt-5.6"),
    )
    instructions, inputs = llm.format_messages_for_responses(
        [
            Message(
                role="system",
                content=[
                    TextContent(text="stable harness and role"),
                    TextContent(text="dynamic project context"),
                ],
            ),
            Message(role="user", content=[TextContent(text="Investigate")]),
        ]
    )

    assert instructions is None
    assert inputs[0]["content"] == [
        {
            "type": "input_text",
            "text": "stable harness and role",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "input_text",
            "text": "dynamic project context",
        },
    ]
    assert llm.litellm_extra_body == {
        "prompt_cache_options": {
            "mode": "explicit",
            "ttl": "30m",
        }
    }


def test_non_openai_models_do_not_force_the_responses_api():
    assert openai_responses_configuration("anthropic/claude-opus-4-8") == {}


def test_openai_conversations_share_stable_role_and_agent_cache_keys(tmp_path):
    main = runtime_config(tmp_path, model="openai/gpt-5.6", role="student")
    child = runtime_config(
        tmp_path,
        model="openai/gpt-5.6",
        role="advisor",
        child=True,
    )
    named = runtime_config(
        tmp_path,
        model="openai/gpt-5.6",
        role="advisor",
        agent_name="explore",
    )
    anthropic = runtime_config(tmp_path, model="anthropic/claude-opus-4-8")

    assert conversation_prompt_cache_key(main) == "senpai:student:main"
    assert conversation_prompt_cache_key(child) == "senpai:advisor:child"
    assert conversation_prompt_cache_key(named) == "senpai:advisor:explore"
    assert conversation_prompt_cache_key(anthropic) is None


def test_local_conversation_accepts_the_openai_prompt_cache_key(tmp_path):
    conversation = LocalConversation(
        agent=Agent(
            llm=LLM(
                model="openai/gpt-5.6",
                api_key=SecretStr("test-key"),
            ),
            tools=[],
        ),
        workspace=tmp_path,
        visualizer=None,
        prompt_cache_key="senpai:student:main",
    )
    try:
        assert (
            conversation.get_llm_call_context().prompt_cache_key
            == "senpai:student:main"
        )
    finally:
        conversation.close()


def test_event_summary_preserves_bounded_reasoning_and_action():
    event = SimpleNamespace(
        source="agent",
        thought="r" * (EVENT_TEXT_LIMIT + 10),
        action={"command": "x" * (EVENT_TEXT_LIMIT + 10)},
        status="running",
    )

    summary = event_summary(event)

    assert summary["thought"] == "r" * EVENT_TEXT_LIMIT
    assert len(summary["action"].encode()) <= EVENT_TEXT_LIMIT
    assert summary["status"] == "running"


@pytest.mark.parametrize(
    ("role", "expected_custom"),
    [
        (
            "advisor",
            {
                "senpai_terminal",
                "get_prs",
                "github_transition",
                "delegate_agent",
            },
        ),
        (
            "student",
            {
                "senpai_terminal",
                "get_prs",
                "github_transition",
                "delegate_agent",
                "senpai_training",
            },
        ),
    ],
)
def test_main_tools_replace_terminal_and_add_role_boundaries(
    tmp_path,
    role,
    expected_custom,
):
    config = runtime_config(
        tmp_path,
        role=role,
    )

    tools = build_main_tools(config)
    by_name = {tool.name: tool for tool in tools}

    assert "terminal" not in by_name
    assert "task_tool_set" not in by_name
    assert expected_custom <= set(by_name)
    assert by_name["senpai_terminal"].params == {"role": role}
    assert by_name["delegate_agent"].params == {
        "event_db_path": str(config.state_dir / f"{role}-events.sqlite3")
    }
    if role == "student":
        assert by_name["senpai_training"].params == {
            "state_dir": str(config.state_dir / "training"),
            "max_timeout_seconds": 1800,
        }
    assert by_name["get_prs"].params == {"state_dir": str(config.state_dir / "github")}


def test_config_keeps_llm_key_secret_and_registers_only_command_secrets(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    state_dir = tmp_path / "advisor-instance" / "openhands-state"
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GITHUB_TOKEN": "github-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": "advisor",
        "GH_TOKEN": "gh-key",
        "WANDB_API_KEY": "wandb-key",
        "EXA_API_KEY": "exa-key",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(state_dir),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
        "SENPAI_TIMEOUT_MINUTES": "0.5",
    }

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.api_key.get_secret_value() == "anthropic-key"
    assert config.github_repo == "acme/widgets"
    assert config.github_token.get_secret_value() == "github-key"
    assert config.command_secrets == {
        "WANDB_API_KEY": "wandb-key",
        "EXA_API_KEY": "exa-key",
    }
    assert "ANTHROPIC_API_KEY" not in config.command_secrets
    assert config.state_dir == state_dir
    assert config.training_max_timeout_seconds == 30


def test_fast_model_defaults_to_same_provider_without_an_explicit_override(
    tmp_path,
):
    env = {
        "OPENAI_API_KEY": "openai-key",
        "GITHUB_TOKEN": "github-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": "advisor",
        "SENPAI_OPENHANDS_API_KEY_ENV": "OPENAI_API_KEY",
        "SENPAI_OPENHANDS_MODEL": "openai/gpt-5.6",
        "SENPAI_OPENHANDS_WORKSPACE": str(tmp_path),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path.parent / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(runtime_config(tmp_path).role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(runtime_config(tmp_path).harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "openai/gpt-5.6"
    assert config.fast_model == "openai/gpt-5.6"


def test_config_consumes_private_one_use_github_token_file(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    token_file = tmp_path / "github-token"
    token_file.write_text("one-use-token", encoding="utf-8")
    token_file.chmod(0o600)
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "SENPAI_GITHUB_TOKEN_FILE": str(token_file),
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": "advisor",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.github_token.get_secret_value() == "one-use-token"
    assert not token_file.exists()


def test_github_token_can_be_consumed_from_an_inherited_pipe():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"pipe-token")
    finally:
        os.close(write_fd)

    assert runner.github_token({"SENPAI_GITHUB_TOKEN_FD": str(read_fd)}) == SecretStr(
        "pipe-token"
    )


def test_delegated_agent_configuration_requires_no_github_credential(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-STUDENT.md"
    role_file.write_text("student role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": "student",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    config = resolve_config(
        parse_runner_args(["--max-turns", "1", "--child", "--agent", "explore"]),
        env,
    )

    assert config.child is True
    assert config.github_token is None


def test_advisor_reuses_one_conversation_without_continue_flag(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness")
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GITHUB_TOKEN": "github-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": "advisor",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    first = resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    second = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert first.conversation_id == second.conversation_id


def test_state_directory_must_be_explicit(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "SENPAI_ROLE": "advisor",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    with pytest.raises(RuntimeError, match="state directory is required"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


def test_state_directory_must_be_outside_target_checkout(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    env = {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "SENPAI_ROLE": "advisor",
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(workspace / ".senpai" / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }

    with pytest.raises(RuntimeError, match="outside the target workspace"):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)


@pytest.mark.parametrize(
    ("role", "logdir"),
    [
        ("advisor", "/var/lib/senpai/$RESEARCH_TAG/advisor"),
        ("student", "/var/lib/senpai"),
    ],
)
def test_entrypoint_pins_state_next_to_role_logs(role, logdir):
    entrypoint = (REPO_ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )

    assert f'LOGDIR="{logdir}"' in entrypoint
    assert 'export SENPAI_OPENHANDS_STATE_DIR="$LOGDIR/openhands_state"' in entrypoint
    assert "SENPAI_OPENHANDS_TIMEOUT_SECONDS" in entrypoint
    assert f"exec python -m senpai_agent.supervisor {role}" in entrypoint


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_entrypoint_renders_the_role_and_delegates_prompting_to_controller(role):
    entrypoint = (REPO_ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'export SENPAI_OPENHANDS_ROLE_FILE="$LOGDIR/'
        f'SENPAI-{role.upper()}.md"' in entrypoint
    )
    assert f"exec python -m senpai_agent.supervisor {role}" in entrypoint


def test_role_and_plugin_are_present_before_first_user_message(tmp_path, monkeypatch):
    captured = {}

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.plugins = kwargs["plugins"]
            self.id = kwargs["conversation_id"]
            captured["secrets"] = kwargs["secrets"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["condenser"] = agent.condenser
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, prompt):
            captured["prompt"] = prompt
            captured["role"] = self.agent.agent_context.system_message_suffix
            captured["plugin"] = self.plugins[0].source
            captured["conversation_id_env"] = runner.os.environ[
                "SENPAI_CONVERSATION_ID"
            ]

        def run(self):
            pass

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])
    config = runtime_config(tmp_path)

    assert run_openhands("first task", config) == 0
    assert captured == {
        "prompt": "first task",
        "role": (
            "# Senpai harness\n\n"
            "harness instructions\n\n"
            "# Senpai role\n\n"
            "advisor role\n"
        ),
        "plugin": str(PLUGIN_DIR),
        "secrets": {"WANDB_API_KEY": "wandb-key"},
        "delete_on_close": False,
        "condenser": None,
        "conversation_id_env": config.conversation_id.hex,
        "closed": True,
    }


def test_child_conversation_is_ephemeral_and_emits_its_terminal_report(
    tmp_path,
    monkeypatch,
    capsys,
):
    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.delete_on_close = kwargs["delete_on_close"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                view=SimpleNamespace(events=[]),
            )

        def send_message(self, _prompt):
            pass

        def run(self):
            self.state.view.events.append(
                runner.MessageEvent(
                    source="agent",
                    llm_message=Message(
                        role="assistant",
                        content=[TextContent(text="bounded child report")],
                    ),
                )
            )
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])
    monkeypatch.setattr(runner, "WEAVE_PROJECT", "wandb-applied-ai-team/senpai-v1")

    config = runtime_config(tmp_path, child=True)
    assert run_openhands("child task", config) == 0

    records = capsys.readouterr().out.splitlines()
    result_record = next(
        line
        for line in records
        if line.startswith("OPENHANDS_RESULT ")
    )
    run_record = next(line for line in records if line.startswith("OPENHANDS_RUN "))
    assert '"result": "bounded child report"' in result_record
    assert json.loads(run_record.removeprefix("OPENHANDS_RUN "))["weave_url"] == (
        "https://wandb.ai/wandb-applied-ai-team/senpai-v1/"
        f"weave/agents/conversations/{config.conversation_id}"
    )


def test_main_student_conversation_is_persisted_for_monitor_wake(
    tmp_path,
    monkeypatch,
):
    captured = {}
    child_runtime = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            captured["delete_on_close"] = kwargs["delete_on_close"]
            captured["tool_concurrency_limit"] = kwargs["agent"].tool_concurrency_limit
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            pass

        def run(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])
    monkeypatch.setattr(runner, "configure_delegation", child_runtime.append)

    assert (
        run_openhands(
            "student task",
            runtime_config(tmp_path, role="student"),
        )
        == 0
    )
    assert captured["delete_on_close"] is False
    assert captured["tool_concurrency_limit"] == 8
    assert child_runtime[0].role == "student"
    assert child_runtime[-1] is None


def test_github_tokens_never_reach_the_agent_environment(
    tmp_path,
    monkeypatch,
):
    observed = {}

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            observed["during_initialization"] = runner.os.environ.get("GITHUB_TOKEN")

        def run(self):
            observed["during_agent_run"] = runner.os.environ.get("GITHUB_TOKEN")

        def close(self):
            pass

    monkeypatch.setenv("GITHUB_TOKEN", "stale-env-secret")
    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])

    assert (
        run_openhands(
            "task",
            runtime_config(tmp_path, role="student"),
        )
        == 0
    )
    assert observed == {
        "during_initialization": None,
        "during_agent_run": None,
    }
    assert "GITHUB_TOKEN" not in runner.os.environ


@pytest.mark.parametrize(
    "status",
    [
        ConversationExecutionStatus.IDLE,
        ConversationExecutionStatus.RUNNING,
        ConversationExecutionStatus.PAUSED,
        ConversationExecutionStatus.WAITING_FOR_CONFIRMATION,
        ConversationExecutionStatus.ERROR,
        ConversationExecutionStatus.STUCK,
        ConversationExecutionStatus.DELETING,
    ],
)
def test_only_finished_conversations_succeed(status, tmp_path, monkeypatch):
    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(execution_status=status)

        def send_message(self, _prompt):
            pass

        def run(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])

    assert run_openhands("task", runtime_config(tmp_path)) == 1


def test_conversation_closes_when_execution_raises(tmp_path, monkeypatch):
    closed = []

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]

        def send_message(self, _prompt):
            pass

        def run(self):
            raise RuntimeError("execution failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])

    with pytest.raises(RuntimeError, match="execution failed"):
        run_openhands("task", runtime_config(tmp_path))

    assert closed == [True]


def test_openhands_turn_has_a_hard_runtime_deadline(tmp_path, monkeypatch):
    interrupted = threading.Event()

    class FakeConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.RUNNING
            )

        def send_message(self, _prompt):
            pass

        def run(self):
            assert interrupted.wait(1)
            self.state.execution_status = ConversationExecutionStatus.ERROR

        def interrupt(self):
            interrupted.set()

        def close(self):
            pass

    monkeypatch.setattr(runner, "LocalConversation", FakeConversation)
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])

    assert (
        run_openhands(
            "task",
            runtime_config(tmp_path, timeout_seconds=0.01),
        )
        == 1
    )
    assert interrupted.is_set()


def test_signals_interrupt_the_conversation_and_restore_handlers(monkeypatch):
    calls = []
    installed = {}
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        installed[signum] = handler
        return previous[signum]

    conversation = SimpleNamespace(interrupt=lambda: calls.append("interrupt"))
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with graceful_interrupts(conversation), pytest.raises(SystemExit) as exit_info:
        installed[signal.SIGTERM](signal.SIGTERM, None)

    assert exit_info.value.code == 128 + signal.SIGTERM
    assert "interrupt" in calls
    assert calls[-2:] == [
        (signal.SIGTERM, previous[signal.SIGTERM]),
        (signal.SIGINT, previous[signal.SIGINT]),
    ]


def test_main_flushes_weave_when_the_run_fails(monkeypatch):
    flushed = []

    def fail_run(prompt, config):
        assert "ANTHROPIC_API_KEY" not in runner.os.environ
        raise RuntimeError("run failed")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(runner.sys, "stdin", StringIO("first task"))
    monkeypatch.setattr(
        runner,
        "resolve_config",
        lambda args: SimpleNamespace(api_key_env="ANTHROPIC_API_KEY"),
    )
    monkeypatch.setattr(runner, "run_openhands", fail_run)
    monkeypatch.setattr(runner, "finish_weave_monitoring", lambda: flushed.append(True))

    with pytest.raises(RuntimeError, match="run failed"):
        main(["--max-turns", "1"])

    assert flushed == [True]


def test_main_flushes_weave_when_the_prompt_is_empty(monkeypatch):
    flushed = []
    monkeypatch.setattr(runner.sys, "stdin", StringIO())
    monkeypatch.setattr(runner, "finish_weave_monitoring", lambda: flushed.append(True))

    with pytest.raises(RuntimeError, match="requires a prompt"):
        main(["--max-turns", "1"])

    assert flushed == [True]


def test_openhands_loads_the_native_senpai_plugin():
    assert resolve_plugin_dir(str(PLUGIN_DIR)) == PLUGIN_DIR
    assert (PLUGIN_DIR / ".plugin" / "plugin.json").is_file()
    assert not (PLUGIN_DIR / ".claude-plugin").exists()

    plugin = Plugin.load(PLUGIN_DIR)
    skill_names = {skill.name for skill in plugin.skills}

    assert plugin.manifest.name == "senpai"
    assert "assign-experiment" in skill_names
    assert "poll-for-work" not in skill_names
    assert "survey-prs" not in skill_names
    assert plugin.mcp_config == {}
    assert not (PLUGIN_DIR / ".mcp.json").exists()


def test_markdown_agents_register_and_construct_through_native_openhands_loader(
    tmp_path,
):
    home = tmp_path / "home"
    workspace = tmp_path / "target"
    workspace.mkdir()
    shutil.copytree(REPO_ROOT / ".agents", home / ".agents")
    program = textwrap.dedent(
        """
        import os
        from pathlib import Path

        from openhands.sdk import LLM
        from openhands.sdk.subagent import (
            agent_definition_to_factory,
            get_registered_agent_definitions,
            register_file_agents,
        )
        from pydantic import SecretStr

        import openhands.tools
        from senpai_agent.tools import register_senpai_tools

        register_senpai_tools()
        workspace = Path(os.environ["SENPAI_TEST_WORKSPACE"])
        registered = register_file_agents(workspace)
        assert set(registered) == {
            "bash-runner",
            "general-purpose",
            "explore",
            "search",
        }
        definitions = {
            definition.name: definition
            for definition in get_registered_agent_definitions()
        }
        assert set(definitions) == set(registered)

        llm = LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
        agents = {
            name: agent_definition_to_factory(
                definition,
                work_dir=workspace,
            )(llm)
            for name, definition in definitions.items()
        }
        assert {tool.name for tool in agents["search"].tools} == {
            "terminal",
            "file_editor",
        }
        assert {tool.name for tool in agents["bash-runner"].tools} == {
            "terminal",
        }
        assert agents["search"].llm.reasoning_effort == "xhigh"
        assert agents["explore"].llm.reasoning_effort == "low"
        print("native-file-agents-ok")
        """
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "OPENHANDS_SUPPRESS_BANNER": "1",
        "SENPAI_TEST_WORKSPACE": str(workspace),
    }

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "native-file-agents-ok" in result.stdout


def test_search_agent_applies_reasoning_effort_and_progressive_skills(
    monkeypatch,
):
    import openhands.sdk.skills.skill as skill_module

    monkeypatch.setattr(
        skill_module,
        "USER_SKILLS_DIRS",
        [REPO_ROOT / ".agents" / "skills", PLUGIN_DIR / "skills"],
    )
    monkeypatch.setenv("SENPAI_ROLE", "advisor")
    register_default_tools(enable_browser=False)
    register_senpai_tools()
    definition = AgentDefinition.load(SEARCH_AGENT)
    agent = agent_definition_to_factory(definition, work_dir=REPO_ROOT)(
        LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
    )

    assert definition.model == "inherit"
    assert definition.reasoning_effort == "xhigh"
    assert agent.llm.reasoning_effort == "xhigh"
    assert definition.skills == [
        "exa-search",
        "alphaxiv-paper-lookup",
    ]
    assert set(definition.tools) == {
        "terminal",
        "file_editor",
    }
    assert {"browser_tool_set", "get_prs", "delegate_agent"}.isdisjoint(
        definition.tools
    )
    assert {skill.name for skill in agent.agent_context.skills} == set(
        definition.skills
    )
    assert all(skill.is_agentskills_format for skill in agent.agent_context.skills)
    assert all(
        skill.content not in agent.agent_context.system_message_suffix
        for skill in agent.agent_context.skills
    )
    assert "# Referenced skills" not in agent.agent_context.system_message_suffix
    assert not definition.mcp_config


@pytest.mark.parametrize(
    "role_file",
    [
        REPO_ROOT / "system_instructions" / "SENPAI-ADVISOR.md",
        REPO_ROOT / "system_instructions" / "SENPAI-STUDENT.md",
    ],
)
def test_inherited_role_context_does_not_direct_search_to_absent_tools(role_file):
    harness = (REPO_ROOT / "system_instructions" / "SENPAI-HARNESS.md").read_text(
        encoding="utf-8"
    )
    role = role_file.read_text(encoding="utf-8")
    suffix = runner.compose_system_instructions(harness, role)
    compact_suffix = " ".join(suffix.split())

    assert "do not independently execute the parent's workflow" in compact_suffix
    assert "only when its named tool is present in your schema" in compact_suffix
    assert "when `delegate_agent` is present" in compact_suffix
    assert (
        "On the main advisor" in compact_suffix
        or "On the main student" in compact_suffix
    )


def test_explore_agent_is_a_concise_low_effort_file_agent():
    definition = AgentDefinition.load(EXPLORE_AGENT)

    assert definition.name == "explore"
    assert definition.model == "inherit"
    assert definition.reasoning_effort == "low"
    assert {"terminal", "file_editor", "delegate_agent"} == set(definition.tools)
    assert "line numbers" in definition.system_prompt
    assert "Large files and conversation logs" in definition.system_prompt


def test_bash_runner_is_a_terminal_only_output_distillation_agent():
    definition = AgentDefinition.load(BASH_RUNNER_AGENT)

    assert definition.name == "bash-runner"
    assert definition.model == "inherit"
    assert definition.reasoning_effort is None
    assert set(definition.tools) == {"terminal"}
    assert "Never dump raw command output" in definition.system_prompt
    assert "Never push" in definition.system_prompt
    assert "passed, failed, skipped, and errored counts" in definition.system_prompt


def test_delegated_agents_have_no_github_credentials_or_mutation_tools():
    for path in (
        BASH_RUNNER_AGENT,
        EXPLORE_AGENT,
        SEARCH_AGENT,
        REPO_ROOT / ".agents/agents/general-purpose.md",
    ):
        definition = AgentDefinition.load(path)

        assert "get_prs" not in definition.tools
        assert "github_transition" not in definition.tools


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_entrypoint_installs_every_markdown_agent_definition(role):
    entrypoint = (REPO_ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )

    assert '"$HOME/.agents/agents/bash-runner.md"' in entrypoint
    assert '"$HOME/.agents/agents/general-purpose.md"' in entrypoint
    assert '"$HOME/.agents/agents/explore.md"' in entrypoint
    assert '"$HOME/.agents/agents/search.md"' in entrypoint


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_entrypoint_does_not_start_hivemind(role):
    entrypoint = (REPO_ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )
    active_lines = [
        line.strip()
        for line in entrypoint.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert not any("start-hivemind.sh" in line for line in active_lines)
    assert "start_hivemind" not in active_lines
