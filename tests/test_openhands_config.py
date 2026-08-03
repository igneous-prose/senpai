import os
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    build_main_agent_context,
    find_role_file,
    parse_runner_args,
    read_role_instructions,
    resolve_config,
    sanitized_agent_definitions,
    sanitized_project_skills,
)
from openhands_support import runtime_env
from test_agent_markdown import HTML_HEADER, PLAIN_HEADER

ROOT = Path(__file__).resolve().parents[1]


def test_browser_is_enabled_by_default_and_can_be_disabled():
    default_args = parse_runner_args(["--max-turns", "1"])
    disabled_args = parse_runner_args(["--max-turns", "1", "--no-browser"])

    assert default_args.enable_browser is True
    assert disabled_args.enable_browser is False


def test_explicit_role_file_is_loaded(tmp_path: Path):
    role_file = tmp_path / "SENPAI-STUDENT.md"
    role_file.write_text(HTML_HEADER + "student role", encoding="utf-8")

    selected = find_role_file(str(role_file))

    assert selected == role_file
    assert read_role_instructions(selected) == "student role"


@pytest.mark.parametrize("explicit", [None, "missing.md"])
def test_role_file_must_be_explicit_and_exist(tmp_path: Path, explicit: str | None):
    path = None if explicit is None else str(tmp_path / explicit)

    with pytest.raises(RuntimeError, match="role instructions|required|does not exist"):
        find_role_file(path)


def test_main_agent_context_places_harness_and_role_before_project_skills():
    context = build_main_agent_context("harness instructions", "advisor role")

    assert context.system_message_suffix == (
        "# Senpai harness\n\nharness instructions\n\n"
        "# Senpai role\n\nadvisor role\n"
    )
    assert context.current_datetime is None
    assert context.load_user_skills is True
    assert context.load_project_skills is False


def test_student_charter_requires_typed_tools_for_every_training_operation():
    instructions = (ROOT / "system_instructions" / "SENPAI-STUDENT.md").read_text()

    assert "must use `run_training`" in instructions
    assert "Never launch training through the terminal" in instructions
    assert "`monitor_training`" in instructions
    assert "`get_training_status`" in instructions
    assert "`cancel_training`" in instructions


def test_project_instructions_and_file_agents_are_sanitized_without_mutation(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    agents = workspace / ".agents" / "agents"
    agents.mkdir(parents=True)
    instructions = workspace / "AGENTS.md"
    definition = agents / "review.md"
    instructions.write_text(HTML_HEADER + "# Project rules\n", encoding="utf-8")
    definition.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\n"
        + PLAIN_HEADER
        + "Review carefully.\n",
        encoding="utf-8",
    )

    skills = sanitized_project_skills(workspace)
    definitions = sanitized_agent_definitions(workspace)

    assert "SPDX-" not in next(skill.content for skill in skills if skill.name == "agents")
    assert "SPDX-" not in next(item.system_prompt for item in definitions if item.name == "review")
    assert instructions.read_text(encoding="utf-8").startswith("<!--\nSPDX-")
    assert "# SPDX-" in definition.read_text(encoding="utf-8")


def test_resolved_config_separates_runtime_credentials_from_command_secrets(
    tmp_path: Path,
):
    env = runtime_env(tmp_path)
    env.update(
        {
            "GH_TOKEN": "secondary-github-key",
            "WANDB_API_KEY": "wandb-key",
            "EXA_API_KEY": "exa-key",
            "SENPAI_TIMEOUT_MINUTES": "0.5",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.api_key.get_secret_value() == "anthropic-key"
    assert config.github_token.get_secret_value() == "github-key"
    assert config.command_secrets == {
        "WANDB_API_KEY": "wandb-key",
        "EXA_API_KEY": "exa-key",
    }
    assert "ANTHROPIC_API_KEY" not in config.command_secrets
    assert config.training_max_timeout_seconds == 30


def test_fast_model_defaults_to_the_selected_non_anthropic_provider(tmp_path: Path):
    env = runtime_env(tmp_path)
    env.update(
        {
            "OPENAI_API_KEY": "openai-key",
            "SENPAI_OPENHANDS_API_KEY_ENV": "OPENAI_API_KEY",
            "SENPAI_OPENHANDS_MODEL": "openai/gpt-5.6",
        }
    )

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.smart_model == "openai/gpt-5.6"
    assert config.fast_model == "openai/gpt-5.6"


def test_config_consumes_a_private_one_use_github_token_file(tmp_path: Path):
    env = runtime_env(tmp_path)
    token_file = tmp_path / "github-token"
    token_file.write_text("one-use-token", encoding="utf-8")
    token_file.chmod(0o600)
    env.pop("GITHUB_TOKEN")
    env["SENPAI_GITHUB_TOKEN_FILE"] = str(token_file)

    config = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert config.github_token.get_secret_value() == "one-use-token"
    assert not token_file.exists()


def test_github_token_rejects_a_non_private_file(tmp_path: Path):
    token_file = tmp_path / "github-token"
    token_file.write_text("exposed-token", encoding="utf-8")
    token_file.chmod(0o644)

    with pytest.raises(RuntimeError, match="private regular file"):
        runner.github_token({"SENPAI_GITHUB_TOKEN_FILE": str(token_file)})

    assert token_file.exists()


def test_github_token_can_be_consumed_from_an_inherited_pipe():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"pipe-token")
    finally:
        os.close(write_fd)

    assert runner.github_token({"SENPAI_GITHUB_TOKEN_FD": str(read_fd)}) == SecretStr(
        "pipe-token"
    )


def test_github_token_ignores_blank_ambient_values():
    assert runner.github_token(
        {"GITHUB_TOKEN": " \n", "GH_TOKEN": "fallback-token"}
    ) == SecretStr("fallback-token")

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN or GH_TOKEN is required"):
        runner.github_token({"GITHUB_TOKEN": " \n"})


def test_child_config_requires_no_github_credential(tmp_path: Path):
    env = runtime_env(tmp_path, role="student")
    env.pop("GITHUB_TOKEN")

    config = resolve_config(
        parse_runner_args(["--max-turns", "1", "--child", "--agent", "explore"]),
        env,
    )

    assert config.child is True
    assert config.github_token is None


def test_advisor_config_reuses_its_durable_conversation_id(tmp_path: Path):
    env = runtime_env(tmp_path)

    first = resolve_config(parse_runner_args(["--max-turns", "1"]), env)
    second = resolve_config(parse_runner_args(["--max-turns", "1"]), env)

    assert first.conversation_id == second.conversation_id


@pytest.mark.parametrize("state_location", [None, "inside-workspace"])
def test_state_directory_is_explicit_and_outside_the_target_checkout(
    tmp_path: Path,
    state_location: str | None,
):
    env = runtime_env(tmp_path)
    if state_location is None:
        env.pop("SENPAI_OPENHANDS_STATE_DIR")
        message = "state directory is required"
    else:
        workspace = Path(env["SENPAI_OPENHANDS_WORKSPACE"])
        env["SENPAI_OPENHANDS_STATE_DIR"] = str(workspace / ".senpai" / "state")
        message = "outside the target workspace"

    with pytest.raises(RuntimeError, match=message):
        resolve_config(parse_runner_args(["--max-turns", "1"]), env)
