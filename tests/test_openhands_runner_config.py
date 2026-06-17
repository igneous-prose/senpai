import argparse
import uuid
from pathlib import Path

import pytest

from senpai_agent.openhands_runner import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    candidate_agent_dirs,
    candidate_skill_dirs,
    default_subagent_tools,
    find_role_file,
    normalize_skill_names,
    normalize_tool_names,
    resolve_api_key,
    resolve_config,
    select_conversation_id,
)


def test_resolve_api_key_uses_requested_env_var():
    assert resolve_api_key({"ANTHROPIC_API_KEY2": "secret"}, "ANTHROPIC_API_KEY2") == "secret"


def test_resolve_api_key_fails_clearly_when_missing():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY2 is required"):
        resolve_api_key({}, "ANTHROPIC_API_KEY2")


def test_find_role_file_walks_workspace_parents(tmp_path):
    repo = tmp_path / "senpai"
    workspace = repo / "target" / "problem"
    workspace.mkdir(parents=True)
    role_file = repo / "CLAUDE.md"
    role_file.write_text("role instructions", encoding="utf-8")

    assert find_role_file(workspace) == role_file


def test_find_role_file_respects_explicit_path(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    role_file = tmp_path / "role.md"
    role_file.write_text("role instructions", encoding="utf-8")

    assert find_role_file(workspace, str(role_file)) == role_file


def test_conversation_id_continuation_uses_marker(tmp_path):
    first = select_conversation_id(tmp_path, continue_session=True)
    second = select_conversation_id(tmp_path, continue_session=True)

    assert first == second
    assert isinstance(first, uuid.UUID)


def test_fresh_conversation_id_does_not_overwrite_continuation_marker(tmp_path):
    continued = select_conversation_id(tmp_path, continue_session=True)
    fresh = select_conversation_id(tmp_path, continue_session=False)
    resumed = select_conversation_id(tmp_path, continue_session=True)

    assert fresh != continued
    assert resumed == continued


def test_explicit_conversation_id_is_used_and_persisted_for_continue(tmp_path):
    known = uuid.uuid4()
    selected = select_conversation_id(
        tmp_path,
        continue_session=True,
        explicit_id=str(known),
    )

    assert selected == known
    assert select_conversation_id(tmp_path, continue_session=True) == known


def test_normalize_skill_names_removes_claude_plugin_namespace():
    assert normalize_skill_names(["senpai:survey-prs", "wandb-primary"]) == [
        "survey-prs",
        "wandb-primary",
    ]


def test_openhands_subagent_tool_names_match_registry():
    assert default_subagent_tools(enable_browser=True) == [
        "terminal",
        "file_editor",
        "task_tracker",
        "browser_tool_set",
    ]
    assert normalize_tool_names(
        ["TerminalTool", "FileEditorTool", "TaskTrackerTool", "BrowserToolSet"],
        enable_browser=True,
    ) == ["terminal", "file_editor", "task_tracker", "browser_tool_set"]


def test_candidate_dirs_include_home_claude_and_senpai_plugin(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "repo" / "target"
    plugin = tmp_path / "repo" / "plugins" / "senpai"
    for path in [
        home / ".claude" / "skills",
        home / ".claude" / "agents",
        tmp_path / "repo" / ".claude" / "skills",
        tmp_path / "repo" / ".claude" / "agents",
        plugin / "skills",
        workspace,
    ]:
        path.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)

    env = {"SENPAI_PLUGIN": str(plugin)}
    assert home / ".claude" / "skills" in candidate_skill_dirs(workspace, env)
    assert plugin / "skills" in candidate_skill_dirs(workspace, env)
    assert tmp_path / "repo" / ".claude" / "skills" in candidate_skill_dirs(workspace, env)
    assert home / ".claude" / "agents" in candidate_agent_dirs(workspace, env)
    assert tmp_path / "repo" / ".claude" / "agents" in candidate_agent_dirs(workspace, env)


def test_resolve_config_collects_core_runtime_settings(tmp_path, monkeypatch):
    workspace = tmp_path / "repo" / "target"
    workspace.mkdir(parents=True)
    role_file = tmp_path / "repo" / "CLAUDE.md"
    role_file.write_text("role", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    args = argparse.Namespace(
        max_turns=11,
        model=DEFAULT_MODEL,
        api_key_env="ANTHROPIC_API_KEY2",
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        workspace=str(workspace),
        state_dir=None,
        conversation_id=None,
        continue_session=True,
        enable_browser=True,
        agent=None,
        role_file=None,
    )

    config = resolve_config(args, {"ANTHROPIC_API_KEY2": "secret"})

    assert config.max_turns == 11
    assert config.model == DEFAULT_MODEL
    assert config.reasoning_effort == DEFAULT_REASONING_EFFORT
    assert config.workspace == workspace
    assert config.enable_browser is True
    assert config.agent_name is None
    assert config.role_file == role_file
    assert isinstance(config.conversation_id, uuid.UUID)
