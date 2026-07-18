import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition
from openhands.tools.browser_use import BrowserToolSet

import senpai_agent.openhands_runner as runner
from senpai_agent.openhands_runner import (
    RunnerConfig,
    build_main_agent_context,
    default_subagent_tools,
    find_role_file,
    load_agent_definition,
    openhands_reasoning_effort,
    parse_runner_args,
    read_role_instructions,
    resolve_plugin_dir,
    run_openhands,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "senpai"
RESEARCHER_AGENT = REPO_ROOT / ".claude" / "agents" / "researcher-agent.md"


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_simple_parsing_accepts_extended_reasoning_effort(effort):
    args = parse_runner_args(["--max-turns", "1", "--reasoning-effort", effort])

    assert args.reasoning_effort == effort
    assert openhands_reasoning_effort(args.reasoning_effort) == "xhigh"


def test_browser_is_enabled_by_default_and_can_be_disabled():
    default_args = parse_runner_args(["--max-turns", "1"])
    disabled_args = parse_runner_args(["--max-turns", "1", "--no-browser"])

    assert default_args.enable_browser is True
    assert BrowserToolSet.name in default_subagent_tools(enable_browser=True)
    assert disabled_args.enable_browser is False


def test_explicit_role_file_cannot_be_replaced_by_target_instructions(tmp_path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("target instructions", encoding="utf-8")
    role_file = tmp_path / "CLAUDE.md"
    role_file.write_text("student role", encoding="utf-8")

    selected = find_role_file(workspace, str(role_file))

    assert selected == role_file
    assert read_role_instructions(selected) == "student role"


def test_explicit_role_file_must_exist(tmp_path):
    with pytest.raises(RuntimeError, match="role file does not exist"):
        find_role_file(tmp_path, str(tmp_path / "missing.md"))


def test_role_is_system_context_loaded_before_project_skills():
    context = build_main_agent_context([], "advisor role")

    assert context.system_message_suffix == "advisor role"
    assert context.load_user_skills is True
    assert context.load_project_skills is True


def test_role_and_plugin_are_present_before_first_user_message(tmp_path, monkeypatch):
    role_file = tmp_path / "CLAUDE.md"
    role_file.write_text("advisor role", encoding="utf-8")
    captured = {}

    class FakeConversation:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.plugins = kwargs["plugins"]
            self.id = kwargs["conversation_id"]
            self.state = SimpleNamespace(
                execution_status=SimpleNamespace(value="finished")
            )

        def send_message(self, prompt):
            captured["prompt"] = prompt
            captured["role"] = self.agent.agent_context.system_message_suffix
            captured["plugin"] = self.plugins[0].source

        def run(self):
            pass

    monkeypatch.setattr(runner, "Conversation", FakeConversation)
    monkeypatch.setattr(runner, "load_skills", lambda _: [])
    monkeypatch.setattr(runner, "register_subagents", lambda *args, **kwargs: [])
    config = RunnerConfig(
        max_turns=1,
        model="anthropic/claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="test-key",
        reasoning_effort="xhigh",
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        conversation_id=uuid.uuid4(),
        continue_session=False,
        enable_browser=False,
        agent_name=None,
        role_file=role_file,
        plugin_dir=PLUGIN_DIR,
        skill_dirs=(),
        agent_dirs=(),
    )

    assert run_openhands("first task", config) == 0
    assert captured == {
        "prompt": "first task",
        "role": "advisor role",
        "plugin": str(PLUGIN_DIR),
    }


def test_openhands_loads_the_native_senpai_plugin():
    assert resolve_plugin_dir(str(PLUGIN_DIR)) == PLUGIN_DIR
    assert (PLUGIN_DIR / ".plugin" / "plugin.json").is_file()
    assert not (PLUGIN_DIR / ".claude-plugin").exists()

    plugin = Plugin.load(PLUGIN_DIR)
    skill_names = {skill.name for skill in plugin.skills}

    assert plugin.manifest.name == "senpai"
    assert "senpai-gh" in skill_names
    assert "survey-prs" in skill_names
    assert "exa" in plugin.mcp_config["mcpServers"]


def test_researcher_agent_has_its_own_exa_server():
    source = AgentDefinition.load(RESEARCHER_AGENT)
    definition = load_agent_definition(RESEARCHER_AGENT, {}, enable_browser=False)

    assert source.mcp_servers == definition.mcp_servers
    assert definition.mcp_servers["exa"]["url"].endswith(
        "tools=web_search_advanced_exa"
    )
    assert definition.mcp_servers["exa"]["headers"]["x-api-key"] == "${EXA_API_KEY}"
