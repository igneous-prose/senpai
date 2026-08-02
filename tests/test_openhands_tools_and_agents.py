import os
import shutil
import subprocess
import sys
import textwrap

import pytest
from openhands.sdk import LLM
from openhands.sdk.plugin import Plugin
from openhands.sdk.subagent import AgentDefinition, agent_definition_to_factory
from openhands.tools.preset.default import register_default_tools
from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    build_main_tools,
    delegation_config,
    resolve_plugin_dir,
)
from senpai_agent.tools import register_senpai_tools
from openhands_support import AGENT_DIR, PLUGIN_DIR, REPO_ROOT, runtime_config


def test_child_mode_keeps_only_foreground_delegation_tools(tmp_path):
    config = runtime_config(tmp_path, child=True)
    names = {tool.name for tool in build_main_tools(config)}

    assert "task_tool_set" not in names
    assert "delegate_agent" in names
    assert "senpai_training" not in names
    assert delegation_config(config).background_allowed is False


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
def test_main_tools_replace_unsafe_defaults_with_role_scoped_boundaries(
    tmp_path,
    role: str,
    expected_custom: set[str],
):
    config = runtime_config(tmp_path, role=role)
    by_name = {tool.name: tool for tool in build_main_tools(config)}

    assert "terminal" not in by_name
    assert "task_tool_set" not in by_name
    assert expected_custom <= set(by_name)
    assert by_name["senpai_terminal"].params == {"role": role}
    assert by_name["delegate_agent"].params == {
        "event_db_path": str(config.state_dir / f"{role}-events.sqlite3")
    }
    assert by_name["get_prs"].params == {
        "state_dir": str(config.state_dir / "github")
    }
    if role == "student":
        assert by_name["senpai_training"].params == {
            "state_dir": str(config.state_dir / "training"),
            "max_timeout_seconds": 1800,
        }


def test_native_senpai_plugin_loads_its_runtime_skills():
    assert resolve_plugin_dir(str(PLUGIN_DIR)) == PLUGIN_DIR

    plugin = Plugin.load(PLUGIN_DIR)

    assert plugin.manifest.name == "senpai"
    assert {skill.name for skill in plugin.skills} == {
        "assign-experiment",
        "bootstrap-target",
        "check-human-issues",
        "close-experiment",
        "merge-winner",
        "submit-experiment-results",
    }
    assert plugin.mcp_config == {}


def test_markdown_agents_register_and_construct_with_the_native_loader(tmp_path):
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
        llm = LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
        agents = {
            name: agent_definition_to_factory(definition, work_dir=workspace)(llm)
            for name, definition in definitions.items()
        }
        assert {tool.name for tool in agents["search"].tools} == {
            "terminal",
            "file_editor",
        }
        assert {tool.name for tool in agents["bash-runner"].tools} == {"terminal"}
        assert agents["search"].llm.reasoning_effort == "xhigh"
        assert agents["explore"].llm.reasoning_effort == "low"
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


def test_search_agent_loads_its_progressive_skills_and_reasoning_override(
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
    definition = AgentDefinition.load(AGENT_DIR / "search.md")
    agent = agent_definition_to_factory(definition, work_dir=REPO_ROOT)(
        LLM(
            model="anthropic/claude-opus-4-8",
            api_key=SecretStr("test-key"),
            reasoning_effort="low",
        )
    )

    assert agent.llm.reasoning_effort == "xhigh"
    assert {skill.name for skill in agent.agent_context.skills} == {
        "exa-search",
        "alphaxiv-paper-lookup",
    }
    assert all(skill.is_agentskills_format for skill in agent.agent_context.skills)
    assert all(
        skill.content not in agent.agent_context.system_message_suffix
        for skill in agent.agent_context.skills
    )


@pytest.mark.parametrize(
    ("filename", "name", "effort", "tools", "skills"),
    [
        ("bash-runner.md", "bash-runner", None, {"terminal"}, set()),
        (
            "explore.md",
            "explore",
            "low",
            {"terminal", "file_editor", "delegate_agent"},
            set(),
        ),
        (
            "general-purpose.md",
            "general-purpose",
            None,
            {"terminal", "file_editor", "task_tracker", "delegate_agent"},
            set(),
        ),
        (
            "search.md",
            "search",
            "xhigh",
            {"terminal", "file_editor"},
            {"exa-search", "alphaxiv-paper-lookup"},
        ),
    ],
)
def test_file_agent_definitions_keep_bounded_tools_and_no_github_mutations(
    filename: str,
    name: str,
    effort: str | None,
    tools: set[str],
    skills: set[str],
):
    definition = AgentDefinition.load(AGENT_DIR / filename)

    assert definition.name == name
    assert definition.model == "inherit"
    assert definition.reasoning_effort == effort
    assert definition.permission_mode == "never_confirm"
    assert set(definition.tools) == tools
    assert set(definition.skills) == skills
    assert {"get_prs", "github_transition"}.isdisjoint(definition.tools)
