import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest
from openhands.sdk.llm import Message, TextContent

from senpai_agent.delegation import (
    DelegationConfig,
    DelegationRequest,
    OpenHandsChildProcess,
    render_child_prompt,
    run_child_process,
)


def delegation_request(
    *,
    parent_context: tuple[Message, ...] | None = None,
    agent: str = "explore",
    model: str = "fast",
    search_mode: str | None = None,
) -> DelegationRequest:
    return DelegationRequest(
        task_id=str(uuid.uuid4()),
        parent_conversation_id=str(uuid.uuid4()),
        parent_context=parent_context
        if parent_context is not None
        else (
            Message(
                role="user",
                content=[
                    TextContent(text="Inspect PR #17."),
                    TextContent(text="Progressively disclosed skill body."),
                ],
            ),
            Message(
                role="assistant",
                content=[TextContent(text="I will compare the evidence.")],
            ),
        ),
        agent=agent,
        model=model,
        search_mode=search_mode,
    )


def delegation_config(tmp_path: Path, **updates) -> DelegationConfig:
    values = {
        "python_executable": Path(sys.executable),
        "workspace": tmp_path / "target",
        "state_dir": tmp_path / "state",
        "smart_model": "anthropic/claude-opus-4-8",
        "smart_reasoning_effort": "xhigh",
        "smart_api_key_env": "ANTHROPIC_API_KEY",
        "smart_api_key": "anthropic-secret",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_reasoning_effort": "low",
        "fast_api_key_env": "ANTHROPIC_API_KEY",
        "fast_api_key": "anthropic-secret",
        "frontier_model": "openai/gpt-5.6",
        "frontier_reasoning_effort": "ultra",
        "frontier_api_key_env": "OPENAI_API_KEY",
        "frontier_api_key": "openai-secret",
        "github_repo": "acme/widgets",
        "github_trusted_actor": None,
        "role_file": tmp_path / "ADVISOR.md",
        "harness_file": tmp_path / "SENPAI-HARNESS.md",
        "plugin_dir": tmp_path / "plugin",
        "enable_browser": True,
        "command_secrets": {"EXA_API_KEY": "exa-secret"},
        "role": "advisor",
        "background_allowed": True,
    }
    values.update(updates)
    return DelegationConfig(**values)


def test_child_prompt_contains_complete_snapshot_and_task():
    request = delegation_request()

    prompt = render_child_prompt(request, "Review the result and report next steps.")

    assert "Review the result and report next steps." in prompt
    assert "Inspect PR #17." in prompt
    assert "Progressively disclosed skill body." in prompt
    assert "I will compare the evidence." in prompt
    payload = prompt.split("<parent_context_json>\n", 1)[1].split(
        "\n</parent_context_json>", 1
    )[0]
    assert [message["role"] for message in json.loads(payload)] == [
        "user",
        "assistant",
    ]


def test_context_free_search_prompt_contains_mode_and_task():
    request = delegation_request(
        parent_context=(),
        agent="search",
        model="smart",
        search_mode="research-publications",
    )

    prompt = render_child_prompt(request, "Find neural operator papers.")

    assert "Search mode: research-publications" in prompt
    assert "Find neural operator papers." in prompt
    assert "parent_context_json" not in prompt


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group deadline")
def test_optional_process_deadline_kills_an_uncooperative_group(tmp_path: Path):
    pid_file = tmp_path / "pid"
    code = (
        "import os,pathlib,signal,time;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="runtime"):
        run_child_process(
            (sys.executable, "-c", code),
            input_text="",
            env=dict(os.environ),
            timeout_seconds=1,
            terminate_grace_seconds=0.05,
        )

    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_child_command_selects_agent_model_effort_and_credential(tmp_path: Path):
    config = delegation_config(tmp_path)

    fast = OpenHandsChildProcess(
        config,
        delegation_request(agent="bash-runner"),
    )
    smart = OpenHandsChildProcess(
        config,
        delegation_request(agent="search", model="smart", search_mode="general-web"),
    )
    frontier = OpenHandsChildProcess(
        config,
        delegation_request(agent="general-purpose", model="frontier"),
    )

    assert "--agent" in fast.command
    assert fast.command[fast.command.index("--agent") + 1] == "bash-runner"
    assert "anthropic/claude-haiku-4-5" in fast.command
    assert fast.command[fast.command.index("--reasoning-effort") + 1] == "low"
    assert fast.command[fast.command.index("--api-key-env") + 1] == (
        "ANTHROPIC_API_KEY"
    )
    assert "anthropic/claude-opus-4-8" in smart.command
    assert smart.command[smart.command.index("--reasoning-effort") + 1] == "xhigh"
    assert "openai/gpt-5.6" in frontier.command
    assert frontier.command[frontier.command.index("--reasoning-effort") + 1] == (
        "ultra"
    )
    assert frontier.command[frontier.command.index("--api-key-env") + 1] == (
        "OPENAI_API_KEY"
    )
    assert fast.state_dir.parent == config.state_dir / "children"
    assert fast.environment["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert fast.environment["OPENAI_API_KEY"] == "openai-secret"
    assert fast.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "ANTHROPIC_API_KEY"
    assert frontier.environment["SENPAI_OPENHANDS_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert fast.environment["GH_REPO"] == "acme/widgets"
    assert "GITHUB_TOKEN" not in fast.environment
    assert "GH_TOKEN" not in fast.environment
    assert fast.environment["EXA_API_KEY"] == "exa-secret"
    assert fast.environment["SENPAI_OPENHANDS_SMART_MODEL"] == config.smart_model
    assert fast.environment["SENPAI_OPENHANDS_SMART_API_KEY_ENV"] == (
        config.smart_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_FAST_MODEL"] == config.fast_model
    assert fast.environment["SENPAI_OPENHANDS_FAST_API_KEY_ENV"] == (
        config.fast_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_MODEL"] == (
        config.frontier_model
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_API_KEY_ENV"] == (
        config.frontier_api_key_env
    )
    assert fast.environment["SENPAI_OPENHANDS_SMART_REASONING_EFFORT"] == (
        config.smart_reasoning_effort
    )
    assert fast.environment["SENPAI_OPENHANDS_FAST_REASONING_EFFORT"] == (
        config.fast_reasoning_effort
    )
    assert fast.environment["SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT"] == (
        config.frontier_reasoning_effort
    )


def test_child_environment_replaces_ambient_model_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "unconfigured-model-key")

    environment = OpenHandsChildProcess(
        delegation_config(tmp_path),
        delegation_request(model="frontier", agent="general-purpose"),
    ).environment

    assert environment["ANTHROPIC_API_KEY"] == "anthropic-secret"
    assert environment["OPENAI_API_KEY"] == "openai-secret"
    assert "GEMINI_API_KEY" not in environment


def test_child_process_never_receives_the_github_write_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = delegation_request()
    config = delegation_config(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")

    def fake_run(argv, *, input_text, env, timeout_seconds, **kwargs):
        assert "GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        assert "SENPAI_GITHUB_TOKEN_FILE" not in env
        assert "ambient-write-token" not in repr(env)
        assert not list(config.state_dir.rglob(".github-token-*"))
        return 'OPENHANDS_RESULT {"status":"finished","result":"done"}'

    monkeypatch.setattr("senpai_agent.delegation.run_child_process", fake_run)

    assert OpenHandsChildProcess(config, request).run("inspect", None) == "done"


def test_child_result_parser_uses_only_terminal_result_record():
    output = (
        'OPENHANDS_EVENT {"text":"intermediate"}\n'
        'OPENHANDS_RESULT {"status":"finished","result":"final report"}'
    )

    assert OpenHandsChildProcess.parse_result(output) == "final report"

    with pytest.raises(RuntimeError, match="terminal result"):
        OpenHandsChildProcess.parse_result('OPENHANDS_EVENT {"text":"not enough"}')
