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
        "fast_model": "anthropic/claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key": "model-secret",
        "github_repo": "acme/widgets",
        "github_trusted_actor": None,
        "smart_reasoning_effort": "xhigh",
        "fast_reasoning_effort": "low",
        "role_file": tmp_path / "SENPAI-ADVISOR.md",
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


def test_child_command_selects_agent_model_and_effort(tmp_path: Path):
    config = delegation_config(tmp_path)

    fast = OpenHandsChildProcess(config, delegation_request())
    smart = OpenHandsChildProcess(
        config,
        delegation_request(agent="search", model="smart", search_mode="general-web"),
    )

    assert "--agent" in fast.command
    assert fast.command[fast.command.index("--agent") + 1] == "explore"
    assert "anthropic/claude-haiku-4-5" in fast.command
    assert fast.command[fast.command.index("--reasoning-effort") + 1] == "low"
    assert "anthropic/claude-opus-4-8" in smart.command
    assert smart.command[smart.command.index("--reasoning-effort") + 1] == "xhigh"
    assert fast.state_dir.parent == config.state_dir / "children"
    assert fast.environment["ANTHROPIC_API_KEY"] == "model-secret"
    assert fast.environment["GH_REPO"] == "acme/widgets"
    assert "GITHUB_TOKEN" not in fast.environment
    assert "GH_TOKEN" not in fast.environment
    assert fast.environment["EXA_API_KEY"] == "exa-secret"
    assert fast.environment["SENPAI_OPENHANDS_SMART_MODEL"] == config.smart_model
    assert fast.environment["SENPAI_OPENHANDS_FAST_MODEL"] == config.fast_model


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
