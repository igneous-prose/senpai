import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest
from openhands.sdk.llm import Message, TextContent

from senpai_agent.child_process import (
    ChildProcessConfig,
    OpenHandsChildProcess,
    render_child_prompt,
    run_bounded_process,
)
from senpai_agent.tools import AgentDispatchRequest


def dispatch_request() -> AgentDispatchRequest:
    return AgentDispatchRequest(
        task_id=str(uuid.uuid4()),
        parent_conversation_id=str(uuid.uuid4()),
        parent_context=(
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
    )


def test_child_prompt_contains_the_complete_snapshot_and_bounded_task():
    request = dispatch_request()

    prompt = render_child_prompt(request, "Review the result and report next steps.")

    assert "Review the result and report next steps." in prompt
    assert "Inspect PR #17." in prompt
    assert "Progressively disclosed skill body." in prompt
    assert "I will compare the evidence." in prompt
    payload = prompt.split("<parent_context_json>\n", 1)[1].split(
        "\n</parent_context_json>", 1
    )[0]
    messages = json.loads(payload)
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_context_free_child_prompt_contains_only_system_context_and_task():
    request = AgentDispatchRequest(
        task_id=str(uuid.uuid4()),
        parent_conversation_id=str(uuid.uuid4()),
        parent_context=(),
    )

    prompt = render_child_prompt(request, "Inspect PR #17.")

    assert "Inspect PR #17." in prompt
    assert "parent_context_json" not in prompt
    assert "complete model-visible parent context" not in prompt


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group deadline")
def test_bounded_process_kills_an_uncooperative_process_group(tmp_path: Path):
    pid_file = tmp_path / "pid"
    code = (
        "import os,pathlib,signal,time;"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="hard runtime"):
        run_bounded_process(
            (sys.executable, "-c", code),
            input_text="",
            env=dict(os.environ),
            timeout_seconds=0.1,
            terminate_grace_seconds=0.05,
        )

    elapsed = time.monotonic() - started
    assert elapsed < 2
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_openhands_child_command_isolated_from_parent_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = dispatch_request()
    config = ChildProcessConfig(
        python_executable=Path(sys.executable),
        workspace=tmp_path / "target",
        state_dir=tmp_path / "state",
        model="anthropic/claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="model-secret",
        github_repo="acme/widgets",
        github_token="github-secret",
        github_trusted_actor=None,
        reasoning_effort="xhigh",
        role_file=tmp_path / "SENPAI-ADVISOR.md",
        harness_file=tmp_path / "SENPAI-HARNESS.md",
        plugin_dir=tmp_path / "plugin",
        enable_browser=True,
        command_secrets={"EXA_API_KEY": "exa-secret"},
    )

    monkeypatch.setenv("SENPAI_OPENHANDS_AGENT", "parent-specialist")
    child = OpenHandsChildProcess(config, request)

    command = child.command
    assert command[:3] == (
        str(Path(sys.executable)),
        "-m",
        "senpai_agent.openhands_runner",
    )
    assert "--child" in command
    assert "--conversation-id" in command
    assert request.task_id not in command
    assert "model-secret" not in repr(command)
    assert child.state_dir.parent == config.state_dir / "children"
    assert child.environment["ANTHROPIC_API_KEY"] == "model-secret"
    assert child.environment["GH_REPO"] == "acme/widgets"
    assert "GITHUB_TOKEN" not in child.environment
    assert "GH_TOKEN" not in child.environment
    assert child.environment["EXA_API_KEY"] == "exa-secret"
    assert "SENPAI_OPENHANDS_AGENT" not in child.environment


def test_child_hands_github_token_over_one_use_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = dispatch_request()
    config = ChildProcessConfig(
        python_executable=Path(sys.executable),
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        model="openai/gpt-5",
        api_key_env="OPENAI_API_KEY",
        api_key="model-secret",
        github_repo="acme/widgets",
        github_token="github-secret",
        github_trusted_actor=None,
        reasoning_effort="high",
        role_file=tmp_path / "role",
        harness_file=tmp_path / "harness",
        plugin_dir=tmp_path / "plugin",
        enable_browser=False,
        command_secrets={},
    )
    observed = {}

    def fake_run(argv, *, input_text, env, timeout_seconds, **kwargs):
        token_path = Path(env["SENPAI_GITHUB_TOKEN_FILE"])
        observed["mode"] = token_path.stat().st_mode & 0o777
        observed["token"] = token_path.read_text()
        token_path.unlink()
        observed["environment"] = env
        return 'OPENHANDS_RESULT {"status":"finished","result":"done"}'

    monkeypatch.setattr(
        "senpai_agent.child_process.run_bounded_process",
        fake_run,
    )

    assert OpenHandsChildProcess(config, request).run("inspect", 10) == "done"
    assert observed["mode"] == 0o600
    assert observed["token"] == "github-secret"
    assert "GITHUB_TOKEN" not in observed["environment"]
    assert "GH_TOKEN" not in observed["environment"]


def test_child_result_parser_uses_only_the_terminal_result_record(tmp_path: Path):
    request = dispatch_request()
    config = ChildProcessConfig(
        python_executable=Path(sys.executable),
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        model="openai/gpt-5",
        api_key_env="OPENAI_API_KEY",
        api_key="secret",
        github_repo="acme/widgets",
        github_token="github-secret",
        github_trusted_actor="senpai-bot",
        reasoning_effort="high",
        role_file=tmp_path / "role",
        harness_file=tmp_path / "harness",
        plugin_dir=tmp_path / "plugin",
        enable_browser=False,
        command_secrets={},
    )
    child = OpenHandsChildProcess(config, request)
    output = (
        'OPENHANDS_EVENT {"text":"intermediate"}\n'
        'OPENHANDS_RESULT {"status":"finished","result":"final report"}'
    )

    assert child.parse_result(output) == "final report"

    with pytest.raises(RuntimeError, match="terminal result"):
        child.parse_result('OPENHANDS_EVENT {"text":"not enough"}')
