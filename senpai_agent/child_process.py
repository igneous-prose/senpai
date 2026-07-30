"""Hard process boundary for generic OpenHands child agents."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from senpai_agent.tools import AgentDispatchRequest


@dataclass(frozen=True)
class ChildProcessConfig:
    python_executable: Path
    workspace: Path
    state_dir: Path
    model: str
    api_key_env: str
    api_key: str
    github_repo: str
    github_token: str
    github_trusted_actor: str | None
    reasoning_effort: str
    role_file: Path
    harness_file: Path
    plugin_dir: Path
    enable_browser: bool
    command_secrets: Mapping[str, str]
    role: str = "advisor"


_CHILD_PROCESS_CONFIG: ChildProcessConfig | None = None


def configure_child_process(config: ChildProcessConfig | None) -> None:
    """Hold secrets only until OpenHands lazily constructs the dispatch tool."""

    global _CHILD_PROCESS_CONFIG
    _CHILD_PROCESS_CONFIG = config


def configured_child_process_factory() -> Callable[
    [AgentDispatchRequest], OpenHandsChildProcess
]:
    if _CHILD_PROCESS_CONFIG is None:
        raise RuntimeError("child process runtime is not configured")
    config = _CHILD_PROCESS_CONFIG
    return lambda request: OpenHandsChildProcess(config, request)


def render_child_prompt(request: AgentDispatchRequest, task: str) -> str:
    """Render one bounded task, optionally with the parent's visible history."""

    if not request.parent_context:
        return (
            "# Generic child-agent task\n\n"
            "You are a fresh, short-lived Senpai child. Use your system "
            "instructions and available tools, perform only the assigned "
            "bounded task, and end with a concise report for the parent.\n\n"
            "# Assigned task\n\n"
            f"{task.strip()}\n"
        )
    context = [message.model_dump(mode="json") for message in request.parent_context]
    return (
        "# Generic child-agent context\n\n"
        "You are a fresh, short-lived child of a main Senpai agent. The JSON "
        "below is the complete model-visible parent context at dispatch time. "
        "Use it as evidence, perform only the assigned bounded task, and end "
        "with a concise report for the parent.\n\n"
        "<parent_context_json>\n"
        f"{json.dumps(context, separators=(',', ':'))}\n"
        "</parent_context_json>\n\n"
        "# Assigned task\n\n"
        f"{task.strip()}\n"
    )


def _stop_process_group(
    process: subprocess.Popen[str],
    *,
    terminate_grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=terminate_grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    process.wait()


def run_bounded_process(
    argv: Sequence[str],
    *,
    input_text: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    terminate_grace_seconds: float = 5,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    on_finish: Callable[[subprocess.Popen[str]], None] | None = None,
) -> str:
    """Run a subprocess group and enforce a non-cooperative wall-clock limit."""

    process = subprocess.Popen(
        tuple(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        start_new_session=os.name != "nt",
    )
    if on_start is not None:
        on_start(process)
    try:
        try:
            output, _ = process.communicate(
                input=input_text,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            _stop_process_group(
                process,
                terminate_grace_seconds=terminate_grace_seconds,
            )
            process.communicate()
            raise TimeoutError(
                f"child agent exceeded its {timeout_seconds:g}-second hard runtime"
            ) from error
    finally:
        if on_finish is not None:
            on_finish(process)
    if process.returncode != 0:
        tail = output[-8192:].strip()
        raise RuntimeError(
            f"child OpenHands process exited {process.returncode}: {tail}"
        )
    return output


class OpenHandsChildProcess:
    """One independently killable OpenHands child invocation."""

    def __init__(
        self,
        config: ChildProcessConfig,
        request: AgentDispatchRequest,
    ):
        self._config = config
        self._request = request
        self._conversation_id = uuid.uuid4()
        self._lock = threading.Lock()
        self._interrupted = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self.state_dir = config.state_dir / "children" / request.task_id

    @property
    def command(self) -> tuple[str, ...]:
        browser_flag = "--browser" if self._config.enable_browser else "--no-browser"
        return (
            str(self._config.python_executable),
            "-m",
            "senpai_agent.openhands_runner",
            "--child",
            "--max-turns",
            "1000",
            "--model",
            self._config.model,
            "--api-key-env",
            self._config.api_key_env,
            "--reasoning-effort",
            self._config.reasoning_effort,
            "--workspace",
            str(self._config.workspace),
            "--state-dir",
            str(self.state_dir),
            "--conversation-id",
            str(self._conversation_id),
            "--role-file",
            str(self._config.role_file),
            "--harness-file",
            str(self._config.harness_file),
            "--plugin-dir",
            str(self._config.plugin_dir),
            browser_flag,
        )

    @property
    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "SENPAI_GITHUB_TOKEN_FILE",
            "SENPAI_OPENHANDS_AGENT",
            "SENPAI_OPENHANDS_CONVERSATION_ID",
        ):
            environment.pop(name, None)
        environment.update(self._config.command_secrets)
        environment.update(
            {
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "SENPAI_ROLE": self._config.role,
                "SENPAI_OPENHANDS_API_KEY_ENV": self._config.api_key_env,
                "SENPAI_PARENT_CONVERSATION_HISTORY_DIR": str(
                    self._config.state_dir
                    / uuid.UUID(self._request.parent_conversation_id).hex
                    / "events"
                ),
                "GH_REPO": self._config.github_repo,
                self._config.api_key_env: self._config.api_key,
            }
        )
        if self._config.github_trusted_actor is not None:
            environment["SENPAI_GITHUB_ACTOR"] = self._config.github_trusted_actor
        return environment

    def run(self, task: str, timeout_seconds: float) -> str:
        if self._interrupted.is_set():
            raise InterruptedError("child agent was interrupted before startup")
        self.state_dir.parent.mkdir(parents=True, exist_ok=True)
        token_path = self.state_dir.parent / f".github-token-{uuid.uuid4()}"
        token_fd = os.open(
            token_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(token_fd, "w", encoding="utf-8") as token_file:
            token_file.write(self._config.github_token)
        environment = {
            **self.environment,
            "SENPAI_GITHUB_TOKEN_FILE": str(token_path),
        }

        def started(process: subprocess.Popen[str]) -> None:
            with self._lock:
                self._process = process
            if self._interrupted.is_set():
                _stop_process_group(process, terminate_grace_seconds=1)

        def finished(process: subprocess.Popen[str]) -> None:
            with self._lock:
                if self._process is process:
                    self._process = None

        try:
            output = run_bounded_process(
                self.command,
                input_text=render_child_prompt(self._request, task),
                env=environment,
                timeout_seconds=timeout_seconds,
                on_start=started,
                on_finish=finished,
            )
            return self.parse_result(output)
        finally:
            token_path.unlink(missing_ok=True)
            shutil.rmtree(self.state_dir, ignore_errors=True)

    def interrupt(self) -> None:
        self._interrupted.set()
        with self._lock:
            process = self._process
        if process is not None:
            _stop_process_group(process, terminate_grace_seconds=1)

    @staticmethod
    def parse_result(output: str) -> str:
        for line in reversed(output.splitlines()):
            if not line.startswith("OPENHANDS_RESULT "):
                continue
            payload = json.loads(line.removeprefix("OPENHANDS_RESULT "))
            result = payload.get("result")
            if (
                payload.get("status") == "finished"
                and isinstance(result, str)
                and result.strip()
            ):
                return result.strip()
            raise RuntimeError(
                "child OpenHands process returned no successful terminal result"
            )
        raise RuntimeError("child OpenHands process emitted no terminal result record")
