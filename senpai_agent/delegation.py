"""One process-isolated delegation path for every Senpai subagent."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

from openhands.sdk.event import ActionEvent, LLMConvertibleEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import Field

from senpai_agent.advisor import AdvisorEvent, AdvisorEventStore
from senpai_agent.processes import terminate_process_group
from senpai_agent.secrets import scrub_github_credentials

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


AgentKind = Literal["general-purpose", "explore", "search", "bash-runner"]
ModelTier = Literal["smart", "fast"]
SearchMode = Literal["general-web", "research-publications"]
MAX_PARALLEL_AGENTS = 8


class AdvisorEventSink(Protocol):
    def enqueue(self, event: AdvisorEvent) -> bool: ...


@dataclass(frozen=True)
class DelegationRequest:
    task_id: str
    parent_conversation_id: str
    parent_context: tuple[Message, ...]
    agent: AgentKind
    model: ModelTier
    search_mode: SearchMode | None


class ChildAgentRunner(Protocol):
    def run(self, task: str, timeout_seconds: float | None) -> str: ...

    def interrupt(self) -> None: ...


class ChildAgentRunnerFactory(Protocol):
    def __call__(self, request: DelegationRequest) -> ChildAgentRunner: ...


@dataclass(frozen=True)
class DelegationConfig:
    python_executable: Path
    workspace: Path
    state_dir: Path
    smart_model: str
    fast_model: str
    api_key_env: str
    api_key: str
    github_repo: str
    github_trusted_actor: str | None
    smart_reasoning_effort: str
    fast_reasoning_effort: str
    role_file: Path
    harness_file: Path
    plugin_dir: Path
    enable_browser: bool
    command_secrets: Mapping[str, str]
    role: str
    background_allowed: bool


_DELEGATION_CONFIG: DelegationConfig | None = None


def configure_delegation(config: DelegationConfig | None) -> None:
    """Hold process-launch secrets outside model-visible tool parameters."""

    global _DELEGATION_CONFIG
    _DELEGATION_CONFIG = config


def configured_child_runner_factory() -> ChildAgentRunnerFactory:
    if _DELEGATION_CONFIG is None:
        raise RuntimeError("subagent runtime is not configured")
    config = _DELEGATION_CONFIG
    return lambda request: OpenHandsChildProcess(config, request)


def render_child_prompt(request: DelegationRequest, task: str) -> str:
    assignment = task.strip()
    if request.search_mode is not None:
        assignment = f"Search mode: {request.search_mode}\n\n{assignment}"
    if not request.parent_context:
        return (
            "# Delegated task\n\n"
            "You are a fresh Senpai subagent. Perform only the assigned task "
            "and return a concise, evidence-linked report to the parent.\n\n"
            f"{assignment}\n"
        )
    context = [message.model_dump(mode="json") for message in request.parent_context]
    return (
        "# Delegated task with parent context\n\n"
        "The JSON below is the complete model-visible parent context at "
        "delegation time. Use it as evidence, perform only the assigned task, "
        "and return a concise, evidence-linked report.\n\n"
        "<parent_context_json>\n"
        f"{json.dumps(context, separators=(',', ':'))}\n"
        "</parent_context_json>\n\n"
        f"{assignment}\n"
    )


def run_child_process(
    argv: Sequence[str],
    *,
    input_text: str,
    env: Mapping[str, str],
    timeout_seconds: float | None,
    terminate_grace_seconds: float = 5,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    on_finish: Callable[[subprocess.Popen[str]], None] | None = None,
) -> str:
    process = subprocess.Popen(
        tuple(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        start_new_session=True,
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
            terminate_process_group(
                process,
                grace_seconds=terminate_grace_seconds,
            )
            process.communicate()
            raise TimeoutError(
                f"subagent exceeded its {timeout_seconds:g}-second runtime"
            ) from error
    finally:
        if on_finish is not None:
            on_finish(process)
    if process.returncode != 0:
        tail = output[-8192:].strip()
        raise RuntimeError(f"subagent process exited {process.returncode}: {tail}")
    return output


class OpenHandsChildProcess:
    """One independently interruptible OpenHands subagent."""

    def __init__(self, config: DelegationConfig, request: DelegationRequest):
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
        model = (
            self._config.smart_model
            if self._request.model == "smart"
            else self._config.fast_model
        )
        effort = (
            self._config.smart_reasoning_effort
            if self._request.model == "smart"
            else self._config.fast_reasoning_effort
        )
        return (
            str(self._config.python_executable),
            "-m",
            "senpai_agent.openhands_runner",
            "--child",
            "--max-turns",
            "1000",
            "--model",
            model,
            "--api-key-env",
            self._config.api_key_env,
            "--reasoning-effort",
            effort,
            "--agent",
            self._request.agent,
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
        scrub_github_credentials(environment)
        for name in (
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
                "SENPAI_OPENHANDS_SMART_MODEL": self._config.smart_model,
                "SENPAI_OPENHANDS_FAST_MODEL": self._config.fast_model,
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

    def run(self, task: str, timeout_seconds: float | None) -> str:
        if self._interrupted.is_set():
            raise InterruptedError("subagent was interrupted before startup")
        self.state_dir.parent.mkdir(parents=True, exist_ok=True)

        def started(process: subprocess.Popen[str]) -> None:
            with self._lock:
                self._process = process
            if self._interrupted.is_set():
                terminate_process_group(process, grace_seconds=1)

        def finished(process: subprocess.Popen[str]) -> None:
            with self._lock:
                if self._process is process:
                    self._process = None

        try:
            output = run_child_process(
                self.command,
                input_text=render_child_prompt(self._request, task),
                env=self.environment,
                timeout_seconds=timeout_seconds,
                on_start=started,
                on_finish=finished,
            )
            return self.parse_result(output)
        finally:
            shutil.rmtree(self.state_dir, ignore_errors=True)

    def interrupt(self) -> None:
        self._interrupted.set()
        with self._lock:
            process = self._process
        if process is not None:
            terminate_process_group(process, grace_seconds=1)

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
            raise RuntimeError("subagent returned no successful terminal result")
        raise RuntimeError("subagent emitted no terminal result record")


class DelegateAgentAction(Action):
    task: str = Field(
        min_length=1,
        description=(
            "Self-contained assignment and requested report. Ask for concise "
            "findings with file paths, line numbers, or source URLs."
        ),
    )
    agent: AgentKind = Field(
        default="general-purpose",
        description=(
            "Agent specialization: general-purpose for mixed work, explore for "
            "code/data/history inspection, search for web and publication research, "
            "or bash-runner for tests, builds, linters, and other CLI output."
        ),
    )
    model: ModelTier = Field(
        default="smart",
        description=(
            "Use fast for mechanical lookup, command execution, grep, and "
            "straightforward extraction. Use smart for ambiguous synthesis, "
            "literature research, code review, or decisions where missing a "
            "subtlety is costly."
        ),
    )
    background: bool = Field(
        default=False,
        description=(
            "False waits and returns the result in this tool call. True returns a "
            "task ID immediately and delivers the result later as a durable event."
        ),
    )
    include_context: bool = Field(
        default=False,
        description=(
            "Copy the complete model-visible parent history into the child. Keep "
            "false for a cheaper self-contained task; the child can search the "
            "parent's durable history files when given a precise question."
        ),
    )
    search_mode: SearchMode | None = Field(
        default=None,
        description=(
            "Required when agent=search: general-web uses Exa's general index; "
            "research-publications uses Exa's publication index and primary papers."
        ),
    )


class DelegateAgentObservation(Observation):
    task_id: str
    status: Literal["finished", "dispatched"]
    result: str | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        if self.status == "finished":
            return [
                TextContent(
                    text=f"Subagent task {self.task_id} finished.\n\n{self.result or ''}"
                )
            ]
        return [
            TextContent(
                text=(
                    f"Subagent task {self.task_id} is running in the background. "
                    "Its result or error will arrive as a durable local event."
                )
            )
        ]


class _DelegateAgentExecutor(
    ToolExecutor[DelegateAgentAction, DelegateAgentObservation]
):
    def __init__(
        self,
        child_runner_factory: ChildAgentRunnerFactory,
        event_sink: AdvisorEventSink | None,
        *,
        event_db_path: Path | None,
        max_workers: int,
        max_runtime_seconds: float | None,
        background_allowed: bool,
    ):
        if not 1 <= max_workers <= MAX_PARALLEL_AGENTS:
            raise ValueError(f"max_workers must be between 1 and {MAX_PARALLEL_AGENTS}")
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        self.child_runner_factory = child_runner_factory
        self.event_sink = event_sink
        self.event_db_path = event_db_path
        self.max_runtime_seconds = max_runtime_seconds
        self.background_allowed = background_allowed
        self._slots = threading.BoundedSemaphore(max_workers)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="senpai-subagent",
        )
        self._background: dict[Future[None], ChildAgentRunner] = {}
        self._active: set[ChildAgentRunner] = set()
        self._lock = threading.Lock()

    def __call__(
        self,
        action: DelegateAgentAction,
        conversation: LocalConversation | None = None,
    ) -> DelegateAgentObservation:
        if conversation is None:
            raise ValueError("delegate_agent requires its parent conversation")
        if (action.agent == "search") != (action.search_mode is not None):
            raise ValueError("search_mode is required only when agent=search")
        if action.background and not self.background_allowed:
            raise ValueError("nested subagents must run with background=false")

        request = DelegationRequest(
            task_id=str(uuid.uuid4()),
            parent_conversation_id=str(conversation.id),
            parent_context=(
                _model_visible_context(conversation) if action.include_context else ()
            ),
            agent=action.agent,
            model=action.model,
            search_mode=action.search_mode,
        )
        if action.background:
            if not self._slots.acquire(blocking=False):
                raise RuntimeError("all eight subagent slots are active")
            try:
                runner = self.child_runner_factory(request)
                future = self._pool.submit(
                    self._run_background,
                    request,
                    action.task,
                    runner,
                )
            except BaseException:
                self._slots.release()
                raise
            with self._lock:
                self._background[future] = runner
            future.add_done_callback(self._forget_background)
            return DelegateAgentObservation(
                task_id=request.task_id,
                status="dispatched",
            )

        self._slots.acquire()
        try:
            runner = self.child_runner_factory(request)
            result = self._run(request, action.task, runner)
        finally:
            self._slots.release()
        return DelegateAgentObservation(
            task_id=request.task_id,
            status="finished",
            result=result,
        )

    def _run(
        self,
        request: DelegationRequest,
        task: str,
        runner: ChildAgentRunner,
    ) -> str:
        with self._lock:
            self._active.add(runner)
        try:
            return runner.run(task, self.max_runtime_seconds)
        finally:
            with self._lock:
                self._active.discard(runner)

    def _run_background(
        self,
        request: DelegationRequest,
        task: str,
        runner: ChildAgentRunner,
    ) -> None:
        try:
            result = self._run(request, task, runner)
            event = AdvisorEvent(
                kind="agent_result",
                dedupe_key=f"agent_result:{request.task_id}",
                payload={
                    "task_id": request.task_id,
                    "parent_conversation_id": request.parent_conversation_id,
                    "task": task,
                    "result": result,
                },
            )
        except BaseException as error:  # noqa: BLE001
            event = AdvisorEvent(
                kind="agent_error",
                dedupe_key=f"agent_result:{request.task_id}",
                payload={
                    "task_id": request.task_id,
                    "parent_conversation_id": request.parent_conversation_id,
                    "task": task,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        finally:
            self._slots.release()
        self._enqueue(event)

    def _enqueue(self, event: AdvisorEvent) -> None:
        if self.event_sink is not None:
            self.event_sink.enqueue(event)
        elif self.event_db_path is not None:
            with AdvisorEventStore(self.event_db_path) as event_sink:
                event_sink.enqueue(event)
        else:
            raise RuntimeError("background delegation has no event sink")

    def _forget_background(self, future: Future[None]) -> None:
        with self._lock:
            self._background.pop(future, None)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)

    def interrupt(self) -> None:
        with self._lock:
            runners = tuple(self._active)
        for runner in runners:
            runner.interrupt()


class DelegateAgentTool(ToolDefinition[DelegateAgentAction, DelegateAgentObservation]):
    name = "delegate_agent"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        child_runner_factory: ChildAgentRunnerFactory | None = None,
        event_sink: AdvisorEventSink | None = None,
        *,
        event_db_path: str | Path | None = None,
        max_workers: int = MAX_PARALLEL_AGENTS,
        max_runtime_seconds: float | None = None,
        background_allowed: bool | None = None,
    ) -> Sequence[Self]:
        resolved_event_path = Path(event_db_path) if event_db_path is not None else None
        if child_runner_factory is None:
            child_runner_factory = configured_child_runner_factory()
            if background_allowed is None:
                background_allowed = bool(_DELEGATION_CONFIG.background_allowed)
        if background_allowed is None:
            background_allowed = True
        return [
            cls(
                description=(
                    "Launch one file-defined Senpai subagent. Up to eight independent "
                    "calls in one response run concurrently. Choose fast for mechanical "
                    "search, CLI execution, and extraction, and smart for subtle "
                    "synthesis. Foreground is the default and returns the result inline; "
                    "background returns a task ID and reports through the durable local "
                    "event stream."
                ),
                action_type=DelegateAgentAction,
                observation_type=DelegateAgentObservation,
                annotations=ToolAnnotations(
                    title="Delegate agent",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=_DelegateAgentExecutor(
                    child_runner_factory,
                    event_sink,
                    event_db_path=resolved_event_path,
                    max_workers=max_workers,
                    max_runtime_seconds=max_runtime_seconds,
                    background_allowed=background_allowed,
                ),
            )
        ]


def _model_visible_context(conversation: LocalConversation) -> tuple[Message, ...]:
    events = list(conversation.state.view.events)
    while events and isinstance(events[-1], ActionEvent):
        events.pop()
    return tuple(
        message.model_copy(deep=True)
        for message in LLMConvertibleEvent.events_to_messages(events)
    )
