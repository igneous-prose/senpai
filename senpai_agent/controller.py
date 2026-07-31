"""Portable Senpai control loop with GitHub as its only remote mailbox."""

from __future__ import annotations

import os
import random
import signal
import sys
import time
from base64 import b64decode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Literal, Protocol
from uuid import UUID

from senpai_agent.advisor import (
    AdvisorEventStore,
    compose_system_instructions,
)
from senpai_agent.github_mailbox import ActiveGitHubWatcher, GitHubMailbox
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    LocalAdvisorMailbox,
    LocalStudentMailbox,
    Mailbox,
)
from senpai_agent.monitor import (
    MonitorMailbox,
    TrainingMonitorEngine,
    WandbMetricSource,
)
from senpai_agent.state import (
    AssignmentConversationRegistry,
    ConversationBatch,
    ConversationStateLedger,
    StudentConversationSelector,
)
from senpai_agent.supervisor import LEASE_ENV, ProgressLease
from senpai_agent.workspace import StudentWorkspaceReconciler


@dataclass(frozen=True, slots=True)
class TurnResult:
    exit_code: int
    delivered_event_keys: frozenset[str] = frozenset()


class TurnRunner(Protocol):
    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        event_keys: frozenset[str],
    ) -> TurnResult: ...


class OpenHandsTurnRunner:
    def __init__(
        self,
        config: object,
        *,
        github_mailbox: GitHubMailbox | None = None,
        active_poll_interval_seconds: float = 30,
    ):
        self.config = config
        self.github_mailbox = github_mailbox
        self.active_poll_interval_seconds = active_poll_interval_seconds

    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        event_keys: frozenset[str],
    ) -> TurnResult:
        from senpai_agent.openhands_runner import run_openhands

        config = replace(
            self.config,
            conversation_id=conversation_id,
        )
        if config.role != "advisor" or self.github_mailbox is None:
            return TurnResult(exit_code=run_openhands(prompt, config))

        store_path = config.state_dir / "advisor-events.sqlite3"
        with ActiveGitHubWatcher(
            self.github_mailbox,
            store_path,
            known_keys=event_keys,
            poll_interval_seconds=self.active_poll_interval_seconds,
        ) as watcher:
            exit_code = run_openhands(prompt, config)
        with AdvisorEventStore(store_path) as store:
            delivered = store.acknowledged(tuple(watcher.observed_keys))
        return TurnResult(
            exit_code=exit_code,
            delivered_event_keys=frozenset(delivered),
        )


class Controller:
    """Poll, reconcile, run one turn, and immediately verify GitHub again."""

    def __init__(
        self,
        *,
        role: Literal["advisor", "student"],
        mailbox: Mailbox,
        turns: TurnRunner,
        conversation_id: UUID,
        full_prompt: str,
        system_context: str = "",
        conversation_state: ConversationStateLedger | None = None,
        conversation_for_events: (
            Callable[[Sequence[ControllerEvent]], Sequence[ConversationBatch]] | None
        ) = None,
        reconcile: Callable[[Sequence[ControllerEvent]], None] | None = None,
        progress: ProgressLease | None = None,
        operation_timeout_seconds: float = 300,
        turn_timeout_seconds: float = 3660,
        start_gate_path: Path | None = None,
        start_gate_poll_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 600,
        jitter_seconds: float = 120,
    ):
        if min(poll_interval_seconds, jitter_seconds) < 0:
            raise ValueError("poll and jitter intervals must not be negative")
        if start_gate_poll_seconds <= 0:
            raise ValueError("start-gate polling interval must be positive")
        if operation_timeout_seconds <= 0 or turn_timeout_seconds <= 0:
            raise ValueError("controller phase timeouts must be positive")
        self.role = role
        self.mailbox = mailbox
        self.turns = turns
        self.conversation_id = conversation_id
        self.conversation_for_events = conversation_for_events
        self.reconcile = reconcile
        self.progress = progress
        self.operation_timeout_seconds = operation_timeout_seconds
        self.turn_timeout_seconds = turn_timeout_seconds
        self.start_gate_path = start_gate_path
        self.start_gate_poll_seconds = start_gate_poll_seconds
        self.full_prompt = full_prompt.strip()
        self.system_context = system_context.strip()
        self.conversation_state = conversation_state
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.jitter_seconds = jitter_seconds
        self._started: set[UUID] = set()
        self._visible: set[str] = set()

    def run(self, *, max_cycles: int | None = None) -> None:
        self._wait_for_start_gate()
        cycles = 0
        turn_failures = 0
        while max_cycles is None or cycles < max_cycles:
            self._publish_progress("poll")
            events = self._new_events(self.mailbox.poll())
            turn_failed = False
            while events:
                batches = self._event_batches(events)
                events = ()
                for batch in batches:
                    batch_events = batch.events
                    conversation_id = batch.conversation_id
                    try:
                        if self.reconcile is not None:
                            self._publish_progress("reconcile")
                            self.reconcile(batch_events)
                        continuing = self._has_started(conversation_id)
                        refresh_system_context = (
                            continuing
                            and self.conversation_state is not None
                            and not self.conversation_state.is_context_current(
                                conversation_id,
                                self.system_context,
                            )
                        )
                        prompt = self._prompt(
                            batch_events,
                            continuing=continuing,
                            refresh_system_context=refresh_system_context,
                        )
                        self._publish_progress(
                            "openhands-turn",
                            self.turn_timeout_seconds,
                        )
                        result = self.turns.run(
                            prompt,
                            conversation_id=conversation_id,
                            event_keys=frozenset(
                                event.dedupe_key for event in batch_events
                            ),
                        )
                    except Exception as error:  # noqa: BLE001
                        turn_failures += 1
                        self._visible.difference_update(
                            event.dedupe_key for event in batch_events
                        )
                        print(
                            f"SENPAI_TURN_EXCEPTION {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        turn_failed = True
                        continue
                    if result.exit_code == 0:
                        self._mark_success(conversation_id)
                        self.mailbox.acknowledge(
                            tuple(event.dedupe_key for event in batch_events)
                        )
                        self._visible.update(result.delivered_event_keys)
                        continue
                    turn_failures += 1
                    self._visible.difference_update(
                        event.dedupe_key for event in batch_events
                    )
                    print(
                        "SENPAI_TURN_ERROR "
                        f"exit_code={result.exit_code} "
                        f"conversation_id={conversation_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    turn_failed = True
                if turn_failed:
                    delay = min(
                        self.poll_interval_seconds,
                        2 ** min(turn_failures, 8),
                    )
                    self._sleep("turn-backoff", delay)
                    break
                turn_failures = 0
                # Post-turn reconciliation avoids waiting one heartbeat for work
                # that appeared while OpenHands was reasoning.
                try:
                    self._publish_progress("poll")
                    events = self._new_events(self.mailbox.poll())
                except Exception as error:  # noqa: BLE001
                    print(
                        f"SENPAI_POST_TURN_POLL_ERROR {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    events = ()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
            if turn_failed:
                continue
            self._sleep(
                "sleep",
                self.poll_interval_seconds + random.uniform(0, self.jitter_seconds),
            )

    def _event_batches(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ConversationBatch, ...]:
        if self.conversation_for_events is not None:
            return tuple(self.conversation_for_events(events))
        return (ConversationBatch(self.conversation_id, tuple(events)),)

    def _wait_for_start_gate(self) -> None:
        while self.start_gate_path is not None and not self.start_gate_path.is_file():
            self._publish_progress(
                "start-gate",
                self.start_gate_poll_seconds + self.operation_timeout_seconds,
            )
            self.sleep(self.start_gate_poll_seconds)

    def _publish_progress(
        self,
        phase: str,
        timeout_seconds: float | None = None,
    ) -> None:
        if self.progress is not None:
            self.progress.update(
                phase,
                timeout_seconds or self.operation_timeout_seconds,
            )

    def _sleep(self, phase: str, seconds: float) -> None:
        self._publish_progress(
            phase,
            max(seconds + self.operation_timeout_seconds, 1),
        )
        self.sleep(seconds)

    def _has_started(self, conversation_id: UUID) -> bool:
        return conversation_id in self._started or (
            self.conversation_state is not None
            and self.conversation_state.has_started(conversation_id)
        )

    def _mark_success(self, conversation_id: UUID) -> None:
        self._started.add(conversation_id)
        if self.conversation_state is not None:
            self.conversation_state.mark_success(
                conversation_id,
                self.system_context,
            )

    def _new_events(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ControllerEvent, ...]:
        current = {event.dedupe_key for event in events}
        new = tuple(event for event in events if event.dedupe_key not in self._visible)
        self._visible = current
        return new

    def _prompt(
        self,
        events: Sequence[ControllerEvent],
        *,
        continuing: bool,
        refresh_system_context: bool = False,
    ) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_prompt = "\n\n".join(event.to_prompt() for event in events)
        if not continuing:
            return (
                f"{self.full_prompt}\n\nCurrent time (UTC): {now}\n\n"
                f"# Current GitHub state\n\n{event_prompt}"
            )
        prompt = (
            f"Continue the {self.role} loop. Current time (UTC): {now}. "
            "GitHub now contains the following actionable state:\n\n"
            f"{event_prompt}"
        )
        if refresh_system_context:
            prompt += (
                "\n\n# Updated Senpai system context\n\n"
                "The deployed harness or role charter changed since this "
                "conversation last ran. Treat the following as the current "
                "Senpai operating context:\n\n"
                f"{self.system_context}"
            )
        return prompt


def _full_prompt(role: Literal["advisor", "student"], env: Mapping[str, str]) -> str:
    workspace = Path(env["SENPAI_OPENHANDS_WORKSPACE"]).resolve()
    instructions = workspace / "instructions" / f"prompt-{role}.md"
    program = workspace / "program.md"
    prompt = (
        "# Research programme\n\n"
        f"{program.read_text(encoding='utf-8').strip()}\n\n"
        f"# {role.title()} task\n\n"
        f"{Template(instructions.read_text(encoding='utf-8')).safe_substitute(env).strip()}"
    )
    encoded_extra = env.get("EXTRA_INSTRUCTIONS_B64")
    if encoded_extra:
        extra = b64decode(encoded_extra, validate=True).decode()
        prompt += f"\n\n# Additional launch instructions\n\n{extra.strip()}"
    identity = (
        f"Role: {role}; repository: {env['GH_REPO']}; "
        f"advisor branch: {env['ADVISOR_BRANCH']}; "
        f"W&B: {env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}."
    )
    if role == "advisor":
        identity += f" Students: {env.get('STUDENT_NAMES', '')}."
    else:
        identity += f" Student: {env['STUDENT_NAME']}."
    return f"{prompt}\n\n# Runtime identity\n\n{identity}"


def _role_interval(
    env: Mapping[str, str],
    role: Literal["advisor", "student"],
    suffix: str,
    default: float,
) -> float:
    role_key = f"SENPAI_{role.upper()}_{suffix}"
    shared_key = f"SENPAI_{suffix}"
    return float(env.get(role_key, env.get(shared_key, str(default))))


def controller_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    import argparse

    progress = ProgressLease(Path(env[LEASE_ENV])) if env.get(LEASE_ENV) else None
    if progress is not None:
        progress.update("startup", 300)

    from senpai_agent.openhands_runner import (
        parse_runner_args,
        read_role_instructions,
        resolve_config,
    )
    from senpai_agent.tools import (
        close_training_runtimes,
        training_runtime,
    )
    from senpai_agent.weave_monitoring import finish_weave_monitoring

    parser = argparse.ArgumentParser(
        description="Run the portable Senpai GitHub/OpenHands controller."
    )
    parser.add_argument("role", choices=("advisor", "student"))
    args = parser.parse_args(argv)
    role = args.role
    if env.get("SENPAI_ROLE") != role:
        raise RuntimeError(f"SENPAI_ROLE must be {role}")
    human_issues = env.get("SENPAI_ENABLE_HUMAN_ISSUES", "true").lower()
    if human_issues not in {"true", "false"}:
        raise RuntimeError("SENPAI_ENABLE_HUMAN_ISSUES must be true or false")

    max_turns = int(env.get("SENPAI_OPENHANDS_MAX_TURNS", "100000"))
    runner_config = resolve_config(
        parse_runner_args(["--max-turns", str(max_turns)]),
        env,
    )
    if runner_config.github_token is None:
        raise RuntimeError("controller worker requires GitHub credentials")
    os.environ.pop(runner_config.api_key_env, None)
    github_mailbox = GitHubMailbox(
        repo=runner_config.github_repo,
        token=runner_config.github_token,
        role=role,
        advisor_branch=env["ADVISOR_BRANCH"],
        students=tuple(
            student.strip()
            for student in env.get("STUDENT_NAMES", "").split(",")
            if student.strip()
        ),
        student_name=env.get("STUDENT_NAME"),
        stale_wip_seconds=int(env.get("SENPAI_STALE_WIP_SECONDS", "7200")),
        trusted_actor=runner_config.github_trusted_actor,
        human_issues_enabled=human_issues == "true",
        feedback_path=(
            runner_config.state_dir / "github-feedback.json"
            if role == "student"
            else None
        ),
    )
    mailbox: Mailbox = github_mailbox
    conversation_selector = None
    reconcile = None

    if role == "advisor":
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalAdvisorMailbox(runner_config.state_dir / "advisor-events.sqlite3"),
        )
    else:
        training, monitor_store = training_runtime(
            runner_config.workspace,
            runner_config.state_dir / "training",
            max_timeout_seconds=runner_config.training_max_timeout_seconds,
        )
        metrics = WandbMetricSource(
            env["WANDB_ENTITY"],
            env["WANDB_PROJECT"],
        )
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalStudentMailbox(runner_config.state_dir / "student-events.sqlite3"),
            MonitorMailbox(
                TrainingMonitorEngine(monitor_store, training, metrics),
                monitor_store,
            ),
        )
        registry = AssignmentConversationRegistry(
            runner_config.state_dir / "student-conversations.json"
        )
        conversation_selector = StudentConversationSelector(registry)
        reconcile = StudentWorkspaceReconciler(runner_config.workspace)

    turns = OpenHandsTurnRunner(
        runner_config,
        github_mailbox=github_mailbox if role == "advisor" else None,
        active_poll_interval_seconds=float(
            env.get("SENPAI_ACTIVE_GITHUB_POLL_INTERVAL_S", "30")
        ),
    )
    controller = Controller(
        role=role,
        mailbox=mailbox,
        turns=turns,
        conversation_id=runner_config.conversation_id,
        system_context=compose_system_instructions(
            read_role_instructions(runner_config.harness_file),
            read_role_instructions(runner_config.role_file),
        ),
        conversation_state=ConversationStateLedger(
            runner_config.state_dir / "conversation-state.json"
        ),
        conversation_for_events=conversation_selector,
        reconcile=reconcile,
        progress=progress,
        operation_timeout_seconds=float(
            env.get("SENPAI_CONTROLLER_OPERATION_TIMEOUT_SECONDS", "300")
        ),
        turn_timeout_seconds=runner_config.timeout_seconds + 60,
        start_gate_path=(
            Path(env["SENPAI_START_GATE_PATH"])
            if env.get("SENPAI_START_GATE_PATH")
            else None
        ),
        start_gate_poll_seconds=float(env.get("SENPAI_START_GATE_POLL_SECONDS", "30")),
        full_prompt=_full_prompt(role, env),
        poll_interval_seconds=_role_interval(
            env,
            role,
            "POLL_INTERVAL_S",
            600,
        ),
        jitter_seconds=_role_interval(
            env,
            role,
            "POLL_JITTER_S",
            120,
        ),
    )

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        controller.run()
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        close_training_runtimes()
        finish_weave_monitoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(controller_main())
