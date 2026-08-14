import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.supervisor as supervisor_module
from senpai_agent.program_context import (
    PROGRAM_SHA256_ENV,
    PROGRAM_SNAPSHOT_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    program_system_prompt_sha256,
)
from senpai_agent.supervisor import (
    SupervisorConfig,
    WorkerLease,
    WorkerSupervisor,
    prepare_program_context_environment,
)


def wait_for(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"{path} was not created")


def run_supervisor(
    supervisor: WorkerSupervisor,
    stop: threading.Event,
) -> tuple[threading.Thread, list[int]]:
    results: list[int] = []
    thread = threading.Thread(target=lambda: results.append(supervisor.run(stop)))
    thread.start()
    return thread, results


def restart_backoffs(stderr: str) -> list[float]:
    return [
        float(match.group(1))
        for match in re.finditer(r"backoff_seconds=([0-9.]+)", stderr)
    ]


def test_supervisor_default_termination_grace_is_sixty_seconds():
    assert SupervisorConfig().terminate_grace_seconds == 60


def test_supervisor_caps_repeated_restart_backoff_at_five_minutes():
    assert SupervisorConfig().max_backoff_seconds == 300


def test_supervisor_pins_committed_programme_before_starting_workers(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    program = workspace / "program.md"
    program.write_text("Committed policy.")
    subprocess.run(("git", "init", "-q"), cwd=workspace, check=True)
    subprocess.run(("git", "add", "program.md"), cwd=workspace, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Senpai Test",
            "-c",
            "user.email=senpai@example.com",
            "commit",
            "-qm",
            "Add programme",
        ),
        cwd=workspace,
        check=True,
    )
    program.write_text("Mutable replacement.")

    environment = prepare_program_context_environment(
        {"SENPAI_OPENHANDS_WORKSPACE": str(workspace)},
        tmp_path / "state",
    )

    snapshot = Path(environment[PROGRAM_SNAPSHOT_ENV])
    prompt = snapshot.read_text()
    manifest = json.loads(
        (tmp_path / "state" / "program-context" / "current.json").read_text()
    )
    assert prompt.endswith("Committed policy.")
    assert "Mutable replacement" not in prompt
    assert environment[PROGRAM_SHA256_ENV] == program_system_prompt_sha256(prompt)
    assert environment[PROGRAM_SOURCE_COMMIT_ENV] == subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert manifest == {
        "version": 1,
        "program_path": "program.md",
        "source_commit": environment[PROGRAM_SOURCE_COMMIT_ENV],
        "prompt_sha256": environment[PROGRAM_SHA256_ENV],
        "snapshot_path": f"program-context/{environment[PROGRAM_SHA256_ENV]}.md",
    }
    automatic_restart = prepare_program_context_environment(
        {"SENPAI_OPENHANDS_WORKSPACE": str(workspace)},
        tmp_path / "state",
    )
    assert automatic_restart["SENPAI_PROGRAM_PATH"] == "program.md"
    assert automatic_restart[PROGRAM_SHA256_ENV] == environment[PROGRAM_SHA256_ENV]
    with pytest.raises(RuntimeError, match="does not match the pinned generation"):
        prepare_program_context_environment(
            {
                "SENPAI_PROGRAM_PATH": "senpai/program.md",
                "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            },
            tmp_path / "state",
        )

    subprocess.run(("git", "add", "program.md"), cwd=workspace, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Senpai Test",
            "-c",
            "user.email=senpai@example.com",
            "commit",
            "-qm",
            "Replace programme",
        ),
        cwd=workspace,
        check=True,
    )
    restarted = prepare_program_context_environment(
        {"SENPAI_OPENHANDS_WORKSPACE": str(workspace)},
        tmp_path / "state",
    )

    assert restarted[PROGRAM_SOURCE_COMMIT_ENV] == environment[
        PROGRAM_SOURCE_COMMIT_ENV
    ]
    assert restarted[PROGRAM_SHA256_ENV] == environment[PROGRAM_SHA256_ENV]
    assert restarted[PROGRAM_SNAPSHOT_ENV] == environment[PROGRAM_SNAPSHOT_ENV]
    assert Path(restarted[PROGRAM_SNAPSHOT_ENV]).read_text() == prompt


def test_pid_one_reaps_adopted_children_without_reaping_its_worker(monkeypatch):
    reaped = []
    monkeypatch.setattr(supervisor_module.os, "getpid", lambda: 1)
    monkeypatch.setattr(
        supervisor_module.psutil,
        "Process",
        lambda: type(
            "SupervisorProcess",
            (),
            {
                "children": lambda self: [
                    type("Child", (), {"pid": 41})(),
                    type("Child", (), {"pid": 42})(),
                ]
            },
        )(),
    )
    monkeypatch.setattr(
        supervisor_module.os,
        "waitpid",
        lambda pid, options: reaped.append((pid, options)),
    )

    WorkerSupervisor._reap_orphaned_children(worker_pid=41)

    assert reaped == [(42, os.WNOHANG)]


def test_restarted_workers_receive_github_token_without_environment_exposure(
    tmp_path: Path,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
token_fd = int(os.environ["SENPAI_GITHUB_TOKEN_FD"])
with os.fdopen(token_fd) as token_stream:
    token = token_stream.read()
with (state / "observations").open("a") as output:
    output.write(json.dumps({
        "token": token,
        "github_env": os.environ.get("GITHUB_TOKEN"),
        "gh_env": os.environ.get("GH_TOKEN"),
        "token_file_env": os.environ.get("SENPAI_GITHUB_TOKEN_FILE"),
    }) + "\\n")

lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "ready",
    "deadline": time.monotonic() + 30,
}))
if count == 1:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        github_token=SecretStr("write-token-sentinel"),
        environment={
            **os.environ,
            "GITHUB_TOKEN": "must-not-survive",
            "GH_TOKEN": "must-not-survive",
        },
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.1,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    observations = [
        json.loads(line)
        for line in (tmp_path / "observations").read_text().splitlines()
    ]
    assert len(observations) == 2
    assert all(item["token"] == "write-token-sentinel" for item in observations)
    assert all(item["github_env"] is None for item in observations)
    assert all(item["gh_env"] is None for item in observations)
    assert all(item["token_file_env"] is None for item in observations)
    assert not list(tmp_path.glob(".github-token-*"))


def test_overdue_worker_is_killed_and_restarted(tmp_path: Path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import signal
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
signal.signal(signal.SIGTERM, signal.SIG_IGN)

lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
temporary = lease.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "wedged-turn" if count == 1 else "healthy-turn",
    "deadline": time.monotonic() + (0.05 if count == 1 else 30),
}))
temporary.replace(lease)

if count > 1:
    (state / "restarted").write_text("restarted")
while True:
    time.sleep(1)
""".strip()
    )
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.01,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.01,
            max_backoff_seconds=0.01,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "restarted")
    stop.set()
    thread.join(5)

    assert not thread.is_alive()
    assert results == [0]
    assert int((tmp_path / "starts").read_text()) >= 2


def test_worker_uptime_without_a_completed_turn_does_not_reset_restart_backoff(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "sleep",
    "deadline": 1e100,
}))
time.sleep(0.03)
if count < 3:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    class AcceleratedClock:
        @staticmethod
        def monotonic():
            return time.monotonic() * 100_000

    monkeypatch.setattr(supervisor_module, "time", AcceleratedClock())
    monkeypatch.setattr(supervisor_module.random, "uniform", lambda _a, _b: 1.2)
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1_000_000_000,
            check_interval_seconds=0.005,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.2,
            max_backoff_seconds=0.4,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    assert restart_backoffs(capsys.readouterr().err) == [0.2, 0.4]


def test_worker_exit_samples_its_final_completed_turn(tmp_path: Path, monkeypatch):
    supervisor = WorkerSupervisor(
        command=("worker",),
        lease_path=tmp_path / "controller-lease.json",
    )
    leases = iter(
        (
            WorkerLease(
                pid=123,
                phase="openhands-turn",
                deadline=time.monotonic() + 30,
            ),
            WorkerLease(
                pid=123,
                phase="turn-complete",
                deadline=time.monotonic() + 30,
                completed_turns=1,
            ),
        )
    )
    monkeypatch.setattr(supervisor, "_read_lease", lambda: next(leases))
    monkeypatch.setattr(supervisor, "_remember_descendants", lambda *_args: None)

    class ExitedProcess:
        pid = 123

        @staticmethod
        def poll():
            return 19

    reason, made_progress = supervisor._wait_for_worker(
        ExitedProcess(),
        {},
        threading.Event(),
        time.monotonic(),
    )

    assert reason == "exit:19"
    assert made_progress is True


def test_completed_turn_resets_restart_backoff(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1])
count_path = state / "starts"
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
lease = Path(os.environ["SENPAI_CONTROLLER_LEASE_PATH"])
lease.write_text(json.dumps({
    "pid": os.getpid(),
    "phase": "poll",
    "deadline": time.monotonic() + 30,
    "completed_turns": 1 if count == 2 else 0,
}))
time.sleep(0.03)
if count < 4:
    raise SystemExit(19)
(state / "ready").write_text("ready")
while True:
    time.sleep(1)
""".strip()
    )
    monkeypatch.setattr(supervisor_module.random, "uniform", lambda _a, _b: 1.0)
    stop = threading.Event()
    supervisor = WorkerSupervisor(
        command=(sys.executable, str(worker), str(tmp_path)),
        lease_path=tmp_path / "controller-lease.json",
        config=SupervisorConfig(
            startup_timeout_seconds=1,
            check_interval_seconds=0.005,
            terminate_grace_seconds=0.05,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=0.4,
        ),
    )

    thread, results = run_supervisor(supervisor, stop)
    wait_for(tmp_path / "ready")
    stop.set()
    thread.join(5)

    assert results == [0]
    assert restart_backoffs(capsys.readouterr().err) == [0.1, 0.1, 0.2]


def test_health_command_reports_live_and_expired_worker_leases(tmp_path: Path):
    lease = tmp_path / "controller-lease.json"
    lease.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "sleep",
                "deadline": time.monotonic() + 30,
            }
        )
    )

    healthy = subprocess.run(
        [
            sys.executable,
            "-m",
            "senpai_agent.supervisor",
            "health",
            str(lease),
        ],
        check=False,
    )
    lease.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "phase": "openhands-turn",
                "deadline": time.monotonic() - 1,
            }
        )
    )
    expired = subprocess.run(
        [
            sys.executable,
            "-m",
            "senpai_agent.supervisor",
            "health",
            str(lease),
        ],
        check=False,
    )

    assert healthy.returncode == 0
    assert expired.returncode == 1


def test_openhands_reopens_durable_events_after_an_unclean_worker_exit(
    tmp_path: Path,
):
    conversation_id = "00000000-0000-0000-0000-000000000049"
    crash = tmp_path / "crash.py"
    crash.write_text(
        """
import os
import sys
from pathlib import Path
from uuid import UUID

from pydantic import SecretStr
from openhands.sdk import Agent, Conversation, LLM

state_dir, workspace, conversation_id = sys.argv[1:]
conversation = Conversation(
    agent=Agent(
        llm=LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key")),
        tools=[],
    ),
    workspace=Path(workspace),
    persistence_dir=Path(state_dir),
    conversation_id=UUID(conversation_id),
    visualizer=None,
    delete_on_close=False,
)
conversation.send_message("resume this durable event")
os._exit(17)
""".strip()
    )
    resume = tmp_path / "resume.py"
    resume.write_text(
        """
import sys
from pathlib import Path
from uuid import UUID

from pydantic import SecretStr
from openhands.sdk import Agent, Conversation, LLM

state_dir, workspace, conversation_id = sys.argv[1:]
conversation = Conversation(
    agent=Agent(
        llm=LLM(model="openai/gpt-4o-mini", api_key=SecretStr("test-key")),
        tools=[],
    ),
    workspace=Path(workspace),
    persistence_dir=Path(state_dir),
    conversation_id=UUID(conversation_id),
    visualizer=None,
    delete_on_close=False,
)
assert any(
    "resume this durable event" in str(event)
    for event in conversation.state.view.events
)
conversation.close()
""".strip()
    )
    environment = {
        **os.environ,
        "OPENHANDS_SUPPRESS_BANNER": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    arguments = [
        str(tmp_path / "openhands-state"),
        str(tmp_path / "workspace"),
        conversation_id,
    ]

    crashed = subprocess.run(
        [sys.executable, str(crash), *arguments],
        env=environment,
        check=False,
    )
    resumed = subprocess.run(
        [sys.executable, str(resume), *arguments],
        env=environment,
        check=False,
    )

    assert crashed.returncode == 17
    assert resumed.returncode == 0
