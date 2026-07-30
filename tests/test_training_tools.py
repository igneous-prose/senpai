import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)


def wait_for_terminal(
    supervisor: TrainingSupervisor,
    training_id: str,
    *,
    timeout: float = 5,
) -> TrainingResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = supervisor.get_training_status(training_id)
        if result.state is not TrainingState.RUNNING:
            return result
        time.sleep(0.02)
    raise AssertionError("training did not reach a terminal state")


def test_training_runs_argv_without_a_shell_and_persists_status(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=state_dir)
    spec = TrainingSpec(
        argv=(
            sys.executable,
            "-c",
            "print('https://wandb.ai/acme/cfd/runs/run-123', flush=True)",
        ),
        cwd=workspace,
        timeout_seconds=20,
    )

    result = supervisor.run_training(spec)
    status = wait_for_terminal(supervisor, result.training_id)

    assert result.state is TrainingState.RUNNING
    assert result.pid is not None
    assert status.state is TrainingState.FINISHED
    assert status.exit_code == 0
    assert status.wandb_run_ids == ("run-123",)
    assert Path(status.log_path).read_text().strip().endswith("/runs/run-123")


def test_running_training_publishes_wandb_id_for_metric_monitoring(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=0.1,
    )
    running = supervisor.run_training(
        TrainingSpec(
            argv=(
                sys.executable,
                "-c",
                (
                    "import time; "
                    "print("
                    "'https://wandb.ai/acme/cfd/runs/live-run', flush=True"
                    "); time.sleep(2)"
                ),
            ),
            cwd=workspace,
            timeout_seconds=20,
        )
    )

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        status = supervisor.get_training_status(running.training_id)
        if status.wandb_run_ids:
            break
        time.sleep(0.02)

    assert status.state is TrainingState.RUNNING
    assert status.wandb_run_ids == ("live-run",)
    supervisor.close()


def test_training_rejects_shell_strings_and_paths_outside_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=tmp_path / "state")

    with pytest.raises(ValueError, match="argv"):
        supervisor.run_training(
            TrainingSpec.model_construct(
                argv="python train.py; rm -rf output",
                cwd=workspace,
                timeout_seconds=20,
            )
        )

    with pytest.raises(ValueError, match="inside"):
        supervisor.run_training(
            TrainingSpec(
                argv=(sys.executable, "-c", "print('no')"),
                cwd=tmp_path,
                timeout_seconds=20,
            )
        )


def test_training_rejects_timeout_above_the_launch_ceiling(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        max_timeout_seconds=30,
    )

    with pytest.raises(ValueError, match="configured maximum"):
        supervisor.run_training(
            TrainingSpec(
                argv=(sys.executable, "-c", "print('never launched')"),
                cwd=workspace,
                timeout_seconds=31,
            )
        )


def test_training_timeout_terminates_the_process_group(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=0.4,
    )
    spec = TrainingSpec(
        argv=(
            sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "time.sleep(60)"
            ),
        ),
        cwd=workspace,
        timeout_seconds=1,
    )

    started = time.monotonic()
    result = supervisor.run_training(spec)
    launch_elapsed = time.monotonic() - started
    terminal = wait_for_terminal(supervisor, result.training_id)

    assert result.state is TrainingState.RUNNING
    assert launch_elapsed < 0.5
    assert terminal.state is TrainingState.TIMED_OUT
    assert terminal.elapsed_seconds < 1.25
    assert time.monotonic() - started < 1.25


def test_training_timeout_kills_term_ignoring_descendants(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_pid_path = workspace / "child.pid"
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=0.1,
    )
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )

    result = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", parent_code),
            cwd=workspace,
            timeout_seconds=5,
        )
    )
    terminal = wait_for_terminal(supervisor, result.training_id, timeout=10)
    child_pid = int(child_pid_path.read_text())

    assert terminal.state is TrainingState.TIMED_OUT
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"training descendant {child_pid} survived process-group cleanup")


def test_successful_training_cleans_up_descendants_before_reporting_terminal(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child_pid_path = workspace / "child.pid"
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=0.1,
    )
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    parent_code = (
        "import pathlib,subprocess,sys;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))"
    )

    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", parent_code),
            cwd=workspace,
            timeout_seconds=20,
        )
    )
    terminal = wait_for_terminal(supervisor, running.training_id)
    child_pid = int(child_pid_path.read_text())

    assert terminal.state is TrainingState.FINISHED
    try:
        child_status = psutil.Process(child_pid).status()
    except psutil.NoSuchProcess:
        child_status = None
    assert child_status in {None, psutil.STATUS_ZOMBIE}


def test_failed_training_returns_bounded_error_tail(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=tmp_path / "state")
    spec = TrainingSpec(
        argv=(
            sys.executable,
            "-c",
            (
                "import sys; "
                "[print('failure-line-' + str(i), file=sys.stderr) "
                "for i in range(1000)]; raise SystemExit(3)"
            ),
        ),
        cwd=workspace,
        timeout_seconds=20,
    )

    result = supervisor.run_training(spec)
    terminal = wait_for_terminal(supervisor, result.training_id)

    assert result.state is TrainingState.RUNNING
    assert terminal.state is TrainingState.FAILED
    assert terminal.exit_code == 3
    assert len(terminal.error_tail.encode()) <= 8192
    assert "failure-line-999" in terminal.error_tail
    assert "failure-line-0" not in terminal.error_tail


def test_training_streams_log_metadata_without_whole_file_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=tmp_path / "state")

    def reject_whole_file_read(_path):
        raise AssertionError("training logs must never be read into memory whole")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
    output_code = (
        "import sys;"
        "print('https://wandb.ai/acme/cfd/runs/first-run', flush=True);"
        "sys.stdout.write('x' * 2_000_000);"
        "print('\\nhttps://wandb.ai/acme/cfd/runs/last-run', flush=True);"
        "raise SystemExit(7)"
    )

    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", output_code),
            cwd=workspace,
            timeout_seconds=20,
        )
    )
    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FAILED
    assert terminal.wandb_run_ids == ("first-run", "last-run")
    assert len(terminal.error_tail.encode()) <= 8192
    assert "last-run" in terminal.error_tail
    assert "first-run" not in terminal.error_tail


def test_supervisor_close_cancels_active_training(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=0.1,
    )
    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(60)"),
            cwd=workspace,
            timeout_seconds=60,
        )
    )

    supervisor.close()
    result = supervisor.get_training_status(running.training_id)

    assert result.state is TrainingState.CANCELLED


def test_supervisor_close_never_races_or_extends_the_training_deadline(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    ready = workspace / "ready"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
        terminate_grace_seconds=1.5,
    )
    started = time.monotonic()
    running = supervisor.run_training(
        TrainingSpec(
            argv=(
                sys.executable,
                "-c",
                (
                    "import pathlib,signal,time;"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    f"pathlib.Path({str(ready)!r}).write_text('ready');"
                    "time.sleep(60)"
                ),
            ),
            cwd=workspace,
            timeout_seconds=2,
        )
    )
    while not ready.exists() and time.monotonic() - started < 0.5:
        time.sleep(0.01)
    assert ready.exists()
    while time.monotonic() - started < 0.7:
        time.sleep(0.01)

    supervisor.close()
    result = supervisor.get_training_status(running.training_id)

    assert result.state in {TrainingState.CANCELLED, TrainingState.TIMED_OUT}
    assert result.elapsed_seconds < 2.1
    assert time.monotonic() - started < 2.1


def test_supervisor_drain_keeps_training_alive_until_terminal(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
    )
    running = supervisor.run_training(
        TrainingSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(0.1)"),
            cwd=workspace,
            timeout_seconds=20,
        )
    )

    supervisor.drain()

    result = supervisor.get_training_status(running.training_id)
    assert result.state is TrainingState.FINISHED


def test_new_supervisor_terminates_verified_orphaned_process_group(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    child_pid_path = workspace / "orphan-child.pid"
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    process = psutil.Popen(
        [sys.executable, "-c", parent_code],
        start_new_session=True,
    )
    deadline = time.monotonic() + 3
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    child_pid = int(child_pid_path.read_text())
    orphan = TrainingResult(
        training_id="d7d0d19f-9961-4dac-b2ff-7382dc463674",
        state=TrainingState.RUNNING,
        pid=process.pid,
        process_group_id=process.pid,
        process_start_time=process.create_time(),
        exit_code=None,
        elapsed_seconds=12,
        log_path=str(state_dir / "orphan.log"),
    )
    (state_dir / f"{orphan.training_id}.json").write_text(orphan.model_dump_json())

    try:
        supervisor = TrainingSupervisor(
            workspace=workspace,
            state_dir=state_dir,
            terminate_grace_seconds=0.1,
        )

        recovered = supervisor.get_training_status(orphan.training_id)
        assert recovered.state is TrainingState.CANCELLED
        assert "supervisor restarted" in recovered.error_tail
        assert process.wait(timeout=3) is not None
        try:
            child_status = psutil.Process(child_pid).status()
        except psutil.NoSuchProcess:
            child_status = None
        assert child_status in {None, psutil.STATUS_ZOMBIE}
    finally:
        if process.is_running():
            process.kill()
            process.wait()


def test_restart_does_not_signal_a_reused_or_unverified_pid(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    unrelated = psutil.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    orphan = TrainingResult(
        training_id="26a7194a-3bea-45b1-a2e5-cd20d99e3a31",
        state=TrainingState.RUNNING,
        pid=unrelated.pid,
        process_group_id=unrelated.pid,
        process_start_time=unrelated.create_time() - 10,
        exit_code=None,
        elapsed_seconds=12,
        log_path=str(state_dir / "orphan.log"),
    )
    (state_dir / f"{orphan.training_id}.json").write_text(orphan.model_dump_json())

    try:
        supervisor = TrainingSupervisor(workspace=workspace, state_dir=state_dir)

        recovered = supervisor.get_training_status(orphan.training_id)
        assert recovered.state is TrainingState.CANCELLED
        assert unrelated.is_running()
    finally:
        unrelated.kill()
        unrelated.wait()
