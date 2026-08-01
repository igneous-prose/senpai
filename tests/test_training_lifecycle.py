from pathlib import Path

import pytest

from senpai_agent.training import TrainingState, TrainingSupervisor
from training_test_support import make_supervisor, run_python, wait_for_terminal


def test_finished_training_persists_its_result_and_log(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    running = run_python(
        supervisor,
        workspace,
        "print('https://wandb.ai/acme/cfd/runs/run-123', flush=True)",
    )

    terminal = wait_for_terminal(supervisor, running.training_id)
    reopened = TrainingSupervisor(
        workspace=workspace,
        state_dir=tmp_path / "state",
    ).get_training_status(running.training_id)

    assert running.state is TrainingState.RUNNING
    assert running.pid is not None
    assert terminal.state is TrainingState.FINISHED
    assert terminal.exit_code == 0
    assert terminal.wandb_run_ids == ("run-123",)
    assert Path(terminal.log_path).read_text().strip().endswith("/runs/run-123")
    assert reopened == terminal


def test_training_passes_shell_metacharacters_as_a_literal_argument(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    literal = "result; $(echo not-a-shell)"
    running = run_python(
        supervisor,
        workspace,
        "import sys; print(sys.argv[1])",
        literal,
    )

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert Path(terminal.log_path).read_text().strip() == literal


def test_training_rejects_a_working_directory_outside_the_workspace(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        run_python(
            supervisor,
            workspace.parent,
            "print('never launched')",
        )


def test_training_rejects_timeout_above_the_launch_ceiling(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path, max_timeout_seconds=30)

    with pytest.raises(ValueError, match="configured maximum"):
        run_python(
            supervisor,
            workspace,
            "print('never launched')",
            timeout_seconds=31,
        )


def test_supervisor_close_cancels_active_training(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    running = run_python(
        supervisor,
        workspace,
        "import time; time.sleep(60)",
        timeout_seconds=60,
    )

    supervisor.close()

    assert supervisor.get_training_status(running.training_id).state is (
        TrainingState.CANCELLED
    )


def test_supervisor_drain_waits_for_training_to_finish(tmp_path: Path):
    workspace, supervisor = make_supervisor(tmp_path)
    running = run_python(
        supervisor,
        workspace,
        "import time; time.sleep(0.1)",
    )

    supervisor.drain()

    assert supervisor.get_training_status(running.training_id).state is (
        TrainingState.FINISHED
    )
