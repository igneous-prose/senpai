import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.tool import Tool, resolve_tool

from senpai_agent.monitor import MonitorStore
from senpai_agent.tools import (
    MonitorTrainingAction,
    MonitorTrainingTool,
    RunTrainingAction,
    RunTrainingTool,
    close_training_runtimes,
    register_senpai_tools,
)
from senpai_agent.training import TrainingResult, TrainingSpec, TrainingState


class StubTraining:
    def __init__(self, workspace: Path, result: TrainingResult):
        self.workspace = workspace
        self.result = result
        self.launched: list[TrainingSpec] = []
        self.status_checks: list[str] = []
        self.closed = False

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        self.launched.append(spec)
        return self.result

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_checks.append(training_id)
        return self.result

    def close(self) -> None:
        self.closed = True


def init_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    return workspace


def finished_result(tmp_path: Path) -> TrainingResult:
    return TrainingResult(
        training_id="training-17",
        state=TrainingState.FINISHED,
        exit_code=0,
        elapsed_seconds=12.5,
        log_path=str(tmp_path / "training.log"),
        wandb_run_ids=("run-abc",),
    )


def test_run_training_registers_a_monitor_for_its_conversation(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors)[0]
    conversation_id = uuid.uuid4()
    spec = TrainingSpec(
        argv=("python", "train.py"),
        cwd=workspace,
        timeout_seconds=600,
    )

    try:
        observation = tool.executor(
            RunTrainingAction(spec=spec),
            SimpleNamespace(id=conversation_id),
        )

        assert training.launched == [spec]
        assert observation.training_id == "training-17"
        assert observation.wandb_run_ids == ("run-abc",)
        monitor = monitors.spec("training-17")
        assert monitor.conversation_id == conversation_id
        assert monitor.metric is None
        assert monitor.gates == ()
    finally:
        monitors.close()


def test_run_training_requires_a_clean_worktree_before_starting(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    (workspace / "candidate.py").write_text("print('uncommitted')\n")
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors)[0]

    try:
        with pytest.raises(RuntimeError, match="clean before training"):
            tool.executor(
                RunTrainingAction(
                    spec=TrainingSpec(
                        argv=("python", "candidate.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                ),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.launched == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_run_training_requires_a_conversation_before_starting(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = RunTrainingTool.create(training, monitors)[0]

    try:
        with pytest.raises(ValueError, match="student conversation"):
            tool.executor(
                RunTrainingAction(
                    spec=TrainingSpec(
                        argv=("python", "train.py"),
                        cwd=workspace,
                        timeout_seconds=20,
                    )
                )
            )

        assert training.launched == []
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_training_validates_the_training_id_before_registration(
    tmp_path: Path,
):
    class MissingTraining(StubTraining):
        def get_training_status(self, training_id: str) -> TrainingResult:
            self.status_checks.append(training_id)
            raise KeyError(training_id)

    workspace = tmp_path / "workspace"
    training = MissingTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    tool = MonitorTrainingTool.create(training, monitors)[0]

    try:
        with pytest.raises(KeyError, match="missing-training"):
            tool.executor(
                MonitorTrainingAction(training_id="missing-training"),
                SimpleNamespace(id=uuid.uuid4()),
            )

        assert training.status_checks == ["missing-training"]
        assert monitors.active() == []
    finally:
        monitors.close()


def test_monitor_training_replaces_the_default_policy(tmp_path: Path):
    workspace = init_workspace(tmp_path)
    training = StubTraining(workspace, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")
    conversation_id = uuid.uuid4()
    run_tool = RunTrainingTool.create(training, monitors)[0]
    monitor_tool = MonitorTrainingTool.create(training, monitors)[0]

    try:
        run_tool.executor(
            RunTrainingAction(
                spec=TrainingSpec(
                    argv=("python", "train.py"),
                    cwd=workspace,
                    timeout_seconds=20,
                )
            ),
            SimpleNamespace(id=conversation_id),
        )
        monitor_tool.executor(
            MonitorTrainingAction(
                training_id="training-17",
                metric="validation/loss",
                direction="min",
                stale_after_seconds=300,
            ),
            SimpleNamespace(id=conversation_id),
        )

        monitor = monitors.spec("training-17")
        assert training.status_checks == ["training-17"]
        assert monitor.metric == "validation/loss"
        assert monitor.direction == "min"
        assert monitor.stale_after_seconds == 300
    finally:
        monitors.close()


def test_interrupting_run_training_closes_its_runtime(tmp_path: Path):
    training = StubTraining(tmp_path, finished_result(tmp_path))
    monitors = MonitorStore(tmp_path / "monitors.sqlite3")

    try:
        RunTrainingTool.create(training, monitors)[0].executor.interrupt()

        assert training.closed is True
    finally:
        monitors.close()


def test_registered_training_tools_share_one_runtime(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=workspace))
    register_senpai_tools()

    tools = resolve_tool(
        Tool(name="senpai_training", params={"state_dir": str(tmp_path / "state")}),
        state,
    )
    by_name = {tool.name: tool for tool in tools}

    try:
        assert set(by_name) == {
            "run_training",
            "get_training_status",
            "monitor_training",
        }
        assert (
            by_name["run_training"].executor.training
            is by_name["get_training_status"].executor.training
        )
        assert (
            by_name["run_training"].executor.training
            is by_name["monitor_training"].executor.training
        )
        assert (
            by_name["run_training"].executor.monitor_store
            is by_name["monitor_training"].executor.store
        )
    finally:
        close_training_runtimes()
