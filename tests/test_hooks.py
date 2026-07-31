import json
import uuid
from pathlib import Path

import pytest
from openhands.sdk.plugin import Plugin

from senpai_agent.hooks import hook_main, terminal_policy
from senpai_agent.training import TrainingResult, TrainingState

REPO_ROOT = Path(__file__).parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "senpai"


@pytest.mark.parametrize(
    "command",
    [
        "git push origin experiment",
        "gh pr merge 17 --squash",
        "gh issue comment 12 --body done",
        "gh api --method PATCH repos/wandb/senpai/pulls/17",
        "gh api --methodPATCH repos/wandb/senpai/pulls/17",
        "gh api -XPOST repos/wandb/senpai/pulls/17",
        "gh api --input=payload.json repos/wandb/senpai/pulls/17",
        "gh api -fstate=closed repos/wandb/senpai/issues/17",
        "gh api -Fbody=@comment.md repos/wandb/senpai/issues/17",
        "gh api repos/wandb/senpai/issues/17 -f state=closed",
        "gh pr checks 17 --watch",
        "env GH_HOST=github.com gh pr merge 17 --squash",
        "env -S 'command gh pr merge 17 --squash'",
        "/usr/bin/env -- command gh issue close 12",
        "command -- gh repo edit wandb/senpai --description compromised",
        "gh alias set ship 'pr merge --squash'",
        "gh workflow run deploy.yml",
        "bash -lc 'env FOO=bar command gh pr ready 17'",
        "curl -X PATCH https://api.github.com/repos/wandb/senpai/pulls/17",
        "curl -XPOST https://API.GitHub.com/repos/wandb/senpai/pulls/17",
        "curl https://api.github.com/repos/wandb/senpai/issues/17 -d state=closed",
        "python train.py --epochs 10",
        "uv run python scripts/train_model.py",
        "torchrun --nproc-per-node 4 train.py",
        "./train_baseline.py --debug",
        "tail -f training.log",
        "timeout 3600 tail -f training.log",
        "setsid sleep 3600",
        "for i in $(seq 120); do sleep 30; done",
        "for (( ; ; )); do echo waiting; done",
        "for id in 1 2; do git push origin experiment; done",
        "for id in 1 2; do python train.py --epochs 10; done",
        "echo inspected;\ngit push origin experiment",
        "watch nvidia-smi",
        "sleep 300",
    ],
)
def test_terminal_policy_denies_workflow_bypasses(command: str, tmp_path: Path):
    decision = terminal_policy(command, "student", tmp_path)

    assert decision.allowed is False
    assert decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "gh pr view 17 --json title",
        "gh pr list --limit 5",
        "gh api repos/wandb/senpai/pulls/17",
        "env GH_HOST=github.com gh pr view 17",
        "command gh repo view wandb/senpai",
        "curl https://api.github.com/repos/wandb/senpai/pulls/17",
        "pytest -q tests/test_train.py",
        "rg 'train.py' README.md",
        "python -c 'print(\"train.py\")'",
        "python train.py --help",
        "python train.py -h",
        "timeout 120 python train.py --help",
        "python -m package.train --help",
        (
            "python train.py --help 2>&1 | grep -E 'slice|weight|loss'; "
            "echo exit=$?"
        ),
        "./train_baseline.py --help",
        "torchrun --help",
        "tail -n 50 training.log",
        "git status --short",
        "timeout 30 pytest -q",
        "setsid pytest -q",
        (
            "for id in run-a run-b; do echo \"$id\"; "
            "grep -n \"$id\" results.log | head -n 3; echo; done"
        ),
        "for id in run-a run-b\ndo\necho \"$id\"\ndone",
    ],
)
def test_terminal_policy_allows_read_only_and_text_references(
    command: str,
    tmp_path: Path,
):
    assert terminal_policy(command, "student", tmp_path).allowed is True


@pytest.mark.parametrize(
    "command",
    [
        "python train.py --help --epochs 10",
        "python train.py -h --epochs 10",
        "python -m package.train --help --epochs 10",
        "python train.py --help 2>&1 --epochs 10 | grep epochs",
        "./train_baseline.py --help --epochs 10",
        "torchrun --help train.py",
    ],
)
def test_terminal_policy_denies_training_arguments_mixed_with_help(
    command: str,
    tmp_path: Path,
):
    assert terminal_policy(command, "student", tmp_path).allowed is False


@pytest.mark.parametrize("tool_name", ["senpai_terminal", "terminal"])
def test_pre_tool_hook_emits_native_deny_json(
    tool_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    event = {
        "event_type": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": "git push origin experiment"},
        "working_dir": str(tmp_path),
    }
    monkeypatch.setattr("sys.stdin.read", lambda: json.dumps(event))
    monkeypatch.setenv("SENPAI_ROLE", "student")

    assert hook_main(["pre-tool-use"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "deny"
    assert "typed" in output["reason"].lower()


def test_pre_tool_hook_fails_closed_on_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr("sys.stdin.read", lambda: "not-json")

    assert hook_main(["pre-tool-use"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "deny"


def test_plugin_registers_only_safety_and_lifecycle_hooks():
    plugin = Plugin.load(PLUGIN_DIR)
    assert plugin.hooks is not None
    assert plugin.hooks.pre_tool_use
    assert plugin.hooks.stop
    assert plugin.hooks.session_end
    hooks = json.loads((PLUGIN_DIR / "hooks" / "hooks.json").read_text())
    assert {hook["matcher"] for hook in hooks["PreToolUse"]} == {
        "senpai_terminal",
        "terminal",
    }
    assert not (PLUGIN_DIR / "scripts" / "check-notifications.sh").exists()


def test_student_stop_hook_denies_while_training_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    workspace = tmp_path / "workspace"
    training_dir = tmp_path / "state" / "training"
    workspace.mkdir()
    training_dir.mkdir(parents=True)
    (workspace / ".git").mkdir()
    training_id = str(uuid.uuid4())
    running = TrainingResult(
        training_id=training_id,
        state=TrainingState.RUNNING,
        exit_code=None,
        elapsed_seconds=10,
        log_path=str(training_dir / f"{training_id}.log"),
    )
    (training_dir / f"{training_id}.json").write_text(running.model_dump_json())
    monkeypatch.setattr(
        "senpai_agent.hooks.subprocess.run",
        lambda *args, **kwargs: type("Completed", (), {"stdout": ""})(),
    )
    monkeypatch.setattr(
        "sys.stdin.read",
        lambda: json.dumps({"working_dir": str(workspace)}),
    )
    monkeypatch.setenv("SENPAI_ROLE", "student")
    monkeypatch.setenv("SENPAI_OPENHANDS_STATE_DIR", str(tmp_path / "state"))

    assert hook_main(["stop"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["decision"] == "deny"
    assert training_id in output["reason"]


def test_student_stop_hook_allows_durable_monitored_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    training_dir = state / "training"
    monitor_dir = training_dir / "monitors"
    workspace.mkdir()
    training_dir.mkdir(parents=True)
    monitor_dir.mkdir()
    (workspace / ".git").mkdir()
    training_id = str(uuid.uuid4())
    running = TrainingResult(
        training_id=training_id,
        state=TrainingState.RUNNING,
        exit_code=None,
        elapsed_seconds=10,
        log_path=str(training_dir / f"{training_id}.log"),
    )
    (training_dir / f"{training_id}.json").write_text(running.model_dump_json())
    (monitor_dir / f"{training_id}.json").write_text(
        json.dumps({"training_id": training_id})
    )
    monkeypatch.setattr(
        "senpai_agent.hooks.subprocess.run",
        lambda *args, **kwargs: type("Completed", (), {"stdout": ""})(),
    )
    monkeypatch.setattr(
        "sys.stdin.read",
        lambda: json.dumps({"working_dir": str(workspace)}),
    )
    monkeypatch.setenv("SENPAI_ROLE", "student")
    monkeypatch.setenv("SENPAI_OPENHANDS_STATE_DIR", str(state))

    assert hook_main(["stop"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "allow"
