from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.tools.terminal import TerminalAction, TerminalObservation

from senpai_agent.tools import SenpaiTerminalExecutor


class FakeTerminal:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.interrupted = False

    def __call__(self, action, conversation=None):
        self.calls.append((action, conversation))
        return TerminalObservation.from_text(
            "allowed",
            command=action.command,
            exit_code=0,
        )

    def close(self) -> None:
        self.closed = True

    def interrupt(self) -> None:
        self.interrupted = True


def test_terminal_executor_delegates_only_after_policy_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from senpai_agent import hooks

    decisions = []

    def allow(command: str, role: str, workspace: Path):
        decisions.append((command, role, workspace))
        return SimpleNamespace(allowed=True, reason="")

    monkeypatch.setattr(hooks, "terminal_policy", allow)
    delegate = FakeTerminal()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git status --short")
    conversation = SimpleNamespace()

    observation = executor(action, conversation)
    executor.interrupt()
    executor.close()

    assert observation.text == "allowed"
    assert decisions == [("git status --short", "student", tmp_path)]
    assert delegate.calls == [(action, conversation)]
    assert delegate.interrupted is True
    assert delegate.closed is True


@pytest.mark.parametrize("policy_error", [False, True])
def test_terminal_executor_fails_closed_without_invoking_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy_error: bool,
):
    from senpai_agent import hooks

    def policy(_command: str, _role: str, _workspace: Path):
        if policy_error:
            raise RuntimeError("parser unavailable")
        return SimpleNamespace(allowed=False, reason="Use the typed GitHub tool.")

    monkeypatch.setattr(hooks, "terminal_policy", policy)
    delegate = FakeTerminal()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git push origin experiment")

    observation = executor(action)

    assert observation.is_error is True
    assert observation.command == action.command
    assert observation.exit_code is None
    assert "denied" in observation.text.lower()
    assert delegate.calls == []
