import signal

import senpai_agent.processes as processes


class ExitedProcess:
    pid = 47

    @staticmethod
    def poll():
        return 0

    @staticmethod
    def wait(timeout=None):
        return 0


class ProcessExitingBeforeSignal:
    pid = 48

    def __init__(self):
        self.poll_results = iter((None, 0))
        self.waited = False

    def poll(self):
        return next(self.poll_results)

    def wait(self, timeout=None):
        self.waited = True
        return 0


def test_finished_process_needs_no_group_cleanup(monkeypatch):
    signals = []
    monkeypatch.setattr(
        processes,
        "signal_process_group",
        lambda process_group_id, sig: signals.append((process_group_id, sig)),
    )

    processes.terminate_process_group(ExitedProcess(), grace_seconds=1)

    assert signals == []


def test_process_is_reaped_when_its_group_disappears_before_sigterm(monkeypatch):
    process = ProcessExitingBeforeSignal()
    monkeypatch.setattr(
        processes,
        "signal_process_group",
        lambda _process_group_id, _sig: False,
    )

    processes.terminate_process_group(process, grace_seconds=1)

    assert process.waited is True


def test_training_waits_full_grace_before_killing_descendants(monkeypatch):
    signals = []
    sleeps = []
    times = iter((10.0, 10.25))
    monkeypatch.setattr(
        processes,
        "signal_process_group",
        lambda process_group_id, sig: (
            signals.append((process_group_id, sig)) or True
        ),
    )
    monkeypatch.setattr(processes.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(processes.time, "sleep", sleeps.append)

    processes.terminate_process_group(
        ExitedProcess(),
        grace_seconds=1,
        wait_full_grace=True,
    )

    assert signals == [
        (47, signal.SIGTERM),
        (47, signal.SIGKILL),
    ]
    assert sleeps == [0.75]
