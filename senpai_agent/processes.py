"""POSIX process-group shutdown shared by every supervised workload."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def signal_process_group(process_group_id: int, sig: signal.Signals) -> bool:
    """Signal a process group, returning false when it no longer exists."""

    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    process_group_id: int | None = None,
    grace_seconds: float,
    wait_full_grace: bool = False,
) -> None:
    """Terminate a process group, escalating after its grace period.

    Training uses ``wait_full_grace`` because its leader can exit while a
    descendant is still cleaning up. Other supervised processes can return as
    soon as their leader exits.
    """

    process_group_id = process_group_id or process.pid
    if process.poll() is not None and not wait_full_grace:
        return
    started = time.monotonic()
    if not signal_process_group(process_group_id, signal.SIGTERM):
        if process.poll() is not None:
            process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    if wait_full_grace:
        remaining = grace_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    if wait_full_grace or process.poll() is None:
        signal_process_group(process_group_id, signal.SIGKILL)
    if process.poll() is None:
        process.wait()
