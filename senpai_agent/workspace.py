"""Target-checkout reconciliation for student assignments."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from senpai_agent.mailbox import ControllerEvent


class StudentWorkspaceReconciler:
    """Check out an assignment without discarding local student commits."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def __call__(self, events: Sequence[ControllerEvent]) -> None:
        assignments = [event for event in events if event.kind == "student_assignment"]
        if not assignments:
            return
        head_ref = str(assignments[0].payload["head_ref"])
        expected_head = str(assignments[0].payload["head_sha"])
        subprocess.run(
            ["git", "fetch", "origin", head_ref],
            cwd=self.workspace,
            check=True,
            timeout=300,
        )
        fetched_head = self._git("rev-parse", "FETCH_HEAD")
        if fetched_head != expected_head:
            raise RuntimeError(
                f"assignment head moved: expected {expected_head}, fetched {fetched_head}"
            )
        local_ref = f"refs/heads/{head_ref}"
        branch_exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", local_ref],
                cwd=self.workspace,
                check=False,
                timeout=30,
            ).returncode
            == 0
        )
        if not branch_exists:
            subprocess.run(
                ["git", "checkout", "-b", head_ref, "FETCH_HEAD"],
                cwd=self.workspace,
                check=True,
                timeout=300,
            )
            return
        local_head = self._git("rev-parse", local_ref)
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", fetched_head, local_head],
            cwd=self.workspace,
            check=False,
            timeout=30,
        ).returncode
        if ancestor != 0:
            raise RuntimeError(
                f"local branch {head_ref} diverged from assignment head {fetched_head}"
            )
        subprocess.run(
            ["git", "checkout", head_ref],
            cwd=self.workspace,
            check=True,
            timeout=300,
        )

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.workspace,
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        ).stdout.strip()
