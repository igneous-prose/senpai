import subprocess
from pathlib import Path

import pytest

from senpai_agent.mailbox import ControllerEvent
from senpai_agent.workspace import StudentWorkspaceReconciler, WorkspaceDivergence


def git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def assigned_workspace(tmp_path: Path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    workspace = tmp_path / "student"
    git("init", "--bare", str(remote))
    git("init", str(seed))
    git("config", "user.name", "test", cwd=seed)
    git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "program.py").write_text("baseline\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "baseline", cwd=seed)
    git("branch", "-M", "student/candidate", cwd=seed)
    git("remote", "add", "origin", str(remote), cwd=seed)
    git("push", "origin", "student/candidate", cwd=seed)
    assigned_head = git("rev-parse", "HEAD", cwd=seed)
    git(
        "clone",
        "--branch",
        "student/candidate",
        str(remote),
        str(workspace),
    )
    git("config", "user.name", "student", cwd=workspace)
    git("config", "user.email", "student@example.com", cwd=workspace)
    return remote, seed, workspace, assigned_head


def assignment_event(head_sha: str):
    return ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:restart",
        payload={
            "head_ref": "student/candidate",
            "head_sha": head_sha,
        },
    )


def test_reconciliation_preserves_unpushed_commits_and_dirty_files(
    tmp_path: Path,
):
    _remote, _seed, workspace, assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("candidate\n")
    git("commit", "-am", "candidate", cwd=workspace)
    local_head = git("rev-parse", "HEAD", cwd=workspace)
    (workspace / "notes.txt").write_text("dirty but recoverable\n")

    StudentWorkspaceReconciler(workspace)((assignment_event(assigned_head),))

    assert git("rev-parse", "HEAD", cwd=workspace) == local_head
    assert (workspace / "notes.txt").read_text() == "dirty but recoverable\n"


def test_reconciliation_rejects_a_remote_head_newer_than_the_assignment(
    tmp_path: Path,
):
    _remote, seed, workspace, assigned_head = assigned_workspace(tmp_path)
    (seed / "program.py").write_text("moved after assignment\n")
    git("commit", "-am", "move assignment branch", cwd=seed)
    git("push", "origin", "student/candidate", cwd=seed)

    with pytest.raises(RuntimeError, match="assignment head moved"):
        StudentWorkspaceReconciler(workspace)((assignment_event(assigned_head),))

    assert git("rev-parse", "HEAD", cwd=workspace) == assigned_head


def test_reconciliation_surfaces_and_preserves_a_diverged_active_branch(
    tmp_path: Path,
):
    _remote, seed, workspace, _assigned_head = assigned_workspace(tmp_path)
    (workspace / "program.py").write_text("rebased experiment\n")
    git("commit", "-am", "local experiment", cwd=workspace)
    local_head = git("rev-parse", "HEAD", cwd=workspace)
    (workspace / "notes.txt").write_text("dirty measurements\n")

    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("new advisor base\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace assignment base", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)

    with pytest.raises(WorkspaceDivergence) as raised:
        StudentWorkspaceReconciler(workspace)((assignment_event(expected_head),))

    assert raised.value.event.kind == "workspace_diverged"
    assert raised.value.event.payload["preserved_local_head"] == local_head
    assert git("rev-parse", "HEAD", cwd=workspace) == local_head
    assert (workspace / "notes.txt").read_text() == "dirty measurements\n"


def test_reconciliation_surfaces_divergence_when_assignment_is_not_checked_out(
    tmp_path: Path,
):
    _remote, seed, workspace, _assigned_head = assigned_workspace(tmp_path)
    git("checkout", "-b", "other-work", cwd=workspace)
    git("checkout", "--orphan", "replacement", cwd=seed)
    git("rm", "-f", "program.py", cwd=seed)
    (seed / "program.py").write_text("new advisor base\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "replace assignment base", cwd=seed)
    git("push", "--force", "origin", "HEAD:student/candidate", cwd=seed)
    expected_head = git("rev-parse", "HEAD", cwd=seed)

    with pytest.raises(WorkspaceDivergence) as raised:
        StudentWorkspaceReconciler(workspace)((assignment_event(expected_head),))

    assert raised.value.event.payload["current_branch"] == "other-work"
    assert git("branch", "--show-current", cwd=workspace) == "other-work"
