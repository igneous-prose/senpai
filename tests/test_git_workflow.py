import base64
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import senpai_agent.git_workflow as git_workflow
from senpai_agent.git_workflow import (
    GitWorkflowPreconditionError,
    create_assignment_branch,
    push_assignment_branch,
)


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    workspace = tmp_path / "workspace"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(workspace))
    git(workspace, "config", "user.name", "Student")
    git(workspace, "config", "user.email", "student@example.com")
    (workspace / "model.py").write_text("baseline = 1\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "baseline")
    git(workspace, "branch", "-M", "experiment-7")
    git(workspace, "remote", "add", "origin", str(remote))
    git(workspace, "push", "-u", "origin", "experiment-7")
    return workspace, remote, git(workspace, "rev-parse", "HEAD")


def test_push_assignment_branch_is_guarded_verified_and_idempotent(tmp_path: Path):
    workspace, remote, previous_sha = repository(tmp_path)
    (workspace / "model.py").write_text("baseline = 2\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "candidate")
    candidate_sha = git(workspace, "rev-parse", "HEAD")

    first = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
    )
    second = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
    )

    assert first.changed is True
    assert first.head_sha == candidate_sha
    assert second.changed is False
    assert second.head_sha == candidate_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == candidate_sha


@pytest.mark.parametrize(
    ("role", "branch"),
    [
        ("advisor", "experiment-7"),
        ("student", "student-one/experiment-7"),
    ],
)
def test_push_assignment_branch_uses_a_validated_ref_accepted_by_the_role_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    branch: str,
):
    workspace, remote, previous_sha = repository(tmp_path)
    if branch != "experiment-7":
        git(workspace, "branch", "-m", branch)
        git(workspace, "push", "-u", "origin", branch)
    guard = (
        Path(__file__).parents[1] / "plugins" / "senpai" / "scripts" / "git-guard.sh"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_senpai_target_git_guard "$2"',
            "install-guard",
            str(guard),
            str(workspace),
        ],
        check=True,
    )
    monkeypatch.setenv("SENPAI_ROLE", role)
    monkeypatch.setenv("ADVISOR_BRANCH", "experiment-7")
    monkeypatch.setenv("STUDENT_NAME", "student-one")
    monkeypatch.setenv("STUDENT_NAMES", "student-one")
    (workspace / "model.py").write_text("baseline = 2\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "candidate")

    pushed = push_assignment_branch(
        workspace,
        branch=branch,
        expected_remote_sha=previous_sha,
    )

    assert pushed.changed is True
    assert git(remote, "rev-parse", f"refs/heads/{branch}") == pushed.head_sha


def test_push_assignment_branch_publishes_only_the_validated_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, remote, previous_sha = repository(tmp_path)
    (workspace / "model.py").write_text("baseline = 2\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "validated candidate")
    validated_sha = git(workspace, "rev-parse", "HEAD")
    real_remote_head = git_workflow._remote_head
    remote_reads = 0

    def advance_branch_after_remote_read(*args, **kwargs):
        nonlocal remote_reads
        remote_sha = real_remote_head(*args, **kwargs)
        remote_reads += 1
        if remote_reads == 1:
            (workspace / "model.py").write_text("baseline = 3\n")
            git(workspace, "add", "model.py")
            git(workspace, "commit", "-m", "concurrent unvalidated candidate")
        return remote_sha

    monkeypatch.setattr(
        git_workflow,
        "_remote_head",
        advance_branch_after_remote_read,
    )

    pushed = push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        expected_local_sha=validated_sha,
    )

    assert pushed.head_sha == validated_sha
    assert git(workspace, "rev-parse", "HEAD") != validated_sha
    assert git(remote, "rev-parse", "refs/heads/experiment-7") == validated_sha


def test_push_assignment_branch_rejects_dirty_or_diverged_worktree(
    tmp_path: Path,
):
    workspace, remote, previous_sha = repository(tmp_path)
    (workspace / "untracked.txt").write_text("dirty")

    with pytest.raises(GitWorkflowPreconditionError, match="clean"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=previous_sha,
        )

    (workspace / "untracked.txt").unlink()
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    git(other, "config", "user.name", "Other")
    git(other, "config", "user.email", "other@example.com")
    git(other, "checkout", "experiment-7")
    (other / "other.py").write_text("remote = True\n")
    git(other, "add", "other.py")
    git(other, "commit", "-m", "remote update")
    git(other, "push", "origin", "experiment-7")
    (workspace / "model.py").write_text("baseline = 3\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "local update")

    with pytest.raises(GitWorkflowPreconditionError, match="remote head"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=previous_sha,
        )


def test_push_assignment_branch_rejects_non_fast_forward_with_current_lease(
    tmp_path: Path,
):
    workspace, remote, _ = repository(tmp_path)
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    git(other, "config", "user.name", "Other")
    git(other, "config", "user.email", "other@example.com")
    git(other, "checkout", "experiment-7")

    (other / "remote.py").write_text("winner = True\n")
    git(other, "add", "remote.py")
    git(other, "commit", "-m", "merge winner")
    git(other, "push", "origin", "experiment-7")
    remote_sha = git(other, "rev-parse", "HEAD")

    (workspace / "notes.md").write_text("stale baseline notes\n")
    git(workspace, "add", "notes.md")
    git(workspace, "commit", "-m", "record result from stale baseline")

    with pytest.raises(GitWorkflowPreconditionError, match="fast-forward"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=remote_sha,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == remote_sha


def test_push_assignment_branch_rejects_the_wrong_local_head_before_publication(
    tmp_path: Path,
):
    workspace, remote, previous_sha = repository(tmp_path)
    (workspace / "model.py").write_text("baseline = 2\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "candidate")

    with pytest.raises(GitWorkflowPreconditionError, match="local head"):
        push_assignment_branch(
            workspace,
            branch="experiment-7",
            expected_remote_sha=previous_sha,
            expected_local_sha="f" * 40,
        )

    assert git(remote, "rev-parse", "refs/heads/experiment-7") == previous_sha


def test_create_assignment_branch_does_not_touch_the_advisor_worktree(
    tmp_path: Path,
):
    workspace, remote, base_sha = repository(tmp_path)
    git(workspace, "branch", "-M", "schmidhuber")
    git(workspace, "push", "origin", "schmidhuber")
    (workspace / "advisor-notes.md").write_text("uncommitted research\n")

    first = create_assignment_branch(
        workspace,
        branch="student-one/lower-lr",
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
        assignment_id="assignment-7",
    )
    second = create_assignment_branch(
        workspace,
        branch="student-one/lower-lr",
        base_branch="schmidhuber",
        expected_base_sha=base_sha,
        assignment_id="assignment-7",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.head_sha == second.head_sha
    assert git(workspace, "branch", "--show-current") == "schmidhuber"
    assert (workspace / "advisor-notes.md").read_text() == "uncommitted research\n"
    assert git(remote, "rev-parse", f"{first.head_sha}^") == base_sha


def test_create_assignment_branch_rejects_stale_base_or_foreign_existing_ref(
    tmp_path: Path,
):
    workspace, _remote, base_sha = repository(tmp_path)
    git(workspace, "branch", "-M", "schmidhuber")
    git(workspace, "push", "origin", "schmidhuber")

    with pytest.raises(GitWorkflowPreconditionError, match="base head"):
        create_assignment_branch(
            workspace,
            branch="student-one/lower-lr",
            base_branch="schmidhuber",
            expected_base_sha="b" * 40,
            assignment_id="assignment-7",
        )

    git(workspace, "checkout", "-b", "student-one/lower-lr")
    (workspace / "foreign.py").write_text("foreign = True\n")
    git(workspace, "add", "foreign.py")
    git(workspace, "commit", "-m", "foreign work")
    git(workspace, "push", "origin", "student-one/lower-lr")

    with pytest.raises(GitWorkflowPreconditionError, match="already exists"):
        create_assignment_branch(
            workspace,
            branch="student-one/lower-lr",
            base_branch="schmidhuber",
            expected_base_sha=base_sha,
            assignment_id="assignment-7",
        )


def test_typed_push_injects_auth_only_into_git_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workspace, _remote, previous_sha = repository(tmp_path)
    (workspace / "model.py").write_text("baseline = 2\n")
    git(workspace, "add", "model.py")
    git(workspace, "commit", "-m", "candidate")
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, str]]] = []

    def recording_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return real_run(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_workflow.subprocess.run", recording_run)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    monkeypatch.setenv("GH_TOKEN", "ambient-gh-token")

    push_assignment_branch(
        workspace,
        branch="experiment-7",
        expected_remote_sha=previous_sha,
        token=SecretStr("typed-write-token"),
    )

    assert calls
    for command, env in calls:
        assert "typed-write-token" not in command
        assert "ambient-write-token" not in env.values()
        assert "ambient-gh-token" not in env.values()
        assert "GITHUB_TOKEN" not in env
        assert "GH_TOKEN" not in env
        if command[1] in {"ls-remote", "fetch", "push"}:
            encoded = env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic ")
            assert base64.b64decode(encoded).decode() == (
                "x-access-token:typed-write-token"
            )
        else:
            assert "GIT_CONFIG_VALUE_0" not in env
