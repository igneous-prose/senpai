import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_GUARD = ROOT / "plugins" / "senpai" / "scripts" / "git-guard.sh"


@pytest.fixture
def target_pre_push_hook(tmp_path: Path) -> Path:
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; install_senpai_target_git_guard "$2"',
            "install-guard",
            str(GIT_GUARD),
            str(target),
        ],
        check=True,
    )
    return target / ".git" / "hooks" / "pre-push"


def run_pre_push(
    hook: Path,
    *,
    role: str,
    local_ref: str,
    remote_ref: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(hook)],
        input=f"{local_ref} {'a' * 40} {remote_ref} {'b' * 40}\n",
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SENPAI_ROLE": role,
            "ADVISOR_BRANCH": "advisor-track",
            "STUDENT_NAME": "student-one",
            "STUDENT_NAMES": "student-one,student-two",
        },
    )


@pytest.mark.parametrize(
    ("role", "local_ref", "remote_ref"),
    [
        ("advisor", "refs/heads/advisor-track", "refs/heads/advisor-track"),
        (
            "advisor",
            "refs/heads/student-one/try-lr",
            "refs/heads/student-one/try-lr",
        ),
        (
            "student",
            "refs/heads/student-one/try-lr",
            "refs/heads/student-one/try-lr",
        ),
    ],
)
def test_target_git_guard_allows_only_role_owned_branches(
    target_pre_push_hook: Path,
    role: str,
    local_ref: str,
    remote_ref: str,
):
    result = run_pre_push(
        target_pre_push_hook,
        role=role,
        local_ref=local_ref,
        remote_ref=remote_ref,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("role", "local_ref", "remote_ref"),
    [
        ("advisor", "refs/heads/main", "refs/heads/main"),
        ("advisor", "refs/heads/foreign/branch", "refs/heads/foreign/branch"),
        ("advisor", "refs/heads/other", "refs/heads/advisor-track"),
        ("advisor", "a" * 40, "refs/heads/advisor-track"),
        ("student", "refs/heads/main", "refs/heads/main"),
        ("student", "refs/heads/advisor-track", "refs/heads/advisor-track"),
        (
            "student",
            "refs/heads/student-two/try-lr",
            "refs/heads/student-two/try-lr",
        ),
        ("unknown", "refs/heads/student-one/try-lr", "refs/heads/student-one/try-lr"),
    ],
)
def test_target_git_guard_rejects_default_foreign_and_cross_role_branches(
    target_pre_push_hook: Path,
    role: str,
    local_ref: str,
    remote_ref: str,
):
    result = run_pre_push(
        target_pre_push_hook,
        role=role,
        local_ref=local_ref,
        remote_ref=remote_ref,
    )

    assert result.returncode == 2
    assert "refusing" in result.stderr.lower()
