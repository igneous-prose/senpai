import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from senpai_agent.git_refs import fetch_github_refs, sync_github_branches


REAL_RUN = subprocess.run
OBJECT_ID = re.compile(r"[0-9a-fA-F]{40}\Z")


def git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def pack_downloader(repo: Path) -> Callable[..., dict[str, str]]:
    def download(
        *,
        sources: Sequence[str],
        destination: Path,
        object_exists: Callable[[str], bool],
        **_kwargs,
    ) -> dict[str, str]:
        resolved = {
            source: (
                source.lower()
                if OBJECT_ID.fullmatch(source)
                else git("rev-parse", source, cwd=repo)
            )
            for source in sources
        }
        if not all(object_exists(object_id) for object_id in resolved.values()):
            revisions = "".join(f"{object_id}\n" for object_id in resolved.values())
            pack = REAL_RUN(
                ["git", "pack-objects", "--stdout", "--revs"],
                cwd=repo,
                check=True,
                input=revisions.encode(),
                capture_output=True,
            ).stdout
            destination.write_bytes(pack)
        return resolved

    return download


def repositories(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    workspace = tmp_path / "workspace"
    git("init", "--bare", str(remote))
    git("init", str(seed))
    git("config", "user.name", "test", cwd=seed)
    git("config", "user.email", "test@example.com", cwd=seed)
    (seed / "program.py").write_text("base\n")
    git("add", "program.py", cwd=seed)
    git("commit", "-m", "base", cwd=seed)
    git("branch", "-M", "research", cwd=seed)
    git("remote", "add", "origin", str(remote), cwd=seed)
    git("push", "origin", "research", cwd=seed)
    git(
        "clone",
        "--no-local",
        "--single-branch",
        "--branch",
        "research",
        str(remote),
        str(workspace),
    )
    git("switch", "-c", "student/candidate", cwd=seed)
    (seed / "candidate.py").write_text("candidate\n")
    git("add", "candidate.py", cwd=seed)
    git("commit", "-m", "candidate", cwd=seed)
    git("push", "origin", "student/candidate", cwd=seed)
    candidate = git("rev-parse", "student/candidate", cwd=seed)
    missing = subprocess.run(
        ["git", "cat-file", "-e", f"{candidate}^{{commit}}"],
        cwd=workspace,
        capture_output=True,
    )
    assert missing.returncode != 0
    return remote, seed, workspace


def test_sync_hydrates_refs_without_changing_the_worktree_or_exposing_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _remote, seed, workspace = repositories(tmp_path)
    (workspace / "notes.txt").write_text("preserve\n")
    before_head = git("rev-parse", "HEAD", cwd=workspace)
    before_status = git("status", "--porcelain", cwd=workspace)
    marker = tmp_path / "alternate-command-ran"
    alternate = tmp_path / "alternate.git"
    git("init", "--bare", str(alternate))
    alternates = workspace / ".git" / "objects" / "info" / "alternates"
    alternates.write_text(f"{alternate / 'objects'}\n")
    git("config", "core.alternateRefsCommand", f"touch {marker}", cwd=workspace)
    git(
        "symbolic-ref",
        "refs/remotes/origin/student/candidate",
        "refs/heads/research",
        cwd=workspace,
    )
    monkeypatch.setattr(
        "senpai_agent.git_refs.download_github_pack",
        pack_downloader(seed),
    )

    def guarded_run(command, **kwargs):
        environment = kwargs.get("env", {})
        values = [str(value) for value in (*command, *environment.values())]
        assert not any("typed-token" in value for value in values)
        assert not any("github.com" in value for value in values)
        assert "GIT_ASKPASS" not in environment
        assert "SENPAI_GITHUB_TOKEN_FD" not in environment
        return REAL_RUN(command, **kwargs)

    monkeypatch.setattr("senpai_agent.git_refs.subprocess.run", guarded_run)

    refs = sync_github_branches(
        workspace,
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        branches=("research", "student/candidate"),
    )

    assert not marker.exists()
    git("config", "--unset", "core.alternateRefsCommand", cwd=workspace)
    alternates.unlink()
    assert refs == {
        "research": git("rev-parse", "research", cwd=seed),
        "student/candidate": git("rev-parse", "student/candidate", cwd=seed),
    }
    assert git("rev-parse", "HEAD", cwd=workspace) == before_head
    assert git("status", "--porcelain", cwd=workspace) == before_status
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/student/candidate"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    assert symbolic.returncode != 0
    common = Path(
        git(
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
            cwd=workspace,
        )
    )
    assert not (common / "FETCH_HEAD").exists()
    assert not tuple((common / "objects" / "pack").glob("*.keep"))


def test_sync_uses_the_common_object_store_for_a_linked_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _remote, seed, workspace = repositories(tmp_path)
    linked = tmp_path / "linked"
    git("worktree", "add", "-b", "linked-work", str(linked), cwd=workspace)
    (linked / "notes.txt").write_text("preserve\n")
    before_head = git("rev-parse", "HEAD", cwd=linked)
    before_status = git("status", "--porcelain", cwd=linked)
    monkeypatch.setattr(
        "senpai_agent.git_refs.download_github_pack",
        pack_downloader(seed),
    )

    sync_github_branches(
        linked,
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        branches=("student/candidate",),
    )

    common = Path(
        git(
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
            cwd=linked,
        )
    )
    administrative = Path(git("rev-parse", "--absolute-git-dir", cwd=linked))
    candidate = git("rev-parse", "student/candidate", cwd=seed)
    assert common != administrative
    assert (common / "senpai-ref-sync.lock").is_file()
    assert not (administrative / "senpai-ref-sync.lock").exists()
    assert not (administrative / "objects").exists()
    assert git(
        "rev-parse", "refs/remotes/origin/student/candidate", cwd=workspace
    ) == candidate
    assert git("cat-file", "-e", f"{candidate}^{{commit}}", cwd=workspace) == ""
    assert git("rev-parse", "HEAD", cwd=linked) == before_head
    assert git("status", "--porcelain", cwd=linked) == before_status


def test_invalid_pack_leaves_refs_and_the_object_store_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _remote, seed, workspace = repositories(tmp_path)
    common = Path(
        git(
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
            cwd=workspace,
        )
    )
    before = tuple(sorted(path.name for path in (common / "objects" / "pack").iterdir()))

    def corrupt_download(*, sources, destination, **_kwargs):
        destination.write_bytes(b"PACK-invalid")
        return {source: git("rev-parse", source, cwd=seed) for source in sources}

    monkeypatch.setattr(
        "senpai_agent.git_refs.download_github_pack",
        corrupt_download,
    )

    with pytest.raises(RuntimeError, match="index-pack"):
        sync_github_branches(
            workspace,
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            branches=("student/candidate",),
        )

    reference = subprocess.run(
        ["git", "show-ref", "--verify", "refs/remotes/origin/student/candidate"],
        cwd=workspace,
        capture_output=True,
    )
    after = tuple(sorted(path.name for path in (common / "objects" / "pack").iterdir()))
    assert reference.returncode != 0
    assert after == before


def test_successful_retry_removes_a_keep_left_by_a_failed_ref_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _remote, seed, workspace = repositories(tmp_path)
    common = Path(
        git(
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
            cwd=workspace,
        )
    )
    monkeypatch.setattr(
        "senpai_agent.git_refs.download_github_pack",
        pack_downloader(seed),
    )
    from senpai_agent import git_refs

    update_refs = git_refs._update_refs
    attempts = 0

    def fail_once(git_directory, refs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated ref transaction failure")
        update_refs(git_directory, refs)

    monkeypatch.setattr("senpai_agent.git_refs._update_refs", fail_once)

    with pytest.raises(RuntimeError, match="simulated"):
        sync_github_branches(
            workspace,
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            branches=("student/candidate",),
        )

    pack_directory = common / "objects" / "pack"
    keeps = tuple(pack_directory.glob("pack-*.keep"))
    assert len(keeps) == 1
    assert keeps[0].read_bytes() == b"senpai-ref-sync\n"

    sync_github_branches(
        workspace,
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        branches=("student/candidate",),
    )

    assert attempts == 2
    assert not tuple(pack_directory.glob("pack-*.keep"))
    assert git(
        "rev-parse", "refs/remotes/origin/student/candidate", cwd=workspace
    ) == git("rev-parse", "student/candidate", cwd=seed)


@pytest.mark.parametrize(
    ("source", "destination"),
    (
        ("refs/tags/release", "refs/remotes/origin/release"),
        ("refs/heads/research", "refs/heads/research"),
    ),
)
def test_fetch_rejects_refs_outside_managed_namespaces(
    tmp_path: Path,
    source: str,
    destination: str,
):
    git("init", str(tmp_path))

    with pytest.raises(ValueError):
        fetch_github_refs(
            tmp_path,
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            refs=((source, destination),),
        )


def test_fetch_rejects_duplicate_destinations(tmp_path: Path):
    git("init", str(tmp_path))

    with pytest.raises(ValueError, match="unique"):
        fetch_github_refs(
            tmp_path,
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            refs=(
                ("refs/heads/one", "refs/remotes/origin/shared"),
                ("refs/heads/two", "refs/remotes/origin/shared"),
            ),
        )


@pytest.mark.parametrize(
    "repo",
    ("owner", "owner/repo/extra", "owner/repo?redirect=attacker", "../repo"),
)
def test_fetch_rejects_unsafe_repository_names(tmp_path: Path, repo: str):
    git("init", str(tmp_path))

    with pytest.raises(ValueError, match="safe owner/name"):
        fetch_github_refs(
            tmp_path,
            repo=repo,
            token=SecretStr("typed-token"),
            refs=(("refs/heads/main", "refs/remotes/origin/main"),),
        )
