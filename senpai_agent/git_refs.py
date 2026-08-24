"""Credential-contained hydration of GitHub branch refs."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.github.upload_pack import download_github_pack


_OBJECT_ID = re.compile(r"[0-9a-fA-F]{40}\Z")
_REPOSITORY = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9_.-]{1,100}\Z"
)
_MAX_PACK_BYTES = 8 * 1024 * 1024 * 1024
_SOURCE_REF_PREFIX = "refs/heads/"
_DESTINATION_REF_PREFIXES = (
    "refs/remotes/origin/",
    "refs/senpai/assignment/",
)
_GIT_EXECUTABLE = str(
    Path("/usr/bin/git")
    if Path("/usr/bin/git").is_file()
    else Path(shutil.which("git") or "git")
)


def sync_github_branches(
    workspace: Path,
    *,
    repo: str,
    token: SecretStr,
    branches: Sequence[str],
) -> dict[str, str]:
    """Fetch named GitHub branches into credential-free remote-tracking refs."""

    workspace = Path(workspace).resolve()
    unique_branches = tuple(dict.fromkeys(branch.strip() for branch in branches))
    if not unique_branches or any(not branch for branch in unique_branches):
        raise ValueError("branches must contain at least one non-empty branch")
    for branch in unique_branches:
        _git(workspace, "check-ref-format", "--branch", branch)

    destinations = {
        branch: f"refs/remotes/origin/{branch}" for branch in unique_branches
    }
    fetched = fetch_github_refs(
        workspace,
        repo=repo,
        token=token,
        refs=tuple(
            (f"refs/heads/{branch}", destination)
            for branch, destination in destinations.items()
        ),
    )
    return {branch: fetched[destination] for branch, destination in destinations.items()}


def fetch_github_refs(
    workspace: Path,
    *,
    repo: str,
    token: SecretStr,
    refs: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Fetch exact refs through controller-owned Git smart HTTP."""

    workspace = Path(workspace).resolve()
    _validate_repo(repo)
    _validate_token(token)
    if not refs:
        raise ValueError("at least one Git ref is required")
    destinations = [destination for _, destination in refs]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Git destinations must be unique")
    for source, destination in refs:
        if _OBJECT_ID.fullmatch(source) is None:
            if not source.startswith(_SOURCE_REF_PREFIX):
                raise ValueError("GitHub sources must be branch refs or object IDs")
            _git(workspace, "check-ref-format", source)
        if not destination.startswith(_DESTINATION_REF_PREFIXES):
            raise ValueError("Git destinations must use a Senpai-managed ref namespace")
        _git(workspace, "check-ref-format", destination)

    with _locked_git_directory(workspace) as git_directory:
        with tempfile.TemporaryDirectory(prefix="senpai-git-fetch-") as directory:
            temporary = Path(directory)
            incoming_pack = temporary / "incoming.pack"
            resolved = download_github_pack(
                repo=repo,
                token=token,
                sources=tuple(source for source, _ in refs),
                destination=incoming_pack,
                object_exists=lambda object_id: _object_exists(
                    git_directory, object_id
                ),
            )
            fetched = {
                destination: resolved[source] for source, destination in refs
            }
            if incoming_pack.exists():
                staging = temporary / "staging.git"
                _index_pack(incoming_pack, staging)
                for object_id in fetched.values():
                    if not _object_exists(staging, object_id, commit=True):
                        raise RuntimeError(
                            f"GitHub pack omitted commit {object_id}"
                        )
                _import_pack(staging / "objects", git_directory / "objects")
            for object_id in fetched.values():
                if not _object_exists(git_directory, object_id, commit=True):
                    raise RuntimeError(
                        f"GitHub object {object_id} is not an available commit"
                    )
            _update_refs(git_directory, fetched)
            _remove_senpai_keep_files(git_directory / "objects" / "pack")
            return fetched


def _git(workspace: Path, *arguments: str) -> str:
    return _run_git(
        workspace,
        *arguments,
        environment=_isolated_git_environment(),
    ).stdout.strip()


def _run_git(
    workspace: Path,
    *arguments: str,
    environment: dict[str, str],
    timeout: int = 30,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [_GIT_EXECUTABLE, *arguments]
    completed = subprocess.run(
        command,
        cwd=workspace,
        check=False,
        text=True,
        input=input_text,
        capture_output=True,
        timeout=timeout,
        env=environment,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments[:2])} failed: {detail[:1000]}"
        )
    return completed


def _validate_repo(repo: str) -> None:
    if _REPOSITORY.fullmatch(repo) is None:
        raise ValueError("repo must use a safe owner/name form")


def _validate_token(token: SecretStr) -> None:
    if not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if not token.get_secret_value().strip():
        raise ValueError("token must not be empty")


def _isolated_git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def _ref_update_environment(git_directory: Path) -> dict[str, str]:
    environment = _isolated_git_environment()
    configuration = (
        ("core.hooksPath", os.devnull),
        ("core.alternateRefsCommand", ""),
        ("core.logAllRefUpdates", "false"),
    )
    environment.update(
        {
            "GIT_DIR": str(git_directory),
            "GIT_CONFIG_COUNT": str(len(configuration)),
        }
    )
    for index, (key, value) in enumerate(configuration):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


@contextmanager
def _locked_git_directory(workspace: Path) -> Iterator[Path]:
    git_directory = Path(
        _git(
            workspace,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(git_directory / "senpai-ref-sync.lock", flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield git_directory
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _object_exists(
    git_directory: Path,
    object_id: str,
    *,
    commit: bool = False,
) -> bool:
    suffix = "^{commit}" if commit else "^{object}"
    completed = _run_git(
        git_directory.parent,
        "cat-file",
        "-e",
        f"{object_id}{suffix}",
        environment=_ref_update_environment(git_directory),
        check=False,
    )
    return completed.returncode == 0


def _index_pack(pack: Path, staging: Path) -> str:
    _run_git(
        pack.parent,
        "init",
        "--bare",
        str(staging),
        environment=_isolated_git_environment(),
    )
    environment = _ref_update_environment(staging)
    with pack.open("rb") as stream:
        completed = subprocess.run(
            [
                _GIT_EXECUTABLE,
                "index-pack",
                "--stdin",
                "--strict",
                "--check-self-contained-and-connected",
                f"--max-input-size={_MAX_PACK_BYTES}",
                "--threads=1",
                "--no-rev-index",
                "--keep=senpai-ref-sync",
            ],
            cwd=pack.parent,
            check=False,
            stdin=stream,
            capture_output=True,
            timeout=300,
            env=environment,
            close_fds=True,
        )
    stdout = completed.stdout.decode(errors="replace").strip()
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git index-pack failed: {(stderr or stdout)[:1000]}")
    prefix, separator, pack_hash = stdout.partition("\t")
    if (
        prefix not in {"keep", "pack"}
        or not separator
        or _OBJECT_ID.fullmatch(pack_hash) is None
    ):
        raise RuntimeError("git index-pack returned an invalid pack ID")
    return pack_hash


def _import_pack(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    pack_source = source / "pack"
    pack_destination = destination / "pack"
    pack_destination.mkdir(exist_ok=True)
    suffix_order = (".keep", ".pack", ".idx", ".rev")
    for suffix in suffix_order:
        for path in sorted(pack_source.glob(f"*{suffix}")):
            _copy_object(path, pack_destination / path.name)


def _copy_object(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.senpai-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(source.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_senpai_keep_files(pack_directory: Path) -> None:
    for path in pack_directory.glob("pack-*.keep"):
        try:
            if path.read_bytes() == b"senpai-ref-sync\n":
                path.unlink()
        except FileNotFoundError:
            continue


def _update_refs(git_directory: Path, refs: dict[str, str]) -> None:
    transaction = "".join(
        f"update {ref}\0{object_id}\0\0" for ref, object_id in refs.items()
    )
    _run_git(
        git_directory.parent,
        "update-ref",
        "--no-deref",
        "--stdin",
        "-z",
        environment=_ref_update_environment(git_directory),
        input_text=transaction,
    )
