# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Load an optional target-repository program into the system prompt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header

PROGRAM_PATH_ENV = "SENPAI_PROGRAM_PATH"
PROGRAM_SNAPSHOT_ENV = "SENPAI_PROGRAM_SYSTEM_PROMPT_FILE"
PROGRAM_SHA256_ENV = "SENPAI_PROGRAM_SYSTEM_PROMPT_SHA256"
PROGRAM_SOURCE_COMMIT_ENV = "SENPAI_PROGRAM_SOURCE_COMMIT"
PROGRAM_PATH_GUIDANCE = (
    "Set --program_path (or program_path in senpai.yaml) to a committed "
    "target-repository-relative path ending in program.md."
)


@dataclass(frozen=True, slots=True)
class ProgramSystemPromptSnapshot:
    program_path: str = ""
    prompt: str = ""
    path: Path | None = None
    sha256: str = ""
    source_commit: str | None = None


def normalize_program_path(value: str) -> str:
    """Return a normalized target-repository-relative program.md path."""

    if not value:
        return ""
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.name != "program.md"
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise ValueError(
            "must be a normalized target-repository-relative "
            "path ending in program.md"
        )
    return value


def program_system_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def snapshot_program_system_prompt(
    workspace: Path,
    value: str,
    state_dir: Path,
    *,
    committed: bool = False,
) -> ProgramSystemPromptSnapshot:
    """Create one immutable, content-addressed programme snapshot."""

    relative_path = normalize_program_path(value)
    if committed:
        relative_path, body, source_commit = _read_committed_program(
            workspace,
            relative_path,
        )
    else:
        if not relative_path:
            return ProgramSystemPromptSnapshot()
        body = _read_workspace_program(workspace, relative_path)
        source_commit = None
    prompt = _format_program(relative_path, body)
    digest = program_system_prompt_sha256(prompt)
    snapshot_path = state_dir / "program-context" / f"{digest}.md"
    _create_private_file(snapshot_path, prompt, "program.md snapshot")
    return ProgramSystemPromptSnapshot(
        program_path=relative_path,
        prompt=prompt,
        path=snapshot_path,
        sha256=digest,
        source_commit=source_commit,
    )


def load_program_system_prompt_snapshot(
    value: str,
    snapshot_path: Path,
    expected_sha256: str,
) -> ProgramSystemPromptSnapshot:
    """Load a snapshot only when its path header and digest still match."""

    relative_path = normalize_program_path(value)
    if not relative_path:
        raise RuntimeError("program.md snapshot requires a configured program path")
    if not expected_sha256:
        raise RuntimeError("program.md snapshot requires an expected SHA-256 digest")

    prompt = _read_private_file(snapshot_path, "program.md snapshot")
    header = f"## program.md - {relative_path}\n\n"
    if not prompt.startswith(header):
        raise RuntimeError(
            "program.md snapshot does not match configured path: "
            f"{relative_path}"
        )
    digest = program_system_prompt_sha256(prompt)
    if digest != expected_sha256:
        raise RuntimeError(
            "program.md snapshot digest does not match the controller snapshot"
        )
    return ProgramSystemPromptSnapshot(
        program_path=relative_path,
        prompt=prompt,
        path=snapshot_path,
        sha256=digest,
    )


def pinned_program_system_prompt(
    workspace: Path,
    value: str,
    state_dir: Path,
) -> ProgramSystemPromptSnapshot:
    """Load one durable generation, or pin committed text for fresh state."""

    state_dir = state_dir.resolve()
    manifest_path = state_dir / "program-context" / "current.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        return _load_program_manifest(manifest_path, value, state_dir)

    relative_path = normalize_program_path(value)
    if _has_conversation_state(state_dir):
        raise RuntimeError(
            "cannot pin program.md without a generation manifest when "
            "conversation state already exists"
        )

    snapshot = snapshot_program_system_prompt(
        workspace,
        relative_path,
        state_dir,
        committed=True,
    )
    if snapshot.path is None or snapshot.source_commit is None:
        raise RuntimeError("configured program.md did not produce a snapshot")
    manifest = json.dumps(
        {
            "version": 1,
            "program_path": snapshot.program_path,
            "source_commit": snapshot.source_commit,
            "prompt_sha256": snapshot.sha256,
            "snapshot_path": snapshot.path.relative_to(state_dir).as_posix(),
        },
        sort_keys=True,
    )
    _create_private_file(manifest_path, f"{manifest}\n", "programme manifest")
    return _load_program_manifest(manifest_path, value, state_dir)


def _load_program_manifest(
    manifest_path: Path,
    configured_path: str,
    state_dir: Path,
) -> ProgramSystemPromptSnapshot:
    try:
        manifest_text = _read_private_file(manifest_path, "programme manifest")
        manifest = json.loads(manifest_text)
        version = manifest["version"]
        program_path = normalize_program_path(manifest["program_path"])
        source_commit = manifest["source_commit"]
        prompt_sha256 = manifest["prompt_sha256"]
        snapshot_path = manifest["snapshot_path"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid programme manifest: {manifest_path}") from error

    configured_path = normalize_program_path(configured_path)
    if configured_path and configured_path != program_path:
        raise RuntimeError(
            "configured program.md path does not match the pinned generation; "
            "use fresh role state or an explicit generation promotion"
        )
    expected_snapshot_path = f"program-context/{prompt_sha256}.md"
    if (
        version != 1
        or not program_path
        or not _is_hex_identifier(source_commit, lengths={40, 64})
        or not _is_hex_identifier(prompt_sha256, lengths={64})
        or snapshot_path != expected_snapshot_path
    ):
        raise RuntimeError(f"invalid programme manifest: {manifest_path}")
    loaded = load_program_system_prompt_snapshot(
        program_path,
        state_dir / expected_snapshot_path,
        prompt_sha256,
    )
    return ProgramSystemPromptSnapshot(
        program_path=program_path,
        prompt=loaded.prompt,
        path=loaded.path,
        sha256=loaded.sha256,
        source_commit=source_commit,
    )


def _is_hex_identifier(value: object, *, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _has_conversation_state(state_dir: Path) -> bool:
    if any(
        (state_dir / name).exists()
        for name in ("advisor-conversation-id", "student-conversations.json")
    ):
        return True
    if not state_dir.exists():
        return False
    return any(
        child.is_dir() and (child / "events").exists()
        for child in state_dir.iterdir()
        if child.name != "program-context"
    )


def _format_program(relative_path: str, body: str) -> str:
    return f"## program.md - {relative_path}\n\n{strip_spdx_header(body).strip()}"


def _read_workspace_program(workspace: Path, relative_path: str) -> str:
    workspace = workspace.resolve()
    try:
        path = (workspace / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"configured program.md does not exist: {relative_path}"
        ) from error
    if not path.is_relative_to(workspace) or not path.is_file():
        raise RuntimeError(
            f"configured program.md must be a file beneath the target workspace: "
            f"{relative_path}"
        )
    return read_agent_markdown(path)


def _read_committed_program(
    workspace: Path,
    relative_path: str,
) -> tuple[str, str, str]:
    workspace = workspace.resolve()
    commit = _git(workspace, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if not relative_path:
        relative_path = _discover_program_path(workspace, commit)
    entry = _git_bytes(
        workspace,
        "ls-tree",
        "-z",
        commit,
        "--",
        relative_path,
    )
    try:
        metadata, listed_path = entry.removesuffix(b"\0").split(b"\t", 1)
        mode, kind, object_id = metadata.split()
    except ValueError as error:
        raise RuntimeError(
            "configured program.md was not found as a committed regular file "
            f"at target commit {commit}: {relative_path}. {PROGRAM_PATH_GUIDANCE}"
        ) from error
    if (
        listed_path.decode() != relative_path
        or mode not in {b"100644", b"100755"}
        or kind != b"blob"
    ):
        raise RuntimeError(
            "configured program.md was not found as a committed regular file "
            f"at target commit {commit}: {relative_path}. {PROGRAM_PATH_GUIDANCE}"
        )
    body = _git_bytes(workspace, "cat-file", "blob", object_id.decode()).decode()
    return relative_path, body, commit


def _discover_program_path(workspace: Path, commit: str) -> str:
    root = ""
    nested = []
    tree = _git_bytes(workspace, "ls-tree", "-r", "-z", commit)
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, _object_id = metadata.split()
        except ValueError as error:
            raise RuntimeError("could not inspect target repository tree") from error
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            continue
        path = raw_path.decode()
        parts = PurePosixPath(path).parts
        if path == "program.md":
            root = path
        elif len(parts) == 2 and parts[-1] == "program.md":
            nested.append(path)
    if root:
        return root
    if len(nested) == 1:
        return nested[0]
    if nested:
        candidates = ", ".join(sorted(nested))
        raise RuntimeError(
            "found multiple committed program.md candidates one directory below "
            f"the repository root at target commit {commit}: {candidates}. "
            f"{PROGRAM_PATH_GUIDANCE}"
        )
    raise RuntimeError(
        f"could not find a committed program.md at target commit {commit}; "
        "searched program.md and */program.md (exactly one directory below the "
        f"repository root). {PROGRAM_PATH_GUIDANCE}"
    )


def _git(workspace: Path, *arguments: str) -> str:
    return _git_bytes(workspace, *arguments).decode()


def _git_bytes(workspace: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(workspace), *arguments),
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"could not load configured program.md from target commit: {message}"
        ) from error


def _create_private_file(path: Path, content: str, description: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists() or path.is_symlink():
        existing = _read_private_file(path, description)
        if existing != content:
            raise RuntimeError(f"{description} content collision: {path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".program-context-",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = _read_private_file(path, description)
            if existing != content:
                raise RuntimeError(f"{description} content collision: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_file(path: Path, description: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{description} is unavailable: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
        ):
            raise RuntimeError(
                f"{description} must be a private regular file: {path}"
            )
        with os.fdopen(descriptor, encoding="utf-8") as source:
            descriptor = -1
            return source.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
