import subprocess
from pathlib import Path

import pytest

from senpai_agent.program_context import (
    load_program_system_prompt_snapshot,
    normalize_program_path,
    pinned_program_system_prompt,
    program_system_prompt_sha256,
    snapshot_program_system_prompt,
)
from test_agent_markdown import HTML_HEADER


def committed_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    workspace = tmp_path / "target"
    workspace.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=workspace, check=True)
    for relative_path, content in files.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(("git", "add", "."), cwd=workspace, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Senpai Test",
            "-c",
            "user.email=senpai@example.com",
            "commit",
            "-qm",
            "Add target files",
        ),
        cwd=workspace,
        check=True,
    )
    return workspace


@pytest.mark.parametrize(
    "path",
    [
        "/program.md",
        "../program.md",
        "senpai/../program.md",
        "./senpai/program.md",
        "senpai//program.md",
        "senpai/PROGRAM.md",
    ],
)
def test_program_path_must_be_normalized_and_repo_relative(path: str):
    with pytest.raises(ValueError, match="target-repository-relative"):
        normalize_program_path(path)


def test_blank_program_path_discovers_root_program(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {"program.md": "Root policy."},
    )

    snapshot = pinned_program_system_prompt(workspace, "", tmp_path / "state")

    assert snapshot.program_path == "program.md"
    assert snapshot.prompt.endswith("Root policy.")


def test_blank_program_path_lists_every_root_and_nested_match(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {
            "program.md": "Root policy.",
            "alpha/program.md": "Alpha policy.",
            "beta/program.md": "Beta policy.",
        },
    )

    with pytest.raises(
        RuntimeError,
        match=r"alpha/program\.md, beta/program\.md, program\.md",
    ) as error:
        pinned_program_system_prompt(workspace, "", tmp_path / "state")

    assert "Only one may exist when program_path is blank" in str(error.value)


def test_explicit_program_path_selects_one_of_multiple_matches(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {
            "program.md": "Root policy.",
            "senpai/program.md": "Nested policy.",
        },
    )

    snapshot = pinned_program_system_prompt(
        workspace,
        "senpai/program.md",
        tmp_path / "state",
    )

    assert snapshot.program_path == "senpai/program.md"
    assert snapshot.prompt.endswith("Nested policy.")


def test_blank_program_path_discovers_one_level_program(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {"senpai/program.md": "Nested policy."},
    )

    snapshot = pinned_program_system_prompt(workspace, "", tmp_path / "state")

    assert snapshot.program_path == "senpai/program.md"
    assert snapshot.prompt.startswith("## program.md - senpai/program.md\n\n")


def test_blank_program_path_does_not_search_deeper_than_one_level(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {"configs/senpai/program.md": "Too deep."},
    )

    with pytest.raises(
        RuntimeError,
        match=r"searched program\.md and \*/program\.md",
    ):
        pinned_program_system_prompt(workspace, "", tmp_path / "state")


def test_blank_program_path_rejects_ambiguous_one_level_matches(tmp_path: Path):
    workspace = committed_repo(
        tmp_path,
        {
            "alpha/program.md": "Alpha policy.",
            "beta/program.md": "Beta policy.",
        },
    )

    with pytest.raises(
        RuntimeError,
        match=r"alpha/program\.md, beta/program\.md",
    ) as error:
        pinned_program_system_prompt(workspace, "", tmp_path / "state")

    assert "--program_path" in str(error.value)


def test_program_snapshot_has_a_source_header_digest_and_private_mode(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    program = workspace / "senpai" / "program.md"
    program.parent.mkdir(parents=True)
    program.write_text(HTML_HEADER + "# Research policy\n\nWin safely.\n")

    snapshot = snapshot_program_system_prompt(
        workspace,
        "senpai/program.md",
        tmp_path / "state",
    )

    assert snapshot.prompt == (
        "## program.md - senpai/program.md\n\n"
        "# Research policy\n\nWin safely."
    )
    assert snapshot.sha256 == program_system_prompt_sha256(snapshot.prompt)
    assert snapshot.path == (
        tmp_path / "state" / "program-context" / f"{snapshot.sha256}.md"
    )
    assert snapshot.path.read_text() == snapshot.prompt
    assert snapshot.path.stat().st_mode & 0o777 == 0o600
    assert "SPDX-" not in snapshot.prompt


def test_program_path_cannot_escape_through_a_symlink(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    outside = tmp_path / "program.md"
    outside.write_text("outside policy")
    (workspace / "program.md").symlink_to(outside)

    with pytest.raises(RuntimeError, match="beneath the target workspace"):
        snapshot_program_system_prompt(
            workspace,
            "program.md",
            tmp_path / "state",
        )


def test_configured_program_must_exist(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="does not exist: senpai/program.md"):
        snapshot_program_system_prompt(
            workspace,
            "senpai/program.md",
            tmp_path / "state",
        )


def test_loaded_snapshot_is_stable_when_the_target_program_changes(tmp_path: Path):
    workspace = tmp_path / "target"
    program = workspace / "senpai" / "program.md"
    program.parent.mkdir(parents=True)
    program.write_text("Initial policy.")
    snapshot = snapshot_program_system_prompt(
        workspace,
        "senpai/program.md",
        tmp_path / "state",
    )
    program.write_text("Changed policy.")

    loaded = load_program_system_prompt_snapshot(
        "senpai/program.md",
        snapshot.path,
        snapshot.sha256,
    )

    assert loaded.prompt == snapshot.prompt
    assert "Changed policy" not in loaded.prompt


def test_snapshot_tampering_is_rejected_by_digest(tmp_path: Path):
    workspace = tmp_path / "target"
    program = workspace / "program.md"
    workspace.mkdir()
    program.write_text("Trusted policy.")
    snapshot = snapshot_program_system_prompt(
        workspace,
        "program.md",
        tmp_path / "state",
    )
    snapshot.path.write_text("## program.md - program.md\n\nReplaced policy.")

    with pytest.raises(RuntimeError, match="digest does not match"):
        load_program_system_prompt_snapshot(
            "program.md",
            snapshot.path,
            snapshot.sha256,
        )


def test_truncated_content_addressed_snapshot_is_not_reused(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Trusted policy.")
    snapshot = snapshot_program_system_prompt(
        workspace,
        "program.md",
        tmp_path / "state",
    )
    snapshot.path.write_text("## program.md - program.md\n\n")

    with pytest.raises(RuntimeError, match="content collision"):
        snapshot_program_system_prompt(
            workspace,
            "program.md",
            tmp_path / "state",
        )


def test_committed_snapshot_ignores_mutable_worktree_content(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    program = workspace / "program.md"
    program.write_text("Committed policy.")
    subprocess.run(("git", "init", "-q"), cwd=workspace, check=True)
    subprocess.run(("git", "add", "program.md"), cwd=workspace, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Senpai Test",
            "-c",
            "user.email=senpai@example.com",
            "commit",
            "-qm",
            "Add programme",
        ),
        cwd=workspace,
        check=True,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    program.write_text("Uncommitted replacement.")

    snapshot = snapshot_program_system_prompt(
        workspace,
        "program.md",
        tmp_path / "state",
        committed=True,
    )

    assert snapshot.source_commit == commit
    assert snapshot.prompt.endswith("Committed policy.")
    assert "Uncommitted replacement" not in snapshot.prompt


def test_missing_generation_manifest_fails_closed_for_existing_conversation(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "advisor-conversation-id").write_text(
        "55555555-5555-5555-5555-555555555555"
    )

    with pytest.raises(RuntimeError, match="conversation state already exists"):
        pinned_program_system_prompt(workspace, "program.md", state_dir)
