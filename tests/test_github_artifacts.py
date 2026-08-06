import os
import time

import pytest

from senpai_agent.github import get_prs

from github_retrieval_support import (
    REPO,
    FakeGitHubReader,
    install_fake_github,
    pull_request,
)


def test_large_results_use_a_stable_private_artifact(monkeypatch, tmp_path):
    fake = FakeGitHubReader({number: pull_request(number) for number in range(1, 7)})
    install_fake_github(monkeypatch, fake)
    target_workspace = tmp_path / "target"
    target_workspace.mkdir()
    artifact_dir = tmp_path / "runtime-state" / "github"

    first = get_prs(
        REPO,
        numbers=[6, 5, 4, 3, 2, 1],
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )
    repeated = get_prs(
        REPO,
        numbers=[1, 2, 3, 4, 5, 6],
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )

    assert first.markdown is None
    assert first.path == repeated.path
    assert first.path is not None and first.path.is_file()
    assert not first.path.is_relative_to(target_workspace)
    assert first.path.stat().st_mode & 0o777 == 0o600
    assert "Complete PR body 6" in first.path.read_text()
    assert sorted(artifact_dir.iterdir()) == [first.path]

    fake.pulls[6] = pull_request(6, head="new-head-6")
    changed = get_prs(
        REPO,
        numbers=range(1, 7),
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )

    assert changed.path != first.path


def test_artifacts_cannot_be_written_inside_the_target_workspace(
    monkeypatch,
    tmp_path,
):
    fake = FakeGitHubReader({number: pull_request(number) for number in range(1, 7)})
    install_fake_github(monkeypatch, fake)
    target_workspace = tmp_path / "target"
    target_workspace.mkdir()

    with pytest.raises(ValueError):
        get_prs(
            REPO,
            numbers=range(1, 7),
            artifact_dir=target_workspace / ".senpai" / "github",
            target_workspace=target_workspace,
        )


def test_artifact_cleanup_removes_only_expired_generated_markdown(
    monkeypatch,
    tmp_path,
):
    fake = FakeGitHubReader({number: pull_request(number) for number in range(1, 7)})
    install_fake_github(monkeypatch, fake)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    expired = artifact_dir / "pull-requests-expired.md"
    fresh = artifact_dir / "pull-requests-fresh.md"
    unrelated = artifact_dir / "operator-notes.md"
    expired.write_text("old generated bundle")
    fresh.write_text("current generated bundle")
    unrelated.write_text("keep me")
    old = time.time() - (25 * 60 * 60)
    os.utime(expired, (old, old))
    os.utime(unrelated, (old, old))

    get_prs(
        REPO,
        numbers=range(1, 7),
        artifact_dir=artifact_dir,
        target_workspace=tmp_path / "target",
    )

    assert not expired.exists()
    assert fresh.read_text() == "current generated bundle"
    assert unrelated.read_text() == "keep me"
