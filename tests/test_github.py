import inspect
import os
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr

from senpai_agent import github
from senpai_agent.github import get_prs

REPO = "acme/widgets"


def pull_request(
    number: int,
    *,
    head: str | None = None,
    body: str | None = None,
) -> dict:
    return {
        "number": number,
        "title": f"Experiment {number}",
        "body": body if body is not None else f"Complete PR body {number}",
        "state": "open",
        "draft": False,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "created_at": f"2026-06-{number:02d}T10:00:00Z",
        "updated_at": f"2026-06-{number:02d}T11:00:00Z",
        "user": {"login": f"student-{number}"},
        "base": {"ref": "advisor", "sha": "base-sha"},
        "head": {"ref": f"experiment-{number}", "sha": head or f"head-{number}"},
    }


def issue_comment(comment_id: int, body: str) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "created_at": f"2026-06-01T10:00:{comment_id:02d}Z",
        "updated_at": f"2026-06-01T10:00:{comment_id:02d}Z",
        "html_url": f"https://github.com/{REPO}/pull/1#issuecomment-{comment_id}",
        "user": {"login": "commenter"},
    }


def review(review_id: int, body: str) -> dict:
    return {
        "id": review_id,
        "body": body,
        "state": "CHANGES_REQUESTED",
        "submitted_at": f"2026-06-02T10:00:{review_id:02d}Z",
        "html_url": f"https://github.com/{REPO}/pull/1#pullrequestreview-{review_id}",
        "user": {"login": "reviewer"},
        "commit_id": "reviewed-head",
    }


def inline_comment(comment_id: int, body: str) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "created_at": f"2026-06-03T10:00:{comment_id:02d}Z",
        "updated_at": f"2026-06-03T10:00:{comment_id:02d}Z",
        "html_url": f"https://github.com/{REPO}/pull/1#discussion_r{comment_id}",
        "user": {"login": "inline-reviewer"},
        "path": "train.py",
        "line": 42,
        "side": "RIGHT",
        "commit_id": "reviewed-head",
        "in_reply_to_id": None,
    }


class FakeGitHubReader:
    """Complete fake for the shared read-only GitHub boundary."""

    def __init__(
        self,
        pulls: dict[int, dict],
        *,
        comments: dict[int, list[list[dict]]] | None = None,
        reviews: dict[int, list[list[dict]]] | None = None,
        inline_comments: dict[int, list[list[dict]]] | None = None,
        search_pages: list[dict] | None = None,
    ):
        self.pulls = pulls
        self.comments = comments or {}
        self.reviews = reviews or {}
        self.inline_comments = inline_comments or {}
        self.search_pages = search_pages or []
        self.search_queries: list[str] = []
        self.tokens: list[SecretStr | None] = []

    def factory(self, token):
        self.tokens.append(token)
        return self

    def get(self, endpoint):
        parsed = urlsplit(endpoint)
        path = parsed.path
        prefix = f"/repos/{REPO}/pulls/"
        if path.startswith(prefix) and path.count("/") == 5:
            return self.pulls[int(path.removeprefix(prefix))]
        raise AssertionError(f"Unexpected non-paginated GitHub endpoint: {endpoint}")

    def pages(self, endpoint):
        parsed = urlsplit(endpoint)
        if parsed.path != "/search/issues":
            raise AssertionError(f"Unexpected GitHub search endpoint: {endpoint}")
        self.search_queries.extend(parse_qs(parsed.query).get("q", []))
        return tuple(self.search_pages)

    def objects(self, endpoint):
        path = urlsplit(endpoint).path
        prefix = f"/repos/{REPO}/pulls/"
        issue_prefix = f"/repos/{REPO}/issues/"
        if path.startswith(issue_prefix) and path.endswith("/comments"):
            number = int(path.removeprefix(issue_prefix).split("/", 1)[0])
            pages = self.comments.get(number, [[]])
        elif path.startswith(prefix) and path.endswith("/reviews"):
            number = int(path.removeprefix(prefix).split("/", 1)[0])
            pages = self.reviews.get(number, [[]])
        elif path.startswith(prefix) and path.endswith("/comments"):
            number = int(path.removeprefix(prefix).split("/", 1)[0])
            pages = self.inline_comments.get(number, [[]])
        else:
            raise AssertionError(f"Unexpected paginated GitHub endpoint: {endpoint}")
        return [item for page in pages for item in page]


def install_fake_github(monkeypatch, fake: FakeGitHubReader) -> None:
    monkeypatch.setattr(github, "GitHubReader", fake.factory)


def test_get_prs_returns_complete_paginated_markdown_in_stable_order(
    monkeypatch, tmp_path
):
    long_body = "body-start\n" + ("untruncated evidence\n" * 2_000) + "body-end"
    fake = FakeGitHubReader(
        {
            1: pull_request(1, body=long_body),
            2: pull_request(2),
        },
        comments={
            1: [
                [issue_comment(1, "first issue comment")],
                [issue_comment(2, "last issue comment from the final page")],
            ]
        },
        reviews={1: [[review(1, "complete review submission")]]},
        inline_comments={1: [[inline_comment(1, "complete inline review comment")]]},
    )
    install_fake_github(monkeypatch, fake)

    result = get_prs(
        REPO,
        numbers=[2, 1, 2],
        artifact_dir=tmp_path / "artifacts",
        target_workspace=tmp_path / "target",
    )

    assert result.path is None
    assert result.markdown is not None
    assert [entry.number for entry in result.manifest] == [1, 2]
    assert result.markdown.index("## PR #1") < result.markdown.index("## PR #2")
    assert "body-start" in result.markdown
    assert "body-end" in result.markdown
    assert "last issue comment from the final page" in result.markdown
    assert "complete review submission" in result.markdown
    assert "complete inline review comment" in result.markdown
    assert "### Issue comments (2)" in result.markdown
    assert "### Review submissions (1)" in result.markdown
    assert "### Inline review comments (1)" in result.markdown


def test_get_prs_unifies_explicit_search_and_date_range_selectors(
    monkeypatch, tmp_path
):
    fake = FakeGitHubReader(
        {
            3: pull_request(3),
            7: pull_request(7),
        },
        search_pages=[
            {"items": [{"number": 7}]},
            {"items": [{"number": 3}, {"number": 7}]},
        ],
    )
    install_fake_github(monkeypatch, fake)

    result = get_prs(
        REPO,
        numbers=[7],
        date_range=("2026-06-01", "2026-06-30"),
        search="label:status:review",
        artifact_dir=tmp_path / "artifacts",
        target_workspace=tmp_path / "target",
    )

    assert [entry.number for entry in result.manifest] == [3, 7]
    assert len(fake.search_queries) == 1
    assert "repo:acme/widgets" in fake.search_queries[0]
    assert "is:pr" in fake.search_queries[0]
    assert "label:status:review" in fake.search_queries[0]
    assert "created:2026-06-01..2026-06-30" in fake.search_queries[0]


def test_get_prs_spills_one_stable_markdown_artifact_outside_target_workspace(
    monkeypatch, tmp_path
):
    pulls = {number: pull_request(number) for number in range(1, 7)}
    fake = FakeGitHubReader(pulls)
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
    second = get_prs(
        REPO,
        numbers=[1, 2, 3, 4, 5, 6],
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )

    assert first.markdown is None
    assert first.path == second.path
    assert first.path is not None and first.path.is_file()
    assert not first.path.is_relative_to(target_workspace)
    assert [entry.number for entry in first.manifest] == [1, 2, 3, 4, 5, 6]
    artifact = first.path.read_text(encoding="utf-8")
    assert artifact.count("\n## PR #") == 6
    assert sorted(artifact_dir.iterdir()) == [first.path]
    assert first.path.suffix == ".md"
    assert first.path.stat().st_mode & 0o777 == 0o600

    fake.pulls[6] = pull_request(6, head="new-head-6")
    changed_head = get_prs(
        REPO,
        numbers=[1, 2, 3, 4, 5, 6],
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )

    assert changed_head.path != first.path
    assert not list(artifact_dir.glob("*.json"))


def test_get_prs_rejects_artifacts_inside_target_workspace(monkeypatch, tmp_path):
    fake = FakeGitHubReader({number: pull_request(number) for number in range(1, 7)})
    install_fake_github(monkeypatch, fake)
    target_workspace = tmp_path / "target"
    target_workspace.mkdir()

    with pytest.raises(ValueError, match="outside the target workspace"):
        get_prs(
            REPO,
            numbers=range(1, 7),
            artifact_dir=target_workspace / ".senpai" / "github",
            target_workspace=target_workspace,
        )


def test_get_prs_removes_only_expired_generated_markdown(monkeypatch, tmp_path):
    fake = FakeGitHubReader({number: pull_request(number) for number in range(1, 7)})
    install_fake_github(monkeypatch, fake)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    expired = artifact_dir / "pull-requests-expired.md"
    unrelated = artifact_dir / "operator-notes.md"
    expired.write_text("old generated bundle")
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
    assert unrelated.read_text() == "keep me"


def test_get_prs_warns_that_raising_inline_limit_can_pollute_agent_context(
    monkeypatch, tmp_path
):
    fake = FakeGitHubReader({1: pull_request(1)})
    install_fake_github(monkeypatch, fake)

    assert inspect.signature(get_prs).parameters["max_inline_prs"].default == 5
    assert "risks polluting agent context" in (get_prs.__doc__ or "")
    with pytest.warns(UserWarning, match="risks polluting agent context"):
        result = get_prs(
            REPO,
            numbers=[1],
            max_inline_prs=6,
            artifact_dir=tmp_path / "artifacts",
            target_workspace=tmp_path / "target",
        )

    assert result.markdown is not None


def test_get_prs_requires_at_least_one_selector(tmp_path):
    with pytest.raises(ValueError, match="at least one selector"):
        get_prs(
            REPO,
            artifact_dir=tmp_path / "artifacts",
            target_workspace=tmp_path / "target",
        )


def test_get_prs_passes_only_the_typed_credential_to_the_shared_reader(
    monkeypatch,
    tmp_path,
):
    fake = FakeGitHubReader({1: pull_request(1)})
    install_fake_github(monkeypatch, fake)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    monkeypatch.setenv("GH_TOKEN", "ambient-gh-token")

    get_prs(
        REPO,
        numbers=[1],
        token=SecretStr("typed-write-token"),
        artifact_dir=tmp_path / "artifacts",
        target_workspace=tmp_path / "target",
    )

    assert len(fake.tokens) == 1
    assert fake.tokens[0].get_secret_value() == "typed-write-token"
