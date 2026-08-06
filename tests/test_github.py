import pytest

from senpai_agent.github import get_prs

from github_retrieval_support import (
    REPO,
    FakeGitHubReader,
    inline_comment,
    install_fake_github,
    issue_comment,
    pull_request,
    review,
)


def test_get_prs_returns_complete_content_in_stable_order(monkeypatch, tmp_path):
    long_body = "body-start\n" + ("untruncated evidence\n" * 2_000) + "body-end"
    fake = FakeGitHubReader(
        {1: pull_request(1, body=long_body), 2: pull_request(2)},
        comments={
            1: [
                issue_comment(2, "later issue comment"),
                issue_comment(1, "earlier issue comment"),
            ]
        },
        reviews={1: [review(1, "complete review submission")]},
        inline_comments={1: [inline_comment(1, "complete inline review comment")]},
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
    assert long_body in result.markdown
    expected_order = (
        "earlier issue comment",
        "later issue comment",
        "complete review submission",
        "complete inline review comment",
        "Complete PR body 2",
    )
    positions = [result.markdown.index(value) for value in expected_order]
    assert positions == sorted(positions)


def test_get_prs_unifies_explicit_search_and_date_range_selectors(
    monkeypatch,
    tmp_path,
):
    fake = FakeGitHubReader(
        {3: pull_request(3), 7: pull_request(7)},
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
    assert fake.search_query == (
        "repo:acme/widgets is:pr label:status:review "
        "created:2026-06-01..2026-06-30"
    )


def test_get_prs_requires_a_bounded_selector(monkeypatch, tmp_path):
    install_fake_github(monkeypatch, FakeGitHubReader({}))

    with pytest.raises(ValueError):
        get_prs(
            REPO,
            artifact_dir=tmp_path / "artifacts",
            target_workspace=tmp_path / "target",
        )
