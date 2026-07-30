import json
from urllib.error import HTTPError

import pytest
from pydantic import SecretStr

from senpai_agent import github_http
from senpai_agent.github_http import GitHubReader, GitHubReadError, next_link


class Response:
    def __init__(self, payload, *, link=None):
        self.payload = payload
        self.headers = {"Link": link} if link else {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def test_reader_follows_pagination_with_typed_auth_and_caches_actor(monkeypatch):
    requests = []
    responses = iter(
        (
            Response(
                [{"id": 1}],
                link=(
                    '<https://api.github.test/items?page=2>; rel="next", '
                    '<https://api.github.test/items?page=2>; rel="last"'
                ),
            ),
            Response([{"id": 2}]),
            Response({"login": "senpai-bot"}),
        )
    )

    def urlopen(github_request, timeout):
        requests.append((github_request, timeout))
        return next(responses)

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(
        SecretStr("github-secret"),
        api_url="https://api.github.test",
    )

    assert reader.objects("/items?per_page=1") == [{"id": 1}, {"id": 2}]
    assert reader.actor() == reader.actor() == "senpai-bot"
    assert len(requests) == 3
    assert all(
        request.headers["Authorization"] == "Bearer github-secret"
        for request, _timeout in requests
    )
    assert all(timeout == 30 for _request, timeout in requests)


def test_reader_rejects_foreign_pagination_origin(monkeypatch):
    monkeypatch.setattr(
        github_http.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(
            [],
            link='<https://attacker.example/items?page=2>; rel="next"',
        ),
    )

    with pytest.raises(GitHubReadError, match="unexpected origin"):
        GitHubReader(
            SecretStr("github-secret"),
            api_url="https://api.github.test",
        ).objects("/items")


def test_reader_errors_do_not_expose_token(monkeypatch):
    def fail(github_request, timeout):
        assert timeout == 30
        raise HTTPError(github_request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubReadError) as raised:
        GitHubReader(SecretStr("github-secret")).get("/user")

    assert "github-secret" not in str(raised.value)
    assert "/user" in str(raised.value)


def test_next_link_extracts_only_the_next_relation():
    assert (
        next_link(
            '<https://api.github.test/items?page=1>; rel="prev", '
            '<https://api.github.test/items?page=3>; rel="next"'
        )
        == "https://api.github.test/items?page=3"
    )
    assert next_link(None) is None
