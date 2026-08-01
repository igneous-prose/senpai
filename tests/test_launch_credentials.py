import base64
import io
import json
import urllib.error

import pytest

from launch_test_support import launch_helpers


class JSONResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def capture_request(monkeypatch, payload):
    captured = {}

    def urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return JSONResponse(payload)

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)
    return captured


def test_exa_preflight_runs_one_instant_publication_search(monkeypatch):
    captured = capture_request(
        monkeypatch,
        {"results": [{"id": "publication"}]},
    )

    launch_helpers.preflight_check_exa_api_key("exa-secret")

    request = captured["request"]
    assert request.full_url == "https://api.exa.ai/search"
    assert request.headers["X-api-key"] == "exa-secret"
    assert json.loads(request.data) == {
        "query": "api credential preflight",
        "type": "instant",
        "category": "publication",
        "numResults": 1,
    }


def test_exa_preflight_rejects_a_success_response_without_search_results(monkeypatch):
    capture_request(monkeypatch, {"status": "ok"})

    with pytest.raises(SystemExit, match="invalid search response"):
        launch_helpers.preflight_check_exa_api_key("exa-secret")


def test_wandb_preflight_authenticates_with_the_minimal_viewer_query(monkeypatch):
    captured = capture_request(
        monkeypatch,
        {"data": {"viewer": {"id": "user"}}},
    )

    launch_helpers.preflight_check_wandb_api_key("wandb-secret")

    request = captured["request"]
    assert request.full_url == "https://api.wandb.ai/graphql"
    assert request.headers["Authorization"] == (
        "Basic " + base64.b64encode(b"api:wandb-secret").decode()
    )
    assert json.loads(request.data) == {
        "query": "query SenpaiPreflight { viewer { id } }"
    }


def test_wandb_preflight_redacts_credentials_from_graphql_errors(monkeypatch):
    basic_auth = base64.b64encode(b"api:wandb-secret").decode()
    capture_request(
        monkeypatch,
        {
            "errors": [
                {"message": f"wandb-secret ({basic_auth}) was rejected"}
            ]
        },
    )

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_wandb_api_key("wandb-secret")

    message = str(raised.value)
    assert "wandb-secret" not in message
    assert basic_auth not in message
    assert "<redacted>" in message


def test_repo_access_uses_an_impossible_ref_write_probe(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured["request"] = request
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            {},
            io.BytesIO(b'{"message":"Object does not exist"}'),
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    launch_helpers.preflight_check_target_repo_access(
        "https://github.com/example/problem.git",
        "github-secret",
    )

    request = captured["request"]
    assert request.full_url == "https://api.github.com/repos/example/problem/git/refs"
    assert request.headers["Authorization"] == "Bearer github-secret"
    assert json.loads(request.data) == {
        "ref": "refs/heads/senpai-write-preflight",
        "sha": "0" * 40,
    }


def test_repo_access_rejects_and_redacts_a_non_validation_error(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"message":"github-secret cannot access this resource"}'),
        )

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(SystemExit) as raised:
        launch_helpers.preflight_check_target_repo_access(
            "https://github.com/example/problem.git",
            "github-secret",
        )

    message = str(raised.value)
    assert "HTTP 403" in message
    assert "Contents: Read and write" in message
    assert "github-secret" not in message
    assert "<redacted>" in message


def test_repo_access_fails_closed_if_the_impossible_write_is_accepted(monkeypatch):
    capture_request(monkeypatch, {})

    with pytest.raises(SystemExit, match="unexpectedly accepted"):
        launch_helpers.preflight_check_target_repo_access(
            "https://github.com/example/problem.git",
            "github-secret",
        )
