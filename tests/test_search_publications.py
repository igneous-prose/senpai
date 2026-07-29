"""Tests for the Exa publication-search skill tool."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / ".claude"
    / "skills"
    / "exa-publication-search"
    / "scripts"
    / "search_publications.py"
)
SPEC = importlib.util.spec_from_file_location("search_publications", SCRIPT)
assert SPEC and SPEC.loader
search_publications = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search_publications)


@dataclasses.dataclass
class FakeCost:
    total: float
    search: dict[str, float]
    contents: dict[str, float] | None = None


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        return self.response


def test_installed_sdk_accepts_publication_category(monkeypatch):
    captured_request = {}
    client = search_publications.Exa("test-key")

    def fake_request(path, options):
        captured_request.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", fake_request)

    client.search("neural operators", category="publication", contents=False)

    assert captured_request["path"] == "/search"
    assert captured_request["options"]["category"] == "publication"


def test_default_search_uses_publication_index_and_compact_highlights():
    args = search_publications.parse_args(["neural operators for PDEs"])
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                title="Fourier Neural Operator",
                url="https://example.com/fno",
                id="publication:fno",
                published_date="2020-10-23",
                author="Zongyi Li",
                score=None,
                highlights=["We propose a Fourier neural operator."],
                summary=None,
                text="full paper text must not be emitted",
            )
        ],
        search_time=123.4,
        cost_dollars=FakeCost(total=0.007, search={"neural": 0.007}),
    )
    client = FakeClient(response)

    payload = search_publications.search_publications(args, client)

    assert client.calls == [
        (
            "neural operators for PDEs",
            {
                "category": "publication",
                "num_results": 30,
                "type": "auto",
                "contents": {"highlights": {"max_characters": 1200}},
            },
        )
    ]
    assert payload["requested_results"] == 30
    assert payload["result_count"] == 1
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["highlights"] == [
        "We propose a Fourier neural operator."
    ]
    assert "text" not in payload["results"][0]
    assert payload["cost_dollars"] == {
        "total": 0.007,
        "search": {"neural": 0.007},
    }


def test_search_options_are_forwarded_without_changing_publication_category():
    args = search_publications.parse_args(
        [
            "equivariant CFD surrogates",
            "--num-results",
            "50",
            "--search-type",
            "deep",
            "--start-published-date",
            "2023-01-01",
            "--end-published-date",
            "2026-01-01",
            "--include-domains",
            "arxiv.org",
            "openreview.net",
            "--exclude-domains",
            "example.com",
            "--include-text",
            "equivariant",
            "--exclude-text",
            "survey",
            "--additional-queries",
            "SE(3) mesh networks",
            "--summary-query",
            "What mechanism improves generalization?",
            "--highlights-max-characters",
            "1800",
        ]
    )
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )

    search_publications.search_publications(args, client)

    assert client.calls == [
        (
            "equivariant CFD surrogates",
            {
                "category": "publication",
                "num_results": 50,
                "type": "deep",
                "contents": {
                    "highlights": {"max_characters": 1800},
                    "summary": {
                        "query": "What mechanism improves generalization?"
                    },
                },
                "start_published_date": "2023-01-01",
                "end_published_date": "2026-01-01",
                "include_domains": ["arxiv.org", "openreview.net"],
                "exclude_domains": ["example.com"],
                "include_text": ["equivariant"],
                "exclude_text": ["survey"],
                "additional_queries": ["SE(3) mesh networks"],
            },
        )
    ]


def test_no_highlights_requests_metadata_only():
    args = search_publications.parse_args(["attention alternatives", "--no-highlights"])
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )

    search_publications.search_publications(args, client)

    assert client.calls[0][1]["contents"] is False


def test_argument_validation_is_reported_as_cli_usage(capsys):
    with pytest.raises(SystemExit):
        search_publications.parse_args(
            ["attention alternatives", "--num-results", "101"]
        )

    assert "--num-results must be between 1 and 100" in capsys.readouterr().err


def test_additional_queries_require_deep_search(capsys):
    with pytest.raises(SystemExit):
        search_publications.parse_args(
            [
                "attention alternatives",
                "--additional-queries",
                "linear attention",
            ]
        )

    assert "--additional-queries requires a deep search type" in (
        capsys.readouterr().err
    )


def test_exa_client_loads_dotenv_before_reading_api_key(monkeypatch):
    events = []
    client = object()

    def fake_load_dotenv():
        events.append("load_dotenv")
        monkeypatch.setenv("EXA_API_KEY", "test-key")

    def fake_exa(api_key):
        events.append(("Exa", api_key))
        return client

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(search_publications, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(search_publications, "Exa", fake_exa)

    assert search_publications.create_exa_client() is client
    assert events == ["load_dotenv", ("Exa", "test-key")]


def test_claude_and_codex_skill_scripts_stay_identical():
    codex_script = (
        Path(__file__).parents[1]
        / ".agents"
        / "skills"
        / "exa-publication-search"
        / "scripts"
        / "search_publications.py"
    )
    assert SCRIPT.read_bytes() == codex_script.read_bytes()
