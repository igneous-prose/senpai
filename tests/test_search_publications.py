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
                "type": "deep",
                "contents": {"highlights": {"max_characters": 2000}},
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
    assert search_publications.render_markdown(payload) == """\
# Exa Publication Search

- **Query:** neural operators for PDEs
- **Category:** publication
- **Search type:** deep
- **Results:** 1 returned / 30 requested
- **Search time:** 123.4 ms
- **Cost (USD):**
  - **Total:** 0.007
  - **Search:**
    - **Neural:** 0.007

## 1. Fourier Neural Operator

- **URL:** <https://example.com/fno>
- **Authors:** Zongyi Li
- **Published:** 2020-10-23
- **Exa ID:** publication:fno
- **Highlights:**
  - We propose a Fourier neural operator."""


def test_main_prints_markdown_not_json(monkeypatch, capsys):
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )
    monkeypatch.setattr(search_publications, "create_exa_client", lambda: client)

    search_publications.main(["attention alternatives", "--num-results", "2"])

    output = capsys.readouterr().out
    assert output.startswith("# Exa Publication Search\n")
    assert "- **Results:** 0 returned / 2 requested" in output
    assert "No publications were returned." in output
    assert '"results":' not in output


def test_summary_lists_become_nested_markdown_bullets():
    assert search_publications.render_summary(
        "Mechanism: - Commutes with *rotations*. - # Preserves [tensor] structure."
    ) == [
        "- **Summary:** Mechanism:",
        "  - Commutes with \\*rotations\\*.",
        "  - \\# Preserves \\[tensor\\] structure.",
    ]


def test_markdown_normalizes_missing_title_and_redundant_summary_label():
    output = search_publications.render_markdown(
        {
            "query": "geometry transfer",
            "category": "publication",
            "search_type": "deep",
            "requested_results": 1,
            "result_count": 1,
            "results": [
                {
                    "rank": 1,
                    "title": "",
                    "summary": "Summary:\n- Maps each shape to a reference domain.",
                }
            ],
        }
    )

    assert "## 1. Untitled publication" in output
    assert "- **Summary:** Maps each shape to a reference domain." in output
    assert "**Summary:** Summary:" not in output


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
                "--search-type",
                "auto",
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


def test_claude_skill_uses_canonical_codex_script():
    codex_script = (
        Path(__file__).parents[1]
        / ".agents"
        / "skills"
        / "exa-publication-search"
        / "scripts"
        / "search_publications.py"
    )
    assert SCRIPT.is_symlink()
    assert SCRIPT.resolve() == codex_script.resolve()
