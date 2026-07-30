"""Executable contract for the two-mode Exa search skill."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / ".agents"
    / "skills"
    / "exa-search"
    / "scripts"
    / "search_exa.py"
)
SPEC = importlib.util.spec_from_file_location("exa_search", SCRIPT)
assert SPEC and SPEC.loader
exa_search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exa_search)


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
    client = exa_search.Exa("test-key")

    def fake_request(path, options):
        captured_request.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", fake_request)

    client.search("neural operators", category="publication", contents=False)

    assert captured_request["path"] == "/search"
    assert captured_request["options"]["category"] == "publication"


def test_installed_sdk_serializes_general_web_without_a_category(monkeypatch):
    captured_request = {}
    client = exa_search.Exa("test-key")

    def fake_request(path, options):
        captured_request.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", fake_request)

    client.search(
        "OpenHands SDK documentation",
        type="auto",
        num_results=10,
        contents={"highlights": True},
    )

    assert captured_request == {
        "path": "/search",
        "options": {
            "query": "OpenHands SDK documentation",
            "type": "auto",
            "numResults": 10,
            "contents": {"highlights": True},
        },
    }


def test_installed_sdk_serializes_general_web_scope_and_freshness(monkeypatch):
    captured_request = {}
    client = exa_search.Exa("test-key")

    def fake_request(path, options):
        captured_request.update(path=path, options=options)
        return {"results": []}

    monkeypatch.setattr(client, "request", fake_request)

    client.search(
        "OpenHands SDK documentation",
        type="auto",
        include_domains=["docs.openhands.dev"],
        contents={"highlights": True, "max_age_hours": 0},
    )

    assert captured_request["options"]["includeDomains"] == ["docs.openhands.dev"]
    assert captured_request["options"]["contents"] == {
        "highlights": True,
        "maxAgeHours": 0,
    }


def test_general_web_uses_exas_agent_recommended_defaults():
    args = exa_search.parse_args(["general-web", "current OpenHands SDK release"])
    response = SimpleNamespace(
        results=[
            SimpleNamespace(
                title="OpenHands SDK",
                url="https://docs.openhands.dev/sdk",
                id="https://docs.openhands.dev/sdk",
                published_date="2026-07-29",
                author="OpenHands",
                score=None,
                highlights=["The SDK supports file-based agents."],
                summary=None,
                text=None,
            )
        ],
        search_time=87,
        cost_dollars=None,
    )
    client = FakeClient(response)

    payload = exa_search.search_exa(args, client)

    assert client.calls == [
        (
            "current OpenHands SDK release",
            {
                "num_results": 10,
                "type": "auto",
                "contents": {"highlights": True},
            },
        )
    ]
    assert payload["mode"] == "general-web"
    assert "category" not in payload
    assert payload["results"][0]["highlights"] == [
        "The SDK supports file-based agents."
    ]
    output = exa_search.render_markdown(payload)
    assert output.startswith("# Exa Web Search\n")
    assert "- **Mode:** general-web" in output
    assert "https://docs.openhands.dev/sdk" in output


def test_general_web_forwards_domain_scope_and_content_freshness():
    args = exa_search.parse_args(
        [
            "general-web",
            "OpenHands SDK documentation",
            "--include-domains",
            "docs.openhands.dev",
            "--max-age-hours",
            "0",
        ]
    )
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )

    exa_search.search_exa(args, client)

    assert client.calls == [
        (
            "OpenHands SDK documentation",
            {
                "num_results": 10,
                "type": "auto",
                "contents": {"highlights": True, "max_age_hours": 0},
                "include_domains": ["docs.openhands.dev"],
            },
        )
    ]


def test_default_search_uses_publication_index_and_compact_highlights():
    args = exa_search.parse_args(["research-publications", "neural operators for PDEs"])
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

    payload = exa_search.search_exa(args, client)

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
    assert (
        exa_search.render_markdown(payload)
        == """\
# Exa Publication Search

- **Query:** neural operators for PDEs
- **Mode:** research-publications
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
    )


def test_main_prints_markdown_not_json(monkeypatch, capsys):
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )
    monkeypatch.setattr(exa_search, "create_exa_client", lambda: client)

    exa_search.main(
        ["research-publications", "attention alternatives", "--num-results", "2"]
    )

    output = capsys.readouterr().out
    assert output.startswith("# Exa Publication Search\n")
    assert "- **Results:** 0 returned / 2 requested" in output
    assert "No publications were returned." in output
    assert '"results":' not in output


def test_summary_lists_become_nested_markdown_bullets():
    assert exa_search.render_summary(
        "Mechanism: - Commutes with *rotations*. - # Preserves [tensor] structure."
    ) == [
        "- **Summary:** Mechanism:",
        "  - Commutes with \\*rotations\\*.",
        "  - \\# Preserves \\[tensor\\] structure.",
    ]


def test_markdown_normalizes_missing_title_and_redundant_summary_label():
    output = exa_search.render_markdown(
        {
            "query": "geometry transfer",
            "mode": "research-publications",
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


def test_markdown_url_percent_encodes_control_characters():
    assert (
        exa_search.markdown_url("https://example.com/good\n- injected")
        == "https://example.com/good%0A-%20injected"
    )


def test_search_options_are_forwarded_without_changing_publication_category():
    args = exa_search.parse_args(
        [
            "research-publications",
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
            "--max-age-hours",
            "0",
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

    exa_search.search_exa(args, client)

    assert client.calls == [
        (
            "equivariant CFD surrogates",
            {
                "category": "publication",
                "num_results": 50,
                "type": "deep",
                "contents": {
                    "highlights": {"max_characters": 1800},
                    "summary": {"query": "What mechanism improves generalization?"},
                    "max_age_hours": 0,
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


def test_publication_search_rejects_include_domains(capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(
            [
                "research-publications",
                "neural operators",
                "--include-domains",
                "arxiv.org",
            ]
        )

    assert "--include-domains is only supported for general-web" in (
        capsys.readouterr().err
    )


def test_no_content_requests_metadata_only():
    args = exa_search.parse_args(
        ["general-web", "attention alternatives", "--no-content"]
    )
    client = FakeClient(
        SimpleNamespace(results=[], search_time=None, cost_dollars=None)
    )

    exa_search.search_exa(args, client)

    assert client.calls[0][1]["contents"] is False


@pytest.mark.parametrize(
    "content_options",
    [
        ["--summary-query", "Summarize it"],
        ["--highlights-max-characters", "100"],
        ["--max-age-hours", "0"],
    ],
)
def test_no_content_rejects_content_options(content_options, capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(
            [
                "general-web",
                "attention alternatives",
                "--no-content",
                *content_options,
            ]
        )

    assert "--no-content cannot be combined with content options" in (
        capsys.readouterr().err
    )


def test_argument_validation_is_reported_as_cli_usage(capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(
            ["general-web", "attention alternatives", "--num-results", "101"]
        )

    assert "--num-results must be between 1 and 100" in capsys.readouterr().err


def test_max_age_hours_rejects_values_below_cache_only_sentinel(capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(
            [
                "general-web",
                "attention alternatives",
                "--max-age-hours",
                "-2",
            ]
        )

    assert "--max-age-hours must be -1 or greater" in capsys.readouterr().err


def test_additional_queries_require_deep_search(capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(
            [
                "general-web",
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

    def fake_find_dotenv(*, usecwd):
        events.append(("find_dotenv", usecwd))
        return "/target/.env"

    def fake_load_dotenv(*, dotenv_path, override):
        events.append(("load_dotenv", dotenv_path, override))
        monkeypatch.setenv("EXA_API_KEY", "test-key")

    def fake_exa(api_key):
        events.append(("Exa", api_key))
        return client

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(exa_search, "find_dotenv", fake_find_dotenv)
    monkeypatch.setattr(exa_search, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(exa_search, "Exa", fake_exa)

    assert exa_search.create_exa_client() is client
    assert events == [
        ("find_dotenv", True),
        ("load_dotenv", "/target/.env", False),
        ("Exa", "test-key"),
    ]


def test_skill_is_installed_only_in_openhands_agent_scope():
    root = Path(__file__).parents[1]

    assert SCRIPT.is_file()
    assert not (root / ".claude").exists()
