import pytest

from exa_search_support import FakeClient, exa_search


def usage_error(argv, message: str, capsys):
    with pytest.raises(SystemExit):
        exa_search.parse_args(argv)
    assert message in capsys.readouterr().err


def test_publication_search_rejects_domain_inclusion(capsys):
    usage_error(
        [
            "research-publications",
            "neural operators",
            "--include-domains",
            "arxiv.org",
        ],
        "--include-domains is only supported for general-web",
        capsys,
    )


def test_no_content_requests_metadata_only():
    client = FakeClient()
    args = exa_search.parse_args(
        ["general-web", "attention alternatives", "--no-content"]
    )

    exa_search.search_exa(args, client)

    assert client.calls[0][1]["contents"] is False


def test_no_content_rejects_content_generation_options(capsys):
    usage_error(
        [
            "general-web",
            "attention alternatives",
            "--no-content",
            "--summary-query",
            "Summarize it",
        ],
        "--no-content cannot be combined with content options",
        capsys,
    )


def test_result_limit_rejects_an_unbounded_request(capsys):
    usage_error(
        ["general-web", "attention alternatives", "--num-results", "101"],
        "--num-results must be between 1 and 100",
        capsys,
    )


def test_additional_queries_require_a_deep_search(capsys):
    usage_error(
        [
            "general-web",
            "attention alternatives",
            "--search-type",
            "auto",
            "--additional-queries",
            "linear attention",
        ],
        "--additional-queries requires a deep search type",
        capsys,
    )


def test_additional_query_fanout_is_bounded(capsys):
    usage_error(
        [
            "research-publications",
            "attention alternatives",
            "--additional-queries",
            *(f"related query {index}" for index in range(11)),
        ],
        "--additional-queries accepts at most 10 queries",
        capsys,
    )


def test_exa_client_loads_dotenv_before_reading_api_key(monkeypatch):
    events = []
    client = object()

    def find_dotenv(*, usecwd):
        events.append(("find_dotenv", usecwd))
        return "/target/.env"

    def load_dotenv(*, dotenv_path, override):
        events.append(("load_dotenv", dotenv_path, override))
        monkeypatch.setenv("EXA_API_KEY", "test-key")

    def exa(api_key):
        events.append(("Exa", api_key))
        return client

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(exa_search, "find_dotenv", find_dotenv)
    monkeypatch.setattr(exa_search, "load_dotenv", load_dotenv)
    monkeypatch.setattr(exa_search, "Exa", exa)

    assert exa_search.create_exa_client() is client
    assert events == [
        ("find_dotenv", True),
        ("load_dotenv", "/target/.env", False),
        ("Exa", "test-key"),
    ]


def test_exa_client_does_not_override_an_exported_api_key(monkeypatch):
    observed = []
    client = object()
    monkeypatch.setenv("EXA_API_KEY", "exported-key")
    monkeypatch.setattr(exa_search, "find_dotenv", lambda *, usecwd: "/target/.env")

    def load_dotenv(*, dotenv_path, override):
        observed.append((dotenv_path, override))

    monkeypatch.setattr(exa_search, "load_dotenv", load_dotenv)
    monkeypatch.setattr(
        exa_search,
        "Exa",
        lambda api_key: observed.append(api_key) or client,
    )

    assert exa_search.create_exa_client() is client
    assert observed == [("/target/.env", False), "exported-key"]


def test_exa_client_reports_a_missing_api_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(exa_search, "find_dotenv", lambda *, usecwd: "")
    monkeypatch.setattr(exa_search, "load_dotenv", lambda **_kwargs: False)

    with pytest.raises(RuntimeError, match="EXA_API_KEY is not set"):
        exa_search.create_exa_client()


def test_exa_client_rejects_a_whitespace_only_api_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "  \t")
    monkeypatch.setattr(exa_search, "find_dotenv", lambda *, usecwd: "")
    monkeypatch.setattr(exa_search, "load_dotenv", lambda **_kwargs: False)

    with pytest.raises(RuntimeError, match="EXA_API_KEY is not set"):
        exa_search.create_exa_client()
