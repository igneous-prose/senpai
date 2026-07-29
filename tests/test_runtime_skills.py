import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_SKILL = ROOT / ".agents" / "skills" / "web-search-advanced-research-paper"
PUBLICATION_SCRIPT = PUBLICATION_SKILL / "scripts" / "search_publications.py"
STATUS_SKILL = ROOT / ".agents" / "skills" / "senpai-status-check" / "SKILL.md"
HUMAN_ISSUES_SKILL = (
    ROOT / "plugins" / "senpai" / "skills" / "check-human-issues" / "SKILL.md"
)
POLL_FOR_WORK_SKILL = (
    ROOT / "plugins" / "senpai" / "skills" / "poll-for-work" / "SKILL.md"
)


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_search_posts_one_bounded_metadata_query(
    monkeypatch,
    capsys,
):
    publication = load_script(PUBLICATION_SCRIPT, "search_publications_contract")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "paper-1",
                            "title": "Useful paper",
                            "url": "https://arxiv.org/abs/1234.5678",
                            "publishedDate": "2026-01-02",
                            "author": "Ada Researcher",
                            "text": "large result body",
                            "score": 0.99,
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(publication, "urlopen", fake_urlopen)

    assert (
        publication.main(
            [
                "--num-results",
                "7",
                "--search-type",
                "fast",
                "--start-published-date",
                "2025-01-01",
                "--end-published-date",
                "2026-06-01",
                "--include-domain",
                "arxiv.org",
                "--include-domain",
                "openreview.net",
                "--exclude-domain",
                "example.invalid",
                "--include-text",
                "operator",
                "--exclude-text",
                "survey",
                "mesh operator learning",
            ],
            {"EXA_API_KEY": "exa-secret"},
        )
        == 0
    )

    request = captured["request"]
    headers = {key.lower(): value for key, value in request.header_items()}
    payload = json.loads(request.data)
    assert request.full_url == "https://api.exa.ai/search"
    assert request.method == "POST"
    assert captured["timeout"] == 20
    assert headers["x-api-key"] == "exa-secret"
    assert payload == {
        "query": "mesh operator learning",
        "type": "fast",
        "category": "publication",
        "numResults": 7,
        "startPublishedDate": "2025-01-01",
        "endPublishedDate": "2026-06-01",
        "includeDomains": ["arxiv.org", "openreview.net"],
        "excludeDomains": ["example.invalid"],
        "includeText": ["operator"],
        "excludeText": ["survey"],
    }
    assert json.loads(capsys.readouterr().out) == {
        "query": "mesh operator learning",
        "results": [
            {
                "id": "paper-1",
                "title": "Useful paper",
                "url": "https://arxiv.org/abs/1234.5678",
                "publishedDate": "2026-01-02",
                "author": "Ada Researcher",
            }
        ],
    }


def test_publication_search_requires_and_redacts_api_key(monkeypatch):
    publication = load_script(PUBLICATION_SCRIPT, "search_publications_errors")

    with pytest.raises(SystemExit, match="EXA_API_KEY is required"):
        publication.main(["bounded query"], {})

    api_key = "exa-secret-value"
    error = HTTPError(
        publication.EXA_SEARCH_URL,
        401,
        "Unauthorized",
        {},
        io.BytesIO(f'{{"error":"invalid {api_key}"}}'.encode()),
    )
    monkeypatch.setattr(
        publication,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError) as caught:
        publication.search_publications(
            "bounded query",
            api_key=api_key,
            num_results=3,
        )

    assert api_key not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_publication_skill_requires_progressive_metadata_search():
    instructions = (PUBLICATION_SKILL / "SKILL.md").read_text()
    lower = instructions.lower()

    assert "context: fork" in instructions
    assert "model: Codex-opus-4-8" in instructions
    assert "effort: high" in instructions
    assert "search_publications.py" in instructions
    assert "metadata-only" in lower
    assert "candidates:" in lower
    assert "refine:" in lower
    assert "read:" in lower
    assert "one to three papers" in lower
    assert "mcp" not in lower
    assert "web_search_advanced_exa" not in lower


def test_status_skill_uses_runtime_scope_without_legacy_assumptions():
    instructions = STATUS_SKILL.read_text()
    lower = instructions.lower()

    for variable in (
        "GH_REPO",
        "ADVISOR_BRANCH",
        "RESEARCH_TAG",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "TARGET_WORKDIR",
        "SENPAI_OPENHANDS_STATE_DIR",
    ):
        assert variable in instructions
    for required in (
        "program.md",
        "get_prs",
        "wandb-primary",
        "$SENPAI_OPENHANDS_STATE_DIR/training/*.json",
        "evidence gap",
        "Do not mutate",
    ):
        assert required.lower() in lower
    for legacy in (
        "wandb/senpai",
        "radford",
        "pai-2",
        ".claude",
        "current_research_state",
        "drivaerml",
        "tandemfoil",
        "airfrans",
        "harvest",
        "shutdown",
        "train.py",
    ):
        assert legacy not in lower


def test_human_issue_skill_uses_the_typed_mutation_boundary():
    instructions = HUMAN_ISSUES_SKILL.read_text()

    assert "human_message_id" in instructions
    assert "respond_to_issue" in instructions
    assert "github_transition" in instructions
    assert "gh issue comment" not in instructions


def test_poll_for_work_skill_uses_the_bounded_pr_reader():
    instructions = POLL_FOR_WORK_SKILL.read_text()

    assert "get_prs" in instructions
    assert "max_inline_prs=5" in instructions
    assert "pr_all_comments" not in instructions
