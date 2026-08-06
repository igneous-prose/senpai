from exa_search_support import FakeClient, exa_search


def test_main_prints_markdown_not_json(monkeypatch, capsys):
    monkeypatch.setattr(exa_search, "create_exa_client", FakeClient)

    exa_search.main(
        ["research-publications", "attention alternatives", "--num-results", "2"]
    )

    output = capsys.readouterr().out
    assert output.startswith("# Exa Publication Search\n")
    assert "- **Results:** 0 returned / 2 requested" in output
    assert "No publications were returned." in output
    assert '"results":' not in output


def test_publication_markdown_renders_metadata_and_cost():
    output = exa_search.render_markdown(
        {
            "query": "neural operators",
            "mode": "research-publications",
            "category": "publication",
            "search_type": "deep",
            "requested_results": 1,
            "result_count": 1,
            "cost_dollars": {"total": 0.007, "search": {"neural": 0.007}},
            "results": [
                {
                    "rank": 1,
                    "title": "Fourier Neural Operator",
                    "url": "https://example.com/fno",
                    "author": "Zongyi Li",
                    "highlights": ["A compact result."],
                }
            ],
        }
    )

    assert output.startswith("# Exa Publication Search\n")
    assert "- **Total:** 0.007" in output
    assert "## 1. Fourier Neural Operator" in output
    assert "- **URL:** <https://example.com/fno>" in output
    assert "  - A compact result." in output


def test_summary_lists_become_escaped_nested_markdown_bullets():
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


def test_untrusted_text_cannot_inject_markdown_structure():
    output = exa_search.render_markdown(
        {
            "query": "safe query\n# injected heading",
            "mode": "general-web",
            "search_type": "auto",
            "requested_results": 1,
            "result_count": 1,
            "results": [
                {
                    "rank": 1,
                    "title": "Result\n## injected result",
                    "highlights": ["finding\n- injected bullet"],
                }
            ],
        }
    )

    assert "safe query \\# injected heading" in output
    assert "## 1. Result \\#\\# injected result" in output
    assert "  - finding - injected bullet" in output
    assert "\n# injected heading" not in output
    assert "\n## injected result" not in output


def test_markdown_url_percent_encodes_control_characters():
    assert (
        exa_search.markdown_url("https://example.com/good\n- injected")
        == "https://example.com/good%0A-%20injected"
    )
