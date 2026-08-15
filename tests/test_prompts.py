from pathlib import Path

import pytest

import senpai_agent.prompts as prompts


def write_prompt_catalog(path: Path, sections: dict[str, str]) -> None:
    path.write_text(
        "# Test prompts\n\n"
        + "\n\n".join(
            f"## {name}\n\n{body}" for name, body in sections.items()
        )
        + "\n"
    )


def test_prompt_catalog_exports_every_named_prompt_as_an_uppercase_string():
    prompt_names = [name for name in prompts.__all__ if name.endswith("_PROMPT")]

    assert prompt_names == list(prompts._PROMPT_NAMES)
    assert all(name.isupper() for name in prompt_names)
    assert all(isinstance(getattr(prompts, name), str) for name in prompt_names)
    assert all(getattr(prompts, name).strip() for name in prompt_names)


def test_prompt_catalog_rejects_duplicate_and_empty_sections():
    with pytest.raises(RuntimeError, match="duplicate prompt section: SAME_PROMPT"):
        prompts._parse_prompt_sections(
            "## SAME_PROMPT\n\nFirst.\n\n## SAME_PROMPT\n\nSecond."
        )

    with pytest.raises(RuntimeError, match="empty prompt section: EMPTY_PROMPT"):
        prompts._parse_prompt_sections("## EMPTY_PROMPT\n")


def test_prompt_catalog_rejects_missing_and_unexpected_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    catalog = tmp_path / "PROMPTS.md"
    write_prompt_catalog(catalog, {"UNEXPECTED_PROMPT": "Unexpected."})
    monkeypatch.setattr(prompts, "PROMPTS_PATH", catalog)

    with pytest.raises(RuntimeError) as raised:
        prompts._load_prompts()

    message = str(raised.value)
    assert "missing: " in message
    assert "CONTEXT_RECOVERY_PROMPT" in message
    assert "unexpected: UNEXPECTED_PROMPT" in message


def test_render_prompt_requires_exact_values_and_does_not_reprocess_insertions():
    template = "Payload: {{PAYLOAD}}"

    assert prompts.render_prompt(
        template,
        PAYLOAD='{"literal": "{{UNCHANGED}}"}',
    ) == 'Payload: {"literal": "{{UNCHANGED}}"}'
    with pytest.raises(ValueError, match="missing: PAYLOAD"):
        prompts.render_prompt(template)
    with pytest.raises(ValueError, match="unexpected: EXTRA"):
        prompts.render_prompt(template, PAYLOAD="value", EXTRA="value")


def test_render_prompt_preserves_placeholder_boundary_whitespace():
    assert prompts.render_prompt("Before\n\n{{VALUE}}", VALUE="value\n\n") == (
        "Before\n\nvalue\n\n"
    )


def test_python_sources_do_not_embed_centralized_prompt_text():
    source_root = Path(prompts.__file__).parent
    fragments = (
        "# Conversation context recovery",
        "Actionable events follow as separately tracked messages.",
        "You are a fresh Senpai subagent.",
        "Senpai restarted before this action completed.",
        "Open feedback_url to read the omitted text.",
        "You may finish this turn; the controller will resume",
        "unfinished sibling tasks keep running unless you cancel",
        "repeating the same long all-results wait will block",
        "delegate_agent is deprecated and cannot launch an agent.",
    )

    for source in source_root.rglob("*.py"):
        text = source.read_text()
        for fragment in fragments:
            assert fragment not in text, f"{fragment!r} remains in {source}"
