import re
from types import SimpleNamespace

import pytest
from openhands.sdk import Agent, LLM, LocalConversation
from openhands.sdk.llm import Message, TextContent
from pydantic import SecretStr

from senpai_agent.openhands_runner import (
    EVENT_TEXT_LIMIT,
    anthropic_compaction_configuration,
    conversation_prompt_cache_key,
    event_summary,
    openai_responses_configuration,
    openhands_reasoning_effort,
    parse_runner_args,
    prompt_cache_configuration,
)
from openhands_support import REPO_ROOT, runtime_config


def test_openhands_fork_revision_is_consistent_across_install_paths():
    pins = []
    for path in (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / ".github" / "workflows" / "test.yaml",
    ):
        matches = {
            match
            for line in path.read_text(encoding="utf-8").splitlines()
            if "morganmcg1/software-agent-sdk" in line
            for match in re.findall(r"[0-9a-f]{40}", line)
        }
        assert len(matches) == 1, f"expected one OpenHands fork revision in {path}"
        pins.extend(matches)

    assert len(set(pins)) == 1


@pytest.mark.parametrize(
    ("effort", "model", "expected"),
    [
        ("max", "openai/gpt-5.6-sol", "max"),
        ("high", "openai/gpt-5.6", "high"),
        ("xhigh", "anthropic/claude-opus-4-8", "xhigh"),
    ],
)
def test_supported_reasoning_effort_is_preserved(
    effort: str,
    model: str,
    expected: str,
):
    args = parse_runner_args(["--max-turns", "1", "--reasoning-effort", effort])

    assert args.reasoning_effort == effort
    assert openhands_reasoning_effort(effort, model) == expected


@pytest.mark.parametrize(
    ("effort", "model"),
    [
        ("max", "anthropic/claude-opus-4-8"),
        ("max", "openai/gpt-5.4"),
        ("max", "openai/gpt-5.60"),
        ("ultra", "openai/gpt-5.6-sol"),
        ("extreme", "openai/gpt-5.6-sol"),
    ],
)
def test_unsupported_reasoning_effort_fails_instead_of_being_rewritten(
    effort: str,
    model: str,
):
    with pytest.raises(ValueError, match="unsupported reasoning effort|unsupported for"):
        openhands_reasoning_effort(effort, model)


def test_ultra_is_not_a_cli_reasoning_effort_alias():
    with pytest.raises(SystemExit):
        parse_runner_args(["--max-turns", "1", "--reasoning-effort", "ultra"])


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "anthropic/claude-opus-4-8",
            {"prompt_cache_ttl": "1h"}
            if "prompt_cache_ttl" in LLM.model_fields
            else {},
        ),
        ("openai/gpt-5.4", {"prompt_cache_retention": "24h"}),
        (
            "openai/gpt-5.6",
            {
                "prompt_cache_retention": None,
                "responses_prompt_cache_breakpoint": True,
                "litellm_extra_body": {
                    "prompt_cache_options": {
                        "mode": "explicit",
                        "ttl": "30m",
                    }
                },
            },
        ),
        ("gemini/gemini-3-pro", {}),
    ],
)
def test_prompt_cache_configuration_is_provider_specific(model: str, expected):
    assert prompt_cache_configuration(model) == expected


def test_openai_response_configuration_is_accepted_by_the_pinned_sdk():
    configuration = openai_responses_configuration("openai/gpt-5.4")
    llm = LLM(
        model="openai/gpt-5.4",
        api_key=SecretStr("test-key"),
        reasoning_effort="xhigh",
        **configuration,
    )

    assert configuration == {
        "api_mode": "responses",
        "reasoning_summary": "auto",
        "reasoning_context": "all_turns",
        "responses_store": True,
        "responses_use_previous_response_id": True,
        "responses_compact_threshold": 200_000,
    }
    assert llm.uses_responses_api() is True
    assert llm.responses_store is True
    assert llm.responses_use_previous_response_id is True
    assert llm.responses_compact_threshold == 200_000
    assert openai_responses_configuration("anthropic/claude-opus-4-8") == {}


def test_anthropic_compaction_configuration_is_accepted_by_the_pinned_sdk():
    configuration = anthropic_compaction_configuration(
        "anthropic/claude-opus-4-8"
    )
    llm = LLM(
        model="anthropic/claude-opus-4-8",
        api_key=SecretStr("test-key"),
        **configuration,
    )

    assert configuration == {"anthropic_compact_threshold": 200_000}
    assert llm.uses_anthropic_compaction() is True
    assert anthropic_compaction_configuration("openai/gpt-5.6") == {}


def test_gpt56_marks_only_the_stable_system_cache_boundary():
    llm = LLM(
        model="openai/gpt-5.6",
        api_key=SecretStr("test-key"),
        **prompt_cache_configuration("openai/gpt-5.6"),
        **openai_responses_configuration("openai/gpt-5.6"),
    )
    instructions, inputs = llm.format_messages_for_responses(
        [
            Message(
                role="system",
                content=[
                    TextContent(text="stable harness and role"),
                    TextContent(text="dynamic project context"),
                ],
            ),
            Message(role="user", content=[TextContent(text="Investigate")]),
        ]
    )

    assert instructions is None
    assert inputs[0]["content"] == [
        {
            "type": "input_text",
            "text": "stable harness and role",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {
            "type": "input_text",
            "text": "dynamic project context",
        },
    ]


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"model": "openai/gpt-5.6", "role": "student"}, "senpai:student:main"),
        (
            {"model": "openai/gpt-5.6", "role": "advisor", "child": True},
            "senpai:advisor:child",
        ),
        (
            {"model": "openai/gpt-5.6", "agent_name": "explore"},
            "senpai:advisor:explore",
        ),
        ({"model": "anthropic/claude-opus-4-8"}, None),
    ],
)
def test_prompt_cache_key_is_scoped_by_role_and_agent(tmp_path, updates, expected):
    assert conversation_prompt_cache_key(runtime_config(tmp_path, **updates)) == expected


def test_local_conversation_exposes_the_configured_prompt_cache_key(tmp_path):
    conversation = LocalConversation(
        agent=Agent(
            llm=LLM(
                model="openai/gpt-5.6",
                api_key=SecretStr("test-key"),
            ),
            tools=[],
        ),
        workspace=tmp_path,
        visualizer=None,
        prompt_cache_key="senpai:student:main",
    )
    try:
        assert (
            conversation.get_llm_call_context().prompt_cache_key
            == "senpai:student:main"
        )
    finally:
        conversation.close()


def test_event_summary_bounds_fields_and_keeps_the_latest_text():
    long_text = "discarded-prefix-" + "x" * EVENT_TEXT_LIMIT + "-latest"
    event = SimpleNamespace(
        source="agent",
        thought=long_text,
        action={"command": long_text},
        status="running",
    )

    summary = event_summary(event)

    assert len(summary["thought"].encode()) <= EVENT_TEXT_LIMIT
    assert summary["thought"].endswith("-latest")
    assert not summary["thought"].startswith("discarded-prefix-")
    assert len(summary["action"].encode()) <= EVENT_TEXT_LIMIT
    assert summary["status"] == "running"
