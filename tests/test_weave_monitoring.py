import pytest
from weave_openhands import instrument, is_instrumented, uninstrument

import senpai_agent.weave_monitoring as monitoring


def test_monitoring_uses_the_senpai_wandb_project_and_student_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(monitoring, "_initialized", False)
    monkeypatch.setattr(monitoring, "_project_name", None)
    monkeypatch.setattr(
        monitoring,
        "weave_init",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(monitoring, "weave_finish", lambda: calls.append("finish"))
    env = {
        "WANDB_ENTITY": "wandb-applied-ai-team",
        "WANDB_PROJECT": "senpai-v1",
        "WANDB_API_KEY": "wandb-secret",
        "SENPAI_ROLE": "student",
        "STUDENT_NAME": "charlie",
    }

    assert monitoring.initialize_weave_monitoring(env) == (
        "wandb-applied-ai-team/senpai-v1"
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("wandb-applied-ai-team/senpai-v1",)
    assert kwargs["agent_name"] == "student-charlie"
    assert kwargs["capture_content"] is True
    assert kwargs["content_transform"]("token=wandb-secret") == "token=<secret-hidden>"

    monitoring.finish_weave_monitoring()

    assert calls[-1] == "finish"


def test_monitoring_requires_complete_wandb_project_configuration():
    with pytest.raises(RuntimeError, match="must be set together"):
        monitoring.weave_project_name({"WANDB_ENTITY": "wandb-applied-ai-team"})


def test_secret_redactor_replaces_overlapping_values_longest_first():
    redact = monitoring.secret_redactor(
        {
            "GITHUB_TOKEN": "token-prefix",
            "GH_TOKEN": "token",
            "WANDB_API_KEY": "wandb-secret",
            "EXA_API_KEY": "exa-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "OPENAI_API_KEY": "openai-secret",
            "SERVICE_PASSWORD": "service-secret",
            "DATABASE_PASSWORD": "database-secret",
            "CLIENT_SECRET": "client-secret",
            "SENPAI_OPENHANDS_API_KEY_ENV": "CUSTOM_MODEL_CREDENTIAL",
            "CUSTOM_MODEL_CREDENTIAL": "custom-model-secret",
        }
    )

    assert redact(
        "token-prefix token wandb-secret exa-secret anthropic-secret "
        "openai-secret service-secret database-secret client-secret "
        "custom-model-secret"
    ) == " ".join(["<secret-hidden>"] * 10)


def test_weave_openhands_instruments_the_pinned_sdk():
    uninstrument()
    try:
        instrument()
        assert is_instrumented()
    finally:
        uninstrument()
