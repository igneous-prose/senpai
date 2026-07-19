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
        "SENPAI_ROLE": "student",
        "STUDENT_NAME": "charlie",
    }

    assert monitoring.initialize_weave_monitoring(env) == (
        "wandb-applied-ai-team/senpai-v1"
    )
    assert calls == [
        (
            ("wandb-applied-ai-team/senpai-v1",),
            {"agent_name": "student-charlie", "capture_content": True},
        )
    ]

    monitoring.finish_weave_monitoring()

    assert calls[-1] == "finish"


def test_monitoring_requires_complete_wandb_project_configuration():
    with pytest.raises(RuntimeError, match="must be set together"):
        monitoring.weave_project_name({"WANDB_ENTITY": "wandb-applied-ai-team"})


def test_weave_openhands_instruments_the_pinned_sdk():
    uninstrument()
    try:
        instrument()
        assert is_instrumented()
    finally:
        uninstrument()
