import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from senpai_agent.monitor import (
    MetricGate,
    MetricSample,
    MonitorDecision,
    MonitorEvaluation,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorSpec,
    WandbMetricSource,
    evaluate_monitor,
)
from senpai_agent.training import TrainingResult, TrainingState


def result(
    tmp_path: Path,
    state: TrainingState = TrainingState.RUNNING,
) -> TrainingResult:
    return TrainingResult(
        training_id="train-1",
        state=state,
        exit_code=0 if state is TrainingState.FINISHED else None,
        elapsed_seconds=20,
        log_path=str(tmp_path / "train.log"),
        wandb_run_ids=("run-1",),
    )


def test_monitor_registration_is_durable_and_bound_to_the_student_conversation(
    tmp_path: Path,
):
    conversation_id = uuid4()
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=conversation_id,
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="lte", threshold=0.2),),
    )

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        assert store.register(spec) is True
        assert store.register(spec) is False

    with MonitorStore(tmp_path / "monitors.sqlite3") as reopened:
        assert reopened.active() == [spec]


def test_ordinary_polls_are_free_and_gate_signals_are_deduplicated(
    tmp_path: Path,
):
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="lte", threshold=0.2),),
    )
    now = datetime.now(UTC)

    quiet, state = evaluate_monitor(
        spec,
        result(tmp_path),
        MetricSample(value=0.3, observed_at=now),
        previous=None,
        emitted=frozenset(),
        now=now,
    )
    crossed, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        MetricSample(value=0.19, observed_at=now + timedelta(minutes=1)),
        previous=state,
        emitted=frozenset(),
        now=now + timedelta(minutes=1),
    )
    duplicate, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        MetricSample(value=0.18, observed_at=now + timedelta(minutes=2)),
        previous=state,
        emitted=frozenset(crossed.dedupe_keys),
        now=now + timedelta(minutes=2),
    )

    assert quiet.signals == ()
    assert [signal.kind for signal in crossed.signals] == ["metric_gate"]
    assert duplicate.signals == ()


def test_stale_and_terminal_changes_are_compact_actionable_signals(tmp_path: Path):
    now = datetime.now(UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="accuracy",
        direction="max",
        stale_after_seconds=60,
    )
    previous_sample = MetricSample(
        value=0.7,
        observed_at=now - timedelta(minutes=2),
    )

    stale, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        None,
        previous=previous_sample,
        emitted=frozenset(),
        now=now,
    )
    terminal, _ = evaluate_monitor(
        spec,
        result(tmp_path, TrainingState.FAILED),
        None,
        previous=previous_sample,
        emitted=frozenset(),
        now=now,
    )

    assert stale.signals[0].kind == "metric_stale"
    assert [signal.kind for signal in terminal.signals] == ["training_status"]
    assert terminal.signals[0].hard_failure is True
    assert len(terminal.signals[0].to_prompt()) < 2_000


def test_old_metric_sample_is_stale_even_when_wandb_still_returns_it(tmp_path: Path):
    now = datetime.now(UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="accuracy",
        stale_after_seconds=60,
    )
    old_sample = MetricSample(
        value=0.7,
        observed_at=now - timedelta(minutes=2),
    )

    evaluation, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        old_sample,
        previous=old_sample,
        emitted=frozenset(),
        now=now,
    )

    assert [signal.kind for signal in evaluation.signals] == ["metric_stale"]


def test_status_only_monitor_never_emits_a_metric_stale_signal(tmp_path: Path):
    now = datetime.now(UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        registered_at=now - timedelta(hours=1),
    )

    evaluation, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        None,
        previous=None,
        emitted=frozenset(),
        now=now,
    )

    assert evaluation.signals == ()


def test_wandb_metric_source_uses_the_latest_metric_history_timestamp(
    monkeypatch,
):
    class FakeRun:
        def history(self, **kwargs):
            assert kwargs == {
                "keys": ["accuracy", "_timestamp"],
                "samples": 2,
                "pandas": False,
            }
            return [
                {"accuracy": 0.6, "_timestamp": 100},
                {"accuracy": 0.7, "_timestamp": 200},
            ]

    def fake_api(**kwargs):
        assert kwargs == {"timeout": 30}
        return SimpleNamespace(run=lambda _path: FakeRun())

    fake_wandb = SimpleNamespace(Api=fake_api)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    sample = WandbMetricSource("entity", "project").latest("run-1", "accuracy")

    assert sample == MetricSample(
        value=0.7,
        observed_at=datetime.fromtimestamp(200, UTC),
    )


def test_monitor_triage_decision_is_typed():
    decision = MonitorDecision.model_validate_json(
        '{"wake_main":true,"summary":"Loss regressed.",'
        '"reason":"The assigned acceptance gate was crossed."}'
    )

    assert decision.wake_main is True
    assert decision.summary == "Loss regressed."


def test_monitor_triage_decision_is_durable_until_signal_is_acknowledged(
    tmp_path: Path,
):
    store_path = tmp_path / "monitors.sqlite3"
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
    )
    signal = MonitorSignal(
        kind="training_status",
        dedupe_key="train-1:status:finished",
        training_id="train-1",
        state=TrainingState.FINISHED,
        detail="Training reached terminal state finished.",
    )
    decision = MonitorDecision(
        wake_main=True,
        summary="Training finished cleanly.",
        reason="The student must inspect and report the result.",
    )

    with MonitorStore(store_path) as store:
        store.register(spec)
        store.record_poll(
            spec,
            MonitorEvaluation(signals=(signal,)),
            sample=None,
        )
        store.record_decision(signal.dedupe_key, decision)

    with MonitorStore(store_path) as reopened:
        assert reopened.decision(signal.dedupe_key) == decision
