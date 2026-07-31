import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from senpai_agent.monitor import (
    MetricGate,
    MetricSample,
    MonitorEvaluation,
    MonitorSignal,
    MonitorStore,
    TrainingMonitorEngine,
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


def test_identical_monitor_policy_is_idempotent_despite_a_new_registration_time(
    tmp_path: Path,
):
    registered_at = datetime(2026, 7, 30, tzinfo=UTC)
    original = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="lte", threshold=0.2),),
        registered_at=registered_at,
    )
    repeated = original.model_copy(
        update={"registered_at": registered_at + timedelta(minutes=5)}
    )

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        assert store.register(original) is True
        assert store.register(repeated) is False
        assert store.spec("train-1") == original
        marker = TrainingMonitorSpec.model_validate_json(
            (store.marker_dir / "train-1.json").read_text()
        )

    assert marker == original


def test_changed_monitor_policy_replaces_policy_and_resets_derived_state(
    tmp_path: Path,
):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    original = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="lte", threshold=0.2),),
        registered_at=now,
    )
    changed = original.model_copy(
        update={
            "gates": (MetricGate(operator="lte", threshold=0.1),),
            "registered_at": now + timedelta(minutes=5),
        }
    )
    old_signal = MonitorSignal(
        kind="metric_gate",
        dedupe_key="train-1:gate:0",
        training_id="train-1",
        metric="val/loss",
        value=0.19,
        state=TrainingState.RUNNING,
        detail="The old policy fired.",
    )

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(original)
        store.record_poll(
            original,
            MonitorEvaluation(signals=(old_signal,)),
            MetricSample(value=0.19, observed_at=now),
            now=now,
        )
        store.complete(original.training_id)

        assert store.register(changed) is True
        assert store.spec("train-1") == changed
        assert store.active() == [changed]
        assert store.previous_sample("train-1") is None
        assert store.pending_signals() == []
        assert store.emitted("train-1") == frozenset()
        assert store.due(now + timedelta(minutes=5)) == [changed]
        marker = TrainingMonitorSpec.model_validate_json(
            (store.marker_dir / "train-1.json").read_text()
        )

    assert marker == changed


@pytest.mark.parametrize(
    "build",
    [
        lambda: MetricGate(operator="lte", threshold=float("nan")),
        lambda: MetricGate(operator="lte", threshold=float("inf")),
        lambda: TrainingMonitorSpec(
            training_id="train-1",
            conversation_id=uuid4(),
            poll_interval_seconds=float("inf"),
        ),
        lambda: TrainingMonitorSpec(
            training_id="train-1",
            conversation_id=uuid4(),
            stale_after_seconds=float("inf"),
        ),
        lambda: MetricSample(
            value=float("nan"),
            observed_at=datetime.now(UTC),
        ),
        lambda: MetricSample(
            value=float("-inf"),
            observed_at=datetime.now(UTC),
        ),
        lambda: MonitorSignal(
            kind="metric_gate",
            dedupe_key="train-1:gate:0",
            training_id="train-1",
            value=float("inf"),
            state=TrainingState.RUNNING,
            detail="Invalid signal.",
        ),
    ],
)
def test_monitor_numeric_contracts_reject_non_finite_values(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize("failure_site", ["status", "metric"])
def test_failed_monitor_poll_is_durable_and_does_not_block_other_monitors(
    tmp_path: Path,
    failure_site: str,
):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    bad = TrainingMonitorSpec(
        training_id="train-bad",
        conversation_id=uuid4(),
        metric="val/loss",
        poll_interval_seconds=60,
        registered_at=now,
    )
    good = TrainingMonitorSpec(
        training_id="train-good",
        conversation_id=uuid4(),
        notify_on_status=frozenset({TrainingState.FINISHED}),
        poll_interval_seconds=60,
        registered_at=now,
    )

    class Training:
        def __init__(self):
            self.calls = []

        def get_training_status(self, training_id):
            self.calls.append(training_id)
            if training_id == bad.training_id and failure_site == "status":
                raise RuntimeError("status backend unavailable")
            return TrainingResult(
                training_id=training_id,
                state=(
                    TrainingState.RUNNING
                    if training_id == bad.training_id
                    else TrainingState.FINISHED
                ),
                exit_code=0 if training_id == good.training_id else None,
                elapsed_seconds=20,
                log_path=str(tmp_path / f"{training_id}.log"),
                wandb_run_ids=(f"run-{training_id}",),
            )

    class Metrics:
        def latest(self, run_id, metric):
            if failure_site == "metric" and run_id == "run-train-bad":
                raise ValueError("val/loss returned non-finite value nan")
            return MetricSample(value=0.2, observed_at=now)

    training = Training()
    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(bad)
        store.register(good)
        produced = TrainingMonitorEngine(store, training, Metrics()).poll(now)

        assert training.calls == ["train-bad", "train-good"]
        assert {signal.kind for signal in produced} == {
            "monitor_error",
            "training_status",
        }
        error = next(signal for signal in produced if signal.kind == "monitor_error")
        assert error.training_id == bad.training_id
        assert error.hard_failure is True
        assert error.state is (
            None if failure_site == "status" else TrainingState.RUNNING
        )
        assert len(error.model_dump_json()) < 2_000
        assert store.pending_signals() == list(produced)
        assert store.due(now + timedelta(seconds=59)) == []


def test_legacy_non_finite_sample_is_repaired_without_blocking_other_monitors(
    tmp_path: Path,
):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    bad = TrainingMonitorSpec(
        training_id="train-bad",
        conversation_id=uuid4(),
        metric="val/loss",
        poll_interval_seconds=60,
        registered_at=now,
    )
    good = TrainingMonitorSpec(
        training_id="train-good",
        conversation_id=uuid4(),
        notify_on_status=frozenset({TrainingState.FINISHED}),
        poll_interval_seconds=60,
        registered_at=now,
    )

    class Training:
        def get_training_status(self, training_id):
            return TrainingResult(
                training_id=training_id,
                state=(
                    TrainingState.RUNNING
                    if training_id == bad.training_id
                    else TrainingState.FINISHED
                ),
                exit_code=0 if training_id == good.training_id else None,
                elapsed_seconds=20,
                log_path=str(tmp_path / f"{training_id}.log"),
                wandb_run_ids=(f"run-{training_id}",),
            )

    class Metrics:
        def latest(self, _run_id, _metric):
            return MetricSample(value=0.2, observed_at=now)

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(bad)
        store.register(good)
        store.connection.execute(
            """
            UPDATE monitors
            SET previous_sample_json = ?
            WHERE training_id = ?
            """,
            (
                '{"value":null,"observed_at":"2026-07-30T00:00:00Z"}',
                bad.training_id,
            ),
        )
        store.connection.commit()

        produced = TrainingMonitorEngine(store, Training(), Metrics()).poll(now)

        assert {signal.kind for signal in produced} == {
            "monitor_error",
            "training_status",
        }
        assert store.previous_sample(bad.training_id) is None
        assert store.due(now + timedelta(seconds=59)) == []


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


@pytest.mark.parametrize(
    ("direction", "operator", "baseline_value", "middle_value", "crossed_value"),
    [
        ("min", "improved_by", 1.0, 0.96, 0.91),
        ("min", "regressed_by", 1.0, 1.04, 1.09),
        ("max", "improved_by", 0.5, 0.54, 0.59),
        ("max", "regressed_by", 0.5, 0.46, 0.41),
    ],
)
def test_change_gates_compare_with_the_first_observed_baseline(
    tmp_path: Path,
    direction: str,
    operator: str,
    baseline_value: float,
    middle_value: float,
    crossed_value: float,
):
    now = datetime.now(UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="score",
        direction=direction,
        gates=(MetricGate(operator=operator, threshold=0.08),),
    )
    baseline = MetricSample(value=baseline_value, observed_at=now)
    middle = MetricSample(
        value=middle_value,
        observed_at=now + timedelta(minutes=1),
    )
    crossed = MetricSample(
        value=crossed_value,
        observed_at=now + timedelta(minutes=2),
    )

    first, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        baseline,
        previous=None,
        baseline=None,
        emitted=frozenset(),
        now=now,
    )
    quiet, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        middle,
        previous=baseline,
        baseline=baseline,
        emitted=frozenset(),
        now=now + timedelta(minutes=1),
    )
    fired, _ = evaluate_monitor(
        spec,
        result(tmp_path),
        crossed,
        previous=middle,
        baseline=baseline,
        emitted=frozenset(),
        now=now + timedelta(minutes=2),
    )

    assert first.signals == ()
    assert quiet.signals == ()
    assert [signal.kind for signal in fired.signals] == ["metric_gate"]


def test_monitor_store_persists_the_first_sample_as_the_change_baseline(
    tmp_path: Path,
):
    now = datetime.now(UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="improved_by", threshold=0.1),),
    )
    first = MetricSample(value=0.8, observed_at=now)
    later = MetricSample(value=0.75, observed_at=now + timedelta(minutes=1))

    with MonitorStore(tmp_path / "monitors.sqlite3") as store:
        store.register(spec)
        store.record_poll(spec, MonitorEvaluation(), first, now=now)
        store.record_poll(
            spec,
            MonitorEvaluation(),
            later,
            now=now + timedelta(minutes=1),
        )

        assert store.baseline_sample(spec.training_id) == first
        assert store.previous_sample(spec.training_id) == later


@pytest.mark.parametrize("baseline_column_exists", [False, True])
def test_legacy_schema_promotes_the_previous_sample_to_the_change_baseline(
    tmp_path: Path,
    baseline_column_exists: bool,
):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    spec = TrainingMonitorSpec(
        training_id="train-1",
        conversation_id=uuid4(),
        metric="val/loss",
        direction="min",
        gates=(MetricGate(operator="improved_by", threshold=0.1),),
        registered_at=now - timedelta(minutes=1),
    )
    previous = MetricSample(
        value=1.0,
        observed_at=now - timedelta(minutes=1),
    )
    current = MetricSample(value=0.89, observed_at=now)
    database = tmp_path / "monitors.sqlite3"

    baseline_column = (
        "baseline_sample_json TEXT," if baseline_column_exists else ""
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"""
            CREATE TABLE monitors (
                training_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                previous_sample_json TEXT,
                {baseline_column}
                next_poll_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            INSERT INTO monitors (
                training_id,
                spec_json,
                previous_sample_json
            )
            VALUES (?, ?, ?)
            """,
            (
                spec.training_id,
                spec.model_dump_json(),
                previous.model_dump_json(),
            ),
        )

    class Training:
        def get_training_status(self, training_id):
            return result(tmp_path).model_copy(update={"training_id": training_id})

    class Metrics:
        def latest(self, _run_id, _metric):
            return current

    with MonitorStore(database) as store:
        assert store.baseline_sample(spec.training_id) == previous

        produced = TrainingMonitorEngine(store, Training(), Metrics()).poll(now)

        assert [signal.kind for signal in produced] == ["metric_gate"]
        assert store.baseline_sample(spec.training_id) == previous
        assert store.previous_sample(spec.training_id) == current


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
        MetricSample(value=0.9, observed_at=now),
        previous=previous_sample,
        emitted=frozenset(),
        now=now,
    )

    assert stale.signals[0].kind == "metric_stale"
    assert [signal.kind for signal in terminal.signals] == ["training_status"]
    assert terminal.signals[0].hard_failure is True
    assert terminal.signals[0].metric is None
    assert terminal.signals[0].value is None
    assert len(terminal.signals[0].model_dump_json()) < 2_000


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
