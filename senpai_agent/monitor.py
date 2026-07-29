"""Durable, programmatic training-monitor state and signal evaluation."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.training import TrainingResult, TrainingState


class MetricGate(BaseModel):
    """One threshold or change that should be surfaced to the student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operator: Literal["lte", "gte", "improved_by", "regressed_by"]
    threshold: float


class TrainingMonitorSpec(BaseModel):
    """Durable monitoring policy for one training process and conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_id: str = Field(min_length=1)
    conversation_id: UUID
    metric: str | None = None
    direction: Literal["min", "max"] | None = None
    gates: tuple[MetricGate, ...] = ()
    poll_interval_seconds: float = Field(default=60, gt=0)
    stale_after_seconds: float = Field(default=600, gt=0)
    notify_on_status: frozenset[TrainingState] = Field(
        default_factory=lambda: frozenset(
            {
                TrainingState.FINISHED,
                TrainingState.FAILED,
                TrainingState.TIMED_OUT,
                TrainingState.CANCELLED,
            }
        )
    )
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, _context: object) -> None:
        if self.gates and self.metric is None:
            raise ValueError("metric gates require a metric")
        if (
            any(gate.operator in {"improved_by", "regressed_by"} for gate in self.gates)
            and self.direction is None
        ):
            raise ValueError("change gates require metric direction")


class MetricSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    observed_at: datetime


class MonitorSignal(BaseModel):
    """Compact event handed to the monitor triage child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["metric_gate", "metric_stale", "training_status"]
    dedupe_key: str
    training_id: str
    metric: str | None = None
    value: float | None = None
    state: TrainingState
    detail: str
    hard_failure: bool = False

    def to_prompt(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )


class MonitorEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signals: tuple[MonitorSignal, ...] = ()

    @property
    def dedupe_keys(self) -> tuple[str, ...]:
        return tuple(signal.dedupe_key for signal in self.signals)


class MonitorDecision(BaseModel):
    """Strict result expected from the no-context monitor triage child."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wake_main: bool
    summary: str = Field(min_length=1, max_length=2_000)
    reason: str = Field(min_length=1, max_length=1_000)


def evaluate_monitor(
    spec: TrainingMonitorSpec,
    result: TrainingResult,
    sample: MetricSample | None,
    *,
    previous: MetricSample | None,
    emitted: frozenset[str],
    now: datetime | None = None,
) -> tuple[MonitorEvaluation, MetricSample | None]:
    """Evaluate one poll without invoking a model."""

    now = (now or datetime.now(UTC)).astimezone(UTC)
    signals: list[MonitorSignal] = []

    if (
        result.state in spec.notify_on_status
        and result.state is not TrainingState.RUNNING
    ):
        key = f"{spec.training_id}:status:{result.state.value}"
        if key not in emitted:
            hard_failure = result.state in {
                TrainingState.FAILED,
                TrainingState.TIMED_OUT,
                TrainingState.CANCELLED,
            }
            signals.append(
                MonitorSignal(
                    kind="training_status",
                    dedupe_key=key,
                    training_id=spec.training_id,
                    metric=spec.metric,
                    value=sample.value if sample is not None else None,
                    state=result.state,
                    detail=(
                        f"Training reached terminal state {result.state.value}"
                        + (
                            f" with exit code {result.exit_code}."
                            if result.exit_code is not None
                            else "."
                        )
                    ),
                    hard_failure=hard_failure,
                )
            )

    if result.state is not TrainingState.RUNNING:
        return MonitorEvaluation(signals=tuple(signals)), sample or previous

    if sample is not None:
        for index, gate in enumerate(spec.gates):
            key = f"{spec.training_id}:gate:{index}"
            if key in emitted or not _gate_crossed(
                gate,
                spec.direction,
                previous,
                sample,
            ):
                continue
            signals.append(
                MonitorSignal(
                    kind="metric_gate",
                    dedupe_key=key,
                    training_id=spec.training_id,
                    metric=spec.metric,
                    value=sample.value,
                    state=result.state,
                    detail=(
                        f"{spec.metric} crossed {gate.operator} "
                        f"{gate.threshold:g} at {sample.value:g}."
                    ),
                )
            )
    if spec.metric is not None:
        latest = sample or previous
        last_update = (
            latest.observed_at.astimezone(UTC)
            if latest is not None
            else spec.registered_at.astimezone(UTC)
        )
        stale_key = f"{spec.training_id}:stale:{last_update.isoformat()}"
        age = (now - last_update).total_seconds()
        if age >= spec.stale_after_seconds and stale_key not in emitted:
            signals.append(
                MonitorSignal(
                    kind="metric_stale",
                    dedupe_key=stale_key,
                    training_id=spec.training_id,
                    metric=spec.metric,
                    value=latest.value if latest is not None else None,
                    state=result.state,
                    detail=(f"{spec.metric} has not updated for {round(age)} seconds."),
                )
            )

    return MonitorEvaluation(signals=tuple(signals)), sample or previous


def _gate_crossed(
    gate: MetricGate,
    direction: Literal["min", "max"] | None,
    previous: MetricSample | None,
    sample: MetricSample,
) -> bool:
    if gate.operator == "lte":
        return sample.value <= gate.threshold and (
            previous is None or previous.value > gate.threshold
        )
    if gate.operator == "gte":
        return sample.value >= gate.threshold and (
            previous is None or previous.value < gate.threshold
        )
    if previous is None or direction is None:
        return False
    improvement = (
        previous.value - sample.value
        if direction == "min"
        else sample.value - previous.value
    )
    if gate.operator == "improved_by":
        return improvement >= gate.threshold
    return -improvement >= gate.threshold


class MonitorStore:
    """SQLite state plus a tiny JSON presence marker used by the Stop hook."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.marker_dir = path.parent / "monitors"
        self.marker_dir.mkdir(exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS monitors (
                training_id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                previous_sample_json TEXT,
                next_poll_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS monitor_signals (
                dedupe_key TEXT PRIMARY KEY,
                training_id TEXT NOT NULL,
                signal_json TEXT NOT NULL,
                decision_json TEXT,
                handled INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(monitor_signals)")
        }
        if "decision_json" not in columns:
            self.connection.execute(
                "ALTER TABLE monitor_signals ADD COLUMN decision_json TEXT"
            )
        self.connection.commit()

    def register(self, spec: TrainingMonitorSpec) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO monitors (training_id, spec_json)
            VALUES (?, ?)
            """,
            (spec.training_id, spec.model_dump_json()),
        )
        self.connection.commit()
        marker = self.marker_dir / f"{spec.training_id}.json"
        marker.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        return cursor.rowcount == 1

    def active(self) -> list[TrainingMonitorSpec]:
        rows = self.connection.execute(
            """
            SELECT spec_json FROM monitors
            WHERE active = 1
            ORDER BY rowid
            """
        ).fetchall()
        return [TrainingMonitorSpec.model_validate_json(row[0]) for row in rows]

    def spec(self, training_id: str) -> TrainingMonitorSpec:
        row = self.connection.execute(
            "SELECT spec_json FROM monitors WHERE training_id = ?",
            (training_id,),
        ).fetchone()
        if row is None:
            raise KeyError(training_id)
        return TrainingMonitorSpec.model_validate_json(row[0])

    def due(
        self,
        now: datetime | None = None,
    ) -> list[TrainingMonitorSpec]:
        timestamp = (now or datetime.now(UTC)).timestamp()
        rows = self.connection.execute(
            """
            SELECT spec_json FROM monitors
            WHERE active = 1 AND next_poll_at <= ?
            ORDER BY rowid
            """,
            (timestamp,),
        ).fetchall()
        return [TrainingMonitorSpec.model_validate_json(row[0]) for row in rows]

    def emitted(self, training_id: str) -> frozenset[str]:
        rows = self.connection.execute(
            "SELECT dedupe_key FROM monitor_signals WHERE training_id = ?",
            (training_id,),
        ).fetchall()
        return frozenset(row[0] for row in rows)

    def previous_sample(self, training_id: str) -> MetricSample | None:
        row = self.connection.execute(
            "SELECT previous_sample_json FROM monitors WHERE training_id = ?",
            (training_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return MetricSample.model_validate_json(row[0])

    def record_poll(
        self,
        spec: TrainingMonitorSpec,
        evaluation: MonitorEvaluation,
        sample: MetricSample | None,
        *,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(UTC)
        for signal in evaluation.signals:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO monitor_signals
                (dedupe_key, training_id, signal_json)
                VALUES (?, ?, ?)
                """,
                (
                    signal.dedupe_key,
                    spec.training_id,
                    signal.model_dump_json(),
                ),
            )
        self.connection.execute(
            """
            UPDATE monitors
            SET previous_sample_json = ?, next_poll_at = ?
            WHERE training_id = ?
            """,
            (
                sample.model_dump_json() if sample is not None else None,
                now.timestamp() + spec.poll_interval_seconds,
                spec.training_id,
            ),
        )
        self.connection.commit()

    def pending_signals(self) -> list[MonitorSignal]:
        rows = self.connection.execute(
            """
            SELECT signal_json FROM monitor_signals
            WHERE handled = 0
            ORDER BY rowid
            """
        ).fetchall()
        return [MonitorSignal.model_validate_json(row[0]) for row in rows]

    def decision(self, dedupe_key: str) -> MonitorDecision | None:
        row = self.connection.execute(
            "SELECT decision_json FROM monitor_signals WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return MonitorDecision.model_validate_json(row[0])

    def record_decision(
        self,
        dedupe_key: str,
        decision: MonitorDecision,
    ) -> None:
        self.connection.execute(
            """
            UPDATE monitor_signals
            SET decision_json = ?
            WHERE dedupe_key = ?
            """,
            (decision.model_dump_json(), dedupe_key),
        )
        self.connection.commit()

    def acknowledge(self, dedupe_key: str) -> None:
        self.connection.execute(
            "UPDATE monitor_signals SET handled = 1 WHERE dedupe_key = ?",
            (dedupe_key,),
        )
        self.connection.commit()

    def complete(self, training_id: str) -> None:
        self.connection.execute(
            "UPDATE monitors SET active = 0 WHERE training_id = ?",
            (training_id,),
        )
        self.connection.commit()
        (self.marker_dir / f"{training_id}.json").unlink(missing_ok=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class TrainingStatusSource(Protocol):
    def get_training_status(self, training_id: str) -> TrainingResult: ...


class MetricSource(Protocol):
    def latest(self, run_id: str, metric: str) -> MetricSample | None: ...


class WandbMetricSource:
    """Fetch one latest metric value without carrying history into the agent."""

    def __init__(self, entity: str, project: str, timeout_seconds: int = 30):
        self.entity = entity
        self.project = project
        self.timeout_seconds = timeout_seconds

    def latest(self, run_id: str, metric: str) -> MetricSample | None:
        import wandb

        run = wandb.Api(timeout=self.timeout_seconds).run(
            f"{self.entity}/{self.project}/{run_id}"
        )
        rows = run.history(
            keys=[metric, "_timestamp"],
            samples=2,
            pandas=False,
        )
        samples = [row for row in rows if row.get(metric) is not None]
        if not samples:
            return None
        latest = samples[-1]
        value = latest[metric]
        timestamp = latest.get("_timestamp")
        observed_at = (
            datetime.fromtimestamp(float(timestamp), UTC)
            if timestamp is not None
            else datetime.now(UTC)
        )
        return MetricSample(value=float(value), observed_at=observed_at)


class TrainingMonitorEngine:
    """Poll due monitors and persist only compact, deduplicated signals."""

    def __init__(
        self,
        store: MonitorStore,
        training: TrainingStatusSource,
        metrics: MetricSource,
    ):
        self.store = store
        self.training = training
        self.metrics = metrics

    def poll(self, now: datetime | None = None) -> tuple[MonitorSignal, ...]:
        now = now or datetime.now(UTC)
        produced: list[MonitorSignal] = []
        for spec in self.store.due(now):
            result = self.training.get_training_status(spec.training_id)
            sample = None
            if spec.metric and result.wandb_run_ids:
                sample = self.metrics.latest(result.wandb_run_ids[-1], spec.metric)
            evaluation, latest = evaluate_monitor(
                spec,
                result,
                sample,
                previous=self.store.previous_sample(spec.training_id),
                emitted=self.store.emitted(spec.training_id),
                now=now,
            )
            self.store.record_poll(spec, evaluation, latest, now=now)
            produced.extend(evaluation.signals)
            if result.state is not TrainingState.RUNNING:
                self.store.complete(spec.training_id)
        return tuple(produced)
