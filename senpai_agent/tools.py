"""Typed OpenHands tools for Senpai's reliable control-plane operations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.terminal import (
    TerminalAction,
    TerminalObservation,
    TerminalTool,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from senpai_agent.delegation import (
    AdvisorEventSink,
    ChildAgentRunnerFactory,
    DelegateAgentTool,
)
from senpai_agent.git_workflow import (
    create_assignment_branch,
    push_assignment_branch,
)
from senpai_agent.github import PRRetrievalResult, get_prs
from senpai_agent.github_workflow import GitHubWorkflow, MutationResult
from senpai_agent.models import (
    AssignmentRecord,
    DispositionRecord,
    ExperimentResult,
    render_disposition_marker,
)
from senpai_agent.monitor import MetricGate, MonitorStore, TrainingMonitorSpec
from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class TrainingService(Protocol):
    def run_training(self, spec: TrainingSpec) -> TrainingResult: ...

    def get_training_status(self, training_id: str) -> TrainingResult: ...

    def close(self) -> None: ...

    def drain(self) -> None: ...


class GitHubWorkflowService(Protocol):
    def create_assignment(
        self,
        assignment: AssignmentRecord,
        *,
        title: str,
        body: str,
    ) -> MutationResult: ...

    def reconcile_labels(
        self,
        number: int,
        *,
        assignment_id: str,
        add: set[str],
        remove: set[str],
        expected_head_sha: str,
    ) -> MutationResult: ...

    def request_revision(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        revision_id: str,
        comment: str,
    ) -> MutationResult: ...

    def respond_to_issue(
        self,
        number: int,
        *,
        human_message_id: int,
        response: str,
    ) -> MutationResult: ...

    def submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult: ...

    def preflight_submit_result(
        self,
        number: int,
        *,
        branch: str,
        current_head_sha: str,
        expected_result_head_sha: str,
        result: ExperimentResult,
    ) -> object: ...

    def close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult: ...

    def merge_experiment(
        self,
        number: int,
        *,
        expected_head_sha: str,
        assignment_id: str,
        merge_method: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MutationResult: ...


@dataclass(frozen=True)
class GitHubCredentials:
    repo: str
    token: SecretStr
    trusted_actor: str | None = None


_GITHUB_CREDENTIALS: GitHubCredentials | None = None
_TRAINING_RUNTIMES: dict[
    Path,
    tuple[TrainingSupervisor, MonitorStore],
] = {}


def configure_github_credentials(
    repo: str,
    token: SecretStr,
    *,
    trusted_actor: str | None = None,
) -> None:
    """Hold write auth outside model-facing tool specs and terminal secrets."""

    global _GITHUB_CREDENTIALS
    if len(repo.split("/")) != 2 or not all(repo.split("/")):
        raise ValueError("repo must use owner/name form")
    if not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
    if trusted_actor is not None and not trusted_actor.strip():
        raise ValueError("trusted actor must not be empty")
    _GITHUB_CREDENTIALS = GitHubCredentials(
        repo=repo,
        token=token,
        trusted_actor=trusted_actor,
    )


def clear_github_credentials() -> None:
    global _GITHUB_CREDENTIALS
    _GITHUB_CREDENTIALS = None


def training_runtime(
    workspace: Path,
    state_dir: Path,
    *,
    max_timeout_seconds: int | None = None,
) -> tuple[TrainingSupervisor, MonitorStore]:
    key = state_dir.resolve()
    runtime = _TRAINING_RUNTIMES.get(key)
    if runtime is None:
        runtime = (
            TrainingSupervisor(
                workspace=workspace,
                state_dir=key,
                max_timeout_seconds=max_timeout_seconds,
            ),
            MonitorStore(key / "monitors.sqlite3"),
        )
        _TRAINING_RUNTIMES[key] = runtime
    return runtime


def close_training_runtimes() -> None:
    for training, monitors in _TRAINING_RUNTIMES.values():
        training.close()
        monitors.close()
    _TRAINING_RUNTIMES.clear()


class RunTrainingAction(Action):
    spec: TrainingSpec = Field(
        description=(
            "Structured process argv, assignment-workspace directory, and hard "
            "timeout. Do not pass a shell command string."
        )
    )


class GetTrainingStatusAction(Action):
    training_id: str = Field(
        min_length=1,
        description="Training ID returned by run_training.",
    )


class MonitorTrainingAction(Action):
    training_id: str = Field(
        min_length=1,
        description="Training ID returned by run_training.",
    )
    metric: str | None = Field(
        default=None,
        description="W&B metric to monitor. Omit for terminal process state only.",
    )
    direction: Literal["min", "max"] | None = Field(
        default=None,
        description="Whether lower or higher values are better for change gates.",
    )
    gates: tuple[MetricGate, ...] = Field(
        default=(),
        description=(
            "Metric thresholds or changes that should be triaged. Ordinary polls "
            "are programmatic and do not consume model tokens."
        ),
    )
    poll_interval_seconds: float = Field(
        default=60,
        gt=0,
        description="Seconds between programmatic monitor polls.",
    )
    stale_after_seconds: float = Field(
        default=600,
        gt=0,
        description="Notify when the selected metric has not updated this long.",
    )
    notify_on_status: frozenset[TrainingState] = Field(
        default_factory=lambda: frozenset(
            {
                TrainingState.FINISHED,
                TrainingState.FAILED,
                TrainingState.TIMED_OUT,
                TrainingState.CANCELLED,
            }
        ),
        description="Training state changes that should be triaged.",
    )


class MonitorTrainingObservation(Observation):
    training_id: str
    conversation_id: str
    status: Literal["monitoring"] = "monitoring"

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=(
                    f"Training {self.training_id} is durably monitored. You may "
                    "finish this turn; the controller will resume this same "
                    f"conversation ({self.conversation_id}) when action is needed."
                )
            )
        ]


class TrainingResultObservation(Observation):
    training_id: str
    state: TrainingState
    pid: int | None = None
    process_group_id: int | None = None
    process_start_time: float | None = None
    exit_code: int | None
    elapsed_seconds: float
    log_path: str
    wandb_run_ids: tuple[str, ...] = ()
    error_tail: str = ""

    @classmethod
    def from_result(cls, result: TrainingResult) -> Self:
        return cls.model_validate(result.model_dump())

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        result = {
            "training_id": self.training_id,
            "state": self.state,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "log_path": self.log_path,
            "wandb_run_ids": self.wandb_run_ids,
        }
        if self.error_tail:
            result["error_tail"] = self.error_tail
        text = json.dumps(result, separators=(",", ":"), default=str)
        return [TextContent(text=text)]


class _RunTrainingExecutor(ToolExecutor[RunTrainingAction, TrainingResultObservation]):
    def __init__(self, training: TrainingService):
        self.training = training

    def __call__(
        self,
        action: RunTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        return TrainingResultObservation.from_result(
            self.training.run_training(action.spec)
        )

    def close(self) -> None:
        return

    def interrupt(self) -> None:
        self.training.close()


class _GetTrainingStatusExecutor(
    ToolExecutor[GetTrainingStatusAction, TrainingResultObservation]
):
    def __init__(self, training: TrainingService):
        self.training = training

    def __call__(
        self,
        action: GetTrainingStatusAction,
        conversation: LocalConversation | None = None,
    ) -> TrainingResultObservation:
        return TrainingResultObservation.from_result(
            self.training.get_training_status(action.training_id)
        )


class RunTrainingTool(ToolDefinition[RunTrainingAction, TrainingResultObservation]):
    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        training: TrainingService | None = None,
        *,
        state_dir: str | Path | None = None,
    ) -> Sequence[Self]:
        if training is None:
            if conv_state is None or state_dir is None:
                raise ValueError("conv_state and state_dir are required")
            workspace = Path(conv_state.workspace.working_dir)
            training = TrainingSupervisor(
                workspace=workspace,
                state_dir=Path(state_dir),
            )
        return [
            cls(
                description=(
                    "Start one supervised training process without blocking and "
                    "return its training ID. Use get_training_status for progress "
                    "and the compact terminal result."
                ),
                action_type=RunTrainingAction,
                observation_type=TrainingResultObservation,
                annotations=ToolAnnotations(
                    title="Run training",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=_RunTrainingExecutor(training),
            )
        ]


class GetTrainingStatusTool(
    ToolDefinition[GetTrainingStatusAction, TrainingResultObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        training: TrainingService | None = None,
        *,
        state_dir: str | Path | None = None,
    ) -> Sequence[Self]:
        if training is None:
            if conv_state is None or state_dir is None:
                raise ValueError("conv_state and state_dir are required")
            training = TrainingSupervisor(
                workspace=Path(conv_state.workspace.working_dir),
                state_dir=Path(state_dir),
            )
        return [
            cls(
                description=(
                    "Read the latest persisted result for one supervised training ID."
                ),
                action_type=GetTrainingStatusAction,
                observation_type=TrainingResultObservation,
                annotations=ToolAnnotations(
                    title="Get training status",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_GetTrainingStatusExecutor(training),
            )
        ]


class _MonitorTrainingExecutor(
    ToolExecutor[MonitorTrainingAction, MonitorTrainingObservation]
):
    def __init__(self, training: TrainingService, store: MonitorStore):
        self.training = training
        self.store = store

    def __call__(
        self,
        action: MonitorTrainingAction,
        conversation: LocalConversation | None = None,
    ) -> MonitorTrainingObservation:
        if conversation is None:
            raise ValueError("monitor_training requires its student conversation")
        self.training.get_training_status(action.training_id)
        spec = TrainingMonitorSpec(
            training_id=action.training_id,
            conversation_id=conversation.id,
            metric=action.metric,
            direction=action.direction,
            gates=action.gates,
            poll_interval_seconds=action.poll_interval_seconds,
            stale_after_seconds=action.stale_after_seconds,
            notify_on_status=action.notify_on_status,
        )
        self.store.register(spec)
        return MonitorTrainingObservation(
            training_id=spec.training_id,
            conversation_id=str(spec.conversation_id),
        )

    def close(self) -> None:
        return


class MonitorTrainingTool(
    ToolDefinition[MonitorTrainingAction, MonitorTrainingObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        training: TrainingService | None = None,
        monitor_store: MonitorStore | None = None,
        *,
        state_dir: str | Path | None = None,
    ) -> Sequence[Self]:
        if training is None or monitor_store is None:
            if conv_state is None or state_dir is None:
                raise ValueError("conv_state and state_dir are required")
            training = training or TrainingSupervisor(
                workspace=Path(conv_state.workspace.working_dir),
                state_dir=Path(state_dir),
            )
            monitor_store = monitor_store or MonitorStore(
                Path(state_dir) / "monitors.sqlite3"
            )
        return [
            cls(
                description=(
                    "Durably monitor one training process without model polling. "
                    "Specify an optional W&B metric, direction, threshold/change "
                    "gates, stale timeout, and terminal states. Senpai sends only "
                    "actionable signals through a small triage child, then resumes "
                    "this same student conversation when needed."
                ),
                action_type=MonitorTrainingAction,
                observation_type=MonitorTrainingObservation,
                annotations=ToolAnnotations(
                    title="Monitor training",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=_MonitorTrainingExecutor(training, monitor_store),
            )
        ]


class TrainingToolSet(ToolDefinition[RunTrainingAction, TrainingResultObservation]):
    """Create both training tools around one process supervisor."""

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        state_dir: str | Path,
        max_timeout_seconds: int | None = None,
    ) -> Sequence[ToolDefinition]:
        training, monitor_store = training_runtime(
            Path(conv_state.workspace.working_dir),
            Path(state_dir),
            max_timeout_seconds=max_timeout_seconds,
        )
        return (
            *RunTrainingTool.create(training=training),
            *GetTrainingStatusTool.create(training=training),
            *MonitorTrainingTool.create(
                training=training,
                monitor_store=monitor_store,
            ),
        )


class GetPRsAction(Action):
    repo: str = Field(
        min_length=3,
        description="GitHub repository in owner/name form.",
    )
    numbers: tuple[int, ...] = Field(
        default=(),
        description="Explicit positive PR numbers to include.",
    )
    date_range: tuple[str | date, str | date] | None = Field(
        default=None,
        description="Optional inclusive PR creation-date range.",
    )
    search: str | None = Field(
        default=None,
        description="Optional GitHub issue-search terms or qualifiers.",
    )
    max_inline_prs: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum PRs returned inline. Do not set this >5 unless explicitly "
            "necessary: more than 5 inline PRs risks severe agent context "
            "pollution. Prefer the returned artifact path."
        ),
    )


class PRManifestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    head_sha: str
    url: str


class GetPRsObservation(Observation):
    manifest: tuple[PRManifestObservation, ...]
    markdown: str | None = None
    path: str | None = None

    @classmethod
    def from_result(cls, result: PRRetrievalResult) -> Self:
        return cls(
            manifest=tuple(
                PRManifestObservation(
                    number=entry.number,
                    title=entry.title,
                    head_sha=entry.head_sha,
                    url=entry.url,
                )
                for entry in result.manifest
            ),
            markdown=result.markdown,
            path=str(result.path) if result.path is not None else None,
        )

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        if self.markdown is not None:
            return [TextContent(text=self.markdown)]

        manifest = "\n".join(
            (f"- #{entry.number} `{entry.head_sha}` {entry.title} ({entry.url})")
            for entry in self.manifest
        )
        return [
            TextContent(
                text=(
                    f"Full PR context is stored at: {self.path}\n"
                    f"Compact manifest:\n{manifest}"
                )
            )
        ]


class _GetPRsExecutor(ToolExecutor[GetPRsAction, GetPRsObservation]):
    def __init__(
        self,
        get_prs_fn: Callable[..., PRRetrievalResult],
        *,
        credentials: GitHubCredentials | None = None,
        artifact_dir: Path,
        target_workspace: Path,
    ):
        self.get_prs = get_prs_fn
        self.credentials = credentials
        self.artifact_dir = artifact_dir
        self.target_workspace = target_workspace

    def __call__(
        self,
        action: GetPRsAction,
        conversation: LocalConversation | None = None,
    ) -> GetPRsObservation:
        if self.credentials is not None and action.repo != self.credentials.repo:
            raise PermissionError(
                "requested repository does not match configured GitHub credentials"
            )
        auth = {"token": self.credentials.token} if self.credentials is not None else {}
        result = self.get_prs(
            action.repo,
            numbers=action.numbers,
            date_range=action.date_range,
            search=action.search,
            max_inline_prs=action.max_inline_prs,
            artifact_dir=self.artifact_dir,
            target_workspace=self.target_workspace,
            **auth,
        )
        return GetPRsObservation.from_result(result)


class GetPRsTool(ToolDefinition[GetPRsAction, GetPRsObservation]):
    name = "get_prs"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        *,
        get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
        state_dir: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> Sequence[Self]:
        credentials = _GITHUB_CREDENTIALS if get_prs_fn is get_prs else None
        if get_prs_fn is get_prs and credentials is None:
            raise RuntimeError(
                "configure GitHub credentials before initializing get_prs"
            )
        if workspace is None and conv_state is not None:
            workspace = Path(conv_state.workspace.working_dir)
        target_workspace = (
            Path(workspace).resolve() if workspace is not None else Path.cwd()
        )
        artifact_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else Path(tempfile.gettempdir()).resolve() / "senpai-pr-artifacts"
        )
        if artifact_dir == target_workspace or artifact_dir.is_relative_to(
            target_workspace
        ):
            raise ValueError("get_prs state_dir must be outside the target workspace")
        return [
            cls(
                description=(
                    "Retrieve complete PR bodies, comments, reviews, and inline "
                    "comments by number, date range, and/or search. Large results "
                    "are returned as one external Markdown artifact."
                ),
                action_type=GetPRsAction,
                observation_type=GetPRsObservation,
                annotations=ToolAnnotations(
                    title="Get pull requests",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_GetPRsExecutor(
                    get_prs_fn,
                    credentials=credentials,
                    artifact_dir=artifact_dir,
                    target_workspace=target_workspace,
                ),
            )
        ]


class _Transition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReconcileLabelsTransition(_Transition):
    operation: Literal["reconcile_labels"]
    pr_number: int = Field(gt=0)
    assignment_id: str = Field(min_length=1)
    expected_head_sha: str = Field(min_length=1)
    add: set[str] = Field(default_factory=set)
    remove: set[str] = Field(default_factory=set)


class RequestRevisionTransition(_Transition):
    operation: Literal["request_revision"]
    pr_number: int = Field(gt=0)
    assignment_id: str = Field(min_length=1)
    expected_head_sha: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    comment: str = Field(min_length=1)


class RespondToIssueTransition(_Transition):
    operation: Literal["respond_to_issue"]
    issue_number: int = Field(gt=0)
    human_message_id: int = Field(
        gt=0,
        description=(
            "Exact numeric ID of the human-authored issue body or comment "
            "being answered."
        ),
    )
    response: str = Field(
        min_length=1,
        max_length=50_000,
        description="Response including the role prefix required by the skill.",
    )


class SubmitResultTransition(_Transition):
    operation: Literal["submit_result"]
    pr_number: int = Field(gt=0)
    branch: str = Field(min_length=1)
    expected_remote_sha: str = Field(min_length=1)
    expected_head_sha: str = Field(min_length=1)
    result: ExperimentResult


class CloseExperimentTransition(_Transition):
    operation: Literal["close_experiment"]
    pr_number: int = Field(gt=0)
    expected_head_sha: str = Field(min_length=1)
    repo: str = Field(min_length=3)
    assignment_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MergeExperimentTransition(_Transition):
    operation: Literal["merge_experiment"]
    pr_number: int = Field(gt=0)
    expected_head_sha: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    merge_method: Literal["merge", "squash", "rebase"] = "squash"


class PushBranchTransition(_Transition):
    operation: Literal["push_branch"]
    branch: str = Field(min_length=1)
    expected_remote_sha: str = Field(min_length=1)


class CreateAssignmentTransition(_Transition):
    operation: Literal["create_assignment"]
    repo: str = Field(
        min_length=3,
        description="Target repository in owner/name form.",
    )
    assignment_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    student: str = Field(min_length=1)
    base_branch: str = Field(min_length=1)
    expected_base_sha: str = Field(min_length=1)
    head_branch: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=50_000)


GitHubTransition = Annotated[
    CreateAssignmentTransition
    | ReconcileLabelsTransition
    | RequestRevisionTransition
    | RespondToIssueTransition
    | SubmitResultTransition
    | CloseExperimentTransition
    | MergeExperimentTransition
    | PushBranchTransition,
    Field(discriminator="operation"),
]


class GitHubTransitionAction(Action):
    transition: GitHubTransition = Field(
        description=(
            "One typed, preconditioned, idempotent GitHub workflow transition."
        )
    )


class GitHubTransitionObservation(Observation):
    changed: bool
    resource_url: str
    state: str
    version: str | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [
            TextContent(
                text=json.dumps(
                    self.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ]


class _GitHubTransitionExecutor(
    ToolExecutor[GitHubTransitionAction, GitHubTransitionObservation]
):
    def __init__(
        self,
        workflow: GitHubWorkflowService,
        role: str,
        workspace: Path,
        git_token: SecretStr | None = None,
    ):
        self.workflow = workflow
        self.role = role
        self.workspace = workspace
        self.git_token = git_token

    def __call__(
        self,
        action: GitHubTransitionAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubTransitionObservation:
        transition = action.transition
        if isinstance(transition, CreateAssignmentTransition):
            self._require_role("advisor")
            branch = create_assignment_branch(
                self.workspace,
                branch=transition.head_branch,
                base_branch=transition.base_branch,
                expected_base_sha=transition.expected_base_sha,
                assignment_id=transition.assignment_id,
                token=self.git_token,
            )
            result = self.workflow.create_assignment(
                AssignmentRecord(
                    repo=transition.repo,
                    assignment_id=transition.assignment_id,
                    revision_id=transition.revision_id,
                    student=transition.student,
                    base_ref=transition.base_branch,
                    base_sha=transition.expected_base_sha,
                    head_ref=transition.head_branch,
                    head_sha=branch.head_sha,
                ),
                title=transition.title,
                body=transition.body,
            )
        elif isinstance(transition, PushBranchTransition):
            self._require_role("advisor")
            pushed = push_assignment_branch(
                self.workspace,
                branch=transition.branch,
                expected_remote_sha=transition.expected_remote_sha,
                token=self.git_token,
            )
            return GitHubTransitionObservation(
                changed=pushed.changed,
                resource_url=f"git:origin/{pushed.branch}",
                state="branch_pushed",
                version=pushed.head_sha,
            )
        elif isinstance(transition, SubmitResultTransition):
            self._require_role("student")
            self.workflow.preflight_submit_result(
                transition.pr_number,
                branch=transition.branch,
                current_head_sha=transition.expected_remote_sha,
                expected_result_head_sha=transition.expected_head_sha,
                result=transition.result,
            )
            pushed = push_assignment_branch(
                self.workspace,
                branch=transition.branch,
                expected_remote_sha=transition.expected_remote_sha,
                token=self.git_token,
            )
            if pushed.head_sha != transition.expected_head_sha:
                raise ValueError(
                    "expected_head_sha must match the local commit being submitted"
                )
            result = self.workflow.submit_result(
                transition.pr_number,
                expected_head_sha=transition.expected_head_sha,
                result=transition.result,
            )
        elif isinstance(transition, ReconcileLabelsTransition):
            self._require_role("advisor")
            result = self.workflow.reconcile_labels(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                add=transition.add,
                remove=transition.remove,
                expected_head_sha=transition.expected_head_sha,
            )
        elif isinstance(transition, RequestRevisionTransition):
            self._require_role("advisor")
            result = self.workflow.request_revision(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                expected_head_sha=transition.expected_head_sha,
                revision_id=transition.revision_id,
                comment=transition.comment,
            )
        elif isinstance(transition, RespondToIssueTransition):
            result = self.workflow.respond_to_issue(
                transition.issue_number,
                human_message_id=transition.human_message_id,
                response=transition.response,
            )
        elif isinstance(transition, CloseExperimentTransition):
            self._require_role("advisor")
            result = self.workflow.close_experiment(
                transition.pr_number,
                assignment_id=transition.assignment_id,
                expected_head_sha=transition.expected_head_sha,
                marker=render_disposition_marker(
                    DispositionRecord(
                        repo=transition.repo,
                        pr_number=transition.pr_number,
                        assignment_id=transition.assignment_id,
                        head_sha=transition.expected_head_sha,
                    )
                ),
                reason=transition.reason,
            )
        else:
            self._require_role("advisor")
            result = self.workflow.merge_experiment(
                transition.pr_number,
                expected_head_sha=transition.expected_head_sha,
                assignment_id=transition.assignment_id,
                merge_method=transition.merge_method,
            )
        return GitHubTransitionObservation(
            changed=result.changed,
            resource_url=result.resource_url,
            state=result.state,
            version=result.version,
        )

    def _require_role(self, expected: str) -> None:
        if self.role != expected:
            raise PermissionError(
                f"{self.role} cannot perform this {expected}-owned transition"
            )


class GitHubTransitionTool(
    ToolDefinition[GitHubTransitionAction, GitHubTransitionObservation]
):
    name = "github_transition"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        workflow: GitHubWorkflowService | None = None,
        *,
        role: str | None = None,
        workspace: str | Path | None = None,
    ) -> Sequence[Self]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        git_token: SecretStr | None = None
        if workflow is None:
            credentials = _GITHUB_CREDENTIALS
            if credentials is None:
                raise RuntimeError(
                    "configure GitHub credentials before initializing workflows"
                )
            workflow = GitHubWorkflow(
                credentials.repo,
                credentials.token,
                trusted_actor=credentials.trusted_actor,
            )
            git_token = credentials.token
        if workspace is None:
            workspace = (
                Path(conv_state.workspace.working_dir)
                if conv_state is not None
                else Path.cwd()
            )
        return [
            cls(
                description=(
                    "Apply one verified GitHub workflow transition. Operations are "
                    "create_assignment, push_branch, reconcile_labels, "
                    "request_revision, respond_to_issue, submit_result, "
                    "close_experiment, and merge_experiment. Every mutation "
                    "verifies its durable identity and converges on replay."
                ),
                action_type=GitHubTransitionAction,
                observation_type=GitHubTransitionObservation,
                annotations=ToolAnnotations(
                    title="GitHub transition",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=_GitHubTransitionExecutor(
                    workflow,
                    role,
                    Path(workspace),
                    git_token,
                ),
            )
        ]


class SenpaiTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Fail-closed policy wrapper around the native terminal executor."""

    def __init__(
        self,
        delegate: ToolExecutor[TerminalAction, TerminalObservation],
        *,
        role: str,
        workspace: Path,
    ):
        self.delegate = delegate
        self.role = role
        self.workspace = Path(workspace)

    @property
    def is_pooled(self) -> bool:
        return bool(getattr(self.delegate, "is_pooled", False))

    def __call__(
        self,
        action: TerminalAction,
        conversation: LocalConversation | None = None,
    ) -> TerminalObservation:
        try:
            from senpai_agent.hooks import terminal_policy

            decision = terminal_policy(
                action.command,
                self.role,
                self.workspace,
            )
            if not decision.allowed:
                reason = decision.reason or "No policy reason was provided."
                return _terminal_denial(action, reason)
        except Exception as error:  # noqa: BLE001
            return _terminal_denial(
                action,
                (f"Policy evaluation failed closed ({type(error).__name__})."),
            )
        return self.delegate(action, conversation)

    def close(self) -> None:
        self.delegate.close()

    def interrupt(self) -> None:
        self.delegate.interrupt()


def _terminal_denial(
    action: TerminalAction,
    reason: str,
) -> TerminalObservation:
    return TerminalObservation.from_text(
        text=f"Terminal command denied by Senpai policy: {reason}",
        is_error=True,
        command=action.command,
        exit_code=None,
    )


def create_senpai_tools(
    *,
    training: TrainingService,
    child_runner_factory: ChildAgentRunnerFactory,
    event_sink: AdvisorEventSink,
    github_workflow: GitHubWorkflowService,
    role: str = "advisor",
    get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
    max_agent_workers: int = 8,
    max_agent_runtime_seconds: float | None = None,
    pr_artifact_dir: str | Path | None = None,
    workspace: str | Path | None = None,
) -> tuple[ToolDefinition, ...]:
    """Create the compact Senpai tool set with all external boundaries injected."""

    return (
        *RunTrainingTool.create(training=training),
        *GetTrainingStatusTool.create(training=training),
        *MonitorTrainingTool.create(
            training=training,
            monitor_store=MonitorStore(
                Path(tempfile.mkdtemp(prefix="senpai-monitor-tests-"))
                / "monitors.sqlite3"
            ),
        ),
        *GetPRsTool.create(
            get_prs_fn=get_prs_fn,
            state_dir=pr_artifact_dir,
            workspace=workspace,
        ),
        *GitHubTransitionTool.create(
            workflow=github_workflow,
            role=role,
        ),
        *DelegateAgentTool.create(
            child_runner_factory=child_runner_factory,
            event_sink=event_sink,
            max_workers=max_agent_workers,
            max_runtime_seconds=max_agent_runtime_seconds,
        ),
    )


class SenpaiTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Create the native terminal behind Senpai's fail-closed policy."""

    @classmethod
    def create(
        cls,
        conv_state: object,
        *,
        role: str | None = None,
    ) -> Sequence[ToolDefinition]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        native = TerminalTool.create(conv_state)[0]
        if native.executor is None:
            raise RuntimeError("native terminal tool has no executor")
        return [
            native.set_executor(
                SenpaiTerminalExecutor(
                    native.executor,
                    role=role,
                    workspace=Path(conv_state.workspace.working_dir),
                )
            )
        ]


_TOOLS_REGISTERED = False


def register_senpai_tools() -> None:
    """Register Senpai's serializable OpenHands tool factories once per process."""

    global _TOOLS_REGISTERED
    if _TOOLS_REGISTERED:
        return
    register_tool("senpai_training", TrainingToolSet)
    register_tool("get_prs", GetPRsTool)
    register_tool("github_transition", GitHubTransitionTool)
    register_tool("delegate_agent", DelegateAgentTool)
    register_tool("senpai_terminal", SenpaiTerminalTool)
    _TOOLS_REGISTERED = True
