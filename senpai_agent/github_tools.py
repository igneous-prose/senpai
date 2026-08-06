"""Role-scoped GitHub tools with one unambiguous schema per workflow action."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from senpai_agent.git_workflow import (
    create_assignment_branch,
    push_assignment_branch,
    require_commit_contains_base,
)
from senpai_agent.github import PRRetrievalResult, get_prs
from senpai_agent.github_workflow import (
    GitHubWorkflow,
    MutationResult,
    PullHeadMismatchError,
    StaleAssignmentRevisionError,
)
from senpai_agent.models import (
    AssignmentRecord,
    DispositionRecord,
    ExperimentResult,
    render_disposition_marker,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


_POST_PUSH_HEAD_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class GitHubCredentials:
    """GitHub authority held by the runtime and never exposed to the model."""

    repo: str
    token: SecretStr
    trusted_actor: str | None = None


_GITHUB_CREDENTIALS: GitHubCredentials | None = None


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
    """Remove the process-local GitHub authority after a conversation turn."""

    global _GITHUB_CREDENTIALS
    _GITHUB_CREDENTIALS = None


class GetPRsAction(Action):
    """Retrieve complete context for a bounded set of pull requests."""

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
            "Maximum PRs returned inline. Do not set this above 5 unless "
            "explicitly necessary; prefer the returned artifact path."
        ),
    )


class PRManifestObservation(BaseModel):
    """Compact identity for one retrieved pull request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    head_sha: str
    url: str


class GetPRsObservation(Observation):
    """Inline pull-request context or a bounded external artifact reference."""

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
        credentials: GitHubCredentials | None,
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
    """Read complete pull-request context without exposing GitHub credentials."""

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
        if workspace is None:
            if conv_state is None:
                raise ValueError("get_prs requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)
        target_workspace = Path(workspace).resolve()
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
                annotations=_annotations("Get pull requests", read_only=True),
                executor=_GetPRsExecutor(
                    get_prs_fn,
                    credentials=credentials,
                    artifact_dir=artifact_dir,
                    target_workspace=target_workspace,
                ),
            )
        ]


class AssignmentVersion(BaseModel):
    """Exact assignment revision and pull-request head a mutation may change."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pr_number: int = Field(
        gt=0,
        description="Pull-request number containing the assignment.",
    )
    assignment_id: str = Field(
        min_length=1,
        description="Stable assignment ID from the trusted assignment marker.",
    )
    revision_id: str = Field(
        min_length=1,
        description="Current revision ID from the trusted assignment marker.",
    )
    expected_pr_head_sha: str = Field(
        min_length=1,
        description=(
            "Current pull-request head SHA. The mutation fails if the PR moved."
        ),
    )


class CreateAssignmentAction(Action):
    """Create one isolated student branch and its typed draft assignment PR."""

    assignment_id: str = Field(
        min_length=1,
        description="New stable assignment ID; reuse it only to replay this assignment.",
    )
    revision_id: str = Field(
        min_length=1,
        description="Initial revision ID for this assignment.",
    )
    student: str = Field(
        min_length=1,
        description="Exact configured student name that will own the assignment.",
    )
    expected_base_sha: str = Field(
        min_length=1,
        description=(
            "Exact current SHA of the runtime-configured advisor branch, used as "
            "the assignment creation precondition."
        ),
    )
    head_branch: str = Field(
        min_length=1,
        description="New remote branch dedicated to this student assignment.",
    )
    title: str = Field(
        min_length=1,
        max_length=256,
        description="Concise falsifiable experiment title for the pull request.",
    )
    body: str = Field(
        min_length=1,
        max_length=50_000,
        description="Complete experiment brief, evidence contract, and stopping rule.",
    )


class PublishAdvisorBranchAction(Action):
    """Lease-publish the configured advisor branch at one exact local commit."""

    remote_branch_sha_before_push: str = Field(
        min_length=1,
        description="Current remote advisor-branch SHA used as the push lease.",
    )
    local_commit_sha: str = Field(
        min_length=1,
        description="Exact local commit to publish; the worktree HEAD must equal it.",
    )


class RepairAssignmentRoutingAction(Action):
    """Restore the desired protocol state for one current assignment revision."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    working_state: Literal["wip", "review"] = Field(
        description=(
            "Desired productive state: wip while the student works, or review "
            "after a terminal result is ready."
        ),
    )
    blockers: set[Literal["blocked", "hold", "needs-rebase"]] = Field(
        default_factory=set,
        description="Exact protocol blockers that should remain on the assignment.",
    )


class SendAssignmentFeedbackAction(Action):
    """Send idempotent guidance without starting a new assignment revision."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    feedback_id: str = Field(
        min_length=1,
        max_length=256,
        description=(
            "Stable ID for this guidance item. Replay is a no-op; changed guidance "
            "must use a new ID."
        ),
    )
    comment: str = Field(
        min_length=1,
        max_length=50_000,
        description="Actionable guidance that does not require a fresh revision.",
    )


class RequestAssignmentRevisionAction(Action):
    """Start a new revision of an existing assignment on an exact research base."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    new_revision_id: str = Field(
        min_length=1,
        description="Fresh revision ID that has never identified another revision.",
    )
    required_base_sha: str = Field(
        min_length=1,
        description=(
            "Exact live base-branch SHA against which the new revision must run."
        ),
    )
    comment: str = Field(
        min_length=1,
        max_length=50_000,
        description="Concrete reason and changed evidence requested for the revision.",
    )


class AcceptResultOnCurrentBaseAction(Action):
    """Record that one exact result remains valid on the current research base."""

    assignment: AssignmentVersion = Field(
        description="Submitted result revision and PR-head precondition.",
    )
    expected_current_base_sha: str = Field(
        min_length=1,
        description=(
            "Exact live SHA of the assignment's recorded base branch after review."
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=50_000,
        description="Scientific reason the existing result remains valid on that base.",
    )


class MergeExperimentAction(Action):
    """Merge one reviewed result after exact head and research-base validation."""

    assignment: AssignmentVersion = Field(
        description="Submitted result revision and PR-head precondition.",
    )
    expected_current_base_sha: str = Field(
        min_length=1,
        description="Exact live base-branch SHA immediately expected for the merge.",
    )
    merge_method: Literal["merge", "squash", "rebase"] = Field(
        default="squash",
        description="GitHub merge method to apply after all workflow checks pass.",
    )


class CloseExperimentAction(Action):
    """Close one reviewed assignment as a durable non-winning experiment."""

    assignment: AssignmentVersion = Field(
        description="Current assignment revision and PR-head precondition.",
    )
    reason: str = Field(
        min_length=1,
        max_length=50_000,
        description="Evidence-backed reason this experiment should close unmerged.",
    )


class RespondToHumanIssueAction(Action):
    """Respond once to one authenticated human-authored issue message."""

    issue_number: int = Field(
        gt=0,
        description=(
            "GitHub issue number containing a human message addressed to this "
            "configured advisor or student."
        ),
    )
    human_message_id: int = Field(
        gt=0,
        description="Exact numeric ID of the human-authored body or comment answered.",
    )
    response: str = Field(
        min_length=1,
        max_length=50_000,
        description="Response text; the runtime adds the authenticated role prefix.",
    )


class SubmitExperimentResultAction(Action):
    """Validate, publish, and submit one student's terminal experiment result."""

    branch: str = Field(
        min_length=1,
        description="Assignment branch named by the current pull request.",
    )
    remote_branch_sha_before_push: str = Field(
        min_length=1,
        description="Current remote branch SHA used as the force-with-lease guard.",
    )
    result: ExperimentResult = Field(
        description=(
            "Complete terminal result. Its assignment PR number and commit SHA are "
            "the publication target and local commit; do not repeat them elsewhere."
        ),
    )


class GitHubMutationObservation(Observation):
    """Verified durable state reached by one GitHub workflow mutation."""

    changed: bool
    resource_url: str
    state: str
    version: str | None = None

    @classmethod
    def from_result(cls, result: MutationResult) -> Self:
        return cls(
            changed=result.changed,
            resource_url=result.resource_url,
            state=result.state,
            version=result.version,
        )

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


@dataclass(frozen=True)
class GitHubToolRuntime:
    """Shared non-model runtime state for one role's GitHub tools."""

    workflow: GitHubWorkflow
    workspace: Path
    git_token: SecretStr | None
    role: Literal["advisor", "student"]
    advisor_branch: str | None
    student_names: frozenset[str]
    student_name: str | None

    def assignment_base_branch(self) -> str:
        """Return the configured advisor branch or fail before a mutation."""

        if self.role != "advisor" or not self.advisor_branch:
            raise RuntimeError("advisor GitHub tools require an advisor branch")
        return self.advisor_branch

    def require_configured_student(self, student: str) -> None:
        """Reject assignment names outside this launch before touching GitHub."""

        if self.role != "advisor" or not self.student_names:
            raise RuntimeError("create_assignment requires configured student names")
        if student not in self.student_names:
            allowed = ", ".join(sorted(self.student_names))
            raise PermissionError(
                f"student {student!r} is outside this launch; choose one of: {allowed}"
            )

    def human_issue_audience(self) -> set[str]:
        """Return the only Issue audience labels this role may answer."""

        if self.role == "advisor":
            return {"team", self.assignment_base_branch()}
        if not self.student_name:
            raise RuntimeError("student GitHub tools require a student name")
        return {"team", f"student:{self.student_name}"}


class _CreateAssignmentExecutor(
    ToolExecutor[CreateAssignmentAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: CreateAssignmentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        base_branch = self.runtime.assignment_base_branch()
        self.runtime.require_configured_student(action.student)
        with self.runtime.workflow.serialized_assignment_mutation():
            branch = create_assignment_branch(
                self.runtime.workspace,
                branch=action.head_branch,
                base_branch=base_branch,
                expected_base_sha=action.expected_base_sha,
                assignment_id=action.assignment_id,
                token=self.runtime.git_token,
            )
            result = self.runtime.workflow.create_assignment(
                AssignmentRecord(
                    repo=self.runtime.workflow.repo,
                    assignment_id=action.assignment_id,
                    revision_id=action.revision_id,
                    student=action.student,
                    base_ref=base_branch,
                    base_sha=action.expected_base_sha,
                    head_ref=action.head_branch,
                    head_sha=branch.head_sha,
                ),
                title=action.title,
                body=action.body,
            )
        return GitHubMutationObservation.from_result(result)


class _PublishAdvisorBranchExecutor(
    ToolExecutor[PublishAdvisorBranchAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: PublishAdvisorBranchAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        if not self.runtime.advisor_branch:
            raise RuntimeError("publish_advisor_branch requires an advisor branch")
        with self.runtime.workflow.serialized_assignment_mutation():
            pushed = push_assignment_branch(
                self.runtime.workspace,
                branch=self.runtime.advisor_branch,
                expected_remote_sha=action.remote_branch_sha_before_push,
                expected_local_sha=action.local_commit_sha,
                token=self.runtime.git_token,
            )
        return GitHubMutationObservation(
            changed=pushed.changed,
            resource_url=f"git:origin/{pushed.branch}",
            state="branch_pushed",
            version=pushed.head_sha,
        )


class _RepairAssignmentRoutingExecutor(
    ToolExecutor[RepairAssignmentRoutingAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: RepairAssignmentRoutingAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.repair_assignment_routing(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            working_state=action.working_state,
            blockers=action.blockers,
        )
        return GitHubMutationObservation.from_result(result)


class _SendAssignmentFeedbackExecutor(
    ToolExecutor[SendAssignmentFeedbackAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: SendAssignmentFeedbackAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.send_assignment_feedback(
            version.pr_number,
            assignment_id=version.assignment_id,
            revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            feedback_id=action.feedback_id,
            comment=action.comment,
        )
        return GitHubMutationObservation.from_result(result)


class _RequestAssignmentRevisionExecutor(
    ToolExecutor[RequestAssignmentRevisionAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: RequestAssignmentRevisionAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.request_revision(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            new_revision_id=action.new_revision_id,
            required_base_sha=action.required_base_sha,
            expected_head_sha=version.expected_pr_head_sha,
            comment=action.comment,
        )
        return GitHubMutationObservation.from_result(result)


class _AcceptResultOnCurrentBaseExecutor(
    ToolExecutor[AcceptResultOnCurrentBaseAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: AcceptResultOnCurrentBaseAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.accept_result_on_current_base(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            expected_current_base_sha=action.expected_current_base_sha,
            reason=action.reason,
        )
        return GitHubMutationObservation.from_result(result)


class _MergeExperimentExecutor(
    ToolExecutor[MergeExperimentAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: MergeExperimentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.merge_experiment(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            expected_current_base_sha=action.expected_current_base_sha,
            merge_method=action.merge_method,
        )
        return GitHubMutationObservation.from_result(result)


class _CloseExperimentExecutor(
    ToolExecutor[CloseExperimentAction, GitHubMutationObservation]
):
    def __init__(self, workflow: GitHubWorkflow):
        self.workflow = workflow

    def __call__(
        self,
        action: CloseExperimentAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        version = action.assignment
        result = self.workflow.close_experiment(
            version.pr_number,
            assignment_id=version.assignment_id,
            current_revision_id=version.revision_id,
            expected_head_sha=version.expected_pr_head_sha,
            marker=render_disposition_marker(
                DispositionRecord(
                    repo=self.workflow.repo,
                    pr_number=version.pr_number,
                    assignment_id=version.assignment_id,
                    head_sha=version.expected_pr_head_sha,
                )
            ),
            reason=action.reason,
        )
        return GitHubMutationObservation.from_result(result)


class _RespondToHumanIssueExecutor(
    ToolExecutor[RespondToHumanIssueAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: RespondToHumanIssueAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        result = self.runtime.workflow.respond_to_issue(
            action.issue_number,
            human_message_id=action.human_message_id,
            response=action.response,
            audience_labels=self.runtime.human_issue_audience(),
        )
        return GitHubMutationObservation.from_result(result)


class _SubmitExperimentResultExecutor(
    ToolExecutor[SubmitExperimentResultAction, GitHubMutationObservation]
):
    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: SubmitExperimentResultAction,
        conversation: LocalConversation | None = None,
    ) -> GitHubMutationObservation:
        number = action.result.assignment.pr_number
        commit_sha = action.result.commit_sha
        with self.runtime.workflow.serialized_assignment_mutation():
            try:
                preflight = self.runtime.workflow.preflight_submit_result(
                    number,
                    branch=action.branch,
                    current_head_sha=action.remote_branch_sha_before_push,
                    expected_result_head_sha=commit_sha,
                    result=action.result,
                )
            except StaleAssignmentRevisionError as error:
                if conversation is not None:
                    conversation.state.execution_status = (
                        ConversationExecutionStatus.FINISHED
                    )
                raise ValueError(
                    f"{error} Ending this stale turn so the controller can resume "
                    "the current assignment revision."
                ) from error
            require_commit_contains_base(
                self.runtime.workspace,
                commit_sha=commit_sha,
                base_sha=preflight.assignment.base_sha,
            )
            push_assignment_branch(
                self.runtime.workspace,
                branch=action.branch,
                expected_remote_sha=action.remote_branch_sha_before_push,
                expected_local_sha=commit_sha,
                token=self.runtime.git_token,
            )
            result = self._submit_after_push(number, action.result)
            return GitHubMutationObservation.from_result(result)

    def _submit_after_push(
        self,
        number: int,
        result: ExperimentResult,
    ) -> MutationResult:
        for delay in _POST_PUSH_HEAD_RETRY_DELAYS:
            try:
                return self.runtime.workflow.submit_result(
                    number,
                    expected_head_sha=result.commit_sha,
                    result=result,
                )
            except PullHeadMismatchError:
                time.sleep(delay)
        return self.runtime.workflow.submit_result(
            number,
            expected_head_sha=result.commit_sha,
            result=result,
        )


def _annotations(title: str, *, read_only: bool = False) -> ToolAnnotations:
    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=not read_only,
        idempotentHint=True,
        openWorldHint=True,
    )


def _configured_student_names(
    value: Sequence[str] | str | None,
) -> frozenset[str]:
    """Normalize an explicit or environment-provided launch allowlist."""

    if value is None:
        value = os.environ.get("STUDENT_NAMES", "")
    items = value.split(",") if isinstance(value, str) else value
    return frozenset(name for item in items if (name := item.strip()))


class CreateAssignmentTool(
    ToolDefinition[CreateAssignmentAction, GitHubMutationObservation]
):
    """Expose only the assignment-creation transaction."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Create or exactly replay one student's typed draft assignment "
                    "PR from the runtime-configured advisor branch and a lease-checked "
                    "base commit. The student must belong to this launch."
                ),
                action_type=CreateAssignmentAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Create assignment"),
                executor=_CreateAssignmentExecutor(runtime),
            )
        ]


class PublishAdvisorBranchTool(
    ToolDefinition[PublishAdvisorBranchAction, GitHubMutationObservation]
):
    """Expose lease-guarded publication of only the configured advisor branch."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Publish the configured advisor branch with force-with-lease. "
                    "Supply its current remote SHA and the exact local commit to push."
                ),
                action_type=PublishAdvisorBranchAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Publish advisor branch"),
                executor=_PublishAdvisorBranchExecutor(runtime),
            )
        ]


class RepairAssignmentRoutingTool(
    ToolDefinition[RepairAssignmentRoutingAction, GitHubMutationObservation]
):
    """Expose bounded repair of protocol-owned assignment routing state."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Repair one current assignment's protocol routing after labels or "
                    "draft state drift. Choose wip or review and only named blockers; "
                    "do not use this for ordinary assignment decisions."
                ),
                action_type=RepairAssignmentRoutingAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Repair assignment routing"),
                executor=_RepairAssignmentRoutingExecutor(runtime.workflow),
            )
        ]


class SendAssignmentFeedbackTool(
    ToolDefinition[SendAssignmentFeedbackAction, GitHubMutationObservation]
):
    """Expose non-revision guidance for one exact assignment version."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Send a clarification, hold, question, or nudge to the current "
                    "assignment without changing its revision or routing state."
                ),
                action_type=SendAssignmentFeedbackAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Send assignment feedback"),
                executor=_SendAssignmentFeedbackExecutor(runtime.workflow),
            )
        ]


class RequestAssignmentRevisionTool(
    ToolDefinition[RequestAssignmentRevisionAction, GitHubMutationObservation]
):
    """Expose a fresh assignment revision with an exact required research base."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Request a scientifically meaningful rerun as a fresh assignment "
                    "revision, bound to the exact live base commit it must evaluate."
                ),
                action_type=RequestAssignmentRevisionAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Request assignment revision"),
                executor=_RequestAssignmentRevisionExecutor(runtime.workflow),
            )
        ]


class AcceptResultOnCurrentBaseTool(
    ToolDefinition[AcceptResultOnCurrentBaseAction, GitHubMutationObservation]
):
    """Expose durable scientific acceptance across a changed research base."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "After comparing the exact submitted result with a changed "
                    "research base, durably record why that result remains valid. "
                    "This does not merge the pull request."
                ),
                action_type=AcceptResultOnCurrentBaseAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Accept result on current base"),
                executor=_AcceptResultOnCurrentBaseExecutor(runtime.workflow),
            )
        ]


class MergeExperimentTool(
    ToolDefinition[MergeExperimentAction, GitHubMutationObservation]
):
    """Expose terminal merge of one verified experiment result."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Merge one review-ready experiment at the exact expected PR head "
                    "and current research base. Changed-base results require a prior "
                    "accept_result_on_current_base call."
                ),
                action_type=MergeExperimentAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Merge experiment"),
                executor=_MergeExperimentExecutor(runtime.workflow),
            )
        ]


class CloseExperimentTool(
    ToolDefinition[CloseExperimentAction, GitHubMutationObservation]
):
    """Expose terminal closure of one exact non-winning experiment."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Close one current experiment without merging it, recording an "
                    "evidence-backed reason and preserving the durable result."
                ),
                action_type=CloseExperimentAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Close experiment"),
                executor=_CloseExperimentExecutor(runtime.workflow),
            )
        ]


class RespondToHumanIssueTool(
    ToolDefinition[RespondToHumanIssueAction, GitHubMutationObservation]
):
    """Expose authenticated, idempotent human-Issue responses to both roles."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Respond exactly once to a verified human-authored GitHub Issue "
                    "body or comment delivered to this configured role. The backend "
                    "rechecks the Issue's team/branch/student audience labels."
                ),
                action_type=RespondToHumanIssueAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Respond to human issue"),
                executor=_RespondToHumanIssueExecutor(runtime),
            )
        ]


class SubmitExperimentResultTool(
    ToolDefinition[SubmitExperimentResultAction, GitHubMutationObservation]
):
    """Expose the student's atomic validate-publish-submit result transaction."""

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Validate one terminal result against its current assignment and "
                    "research base, lease-push result.commit_sha, then publish the "
                    "typed result and make the PR review-ready."
                ),
                action_type=SubmitExperimentResultAction,
                observation_type=GitHubMutationObservation,
                annotations=_annotations("Submit experiment result"),
                executor=_SubmitExperimentResultExecutor(runtime),
            )
        ]


class GitHubWorkflowToolSet(
    ToolDefinition[GetPRsAction, GetPRsObservation]
):
    """Resolve the GitHub reader and only the workflow tools allowed for one role."""

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        workflow: GitHubWorkflow | None = None,
        *,
        role: str | None = None,
        state_dir: str | Path | None = None,
        workspace: str | Path | None = None,
        advisor_branch: str | None = None,
        student_names: Sequence[str] | str | None = None,
        student_name: str | None = None,
        get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
    ) -> Sequence[ToolDefinition]:
        role = role or os.environ.get("SENPAI_ROLE")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        if workspace is None:
            if conv_state is None:
                raise ValueError("senpai_github requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)
        credentials = _GITHUB_CREDENTIALS
        git_token: SecretStr | None = None
        if workflow is None:
            if credentials is None:
                raise RuntimeError(
                    "configure GitHub credentials before initializing workflows"
                )
            workflow = GitHubWorkflow(
                credentials.repo,
                credentials.token,
                role=role,
                trusted_actor=credentials.trusted_actor,
            )
            git_token = credentials.token
        elif workflow.role != role:
            raise ValueError("workflow role must match the GitHub tool role")
        configured_advisor_branch = advisor_branch or os.environ.get("ADVISOR_BRANCH")
        configured_student_names = _configured_student_names(student_names)
        configured_student_name = student_name or os.environ.get("STUDENT_NAME")
        runtime = GitHubToolRuntime(
            workflow=workflow,
            workspace=Path(workspace),
            git_token=git_token,
            role=role,
            advisor_branch=configured_advisor_branch,
            student_names=configured_student_names,
            student_name=configured_student_name,
        )
        if role == "advisor":
            runtime.assignment_base_branch()
            if not runtime.student_names:
                raise ValueError("advisor GitHub tools require configured student names")
        else:
            runtime.human_issue_audience()
        common = (
            *GetPRsTool.create(
                conv_state,
                get_prs_fn=get_prs_fn,
                state_dir=state_dir,
                workspace=workspace,
            ),
            *RespondToHumanIssueTool.create(runtime),
        )
        if role == "student":
            return (*common, *SubmitExperimentResultTool.create(runtime))
        return (
            *common,
            *CreateAssignmentTool.create(runtime),
            *PublishAdvisorBranchTool.create(runtime),
            *RepairAssignmentRoutingTool.create(runtime),
            *SendAssignmentFeedbackTool.create(runtime),
            *RequestAssignmentRevisionTool.create(runtime),
            *AcceptResultOnCurrentBaseTool.create(runtime),
            *MergeExperimentTool.create(runtime),
            *CloseExperimentTool.create(runtime),
        )
