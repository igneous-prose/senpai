import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import Tool, resolve_tool
from openhands.tools.terminal import TerminalAction, TerminalObservation
from pydantic import SecretStr

from senpai_agent.delegation import (
    DelegateAgentAction,
    DelegateAgentTool,
    DelegationRequest,
)
from senpai_agent.git_workflow import PushResult
from senpai_agent.github import (
    PRManifestEntry,
    PRRetrievalResult,
)
from senpai_agent.github_workflow import MutationResult
from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    ResultStatus,
)
from senpai_agent.tools import (
    CreateAssignmentTransition,
    GetPRsAction,
    GetTrainingStatusAction,
    GitHubTransitionAction,
    PushBranchTransition,
    ReconcileLabelsTransition,
    RequestRevisionTransition,
    RespondToIssueTransition,
    RunTrainingAction,
    SenpaiTerminalExecutor,
    SubmitResultTransition,
    clear_github_credentials,
    configure_github_credentials,
    create_senpai_tools,
    register_senpai_tools,
)
from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
)


class FakeTraining:
    def __init__(self, result: TrainingResult):
        self.result = result
        self.run_specs: list[TrainingSpec] = []
        self.status_ids: list[str] = []
        self.cancelled = False

    def run_training(self, spec: TrainingSpec) -> TrainingResult:
        self.run_specs.append(spec)
        return self.result

    def get_training_status(self, training_id: str) -> TrainingResult:
        self.status_ids.append(training_id)
        return self.result

    def close(self) -> None:
        self.cancelled = True


class EventSink:
    def __init__(self):
        self.events = []
        self.received = threading.Event()

    def enqueue(self, event) -> bool:
        self.events.append(event)
        self.received.set()
        return True


class FakeGitHubWorkflow:
    def __init__(self):
        self.calls = []

    def reconcile_labels(self, number, **kwargs):
        self.calls.append(("reconcile_labels", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="labels_reconciled",
            version=kwargs["expected_head_sha"],
        )

    def create_assignment(self, assignment, **kwargs):
        self.calls.append(("create_assignment", assignment, kwargs))
        return MutationResult(
            changed=True,
            resource_url="https://github.test/pull/18",
            state="assignment_created",
            version=assignment.head_sha,
        )

    def preflight_submit_result(self, number, **kwargs):
        self.calls.append(("preflight_submit_result", number, kwargs))

    def submit_result(self, number, **kwargs):
        self.calls.append(("submit_result", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="result_submitted",
            version=kwargs["expected_head_sha"],
        )

    def request_revision(self, number, **kwargs):
        self.calls.append(("request_revision", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="revision_requested",
            version=kwargs["expected_head_sha"],
        )

    def respond_to_issue(self, number, **kwargs):
        self.calls.append(("respond_to_issue", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/issues/{number}",
            state="issue_response_upserted",
            version=str(kwargs["human_message_id"]),
        )


def training_result(tmp_path: Path) -> TrainingResult:
    return TrainingResult(
        training_id="training-17",
        state=TrainingState.FINISHED,
        exit_code=0,
        elapsed_seconds=12.5,
        log_path=str(tmp_path / "training.log"),
        wandb_run_ids=("run-abc",),
    )


def experiment_result(*, head_sha: str = "c" * 40) -> ExperimentResult:
    return ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_head_sha=head_sha,
            student="student-one",
        ),
        status=ResultStatus.INCONCLUSIVE,
        hypothesis="Check the bounded candidate.",
        summary="The bounded comparison completed.",
        runs=(),
        primary_metric=None,
        commit_sha=head_sha,
    )


def close_tools(tools) -> None:
    for tool in tools:
        if tool.executor is not None:
            tool.executor.close()


def test_tool_schemas_are_typed_explicit_and_context_bounded(tmp_path: Path):
    fake_training = FakeTraining(training_result(tmp_path))
    tools = create_senpai_tools(
        training=fake_training,
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
    )

    try:
        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == {
            "run_training",
            "get_training_status",
            "monitor_training",
            "get_prs",
            "github_transition",
            "delegate_agent",
        }
        assert (
            "spec" in by_name["run_training"].action_type.to_mcp_schema()["properties"]
        )
        assert (
            "training_id"
            in by_name["get_training_status"].action_type.to_mcp_schema()["properties"]
        )
        monitor_schema = by_name["monitor_training"].action_type.to_mcp_schema()
        assert {
            "training_id",
            "metric",
            "direction",
            "gates",
            "poll_interval_seconds",
            "stale_after_seconds",
            "notify_on_status",
        } == set(monitor_schema["properties"])

        pr_schema = by_name["get_prs"].action_type.to_mcp_schema()
        inline_description = pr_schema["properties"]["max_inline_prs"]["description"]
        assert ">5" in inline_description
        assert "context pollution" in inline_description.lower()
        assert "artifact" in inline_description.lower()
        delegation_schema = by_name["delegate_agent"].action_type.to_mcp_schema()
        serialized_delegation = json.dumps(
            {
                "description": by_name["delegate_agent"].description,
                "schema": delegation_schema,
            }
        ).lower()
        assert set(delegation_schema["properties"]) == {
            "task",
            "agent",
            "model",
            "background",
            "include_context",
            "search_mode",
        }
        assert "include_context" not in delegation_schema.get("required", ())
        defaults = DelegateAgentAction(task="bounded")
        assert defaults.include_context is False
        assert defaults.background is False
        assert defaults.model == "smart"
        assert "eight" in serialized_delegation
        assert "reviewer" not in serialized_delegation
        transition_schema = by_name["github_transition"].action_type.to_mcp_schema()
        assert "transition" in transition_schema["properties"]
    finally:
        close_tools(tools)


def test_training_and_github_tools_delegate_existing_typed_interfaces(
    tmp_path: Path,
):
    result = training_result(tmp_path)
    fake_training = FakeTraining(result)
    pr_calls: list[tuple[str, dict]] = []
    pr_result = PRRetrievalResult(
        manifest=(
            PRManifestEntry(
                number=17,
                title="Try spectral loss",
                head_sha="abc123",
                url="https://github.com/acme/widgets/pull/17",
            ),
        ),
        markdown="# PR #17\n\nComplete context.\n",
        path=None,
    )

    def fake_get_prs(repo: str, **kwargs) -> PRRetrievalResult:
        pr_calls.append((repo, kwargs))
        return pr_result

    tools = create_senpai_tools(
        training=fake_training,
        get_prs_fn=fake_get_prs,
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
        pr_artifact_dir=tmp_path / "state" / "github",
        workspace=tmp_path / "target",
    )
    by_name = {tool.name: tool for tool in tools}
    spec = TrainingSpec(
        argv=("python", "train.py"),
        cwd=tmp_path,
        timeout_seconds=600,
    )

    try:
        run_observation = by_name["run_training"](RunTrainingAction(spec=spec))
        status_observation = by_name["get_training_status"](
            GetTrainingStatusAction(training_id=result.training_id)
        )
        pr_observation = by_name["get_prs"].executor(
            GetPRsAction(
                repo="acme/widgets",
                numbers=(17,),
                date_range=("2026-07-01", "2026-07-29"),
                search="label:status:review",
            ),
            SimpleNamespace(
                state=SimpleNamespace(
                    workspace=SimpleNamespace(
                        working_dir=tmp_path / "target",
                    )
                )
            ),
        )

        assert fake_training.run_specs == [spec]
        assert fake_training.status_ids == ["training-17"]
        assert run_observation.training_id == result.training_id
        assert run_observation.state is TrainingState.FINISHED
        assert status_observation.wandb_run_ids == ("run-abc",)
        assert pr_calls == [
            (
                "acme/widgets",
                {
                    "numbers": (17,),
                    "date_range": ("2026-07-01", "2026-07-29"),
                    "search": "label:status:review",
                    "max_inline_prs": 5,
                    "artifact_dir": tmp_path / "state" / "github",
                    "target_workspace": tmp_path / "target",
                },
            )
        ]
        assert pr_observation.markdown == pr_result.markdown
        assert pr_observation.manifest[0].head_sha == "abc123"
        assert pr_observation.to_llm_content[0].text == pr_result.markdown
    finally:
        close_tools(tools)


def test_training_tool_interrupt_cancels_the_supervisor(tmp_path: Path):
    training = FakeTraining(training_result(tmp_path))
    tools = create_senpai_tools(
        training=training,
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
    )
    run_training = {tool.name: tool for tool in tools}["run_training"]

    run_training.executor.interrupt()

    assert training.cancelled is True
    close_tools(tools)


def test_github_transition_delegates_a_typed_idempotent_operation(
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="advisor",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        observation = transition(
            GitHubTransitionAction(
                transition=ReconcileLabelsTransition(
                    operation="reconcile_labels",
                    pr_number=17,
                    assignment_id="assignment-17",
                    expected_head_sha="abc123",
                    add={"status:review"},
                    remove={"status:wip"},
                )
            )
        )
        assert observation.changed is True
        assert observation.state == "labels_reconciled"
        assert github.calls == [
            (
                "reconcile_labels",
                17,
                {
                    "add": {"status:review"},
                    "remove": {"status:wip"},
                    "expected_head_sha": "abc123",
                    "assignment_id": "assignment-17",
                },
            )
        ]
    finally:
        close_tools(tools)


def test_request_revision_transition_carries_the_assignment_identity(
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="advisor",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        observation = transition(
            GitHubTransitionAction(
                transition=RequestRevisionTransition(
                    operation="request_revision",
                    pr_number=17,
                    assignment_id="assignment-17",
                    expected_head_sha="abc123",
                    revision_id="revision-2",
                    comment="Run the requested ablation.",
                )
            )
        )
        assert observation.state == "revision_requested"
        assert github.calls == [
            (
                "request_revision",
                17,
                {
                    "assignment_id": "assignment-17",
                    "expected_head_sha": "abc123",
                    "revision_id": "revision-2",
                    "comment": "Run the requested ablation.",
                },
            )
        ]
    finally:
        close_tools(tools)


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_both_roles_can_respond_to_a_verified_human_message(
    role: str,
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role=role,
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        observation = transition(
            GitHubTransitionAction(
                transition=RespondToIssueTransition(
                    operation="respond_to_issue",
                    issue_number=23,
                    human_message_id=987,
                    response=f"{role.upper()}: bounded response",
                )
            )
        )
        assert observation.state == "issue_response_upserted"
        assert github.calls == [
            (
                "respond_to_issue",
                23,
                {
                    "human_message_id": 987,
                    "response": f"{role.upper()}: bounded response",
                },
            )
        ]
    finally:
        close_tools(tools)


def test_student_cannot_bypass_result_submission_with_direct_label_change(
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="student",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]
    action = GitHubTransitionAction(
        transition=ReconcileLabelsTransition(
            operation="reconcile_labels",
            pr_number=17,
            assignment_id="assignment-17",
            expected_head_sha="abc123",
            add={"status:review"},
            remove={"status:wip"},
        )
    )

    try:
        with pytest.raises(PermissionError, match="advisor-owned"):
            transition(action)
        assert github.calls == []
    finally:
        close_tools(tools)


def test_student_cannot_use_the_standalone_push_transition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    push_calls = []
    monkeypatch.setattr(
        "senpai_agent.tools.push_assignment_branch",
        lambda *args, **kwargs: push_calls.append((args, kwargs)),
    )
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
        role="student",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        with pytest.raises(PermissionError, match="advisor-owned"):
            transition(
                GitHubTransitionAction(
                    transition=PushBranchTransition(
                        operation="push_branch",
                        branch="advisor-branch",
                        expected_remote_sha="a" * 40,
                    )
                )
            )
        assert push_calls == []
    finally:
        close_tools(tools)


def test_advisor_push_transition_is_limited_to_the_configured_advisor_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    push_calls = []

    def push(workspace, **kwargs):
        push_calls.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha="b" * 40,
        )

    monkeypatch.setattr("senpai_agent.tools.push_assignment_branch", push)
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
        role="advisor",
        advisor_branch="schmidhuber",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        with pytest.raises(PermissionError, match="advisor branch"):
            transition(
                GitHubTransitionAction(
                    transition=PushBranchTransition(
                        operation="push_branch",
                        branch="main",
                        expected_remote_sha="a" * 40,
                    )
                )
            )
        assert push_calls == []

        observation = transition(
            GitHubTransitionAction(
                transition=PushBranchTransition(
                    operation="push_branch",
                    branch="schmidhuber",
                    expected_remote_sha="a" * 40,
                )
            )
        )
        assert observation.state == "branch_pushed"
        assert push_calls[0][1]["branch"] == "schmidhuber"
    finally:
        close_tools(tools)


def test_student_result_is_preflighted_before_its_branch_is_pushed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    push_calls = []

    def fail_preflight(number, **kwargs):
        github.calls.append(("preflight_submit_result", number, kwargs))
        raise ValueError("invalid assignment")

    github.preflight_submit_result = fail_preflight
    monkeypatch.setattr(
        "senpai_agent.tools.push_assignment_branch",
        lambda *args, **kwargs: push_calls.append((args, kwargs)),
    )
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="student",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]
    result = experiment_result()

    try:
        with pytest.raises(ValueError, match="invalid assignment"):
            transition(
                GitHubTransitionAction(
                    transition=SubmitResultTransition(
                        operation="submit_result",
                        pr_number=17,
                        branch="student-one/candidate",
                        expected_remote_sha="a" * 40,
                        expected_head_sha="c" * 40,
                        result=result,
                    )
                )
            )
        assert github.calls == [
            (
                "preflight_submit_result",
                17,
                {
                    "branch": "student-one/candidate",
                    "current_head_sha": "a" * 40,
                    "expected_result_head_sha": "c" * 40,
                    "result": result,
                },
            )
        ]
        assert push_calls == []
    finally:
        close_tools(tools)


def test_student_result_validates_the_declared_local_head_before_push(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    push_calls = []

    def push(workspace, **kwargs):
        push_calls.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha=kwargs["expected_local_sha"],
        )

    monkeypatch.setattr("senpai_agent.tools.push_assignment_branch", push)
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="student",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        result = experiment_result()
        observation = transition(
            GitHubTransitionAction(
                transition=SubmitResultTransition(
                    operation="submit_result",
                    pr_number=17,
                    branch="student-one/candidate",
                    expected_remote_sha="a" * 40,
                    expected_head_sha="c" * 40,
                    result=result,
                )
            )
        )

        assert observation.state == "result_submitted"
        assert push_calls[0][1]["expected_local_sha"] == "c" * 40
    finally:
        close_tools(tools)


def test_advisor_create_assignment_owns_git_and_pr_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    github = FakeGitHubWorkflow()
    branch_calls = []

    def fake_create_assignment_branch(workspace, **kwargs):
        branch_calls.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha="c" * 40,
        )

    monkeypatch.setattr(
        "senpai_agent.tools.create_assignment_branch",
        fake_create_assignment_branch,
    )
    tools = create_senpai_tools(
        training=FakeTraining(training_result(tmp_path)),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: None,
        event_sink=EventSink(),
        github_workflow=github,
        role="advisor",
    )
    transition = {tool.name: tool for tool in tools}["github_transition"]

    try:
        observation = transition(
            GitHubTransitionAction(
                transition=CreateAssignmentTransition(
                    operation="create_assignment",
                    repo="acme/widgets",
                    assignment_id="assignment-18",
                    revision_id="revision-1",
                    student="student-one",
                    base_branch="schmidhuber",
                    expected_base_sha="b" * 40,
                    head_branch="student-one/lower-lr",
                    title="Try a lower learning rate",
                    body="Run one bounded comparison.",
                )
            )
        )
        assert observation.state == "assignment_created"
        assert branch_calls == [
            (
                Path.cwd(),
                {
                    "branch": "student-one/lower-lr",
                    "base_branch": "schmidhuber",
                    "expected_base_sha": "b" * 40,
                    "assignment_id": "assignment-18",
                    "token": None,
                },
            )
        ]
        operation, assignment, arguments = github.calls[0]
        assert operation == "create_assignment"
        assert assignment.head_sha == "c" * 40
        assert assignment.repo == "acme/widgets"
        assert arguments == {
            "title": "Try a lower learning rate",
            "body": "Run one bounded comparison.",
        }
    finally:
        close_tools(tools)


def test_registered_training_spec_resolves_to_one_shared_supervisor(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=workspace),
    )
    register_senpai_tools()

    tools = resolve_tool(
        Tool(
            name="senpai_training",
            params={"state_dir": str(tmp_path / "state")},
        ),
        state,
    )
    by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {
        "run_training",
        "get_training_status",
        "monitor_training",
    }
    assert (
        by_name["run_training"].executor.training
        is by_name["get_training_status"].executor.training
    )
    close_tools(tools)


def test_registered_github_tool_keeps_tokens_out_of_the_serialized_spec(
    monkeypatch: pytest.MonkeyPatch,
):
    configure_github_credentials(
        "acme/widgets",
        SecretStr("github-secret"),
        trusted_actor="senpai-bot",
    )
    register_senpai_tools()
    spec = Tool(name="github_transition", params={"role": "student"})

    try:
        tools = resolve_tool(
            spec,
            SimpleNamespace(
                workspace=SimpleNamespace(working_dir=Path.cwd()),
            ),
        )

        assert [tool.name for tool in tools] == ["github_transition"]
        assert spec.model_dump() == {
            "name": "github_transition",
            "params": {"role": "student"},
        }
        serialized = json.dumps(spec.model_dump())
        assert "github-secret" not in serialized
        workflow = tools[0].executor.workflow
        assert "github-secret" not in repr(workflow)
    finally:
        clear_github_credentials()


def test_registered_github_tools_never_fall_back_to_ambient_write_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    clear_github_credentials()
    monkeypatch.setenv("GH_REPO", "acme/widgets")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-write-token")
    register_senpai_tools()
    state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=Path.cwd()),
    )

    with pytest.raises(RuntimeError, match="configure GitHub credentials"):
        resolve_tool(Tool(name="get_prs"), state)
    with pytest.raises(RuntimeError, match="configure GitHub credentials"):
        resolve_tool(
            Tool(name="github_transition", params={"role": "student"}),
            state,
        )


class FakeChild:
    def __init__(
        self,
        *,
        conversation_id: uuid.UUID,
        release: threading.Event,
        started: threading.Event,
        fail: bool = False,
    ):
        self.id = conversation_id
        self.release = release
        self.started = started
        self.fail = fail
        self.messages: list[str] = []
        self.closed = False
        self.interrupted = False

    def run(self, task: str, timeout_seconds: float | None) -> str:
        self.messages.append(task)
        self.started.set()
        try:
            if not self.release.wait(timeout_seconds or 2):
                self.interrupted = True
                raise TimeoutError(
                    f"subagent exceeded its {timeout_seconds:g}-second runtime"
                )
            if self.fail:
                raise RuntimeError("child disappeared")
            return "Child research result"
        finally:
            self.closed = True

    def interrupt(self) -> None:
        self.interrupted = True
        self.release.set()


def parent_conversation() -> SimpleNamespace:
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(text="Investigate the regression")],
                ),
                activated_skills=["experiment-report"],
                extended_content=[
                    TextContent(text="Progressively disclosed skill instructions")
                ],
            ),
            MessageEvent(
                source="agent",
                llm_message=Message(
                    role="assistant",
                    content=[TextContent(text="I will inspect the evidence.")],
                ),
            ),
        ]
    )
    return SimpleNamespace(id=uuid.uuid4(), state=SimpleNamespace(view=view))


def test_delegate_agent_foreground_returns_result_inline():
    parent = parent_conversation()
    release = threading.Event()
    release.set()
    child = FakeChild(
        conversation_id=uuid.uuid4(),
        release=release,
        started=threading.Event(),
    )
    tools = create_senpai_tools(
        training=FakeTraining(
            TrainingResult(
                training_id="unused",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=0,
                log_path="/tmp/unused",
            )
        ),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: child,
        event_sink=EventSink(),
        github_workflow=FakeGitHubWorkflow(),
    )
    delegate = {tool.name: tool for tool in tools}["delegate_agent"]

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Locate the implementation.",
                agent="explore",
                model="fast",
            ),
            parent,
        )

        assert observation.status == "finished"
        assert observation.result == "Child research result"
        assert child.closed is True
    finally:
        close_tools(tools)


def test_delegate_agent_allows_eight_parallel_background_children():
    parent = parent_conversation()
    release = threading.Event()
    children: list[FakeChild] = []
    sink = EventSink()

    def child_factory(_request: DelegationRequest) -> FakeChild:
        child = FakeChild(
            conversation_id=uuid.uuid4(),
            release=release,
            started=threading.Event(),
        )
        children.append(child)
        return child

    delegate = DelegateAgentTool.create(
        child_runner_factory=child_factory,
        event_sink=sink,
        max_workers=8,
    )[0]

    try:
        for index in range(8):
            observation = delegate(
                DelegateAgentAction(
                    task=f"Inspect area {index}.",
                    agent="explore",
                    model="fast",
                    background=True,
                ),
                parent,
            )
            assert observation.status == "dispatched"

        assert all(child.started.wait(1) for child in children)
        with pytest.raises(RuntimeError, match="all eight"):
            delegate(
                DelegateAgentAction(task="Ninth task.", background=True),
                parent,
            )
    finally:
        release.set()
        delegate.executor.close()

    assert len(sink.events) == 8


@pytest.mark.parametrize(
    ("include_context", "expected_roles"),
    [(False, []), (True, ["user", "assistant"])],
)
def test_delegate_agent_background_is_nonblocking_and_context_is_explicit(
    include_context,
    expected_roles,
):
    parent = parent_conversation()
    release = threading.Event()
    started = threading.Event()
    child = FakeChild(
        conversation_id=uuid.uuid4(),
        release=release,
        started=started,
    )
    requests: list[DelegationRequest] = []
    sink = EventSink()

    def child_factory(request: DelegationRequest) -> FakeChild:
        requests.append(request)
        return child

    tools = create_senpai_tools(
        training=FakeTraining(
            TrainingResult(
                training_id="unused",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=0,
                log_path="/tmp/unused",
            )
        ),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=child_factory,
        event_sink=sink,
        github_workflow=FakeGitHubWorkflow(),
    )
    delegate = {tool.name: tool for tool in tools}["delegate_agent"]

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Compare the candidate runs.",
                agent="explore",
                model="fast",
                background=True,
                include_context=include_context,
            ),
            parent,
        )

        assert observation.status == "dispatched"
        assert observation.task_id
        assert len(requests) == 1
        assert started.wait(1)
        assert not sink.received.is_set()
        assert child.messages == ["Compare the candidate runs."]
        request = requests[0]
        assert request.task_id == observation.task_id
        assert request.parent_conversation_id == str(parent.id)
        assert [message.role for message in request.parent_context] == expected_roles
        if include_context:
            assert [
                content.text
                for content in request.parent_context[0].content
                if isinstance(content, TextContent)
            ] == [
                "Investigate the regression",
                "Progressively disclosed skill instructions",
            ]

        release.set()
        assert sink.received.wait(1)
        assert child.closed is True
        event = sink.events[0]
        assert event.kind == "agent_result"
        assert event.dedupe_key == f"agent_result:{observation.task_id}"
        assert event.payload == {
            "task_id": observation.task_id,
            "parent_conversation_id": str(parent.id),
            "task": "Compare the candidate runs.",
            "result": "Child research result",
        }
    finally:
        release.set()
        close_tools(tools)


def test_delegate_agent_background_enqueues_error_when_child_disappears():
    parent = parent_conversation()
    release = threading.Event()
    release.set()
    child = FakeChild(
        conversation_id=uuid.uuid4(),
        release=release,
        started=threading.Event(),
        fail=True,
    )
    sink = EventSink()
    tools = create_senpai_tools(
        training=FakeTraining(
            TrainingResult(
                training_id="unused",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=0,
                log_path="/tmp/unused",
            )
        ),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: child,
        event_sink=sink,
        github_workflow=FakeGitHubWorkflow(),
    )
    delegate = {tool.name: tool for tool in tools}["delegate_agent"]

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Check one hypothesis.",
                background=True,
            ),
            parent,
        )

        assert sink.received.wait(1)
        assert child.closed is True
        event = sink.events[0]
        assert event.kind == "agent_error"
        assert event.dedupe_key == f"agent_result:{observation.task_id}"
        assert event.payload["task_id"] == observation.task_id
        assert event.payload["parent_conversation_id"] == str(parent.id)
        assert event.payload["task"] == "Check one hypothesis."
        assert event.payload["error"] == "RuntimeError: child disappeared"
    finally:
        close_tools(tools)


def test_delegate_agent_supports_an_explicit_runtime_limit():
    parent = parent_conversation()
    child = FakeChild(
        conversation_id=uuid.uuid4(),
        release=threading.Event(),
        started=threading.Event(),
    )
    sink = EventSink()
    tools = create_senpai_tools(
        training=FakeTraining(
            TrainingResult(
                training_id="unused",
                state=TrainingState.RUNNING,
                exit_code=None,
                elapsed_seconds=0,
                log_path="/tmp/unused",
            )
        ),
        get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
        child_runner_factory=lambda _request: child,
        event_sink=sink,
        github_workflow=FakeGitHubWorkflow(),
        max_agent_runtime_seconds=0.05,
    )
    delegate = {tool.name: tool for tool in tools}["delegate_agent"]

    try:
        observation = delegate(
            DelegateAgentAction(
                task="Bound this investigation.",
                background=True,
            ),
            parent,
        )

        assert sink.received.wait(1)
        assert child.interrupted is True
        assert child.closed is True
        event = sink.events[0]
        assert event.kind == "agent_error"
        assert event.dedupe_key == f"agent_result:{observation.task_id}"
        assert event.payload["error"] == (
            "TimeoutError: subagent exceeded its 0.05-second runtime"
        )
    finally:
        close_tools(tools)


class FakeTerminalDelegate:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.interrupted = False

    def __call__(self, action, conversation=None):
        self.calls.append((action, conversation))
        return TerminalObservation.from_text(
            "allowed",
            command=action.command,
            exit_code=0,
        )

    def close(self) -> None:
        self.closed = True

    def interrupt(self) -> None:
        self.interrupted = True


def test_terminal_executor_delegates_only_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from senpai_agent import hooks

    decisions = []

    def allow(command: str, role: str, workspace: Path):
        decisions.append((command, role, workspace))
        return SimpleNamespace(allowed=True, reason="")

    monkeypatch.setattr(hooks, "terminal_policy", allow)
    delegate = FakeTerminalDelegate()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git status --short")
    conversation = SimpleNamespace()

    observation = executor(action, conversation)
    executor.interrupt()
    executor.close()

    assert observation.text == "allowed"
    assert decisions == [("git status --short", "student", tmp_path)]
    assert delegate.calls == [(action, conversation)]
    assert delegate.interrupted is True
    assert delegate.closed is True


@pytest.mark.parametrize("policy_failure", [False, True])
def test_terminal_executor_fails_closed_on_denial_or_policy_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy_failure: bool,
):
    from senpai_agent import hooks

    def policy(_command: str, _role: str, _workspace: Path):
        if policy_failure:
            raise RuntimeError("parser unavailable")
        return SimpleNamespace(allowed=False, reason="Use the typed GitHub tool.")

    monkeypatch.setattr(hooks, "terminal_policy", policy)
    delegate = FakeTerminalDelegate()
    executor = SenpaiTerminalExecutor(
        delegate,
        role="student",
        workspace=tmp_path,
    )
    action = TerminalAction(command="git push origin experiment")

    observation = executor(action)

    assert observation.is_error is True
    assert observation.command == action.command
    assert observation.exit_code is None
    assert "denied" in observation.text.lower()
    assert delegate.calls == []
