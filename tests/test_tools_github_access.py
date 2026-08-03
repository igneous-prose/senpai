import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openhands.sdk.tool import Tool, resolve_tool
from pydantic import SecretStr

from senpai_agent.github import PRManifestEntry, PRRetrievalResult
from senpai_agent.github_workflow import MutationResult
from senpai_agent.tools import (
    CloseExperimentTransition,
    CreateAssignmentTransition,
    GetPRsAction,
    GetPRsTool,
    GitHubTransitionAction,
    GitHubTransitionTool,
    MergeExperimentTransition,
    PushBranchTransition,
    ReconcileLabelsTransition,
    RequestRevisionTransition,
    RespondToIssueTransition,
    SendAssignmentFeedbackTransition,
    clear_github_credentials,
    configure_github_credentials,
    register_senpai_tools,
)


class ForbiddenWorkflow:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            raise AssertionError(f"student reached advisor workflow method {name}")

        return call


ADVISOR_TRANSITIONS = [
    pytest.param(
        CreateAssignmentTransition(
            operation="create_assignment",
            assignment_id="assignment-17",
            revision_id="revision-1",
            student="student-one",
            base_branch="main",
            expected_base_sha="a" * 40,
            head_branch="student-one/candidate",
            title="Try one candidate",
            body="Run the bounded comparison.",
        ),
        id="create-assignment",
    ),
    pytest.param(
        PushBranchTransition(
            operation="push_branch",
            branch="advisor-branch",
            expected_remote_sha="a" * 40,
            expected_head_sha="b" * 40,
        ),
        id="push-branch",
    ),
    pytest.param(
        ReconcileLabelsTransition(
            operation="reconcile_labels",
            pr_number=17,
            assignment_id="assignment-17",
            expected_head_sha="a" * 40,
            add={"status:review"},
        ),
        id="reconcile-labels",
    ),
    pytest.param(
        RequestRevisionTransition(
            operation="request_revision",
            pr_number=17,
            assignment_id="assignment-17",
            expected_head_sha="a" * 40,
            revision_id="revision-2",
            comment="Run one more seed.",
        ),
        id="request-revision",
    ),
    pytest.param(
        SendAssignmentFeedbackTransition(
            operation="send_assignment_feedback",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_head_sha="a" * 40,
            feedback_id="inspect-seed",
            comment="Inspect the failed seed.",
        ),
        id="send-feedback",
    ),
    pytest.param(
        CloseExperimentTransition(
            operation="close_experiment",
            pr_number=17,
            expected_head_sha="a" * 40,
            assignment_id="assignment-17",
            reason="The hypothesis was falsified.",
        ),
        id="close-experiment",
    ),
    pytest.param(
        MergeExperimentTransition(
            operation="merge_experiment",
            pr_number=17,
            expected_head_sha="a" * 40,
            assignment_id="assignment-17",
        ),
        id="merge-experiment",
    ),
]


@pytest.mark.parametrize("transition", ADVISOR_TRANSITIONS)
def test_students_cannot_reach_advisor_owned_side_effects(
    transition,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workflow = ForbiddenWorkflow()

    def forbidden_git(*_args, **_kwargs):
        raise AssertionError("student reached an advisor-owned Git operation")

    monkeypatch.setattr("senpai_agent.tools.create_assignment_branch", forbidden_git)
    monkeypatch.setattr("senpai_agent.tools.push_assignment_branch", forbidden_git)
    tool = GitHubTransitionTool.create(
        workflow=workflow,
        role="student",
        workspace=tmp_path,
        advisor_branch="advisor-branch",
    )[0]

    with pytest.raises(PermissionError, match="advisor-owned"):
        tool(GitHubTransitionAction(transition=transition))

    assert workflow.calls == []


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_both_roles_can_respond_to_a_verified_human_message(
    role: str,
    tmp_path: Path,
):
    calls = []

    class Workflow:
        def respond_to_issue(self, number, **kwargs):
            calls.append((number, kwargs))
            return MutationResult(
                changed=True,
                resource_url=f"https://github.test/issues/{number}",
                state="issue_response_upserted",
                version=str(kwargs["human_message_id"]),
            )

    tool = GitHubTransitionTool.create(
        workflow=Workflow(),
        role=role,
        workspace=tmp_path,
    )[0]

    observation = tool(
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
    assert calls == [
        (
            23,
            {
                "human_message_id": 987,
                "response": f"{role.upper()}: bounded response",
            },
        )
    ]


def test_get_prs_is_scoped_to_the_configured_repo_and_credential(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    calls = []
    configure_github_credentials(
        "acme/widgets",
        SecretStr("github-secret"),
        trusted_actor="senpai-bot",
    )

    try:
        tool = GetPRsTool.create(
            state_dir=tmp_path / "state" / "github",
            workspace=workspace,
        )[0]

        def get_prs(repo: str, **kwargs) -> PRRetrievalResult:
            calls.append((repo, kwargs))
            return PRRetrievalResult(
                manifest=(
                    PRManifestEntry(
                        number=17,
                        title="Try spectral loss",
                        head_sha="abc123",
                        url="https://github.test/acme/widgets/pull/17",
                    ),
                ),
                markdown="# PR #17\n\nComplete context.\n",
                path=None,
            )

        tool.executor.get_prs = get_prs
        with pytest.raises(PermissionError, match="configured GitHub credentials"):
            tool(GetPRsAction(repo="other/widgets"))

        observation = tool(
            GetPRsAction(
                repo="acme/widgets",
                numbers=(17,),
                search="label:status:review",
            )
        )

        assert observation.manifest[0].head_sha == "abc123"
        assert observation.to_llm_content[0].text.startswith("# PR #17")
        assert calls[0][0] == "acme/widgets"
        assert calls[0][1]["numbers"] == (17,)
        assert calls[0][1]["search"] == "label:status:review"
        assert calls[0][1]["target_workspace"] == workspace.resolve()
        assert calls[0][1]["token"].get_secret_value() == "github-secret"
    finally:
        clear_github_credentials()


def test_get_prs_artifacts_must_live_outside_the_target_checkout(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside the target workspace"):
        GetPRsTool.create(
            get_prs_fn=lambda *_args, **_kwargs: PRRetrievalResult((), "", None),
            state_dir=workspace / "state",
            workspace=workspace,
        )


def test_registered_github_tool_keeps_credentials_out_of_the_tool_spec():
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
            SimpleNamespace(workspace=SimpleNamespace(working_dir=Path.cwd())),
        )

        assert [tool.name for tool in tools] == ["github_transition"]
        assert spec.model_dump() == {
            "name": "github_transition",
            "params": {"role": "student"},
        }
        assert "github-secret" not in json.dumps(spec.model_dump())
        assert "github-secret" not in repr(tools[0].executor.workflow)
    finally:
        clear_github_credentials()


def test_registered_github_tools_ignore_ambient_write_tokens(
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
