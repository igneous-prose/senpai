from pathlib import Path

import pytest

from senpai_agent.git_workflow import PushResult
from senpai_agent.github_workflow import MutationResult
from senpai_agent.tools import (
    CloseExperimentTransition,
    CreateAssignmentTransition,
    GitHubTransitionAction,
    GitHubTransitionTool,
    MergeExperimentTransition,
    PushBranchTransition,
)


class Workflow:
    def __init__(self):
        self.repo = "acme/widgets"
        self.calls = []

    def create_assignment(self, assignment, **kwargs):
        self.calls.append(("create_assignment", assignment, kwargs))
        return MutationResult(
            changed=True,
            resource_url="https://github.test/pull/18",
            state="assignment_created",
            version=assignment.head_sha,
        )

    def merge_experiment(self, number, **kwargs):
        self.calls.append(("merge_experiment", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="experiment_merged",
            version="merge-sha",
        )

    def close_experiment(self, number, **kwargs):
        self.calls.append(("close_experiment", number, kwargs))
        return MutationResult(
            changed=True,
            resource_url=f"https://github.test/pull/{number}",
            state="experiment_closed",
            version=kwargs["expected_head_sha"],
        )


def advisor_tool(workflow: Workflow, workspace: Path, **kwargs):
    return GitHubTransitionTool.create(
        workflow=workflow,
        role="advisor",
        workspace=workspace,
        **kwargs,
    )[0]


def test_advisor_push_is_limited_to_the_configured_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pushes = []

    def push(workspace, **kwargs):
        pushes.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha="b" * 40,
        )

    monkeypatch.setattr("senpai_agent.tools.push_assignment_branch", push)
    tool = advisor_tool(Workflow(), tmp_path, advisor_branch="schmidhuber")

    with pytest.raises(PermissionError, match="advisor branch"):
        tool(
            GitHubTransitionAction(
                transition=PushBranchTransition(
                    operation="push_branch",
                    branch="main",
                    expected_remote_sha="a" * 40,
                    expected_head_sha="b" * 40,
                )
            )
        )

    observation = tool(
        GitHubTransitionAction(
            transition=PushBranchTransition(
                operation="push_branch",
                repo="acme/widgets",
                branch="schmidhuber",
                expected_remote_sha="a" * 40,
                expected_head_sha="b" * 40,
            )
        )
    )

    assert observation.state == "branch_pushed"
    assert observation.version == "b" * 40
    assert pushes == [
        (
            tmp_path,
            {
                "branch": "schmidhuber",
                "expected_remote_sha": "a" * 40,
                "expected_local_sha": "b" * 40,
                "token": None,
            },
        )
    ]


def test_transition_rejects_a_repository_outside_the_bound_runtime(tmp_path: Path):
    tool = advisor_tool(Workflow(), tmp_path)

    with pytest.raises(PermissionError, match="repository"):
        tool(
            GitHubTransitionAction(
                transition=MergeExperimentTransition(
                    operation="merge_experiment",
                    repo="other/widgets",
                    pr_number=17,
                    assignment_id="assignment-17",
                    expected_head_sha="a" * 40,
                )
            )
        )


def test_create_assignment_uses_the_created_branch_head_for_the_pr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    workflow = Workflow()
    branch_calls = []

    def create_branch(workspace, **kwargs):
        branch_calls.append((workspace, kwargs))
        return PushResult(
            changed=True,
            branch=kwargs["branch"],
            head_sha="c" * 40,
        )

    monkeypatch.setattr("senpai_agent.tools.create_assignment_branch", create_branch)
    tool = advisor_tool(workflow, tmp_path)

    observation = tool(
        GitHubTransitionAction(
            transition=CreateAssignmentTransition(
                operation="create_assignment",
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
    assert branch_calls[0][1] == {
        "branch": "student-one/lower-lr",
        "base_branch": "schmidhuber",
        "expected_base_sha": "b" * 40,
        "assignment_id": "assignment-18",
        "token": None,
    }
    _, assignment, pr_fields = workflow.calls[0]
    assert assignment.head_sha == "c" * 40
    assert assignment.repo == "acme/widgets"
    assert pr_fields == {
        "title": "Try a lower learning rate",
        "body": "Run one bounded comparison.",
    }


def test_close_stamps_the_runtime_repository_without_repeated_input(tmp_path: Path):
    workflow = Workflow()
    tool = advisor_tool(workflow, tmp_path)

    observation = tool(
        GitHubTransitionAction(
            transition=CloseExperimentTransition(
                operation="close_experiment",
                pr_number=17,
                expected_head_sha="a" * 40,
                assignment_id="assignment-17",
                reason="The hypothesis was falsified.",
            )
        )
    )

    assert observation.state == "experiment_closed"
    _, _, fields = workflow.calls[0]
    assert '"repo":"acme/widgets"' in fields["marker"]


def test_merge_forwards_explicit_acceptance_of_an_advanced_baseline(
    tmp_path: Path,
):
    workflow = Workflow()
    tool = advisor_tool(workflow, tmp_path)

    observation = tool(
        GitHubTransitionAction(
            transition=MergeExperimentTransition(
                operation="merge_experiment",
                pr_number=17,
                assignment_id="assignment-17",
                expected_head_sha="a" * 40,
                accepted_base_sha="b" * 40,
            )
        )
    )

    assert observation.state == "experiment_merged"
    assert workflow.calls == [
        (
            "merge_experiment",
            17,
            {
                "expected_head_sha": "a" * 40,
                "assignment_id": "assignment-17",
                "merge_method": "squash",
                "accepted_base_sha": "b" * 40,
            },
        )
    ]
