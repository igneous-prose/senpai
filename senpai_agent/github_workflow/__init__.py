"""Typed, role-safe GitHub workflow operations."""

from senpai_agent.github_workflow.errors import (
    GitHubAPIError,
    GitHubTransportError,
    GitHubWorkflowError,
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.github_workflow.responses import (
    HttpResponse,
    HttpTransport,
    MutationResult,
    PullRequestSnapshot,
    SubmitResultPreflight,
)
from senpai_agent.github_workflow.workflow import GitHubWorkflow

__all__ = [
    "GitHubAPIError",
    "GitHubTransportError",
    "GitHubWorkflow",
    "GitHubWorkflowError",
    "HttpResponse",
    "HttpTransport",
    "MutationResult",
    "PullHeadMismatchError",
    "PullRequestSnapshot",
    "ReconciliationError",
    "StaleAssignmentRevisionError",
    "StaleResearchBaseError",
    "SubmitResultPreflight",
    "WorkflowPreconditionError",
]
