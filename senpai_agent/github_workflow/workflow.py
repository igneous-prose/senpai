"""Public GitHub workflow client assembled from cohesive domain operations."""

from senpai_agent.github_workflow.assignments import AssignmentMixin
from senpai_agent.github_workflow.comments import CommentsMixin
from senpai_agent.github_workflow.core import WorkflowCore
from senpai_agent.github_workflow.issues import HumanIssueMixin
from senpai_agent.github_workflow.lookup import LookupMixin
from senpai_agent.github_workflow.merge import MergeMixin
from senpai_agent.github_workflow.results import ResultMixin
from senpai_agent.github_workflow.review import ReviewMixin
from senpai_agent.github_workflow.revisions import RevisionMixin


class GitHubWorkflow(
    HumanIssueMixin,
    MergeMixin,
    ReviewMixin,
    ResultMixin,
    RevisionMixin,
    AssignmentMixin,
    LookupMixin,
    CommentsMixin,
    WorkflowCore,
):
    """Desired-state GitHub client for Senpai research workflows."""

    __slots__ = ()
