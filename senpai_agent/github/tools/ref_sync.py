"""Credential-contained Git ref synchronization tool."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor
from pydantic import Field

from senpai_agent.git_refs import sync_github_branches

from .runtime import GitHubToolRuntime, tool_annotations

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class SyncGitRefsAction(Action):
    """Hydrate named branches from the configured GitHub repository."""

    branches: tuple[str, ...] = Field(
        min_length=1,
        max_length=32,
        description=(
            "One to 32 exact branch names from the configured repository. "
            "Do not supply URLs, refspecs, wildcards, or local ref names."
        ),
    )


class SyncGitRefsObservation(Observation):
    """Exact remote-tracking refs hydrated without changing the worktree."""

    refs: dict[str, str]

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


class SyncGitRefsExecutor(ToolExecutor[SyncGitRefsAction, SyncGitRefsObservation]):
    """Fetch through controller-owned HTTP without a credentialed Git child."""

    def __init__(self, runtime: GitHubToolRuntime):
        self.runtime = runtime

    def __call__(
        self,
        action: SyncGitRefsAction,
        conversation: LocalConversation | None = None,
    ) -> SyncGitRefsObservation:
        if self.runtime.git_token is None:
            raise RuntimeError("sync_git_refs requires configured Git credentials")
        refs = sync_github_branches(
            self.runtime.workspace,
            repo=self.runtime.workflow.repo,
            token=self.runtime.git_token,
            branches=action.branches,
        )
        return SyncGitRefsObservation(refs=refs)


class SyncGitRefsTool(ToolDefinition[SyncGitRefsAction, SyncGitRefsObservation]):
    """Synchronize remote branches without exposing GitHub credentials."""

    name = "sync_git_refs"

    @classmethod
    def create(cls, runtime: GitHubToolRuntime) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Fetch named branches from the configured GitHub repository into "
                    "refs/remotes/origin/<branch>. The operation does not change HEAD, "
                    "the current branch, index, worktree, remotes, tags, or credentials."
                ),
                action_type=SyncGitRefsAction,
                observation_type=SyncGitRefsObservation,
                annotations=tool_annotations("Sync Git refs", destructive=False),
                executor=SyncGitRefsExecutor(runtime),
            )
        ]
