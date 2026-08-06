from typing import cast

import pytest

from senpai_agent.github_workflow import WorkflowPreconditionError
from github_workflow_support import (
    REPO,
    FakeGitHub,
    comment,
    human_issue,
    pull_request,
    workflow,
)


def test_respond_to_issue_writes_one_verified_idempotent_reply():
    fake = FakeGitHub(pull_request(), issue=human_issue())
    client = workflow(fake)

    first = client.respond_to_issue(
        7,
        human_message_id=700,
        response="I will investigate this now.",
    )
    mutations_after_first = list(fake.mutations)
    second = client.respond_to_issue(
        7,
        human_message_id=700,
        response="I will investigate this now.",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.state == "issue_response_upserted"
    assert fake.comments == [
        comment(
            1,
            "<!-- senpai-human-response:700 -->\n\n"
            "ADVISOR: I will investigate this now.",
        )
    ]
    assert fake.mutations == mutations_after_first


def test_respond_to_issue_accepts_a_specific_human_comment():
    fake = FakeGitHub(
        pull_request(),
        issue=human_issue(),
        comments=[comment(42, "Please also compare memory use.", author="ada")],
    )

    result = workflow(fake, role="student").respond_to_issue(
        7,
        human_message_id=42,
        response="STUDENT fern: I included memory in the comparison.",
    )

    assert result.changed is True
    assert len(fake.comments) == 2
    assert cast(str, fake.comments[-1]["body"]).endswith(
        "\n\nSTUDENT: I included memory in the comparison."
    )


@pytest.mark.parametrize(
    ("issue", "comments", "message_id", "match"),
    [
        (human_issue(state="closed"), [], 700, "must be open"),
        (human_issue(labels={"team"}), [], 700, "human"),
        (
            human_issue(
                pull_request_url=f"https://api.github.test/repos/{REPO}/pulls/7"
            ),
            [],
            700,
            "pull request",
        ),
        (human_issue(author="senpai-bot"), [], 700, "authenticated actor"),
        (
            human_issue(),
            [comment(42, "Already answered.", author="senpai-bot")],
            42,
            "authenticated actor",
        ),
        (human_issue(), [], 999, "not present"),
    ],
    ids=(
        "closed-issue",
        "missing-human-label",
        "pull-request",
        "bot-authored-issue",
        "bot-authored-comment",
        "unknown-message",
    ),
)
def test_respond_to_issue_rejects_untrusted_sources_before_writing(
    issue,
    comments,
    message_id,
    match,
):
    fake = FakeGitHub(pull_request(), issue=issue, comments=comments)

    with pytest.raises(WorkflowPreconditionError, match=match):
        workflow(fake).respond_to_issue(
            7,
            human_message_id=message_id,
            response="ADVISOR: bounded response",
        )

    assert fake.mutations == []
