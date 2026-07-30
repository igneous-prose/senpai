from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_SKILL = ROOT / ".agents" / "skills" / "senpai-status-check" / "SKILL.md"
HUMAN_ISSUES_SKILL = (
    ROOT / "plugins" / "senpai" / "skills" / "check-human-issues" / "SKILL.md"
)


def test_status_skill_uses_runtime_scope_without_legacy_assumptions():
    instructions = STATUS_SKILL.read_text()
    lower = instructions.lower()

    for variable in (
        "GH_REPO",
        "ADVISOR_BRANCH",
        "RESEARCH_TAG",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "TARGET_WORKDIR",
        "SENPAI_OPENHANDS_STATE_DIR",
    ):
        assert variable in instructions
    for required in (
        "program.md",
        "get_prs",
        "wandb-primary",
        "$SENPAI_OPENHANDS_STATE_DIR/training/*.json",
        "evidence gap",
        "Do not mutate",
    ):
        assert required.lower() in lower
    for legacy in (
        "wandb/senpai",
        "radford",
        "pai-2",
        ".claude",
        "current_research_state",
        "drivaerml",
        "tandemfoil",
        "airfrans",
        "harvest",
        "shutdown",
        "train.py",
    ):
        assert legacy not in lower


def test_human_issue_skill_uses_the_typed_mutation_boundary():
    instructions = HUMAN_ISSUES_SKILL.read_text()

    assert "human_message_id" in instructions
    assert "respond_to_issue" in instructions
    assert "github_transition" in instructions
    assert "gh issue comment" not in instructions
