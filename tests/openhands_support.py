import uuid
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.openhands_runner import RunnerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "senpai"
AGENT_DIR = REPO_ROOT / ".agents" / "agents"


def runtime_config(tmp_path: Path, **updates) -> RunnerConfig:
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    role_file = tmp_path / "SENPAI-ADVISOR.md"
    role_file.write_text("advisor role", encoding="utf-8")
    values = {
        "max_turns": 1,
        "model": "anthropic/claude-opus-4-8",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key": SecretStr("test-key"),
        "github_repo": "acme/widgets",
        "github_token": SecretStr("github-key"),
        "github_trusted_actor": None,
        "command_secrets": {"WANDB_API_KEY": "wandb-key"},
        "reasoning_effort": "xhigh",
        "smart_model": "anthropic/claude-opus-4-8",
        "fast_model": "anthropic/claude-haiku-4-5",
        "fast_reasoning_effort": "low",
        "workspace": tmp_path,
        "state_dir": tmp_path / "state",
        "conversation_id": uuid.uuid4(),
        "role": "advisor",
        "enable_browser": False,
        "agent_name": None,
        "harness_file": harness_file,
        "role_file": role_file,
        "plugin_dir": PLUGIN_DIR,
    }
    values.update(updates)
    return RunnerConfig(**values)


def runtime_env(tmp_path: Path, *, role: str = "advisor") -> dict[str, str]:
    workspace = tmp_path / "target"
    workspace.mkdir(exist_ok=True)
    role_file = tmp_path / f"SENPAI-{role.upper()}.md"
    role_file.write_text(f"{role} role", encoding="utf-8")
    harness_file = tmp_path / "SENPAI-HARNESS.md"
    harness_file.write_text("harness instructions", encoding="utf-8")
    return {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GITHUB_TOKEN": "github-key",
        "GH_REPO": "acme/widgets",
        "SENPAI_ROLE": role,
        "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
        "SENPAI_OPENHANDS_STATE_DIR": str(tmp_path / "state"),
        "SENPAI_OPENHANDS_ROLE_FILE": str(role_file),
        "SENPAI_OPENHANDS_HARNESS_FILE": str(harness_file),
        "SENPAI_PLUGIN": str(PLUGIN_DIR),
    }


def isolate_agent_discovery(monkeypatch, runner) -> None:
    monkeypatch.setattr(runner, "discover_agents", lambda _: [])
    monkeypatch.setattr(runner, "register_file_agents", lambda _: [])
