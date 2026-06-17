import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "k8s" / "run-senpai-claude.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_helper(tmp_path: Path, *, runtime: str | None = None, extra: str = ""):
    capture_dir = tmp_path / "capture"
    bin_dir = tmp_path / "bin"
    plugin_dir = tmp_path / "plugin"
    capture_dir.mkdir()
    bin_dir.mkdir()
    plugin_dir.mkdir()

    _write_executable(
        bin_dir / "claude",
        """#!/bin/sh
printf '%s\n' "$@" > "$CAPTURE_DIR/claude_argv"
printf '%s\n' "$CLAUDE_PLUGIN_ROOT" > "$CAPTURE_DIR/claude_plugin_root"
cat > "$CAPTURE_DIR/claude_stdin"
""",
    )
    _write_executable(
        bin_dir / "python",
        """#!/bin/sh
printf '%s\n' "$@" > "$CAPTURE_DIR/python_argv"
printf '%s\n' "$CLAUDE_PLUGIN_ROOT" > "$CAPTURE_DIR/python_plugin_root"
cat > "$CAPTURE_DIR/python_stdin"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_DIR": str(capture_dir),
            "LOGFILE": str(tmp_path / "agent.log"),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SENPAI_PLUGIN": str(plugin_dir),
            "SENPAI_TEST_PROMPT": "fix train.py without killing yourself",
        }
    )
    if runtime:
        env["SENPAI_AGENT_RUNTIME"] = runtime

    command = f'source "{RUNNER}" && run_senpai_claude 7 "$SENPAI_TEST_PROMPT" {extra}'
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, capture_dir, plugin_dir


def test_claude_runtime_is_default_and_keeps_prompt_on_stdin(tmp_path):
    result, capture_dir, plugin_dir = _run_helper(tmp_path)

    assert result.returncode == 0, result.stderr
    argv = (capture_dir / "claude_argv").read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-p", "-"]
    assert "--max-turns" in argv
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert "--plugin-dir" in argv
    assert argv[argv.index("--plugin-dir") + 1] == str(plugin_dir)
    assert "fix train.py without killing yourself" not in argv
    assert (capture_dir / "claude_stdin").read_text(encoding="utf-8") == "fix train.py without killing yourself"
    assert (capture_dir / "claude_plugin_root").read_text(encoding="utf-8").strip() == str(plugin_dir)


def test_openhands_runtime_receives_continue_flag_and_stdin_prompt(tmp_path):
    result, capture_dir, plugin_dir = _run_helper(tmp_path, runtime="openhands", extra="-c")

    assert result.returncode == 0, result.stderr
    argv = (capture_dir / "python_argv").read_text(encoding="utf-8").splitlines()
    assert argv == ["-m", "senpai_agent.openhands_runner", "-c", "--max-turns", "7"]
    assert (capture_dir / "python_stdin").read_text(encoding="utf-8") == "fix train.py without killing yourself"
    assert (capture_dir / "python_plugin_root").read_text(encoding="utf-8").strip() == str(plugin_dir)


def test_unknown_runtime_fails_clearly(tmp_path):
    result, _, _ = _run_helper(tmp_path, runtime="mystery")

    assert result.returncode == 2
    assert "SENPAI_AGENT_RUNTIME must be 'claude' or 'openhands'" in result.stderr
