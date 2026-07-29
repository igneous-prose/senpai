from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


_SHELL_SEPARATORS = {";", "&&", "||", "|", "&"}
_GH_READ_ONLY = {
    "auth": {"status"},
    "issue": {"list", "status", "view"},
    "pr": {"checks", "diff", "list", "status", "view"},
    "repo": {"list", "view"},
    "run": {"list", "view"},
    "search": {"commits", "issues", "prs", "repos"},
    "workflow": {"list", "view"},
}
_TRAIN_LAUNCHERS = {"accelerate", "deepspeed", "torchrun"}
_TRAIN_SCRIPT = re.compile(r"^train[^/]*[.]py$")


def _command_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.commenters = ""
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token in _SHELL_SEPARATORS:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _program_index(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if "=" in token and not token.startswith(("/", "./")):
            name, _, _value = token.partition("=")
            if name.replace("_", "").isalnum():
                continue
        return index
    return None


def _gh_policy(tokens: list[str], index: int) -> PolicyDecision:
    arguments = tokens[index + 1 :]
    noun_index = next(
        (
            position
            for position, value in enumerate(arguments)
            if value == "api" or value in _GH_READ_ONLY
        ),
        None,
    )
    if noun_index is None:
        if any(value in {"--help", "--version"} for value in arguments):
            return PolicyDecision(True)
        return PolicyDecision(
            False,
            "Only explicitly read-only gh commands may use the terminal.",
        )

    noun = arguments[noun_index]
    remaining = arguments[noun_index + 1 :]
    if noun == "api":
        method = "GET"
        for position, value in enumerate(remaining):
            if value in {"-X", "--method"} and position + 1 < len(remaining):
                method = remaining[position + 1].upper()
            elif value.startswith("--method="):
                method = value.partition("=")[2].upper()
            elif value in {
                "-f",
                "-F",
                "--field",
                "--raw-field",
                "--input",
            } or value.startswith(("-f=", "-F=", "--field=", "--raw-field=")):
                method = "POST"
        if method != "GET":
            return PolicyDecision(
                False,
                "Use a typed Senpai GitHub tool for mutating GitHub API calls.",
            )
        return PolicyDecision(True)

    operation = next((value for value in remaining if not value.startswith("-")), "")
    if operation not in _GH_READ_ONLY[noun]:
        return PolicyDecision(
            False,
            f"Use a typed Senpai GitHub tool for `gh {noun}` mutations.",
        )
    return PolicyDecision(True)


def _wrapper_command(
    arguments: list[str],
    *,
    value_options: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    position = 0
    while position < len(arguments):
        value = arguments[position]
        if value == "--":
            return arguments[position + 1 :]
        if value in value_options:
            position += 2
            continue
        if value.startswith("-") or _program_index([value]) is None:
            position += 1
            continue
        return arguments[position:]
    return []


def _env_command(arguments: list[str]) -> list[str]:
    for position, value in enumerate(arguments):
        if value in {"-S", "--split-string"} and position + 1 < len(arguments):
            return shlex.split(arguments[position + 1]) + arguments[position + 2 :]
        if value.startswith("--split-string="):
            return shlex.split(value.partition("=")[2]) + arguments[position + 1 :]
    return _wrapper_command(
        arguments,
        value_options={"-u", "--unset", "-C", "--chdir"},
    )


def _shell_command(arguments: list[str]) -> str | None:
    for position, value in enumerate(arguments):
        is_command_option = value == "-c" or (
            value.startswith("-") and not value.startswith("--") and "c" in value[1:]
        )
        if is_command_option and position + 1 < len(arguments):
            return arguments[position + 1]
    return None


def _curl_policy(tokens: list[str], index: int) -> PolicyDecision:
    arguments = tokens[index + 1 :]
    if not any("api.github.com" in value for value in arguments):
        return PolicyDecision(True)

    method = "GET"
    for position, value in enumerate(arguments):
        if value in {"-X", "--request"} and position + 1 < len(arguments):
            method = arguments[position + 1].upper()
        elif value.startswith(("-X", "--request=")):
            method = value.removeprefix("-X").partition("=")[-1].upper()
        elif value in {
            "-d",
            "--data",
            "--data-ascii",
            "--data-binary",
            "--data-raw",
            "--json",
        } or value.startswith(
            (
                "-d",
                "--data=",
                "--data-ascii=",
                "--data-binary=",
                "--data-raw=",
                "--json=",
            )
        ):
            if method == "GET":
                method = "POST"
        elif (
            value in {"-T", "--upload-file"}
            or value.startswith(("-T", "--upload-file="))
        ) and method == "GET":
            method = "PUT"
    if method != "GET":
        return PolicyDecision(
            False,
            "Use a typed Senpai GitHub tool for mutating GitHub API calls.",
        )
    return PolicyDecision(True)


def _python_launches_training(tokens: list[str], index: int) -> bool:
    arguments = tokens[index + 1 :]
    if "-c" in arguments:
        return False
    if "-m" in arguments:
        position = arguments.index("-m")
        if position + 1 < len(arguments):
            module = arguments[position + 1]
            return "train" in module.lower() or module == "torch.distributed.run"
    script = next((value for value in arguments if not value.startswith("-")), "")
    return bool(_TRAIN_SCRIPT.fullmatch(Path(script).name))


def _segment_policy(tokens: list[str]) -> PolicyDecision:
    index = _program_index(tokens)
    if index is None:
        return PolicyDecision(True)
    program = Path(tokens[index]).name

    arguments = tokens[index + 1 :]
    if program == "env":
        command = _env_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program in {"command", "exec", "nohup"}:
        command = _wrapper_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program in {"bash", "dash", "sh", "zsh"}:
        command = _shell_command(arguments)
        if command is not None:
            return terminal_policy(command, "", Path.cwd())

    if program == "git" and "push" in arguments:
        return PolicyDecision(
            False,
            "Use the typed Senpai branch-push tool; raw git push bypasses guards.",
        )
    if program == "gh":
        return _gh_policy(tokens, index)
    if program == "curl":
        return _curl_policy(tokens, index)

    if program == "uv" and arguments[:1] == ["run"]:
        return _segment_policy(tokens[index + 2 :])
    if program.startswith("python") and _python_launches_training(tokens, index):
        return PolicyDecision(
            False,
            "Use run_training so timeouts, logs, status, and W&B IDs are supervised.",
        )
    if program in _TRAIN_LAUNCHERS or _TRAIN_SCRIPT.fullmatch(program):
        return PolicyDecision(
            False,
            "Use run_training so timeouts, logs, status, and W&B IDs are supervised.",
        )

    if program in {"sleep", "watch", "while", "until"}:
        return PolicyDecision(
            False,
            "Do not run foreground polling loops; use Senpai events or status tools.",
        )
    if program == "tail" and any(
        argument == "--follow" or argument.startswith("-") and "f" in argument[1:]
        for argument in tokens[index + 1 :]
    ):
        return PolicyDecision(
            False,
            "Do not stream logs; use get_training_status for bounded updates.",
        )
    return PolicyDecision(True)


def terminal_policy(
    command: str,
    role: str,
    workspace: Path,
) -> PolicyDecision:
    del role, workspace
    for segment in _command_segments(command):
        decision = _segment_policy(segment)
        if not decision.allowed:
            return decision
    return PolicyDecision(True)


def _stop_policy(
    role: str,
    working_dir: Path,
    state_dir: Path | None,
) -> PolicyDecision:
    if role != "student" or not (working_dir / ".git").exists():
        return PolicyDecision(True)
    if state_dir is not None:
        running = {
            path.stem
            for path in (state_dir / "training").glob("*.json")
            if json.loads(path.read_text()).get("state") == "running"
        }
        monitored = {
            path.stem for path in (state_dir / "training" / "monitors").glob("*.json")
        }
        unmonitored = running - monitored
        if unmonitored:
            return PolicyDecision(
                False,
                "Training is still running without a durable monitor; call "
                "monitor_training before finishing: "
                f"{', '.join(sorted(unmonitored))}",
            )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=working_dir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if status.strip():
        return PolicyDecision(
            False,
            "Commit or discard the assignment changes before finishing.",
        )
    return PolicyDecision(True)


def _emit(decision: PolicyDecision) -> int:
    output = {"decision": "allow" if decision.allowed else "deny"}
    if decision.reason:
        output["reason"] = decision.reason
        output["additionalContext"] = decision.reason
    print(json.dumps(output))
    return 0 if decision.allowed else 2


def hook_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    command = (argv or sys.argv[1:])[0]
    try:
        event = json.loads(sys.stdin.read() or "{}")
        working_dir = Path(event.get("working_dir") or os.getcwd()).resolve()
        role = env.get("SENPAI_ROLE", "")
        if command == "pre-tool-use":
            tool_input = event.get("tool_input") or {}
            return _emit(terminal_policy(str(tool_input["command"]), role, working_dir))
        if command == "stop":
            state_dir_value = env.get("SENPAI_OPENHANDS_STATE_DIR")
            state_dir = Path(state_dir_value).resolve() if state_dir_value else None
            return _emit(_stop_policy(role, working_dir, state_dir))
        if command == "session-end":
            return _emit(PolicyDecision(True))
        raise ValueError(f"unknown hook command: {command}")
    except Exception:  # noqa: BLE001
        return _emit(
            PolicyDecision(False, "Senpai safety policy could not be evaluated.")
        )


if __name__ == "__main__":
    raise SystemExit(hook_main())
