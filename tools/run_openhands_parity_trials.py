#!/usr/bin/env python3
"""Run live OpenHands parity trials for the Senpai agent runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "anthropic/claude-opus-4-8"


@dataclass(frozen=True)
class TrialTask:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    validate: Callable[[Path, subprocess.CompletedProcess[str]], tuple[bool, str]]
    max_turns: int = 30


@dataclass(frozen=True)
class TrialResult:
    task: str
    index: int
    ok: bool
    detail: str
    seconds: float
    workspace: str
    returncode: int


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def setup_context_contract(workspace: Path) -> None:
    write(
        workspace / "program.md",
        """
        # Program

        Target problem: cfd-surrogate-smoke.
        Primary metric: test/pressure_mse.
        Budget: every trial must finish under 20 GPU minutes.
        Boundary: only edit files under src/ and reports/.
        Stop condition: abandon ideas that do not beat 0.042 on the test metric.
        """,
    )
    write(
        workspace / "instructions" / "prompt-student.md",
        """
        You are a student agent. Read program.md before making changes.
        Keep outputs structured and do not invent external experiment results.
        """,
    )


CONTEXT_PROMPT = """
Read program.md and instructions/prompt-student.md, then write reports/context.json.
The JSON object must contain exactly these keys: target_problem, primary_metric,
budget, editable_boundary, stop_condition. Use the exact facts from the files.
Do not add prose outside the JSON file.
"""


def validate_context_contract(workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    path = workspace / "reports" / "context.json"
    if not path.exists():
        return False, "reports/context.json was not created"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"context.json is invalid JSON: {exc}"
    expected = {
        "target_problem": "cfd-surrogate-smoke",
        "primary_metric": "test/pressure_mse",
        "budget": "every trial must finish under 20 GPU minutes",
        "editable_boundary": "only edit files under src/ and reports/",
        "stop_condition": "abandon ideas that do not beat 0.042 on the test metric",
    }
    return (data == expected, f"context data={data!r}")


def setup_code_patch(workspace: Path) -> None:
    write(
        workspace / "src" / "metrics.py",
        """
        def normalized_mse(pred, target, scale):
            # Bug: this forgets to normalize by scale.
            return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)
        """,
    )
    write(
        workspace / "check.py",
        """
        from src.metrics import normalized_mse

        value = normalized_mse([3.0, 5.0], [1.0, 1.0], scale=2.0)
        assert abs(value - 2.5) < 1e-9, value
        print("checks passed")
        """,
    )


CODE_PATCH_PROMPT = """
Fix src/metrics.py so python check.py passes. Keep the implementation simple:
mean squared error divided by scale squared. Run python check.py after editing.
Do not create unrelated files.
"""


def validate_code_patch(workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    check = subprocess.run(
        [sys.executable, "check.py"],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if check.returncode != 0:
        return False, f"check.py failed: {check.stderr or check.stdout}"
    source = (workspace / "src" / "metrics.py").read_text(encoding="utf-8")
    return ("/ scale" in source or "scale ** 2" in source or "scale**2" in source, "check.py passed")


def setup_result_marker(workspace: Path) -> None:
    write(
        workspace / "training_log.jsonl",
        """
        {"epoch": 1, "val/pressure_mse": 0.061, "test/pressure_mse": 0.070}
        {"epoch": 2, "val/pressure_mse": 0.048, "test/pressure_mse": 0.052}
        {"epoch": 3, "val/pressure_mse": 0.045, "test/pressure_mse": 0.041}
        """,
    )
    write(
        workspace / "assignment.json",
        """
        {
          "pr": 123,
          "wandb_run_id": "smoke-run-123",
          "primary_metric": "val/pressure_mse",
          "test_metric": "test/pressure_mse"
        }
        """,
    )


RESULT_MARKER_PROMPT = """
Read assignment.json and training_log.jsonl. Write reports/pr-comment.md with
a concise results comment. It must include exactly one line beginning
SENPAI-RESULT: followed by valid JSON. The JSON must set terminal=true,
status="complete", pending_arms=false, wandb_run_ids=["smoke-run-123"],
primary_metric.name="val/pressure_mse" with value 0.045, and
test_metric.name="test/pressure_mse" with value 0.041.
"""


def validate_result_marker(workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    path = workspace / "reports" / "pr-comment.md"
    if not path.exists():
        return False, "reports/pr-comment.md was not created"
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_lines = [line for line in lines if line.startswith("SENPAI-RESULT:")]
    if len(marker_lines) != 1:
        return False, f"expected one SENPAI-RESULT line, got {len(marker_lines)}"
    try:
        payload = json.loads(marker_lines[0].split(":", 1)[1].strip())
    except json.JSONDecodeError as exc:
        return False, f"marker JSON is invalid: {exc}"
    checks = [
        payload.get("terminal") is True,
        payload.get("status") == "complete",
        payload.get("pending_arms") is False,
        payload.get("wandb_run_ids") == ["smoke-run-123"],
        payload.get("primary_metric", {}).get("name") == "val/pressure_mse",
        payload.get("primary_metric", {}).get("value") == 0.045,
        payload.get("test_metric", {}).get("name") == "test/pressure_mse",
        payload.get("test_metric", {}).get("value") == 0.041,
    ]
    return (all(checks), f"payload={payload!r}")


def setup_advisor_triage(workspace: Path) -> None:
    write(
        workspace / "fleet_state.json",
        """
        {
          "advisor_branch": "noam",
          "students": ["fern", "willow", "sage"],
          "prs": [
            {"number": 10, "title": "active run", "labels": ["noam", "student:fern", "status:wip"]},
            {"number": 11, "title": "ready result", "labels": ["noam", "student:willow", "status:review"]},
            {"number": 12, "title": "blocked run", "labels": ["noam", "student:sage", "status:blocked"]}
          ],
          "candidate_hypotheses": [
            {"slug": "pressure-head-probe", "target": "test/pressure_mse", "why": "cheap diagnostic for pressure head underfit"},
            {"slug": "mesh-token-dropout", "target": "test/pressure_mse", "why": "regularization for sparse mesh regions"}
          ]
        }
        """,
    )


ADVISOR_TRIAGE_PROMPT = """
Read fleet_state.json. Write reports/advisor-plan.json with keys:
idle_students, review_prs, blocked_prs, assignments. A student is idle only if
they have no status:wip PR. Assign exactly one candidate hypothesis to each idle
student, in student order, without assigning fern. Include labels
["noam", "student:<name>", "status:wip"] for each assignment.
"""


def validate_advisor_triage(workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    path = workspace / "reports" / "advisor-plan.json"
    if not path.exists():
        return False, "reports/advisor-plan.json was not created"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"advisor-plan.json is invalid JSON: {exc}"
    assignments = data.get("assignments", [])
    assigned_students = [assignment.get("student") for assignment in assignments]
    ok = (
        data.get("idle_students") == ["willow", "sage"]
        and data.get("review_prs") == [11]
        and data.get("blocked_prs") == [12]
        and assigned_students == ["willow", "sage"]
        and all("student:fern" not in assignment.get("labels", []) for assignment in assignments)
    )
    return (ok, f"advisor plan={data!r}")


def setup_skill_subagent(workspace: Path) -> None:
    setup_context_contract(workspace)
    write(
        workspace / ".claude" / "skills" / "smoke-research" / "SKILL.md",
        """
        ---
        name: smoke-research
        description: Use this skill when asked for the smoke research brief.
        ---

        Write a JSON file at reports/research-brief.json with keys mechanism,
        evidence, and next_experiment. Use only local files.
        """,
    )
    write(
        workspace / ".claude" / "agents" / "researcher-agent.md",
        """
        ---
        name: researcher-agent
        description: Local research synthesis agent for smoke tests.
        tools:
          - TerminalTool
          - FileEditorTool
        skills:
          - smoke-research
        permission_mode: never_confirm
        ---

        You are a local research specialist. Read the workspace files and produce
        concise structured evidence. Do not browse the internet.
        """,
    )


SKILL_SUBAGENT_PROMPT = """
Use the task tool with subagent_type researcher-agent to create the smoke
research brief requested by the smoke-research skill. The final artifact must be
reports/research-brief.json and must mention cfd-surrogate-smoke as evidence.
"""


def validate_skill_subagent(workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    path = workspace / "reports" / "research-brief.json"
    if not path.exists():
        return False, "reports/research-brief.json was not created"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"research-brief.json is invalid JSON: {exc}"
    combined = json.dumps(data, sort_keys=True)
    log = result.stdout + result.stderr
    ok = (
        {"mechanism", "evidence", "next_experiment"}.issubset(data)
        and "cfd-surrogate-smoke" in combined
        and "researcher-agent" in log
    )
    return (ok, f"brief={data!r}; subagent_logged={'researcher-agent' in log}")


TASKS = {
    task.name: task
    for task in [
        TrialTask("context_contract", CONTEXT_PROMPT, setup_context_contract, validate_context_contract, 20),
        TrialTask("code_patch", CODE_PATCH_PROMPT, setup_code_patch, validate_code_patch, 25),
        TrialTask("result_marker", RESULT_MARKER_PROMPT, setup_result_marker, validate_result_marker, 25),
        TrialTask("advisor_triage", ADVISOR_TRIAGE_PROMPT, setup_advisor_triage, validate_advisor_triage, 25),
        TrialTask("skill_subagent", SKILL_SUBAGENT_PROMPT, setup_skill_subagent, validate_skill_subagent, 35),
    ]
}


def run_trial(task: TrialTask, index: int, output_dir: Path, env: dict[str, str], timeout: int) -> TrialResult:
    workspace = output_dir / task.name / f"run_{index:02d}" / "workspace"
    state_dir = output_dir / task.name / f"run_{index:02d}" / "state"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    task.setup(workspace)

    command = [
        sys.executable,
        "-m",
        "senpai_agent.openhands_runner",
        "--max-turns",
        str(task.max_turns),
        "--workspace",
        str(workspace),
        "--state-dir",
        str(state_dir),
        "--api-key-env",
        "ANTHROPIC_API_KEY2",
    ]
    start = time.time()
    try:
        result = subprocess.run(
            command,
            input=task.prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=ROOT,
        )
        ok, detail = task.validate(workspace, result)
        if result.returncode != 0:
            ok = False
            detail = f"runner exited {result.returncode}: {detail}; stderr={result.stderr[-1000:]}"
    except subprocess.TimeoutExpired as exc:
        return TrialResult(task.name, index, False, f"timeout after {timeout}s", time.time() - start, str(workspace), 124)
    except Exception as exc:
        return TrialResult(task.name, index, False, f"{exc.__class__.__name__}: {exc}", time.time() - start, str(workspace), 1)

    return TrialResult(task.name, index, ok, detail, time.time() - start, str(workspace), result.returncode)


def summarize(results: list[TrialResult]) -> dict[str, object]:
    by_task: dict[str, list[TrialResult]] = {}
    for result in results:
        by_task.setdefault(result.task, []).append(result)
    return {
        "results": [result.__dict__ for result in results],
        "summary": {
            task: {
                "passed": sum(result.ok for result in task_results),
                "total": len(task_results),
                "pass_rate": sum(result.ok for result in task_results) / len(task_results),
                "avg_seconds": sum(result.seconds for result in task_results) / len(task_results),
                "failures": [
                    {"index": result.index, "detail": result.detail, "workspace": result.workspace}
                    for result in task_results
                    if not result.ok
                ],
            }
            for task, task_results in sorted(by_task.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task", choices=sorted(TASKS), action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dotenv = load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    env.update({key: value for key, value in dotenv.items() if key not in env})
    if not env.get("ANTHROPIC_API_KEY2"):
        raise RuntimeError("ANTHROPIC_API_KEY2 is required in .env or the environment")

    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["SENPAI_OPENHANDS_MODEL"] = args.model
    env["SENPAI_OPENHANDS_API_KEY_ENV"] = "ANTHROPIC_API_KEY2"
    env["SENPAI_PLUGIN"] = str(ROOT / "plugins" / "senpai")
    env["OPENHANDS_SUPPRESS_BANNER"] = "1"

    selected = [TASKS[name] for name in (args.task or sorted(TASKS))]
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="senpai-openhands-parity-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    results: list[TrialResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task in selected:
            for index in range(1, args.runs + 1):
                futures.append(pool.submit(run_trial, task, index, output_dir, env, args.timeout))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {result.task}#{result.index} {result.seconds:.1f}s {result.detail}", flush=True)

    report = summarize(results)
    report["output_dir"] = str(output_dir)
    report["model"] = args.model
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
