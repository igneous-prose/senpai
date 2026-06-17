#!/usr/bin/env python3
"""Compare OpenHands and Claude Code on qualitative Senpai research judgment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
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
DEFAULT_MODEL = "claude-opus-4-8"
OPENHANDS_MODEL = f"anthropic/{DEFAULT_MODEL}"
REPO = "morganmcg1/gemma-challenge-senpai"
PR_NUMBERS = [558, 561, 573, 577, 582]
REQUIRED_KEYS = [
    "recommendation",
    "mechanism",
    "evidence",
    "ruled_out_paths",
    "next_experiment",
    "stop_condition",
    "confidence",
]


@dataclass(frozen=True)
class QualityTask:
    name: str
    title: str
    context_prs: tuple[int, ...]
    prompt: str
    expected_terms: tuple[str, ...]
    forbidden_next_terms: tuple[str, ...]
    max_turns: int = 24


@dataclass(frozen=True)
class TrialResult:
    runtime: str
    task: str
    index: int
    score: int
    passed: bool
    valid_json: bool
    subagent_logged: bool
    detail: str
    seconds: float
    workspace: str
    output_path: str
    scored_path: str


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


def fetch_pr_context(numbers: list[int]) -> dict[int, dict[str, object]]:
    contexts: dict[int, dict[str, object]] = {}
    for number in numbers:
        raw = subprocess.check_output(
            [
                "gh",
                "pr",
                "view",
                "-R",
                REPO,
                str(number),
                "--json",
                "number,title,state,mergedAt,closedAt,labels,body,comments,reviews,url",
            ],
            text=True,
            cwd=ROOT,
        )
        data = json.loads(raw)
        data["labels"] = [label["name"] for label in data.get("labels", [])]
        data["comments"] = [
            {
                "author": comment.get("author", {}).get("login"),
                "body": truncate(comment.get("body", ""), 1600),
                "createdAt": comment.get("createdAt"),
            }
            for comment in data.get("comments", [])[-2:]
        ]
        data["reviews"] = [
            {
                "author": review.get("author", {}).get("login"),
                "state": review.get("state"),
                "body": truncate(review.get("body", ""), 1000),
                "submittedAt": review.get("submittedAt"),
            }
            for review in data.get("reviews", [])[-2:]
        ]
        data["body"] = truncate(data.get("body", ""), 2200)
        contexts[number] = data
    return contexts


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...truncated...]\n" + text[-limit // 4 :]


def render_pr_context(contexts: dict[int, dict[str, object]], numbers: tuple[int, ...]) -> str:
    parts = []
    for number in numbers:
        data = contexts[number]
        parts.append(
            "\n".join(
                [
                    f"## PR #{number}: {data['title']}",
                    f"URL: {data['url']}",
                    f"State: {data['state']} mergedAt={data.get('mergedAt')} closedAt={data.get('closedAt')}",
                    f"Labels: {', '.join(data.get('labels', []))}",
                    "",
                    "### Assignment Body",
                    str(data.get("body", "")),
                    "",
                    "### Recent Comments",
                    json.dumps(data.get("comments", []), indent=2),
                    "",
                    "### Recent Reviews",
                    json.dumps(data.get("reviews", []), indent=2),
                ]
            )
        )
    return "\n\n".join(parts)


def copy_claude_assets(workspace: Path) -> Path:
    home = workspace / "home"
    claude_home = home / ".claude"
    claude_home.mkdir(parents=True)
    for child in ["agents", "skills", "settings.json"]:
        src = ROOT / ".claude" / child
        dst = claude_home / child
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("worktrees"))
        elif src.exists():
            shutil.copy2(src, dst)
    return home


TASKS = {
    task.name: task
    for task in [
        QualityTask(
            name="researcher_axis_synthesis",
            title="Researcher synthesis after framework/FlashInfer closures",
            context_prs=(558, 582),
            expected_terms=("#558", "#582", "#481", "batch-invariant", "framework-robust"),
            forbidden_next_terms=("try sglang", "try flashinfer", "switch to flashinfer"),
            prompt="""
            You are running as the researcher-agent. You may search the web for
            at most two high-signal references, but do not mutate GitHub or
            source repo state. Only write the required answer.json file.

            Given the PR context, synthesize the next research direction after
            the framework zoom-out and FlashInfer free-lever axes. Should the
            advisor keep probing alternate frameworks, scope purpose-built
            batch-invariant kernels, or pivot elsewhere?
            """,
        ),
        QualityTask(
            name="post_result_next_steps",
            title="Advisor next steps after an experiment result",
            context_prs=(582,),
            expected_terms=("#582", "flashinfer", "not loadable", "identity", "close"),
            forbidden_next_terms=("rerun flashinfer", "try flashinfer again", "assign sglang"),
            prompt="""
            You are running as the researcher-agent. Do not call gh or mutate
            any PR. Only write the required answer.json file.

            You are the advisor reviewing this completed result. Decide what
            should happen next in the research loop. Describe whether to bank,
            merge, send back, close a direction, assign a follow-up, or update
            the research ledger, and justify the choice from evidence.
            """,
        ),
        QualityTask(
            name="direction_selection",
            title="Pick the next best research direction from recent program state",
            context_prs=(558, 561, 573, 577, 582),
            expected_terms=("worst-case", "quality", "spec-dec", "denominator", "stop"),
            forbidden_next_terms=("framework zoom-out", "free flashinfer", "try sglang"),
            prompt="""
            You are running as the researcher-agent. You may search the web for
            at most two references if useful, but do not mutate GitHub or source
            repo state. Only write the required answer.json file.

            Given these recent merged/closed PRs, choose the single highest-value
            next research direction. Prefer a discriminating experiment over a
            broad idea list. Explain why it is better than repeating the dead
            axes, what result would falsify it, and what the advisor should ask
            a student to do.
            """,
        ),
    ]
}


def task_prompt(task: QualityTask, contexts: dict[int, dict[str, object]]) -> str:
    return textwrap.dedent(
        f"""
        You are evaluating Senpai research direction quality. This is a read-only
        qualitative parity test. Do not run mutating GitHub commands. Do not
        comment, label, merge, close, or push. It is okay to describe what the
        advisor would do next. All required PR context is included below; do not
        inspect the filesystem except to write `answer.json`.

        Write your final answer to `answer.json` in the current workspace, then
        print the same JSON object. Use exactly these keys:
        {json.dumps(REQUIRED_KEYS)}

        The `evidence` value must cite concrete PR numbers. The
        `next_experiment` value must be a concrete, discriminating experiment or
        explicit no-run decision. The `stop_condition` must say what result
        would make the advisor abandon the direction.

        # Question
        {task.prompt}

        # PR context
        {render_pr_context(contexts, task.context_prs)}
        """
    ).strip()


def base_env(workspace: Path) -> dict[str, str]:
    dotenv = load_dotenv(ROOT / ".env")
    env = os.environ.copy()
    env.update({key: value for key, value in dotenv.items() if key not in env})
    home = copy_claude_assets(workspace)
    env["HOME"] = str(home)
    env["CLAUDE_PLUGIN_ROOT"] = str(ROOT / "plugins" / "senpai")
    env["SENPAI_PLUGIN"] = str(ROOT / "plugins" / "senpai")
    env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    env["SENPAI_OPENHANDS_MODEL"] = OPENHANDS_MODEL
    env["SENPAI_OPENHANDS_REASONING_EFFORT"] = "xhigh"
    env["SENPAI_OPENHANDS_ENABLE_BROWSER"] = "1"
    env["SENPAI_OPENHANDS_API_KEY_ENV"] = "ANTHROPIC_API_KEY2"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["GH_CONFIG_DIR"] = str(workspace / "gh_config")
    if env.get("ANTHROPIC_API_KEY2"):
        env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_API_KEY2"]
    env.pop("GITHUB_TOKEN", None)
    env.pop("GH_TOKEN", None)
    return env


def read_scored_output(workspace: Path, fallback: str) -> tuple[str, Path]:
    answer_path = workspace / "answer.json"
    if answer_path.exists():
        return answer_path.read_text(encoding="utf-8"), answer_path
    return fallback, workspace / "output.txt"


def run_openhands(prompt: str, workspace: Path, env: dict[str, str], max_turns: int, timeout: int) -> tuple[str, bool]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "senpai_agent.openhands_runner",
            "--max-turns",
            str(max_turns),
            "--workspace",
            str(workspace),
            "--state-dir",
            str(workspace / "openhands_state"),
            "--api-key-env",
            "ANTHROPIC_API_KEY2",
            "--reasoning-effort",
            "xhigh",
            "--enable-browser",
            "--agent",
            "researcher-agent",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=ROOT,
        env=env,
    )
    output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        output += f"\nRUNTIME_EXIT={result.returncode}"
    return output, True


def run_claude(prompt: str, workspace: Path, env: dict[str, str], max_turns: int, timeout: int) -> tuple[str, bool]:
    result = subprocess.run(
        [
            "claude",
            "-p",
            "-",
            "--bare",
            "--no-session-persistence",
            "--model",
            DEFAULT_MODEL,
            "--agent",
            "researcher-agent",
            "--effort",
            "max",
            "--max-turns",
            str(max_turns),
            "--output-format",
            "stream-json",
            "--verbose",
            "--plugin-dir",
            str(ROOT / "plugins" / "senpai"),
            "--tools",
            "Read",
            "Write",
            "WebSearch",
            "WebFetch",
            "--dangerously-skip-permissions",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=workspace,
        env=env,
    )
    output = result.stdout + "\n" + result.stderr
    final = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            final.append(event["result"])
        message = event.get("message", {})
        for content in message.get("content", []) if isinstance(message, dict) else []:
            if content.get("type") == "text":
                final.append(content.get("text", ""))
    if result.returncode != 0:
        output += f"\nRUNTIME_EXIT={result.returncode}"
    return "\n".join(final) + "\n\n" + output, True


def extract_json_object(text: str) -> tuple[dict[str, object] | None, str]:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = blocks or [text]
    decoder = json.JSONDecoder()
    found: list[dict[str, object]] = []
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                found.append(obj)
    for obj in found:
        if any(key in obj for key in REQUIRED_KEYS):
            return obj, ""
    return None, "no JSON object found"


def timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def flatten(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).lower()


def has_forbidden_next(text: str, forbidden: tuple[str, ...]) -> bool:
    compact = re.sub(r"\s+", " ", text.lower())
    nextish = compact
    for marker in ["next_experiment", "recommendation"]:
        idx = compact.find(marker)
        if idx >= 0:
            nextish = compact[idx : idx + 1800]
            break
    return any(term in nextish for term in forbidden)


def score_output(task: QualityTask, output: str, subagent_logged: bool) -> tuple[int, bool, bool, str]:
    obj, error = extract_json_object(output)
    if obj is None:
        return 0, False, False, error

    text = flatten(obj)
    score = 0
    details = []

    score += 4
    missing = [key for key in REQUIRED_KEYS if key not in obj]
    if not missing:
        score += 4
    else:
        details.append(f"missing_keys={missing}")

    evidence_hits = sum(1 for term in task.expected_terms if term.lower() in text)
    pr_hits = len(set(re.findall(r"#\d+", text)))
    if evidence_hits >= 3 and pr_hits >= 1:
        score += 4
    else:
        details.append(f"weak_evidence_terms={evidence_hits}, pr_hits={pr_hits}")

    if any(word in text for word in ["mechanism", "because", "therefore", "causal", "bottleneck"]):
        score += 3
    else:
        details.append("weak_mechanism")

    if any(word in text for word in ["falsify", "abandon", "stop", "threshold", "gate"]):
        score += 3
    else:
        details.append("weak_stop_condition")

    if not has_forbidden_next(text, task.forbidden_next_terms):
        score += 4
    else:
        details.append("forbidden_dead_axis_repeated")

    if subagent_logged:
        score += 2
    else:
        details.append("researcher_agent_not_logged")

    passed = score >= 20 and not has_forbidden_next(text, task.forbidden_next_terms)
    return score, True, passed, "; ".join(details) or "ok"


def run_trial(
    runtime: str,
    task: QualityTask,
    index: int,
    contexts: dict[int, dict[str, object]],
    output_dir: Path,
    timeout: int,
) -> TrialResult:
    workspace = output_dir / runtime / task.name / f"run_{index:02d}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    prompt = task_prompt(task, contexts)
    (workspace / "PROMPT.md").write_text(prompt, encoding="utf-8")
    env = base_env(workspace)

    start = time.time()
    try:
        if runtime == "openhands":
            output, subagent_logged = run_openhands(prompt, workspace, env, task.max_turns, timeout)
        elif runtime == "claude":
            output, subagent_logged = run_claude(prompt, workspace, env, task.max_turns, timeout)
        else:
            raise ValueError(f"unknown runtime {runtime}")
    except subprocess.TimeoutExpired as exc:
        partial_stdout = timeout_text(exc.stdout)
        partial_stderr = timeout_text(exc.stderr)
        output = f"{partial_stdout}\n{partial_stderr}\ntimeout after {timeout}s"
        subagent_logged = "researcher-agent" in output
    except Exception as exc:
        output = f"{exc.__class__.__name__}: {exc}"
        subagent_logged = False

    output_path = workspace / "output.txt"
    output_path.write_text(output, encoding="utf-8")
    scored_output, scored_path = read_scored_output(workspace, output)
    score, valid_json, passed, detail = score_output(task, scored_output, subagent_logged)
    return TrialResult(
        runtime=runtime,
        task=task.name,
        index=index,
        score=score,
        passed=passed,
        valid_json=valid_json,
        subagent_logged=subagent_logged,
        detail=detail,
        seconds=time.time() - start,
        workspace=str(workspace),
        output_path=str(output_path),
        scored_path=str(scored_path),
    )


def summarize(results: list[TrialResult]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[TrialResult]] = {}
    for result in results:
        groups.setdefault((result.runtime, result.task), []).append(result)
    return {
        "results": [result.__dict__ for result in results],
        "summary": {
            f"{runtime}/{task}": {
                "passed": sum(result.passed for result in group),
                "total": len(group),
                "pass_rate": sum(result.passed for result in group) / len(group),
                "avg_score": sum(result.score for result in group) / len(group),
                "avg_seconds": sum(result.seconds for result in group) / len(group),
                "valid_json": sum(result.valid_json for result in group),
                "subagent_logged": sum(result.subagent_logged for result in group),
                "failures": [
                    {
                        "index": result.index,
                        "score": result.score,
                        "detail": result.detail,
                        "output_path": result.output_path,
                        "scored_path": result.scored_path,
                    }
                    for result in group
                    if not result.passed
                ],
            }
            for (runtime, task), group in sorted(groups.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runtime", choices=["openhands", "claude"], action="append")
    parser.add_argument("--task", choices=sorted(TASKS), action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_runtimes = args.runtime or ["openhands", "claude"]
    selected_tasks = [TASKS[name] for name in (args.task or sorted(TASKS))]
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="senpai-research-quality-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = fetch_pr_context(PR_NUMBERS)
    (output_dir / "pr_context.json").write_text(
        json.dumps(contexts, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    futures = []
    results: list[TrialResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for runtime in selected_runtimes:
            for task in selected_tasks:
                for index in range(1, args.runs + 1):
                    futures.append(
                        pool.submit(run_trial, runtime, task, index, contexts, output_dir, args.timeout)
                    )
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"{status} {result.runtime}/{result.task}#{result.index} "
                f"score={result.score} {result.seconds:.1f}s {result.detail}",
                flush=True,
            )

    report = summarize(results)
    report["repo"] = REPO
    report["prs"] = PR_NUMBERS
    report["model"] = DEFAULT_MODEL
    report["openhands_model"] = OPENHANDS_MODEL
    report["output_dir"] = str(output_dir)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
