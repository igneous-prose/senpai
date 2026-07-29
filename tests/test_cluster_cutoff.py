import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUTOFF = REPO_ROOT / "scripts" / "arm_senpai_cluster_cutoff.sh"


def run_cutoff(*args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(CUTOFF), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )


def test_cutoff_cli_has_no_conversation_harvest_or_local_archive_options():
    result = run_cutoff("--help")

    assert result.returncode == 0
    output = result.stdout.lower()
    assert "harvest" not in output
    assert "local pull" not in output
    assert "parallel copies" not in output


def test_cutoff_dry_run_keeps_readiness_and_delete_without_archive_rbac(tmp_path):
    captured_script = tmp_path / "cutoff-job.sh"
    fake_kubectl = tmp_path / "kubectl"
    fake_kubectl.write_text(
        """#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    --from-file=cutoff-job.sh=*)
      cp "${arg#--from-file=cutoff-job.sh=}" "$CAPTURED_CUTOFF_SCRIPT"
      ;;
  esac
done
printf '%s\n' 'apiVersion: v1' 'kind: ConfigMap'
""",
        encoding="utf-8",
    )
    fake_kubectl.chmod(0o755)

    result = run_cutoff(
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
        "--dry-run",
        env={
            "KUBECTL": str(fake_kubectl),
            "CAPTURED_CUTOFF_SCRIPT": str(captured_script),
        },
    )

    assert result.returncode == 0, result.stderr
    assert captured_script.is_file()
    rendered = result.stdout
    job_script = captured_script.read_text(encoding="utf-8")
    assert "Waiting for ready gate" in job_script
    assert 'sleep_until "$KILL_AT_EPOCH" "hard cutoff delete"' in job_script
    assert "delete deployments,configmaps,secrets" in job_script
    assert "harvest" not in job_script.lower()
    assert "kubectl exec" not in job_script
    assert 'resources: ["pods/log"]' not in rendered
    assert 'resources: ["pods/exec"]' not in rendered


def test_generated_cutoff_waits_for_readiness_then_deletes_selected_resources(tmp_path):
    captured_script = tmp_path / "cutoff-job.sh"
    generator_kubectl = tmp_path / "generator-kubectl"
    generator_kubectl.write_text(
        """#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    --from-file=cutoff-job.sh=*)
      cp "${arg#--from-file=cutoff-job.sh=}" "$CAPTURED_CUTOFF_SCRIPT"
      ;;
  esac
done
printf '%s\n' 'apiVersion: v1' 'kind: ConfigMap'
""",
        encoding="utf-8",
    )
    generator_kubectl.chmod(0o755)
    generated = run_cutoff(
        "--run-slug",
        "acceptance",
        "--tags-csv",
        "track-a",
        "--expected-pods",
        "1",
        "--expected-deployments",
        "1",
        "--budget-hours",
        "0",
        "--dry-run",
        env={
            "KUBECTL": str(generator_kubectl),
            "CAPTURED_CUTOFF_SCRIPT": str(captured_script),
        },
    )
    assert generated.returncode == 0, generated.stderr

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    delete_log = tmp_path / "delete.log"
    runtime_kubectl = bin_dir / "kubectl"
    runtime_kubectl.write_text(
        """#!/bin/sh
case "$*" in
  *"get pods"*)
    printf '%s\n' '{"items":[{"status":{"containerStatuses":[{"ready":true}]}}]}'
    ;;
  *"get deployments"*)
    printf '%s\n' 'senpai-track-a'
    ;;
  *"delete deployments,configmaps,secrets"*)
    printf '%s\n' "$*" > "$DELETE_LOG"
    ;;
  *)
    printf 'unexpected kubectl call: %s\n' "$*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    runtime_kubectl.chmod(0o755)
    state_root = tmp_path / "state"
    gate = tmp_path / "start-gate"
    result = subprocess.run(
        ["bash", str(captured_script)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DELETE_LOG": str(delete_log),
            "RUN_SLUG": "acceptance",
            "TAGS_CSV": "track-a",
            "EXPECTED_PODS": "1",
            "EXPECTED_DEPLOYMENTS": "1",
            "BUDGET_SECONDS": "0",
            "PVC_LOG_ROOT": str(state_root),
            "START_GATE_PATH": str(gate),
            "NAMESPACE": "test-ns",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert gate.is_file()
    deleted = delete_log.read_text(encoding="utf-8")
    assert "delete deployments,configmaps,secrets" in deleted
    assert "research-tag in (track-a)" in deleted
