import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
TEMPLATE_TOKEN = re.compile(r"\{\{[A-Z0-9_]+\}\}")


def load_kubernetes_template(name: str) -> dict:
    """Render Go-template tokens before asking PyYAML to parse the manifest."""
    template = (ROOT / "k8s" / name).read_text(encoding="utf-8")
    return yaml.safe_load(TEMPLATE_TOKEN.sub("fixture", template))


def container_for(manifest: dict) -> dict:
    return manifest["spec"]["template"]["spec"]["containers"][0]


def named_items(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items}


def test_advisor_image_is_a_locked_lightweight_openhands_runtime():
    dockerfile = (ROOT / "Dockerfile.advisor").read_text(encoding="utf-8")
    lowered = dockerfile.lower()

    assert dockerfile.startswith("FROM python:3.13-slim")
    assert "uv export --locked" in dockerfile
    assert "--prune torch" in dockerfile
    assert "--prune torchvision" in dockerfile
    assert "--prune torch-geometric" in dockerfile
    assert "coreweave/ml-containers" not in lowered
    assert "nvidia_" not in lowered
    assert "senpai-gpu-smoke-test" not in lowered
    assert "import torch" not in lowered
    assert "@anthropic-ai/claude-code" not in lowered


def test_student_image_keeps_the_locked_cuda_training_runtime():
    dockerfile = (ROOT / "Dockerfile.student").read_text(encoding="utf-8")
    lowered = dockerfile.lower()

    assert "coreweave/ml-containers" in lowered
    assert "uv export --locked" in dockerfile
    assert "import importlib.metadata, openhands.sdk, sys, torch" in dockerfile
    assert "NVIDIA_VISIBLE_DEVICES=all" in dockerfile
    assert "senpai-gpu-smoke-test" in dockerfile
    assert "@anthropic-ai/claude-code" not in lowered


def test_both_images_include_gh_and_working_chromium():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "apt-get install -y gh" in dockerfile
        assert "playwright install chromium" in dockerfile
        assert "senpai-browser-smoke-test" in dockerfile
        assert "senpai-browser-smoke-test &&" in dockerfile


def test_both_images_expose_the_controller_lease_as_their_healthcheck():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "HEALTHCHECK" in dockerfile
        assert "senpai_agent.supervisor health" in dockerfile


def test_neither_role_image_depends_on_kubernetes_tools():
    advisor = (ROOT / "Dockerfile.advisor").read_text(encoding="utf-8")
    student = (ROOT / "Dockerfile.student").read_text(encoding="utf-8")

    assert "kubectl" not in advisor
    assert "kubectl" not in student


def test_both_images_record_the_exact_source_revision():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "ARG SENPAI_SOURCE_REVISION=unknown" in dockerfile
        assert (
            'LABEL org.opencontainers.image.revision="${SENPAI_SOURCE_REVISION}"'
            in dockerfile
        )
        assert 'SENPAI_IMAGE_REVISION="${SENPAI_SOURCE_REVISION}"' in dockerfile


def test_build_workflow_builds_both_images_from_the_exact_checked_out_commit():
    source = (ROOT / ".github" / "workflows" / "build.yaml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    events = yaml.load(source, Loader=yaml.BaseLoader)["on"]
    build = workflow["jobs"]["build"]
    roles = build["strategy"]["matrix"]["role"]
    steps = {step["name"]: step for step in build["steps"]}

    assert set(roles) == {"advisor", "student"}
    assert events["pull_request"] == {}
    assert build["env"]["IMAGE_NAME"] == ("${{ github.repository }}-${{ matrix.role }}")
    assert steps["Checkout"]["with"]["ref"] == "${{ env.SOURCE_REVISION }}"
    assert steps["Extract metadata"]["with"]["images"] == (
        "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}"
    )
    assert (
        "type=raw,value=sha-${{ env.SOURCE_REVISION }}"
        in steps["Extract metadata"]["with"]["tags"]
    )
    build_inputs = steps["Build and push"]["with"]
    assert build_inputs["file"] == "Dockerfile.${{ matrix.role }}"
    assert (
        "SENPAI_SOURCE_REVISION=${{ env.SOURCE_REVISION }}"
        in build_inputs["build-args"]
    )


def test_runtime_workflow_uses_the_lockfile_uv_and_exa_versions():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test.yaml").read_text(encoding="utf-8")
    )
    steps = {
        step["name"]: step for step in workflow["jobs"]["runtime"]["steps"]
    }

    assert steps["Install uv and Python"]["with"]["version"] == "0.10.9"
    install = steps["Install runtime test dependencies"]["run"]
    assert "uv lock --check" in install
    assert "exa-py @ https://github.com/exa-labs/exa-py/archive/" in install


def test_advisor_state_is_persistent_and_student_state_is_ephemeral():
    advisor = load_kubernetes_template("advisor-deployment.yaml")
    student = load_kubernetes_template("student-deployment.yaml")

    advisor_container = container_for(advisor)
    advisor_mounts = named_items(advisor_container["volumeMounts"])
    advisor_volumes = named_items(advisor["spec"]["template"]["spec"]["volumes"])
    assert advisor_mounts["state"]["mountPath"] == "/var/lib/senpai"
    assert advisor_volumes["state"]["persistentVolumeClaim"] == {"claimName": "fixture"}
    assert "serve-events" not in advisor_container["args"][0]

    student_container = container_for(student)
    student_mounts = named_items(student_container["volumeMounts"])
    student_volumes = named_items(student["spec"]["template"]["spec"]["volumes"])
    assert student_mounts["state"]["mountPath"] == "/var/lib/senpai"
    assert student_volumes["state"]["emptyDir"] == {}
    assert "student_logs" not in student_container["args"][0]


def test_entrypoints_delegate_runtime_lifecycle_to_the_python_supervisor():
    entrypoint = (ROOT / "k8s" / "entrypoint-advisor.sh").read_text(encoding="utf-8")
    advisor = load_kubernetes_template("advisor-deployment.yaml")
    container = container_for(advisor)

    assert 'LOGDIR="/var/lib/senpai/$RESEARCH_TAG/advisor"' in entrypoint
    assert "serve-events" not in entrypoint
    assert "SENPAI_ADVISOR_EVENT_TOKEN" not in entrypoint
    assert "exec python -m senpai_agent.supervisor" in entrypoint
    assert "readinessProbe" not in container
    assert container["livenessProbe"]["exec"]["command"][:2] == [
        "/bin/sh",
        "-c",
    ]
    assert (
        "senpai_agent.supervisor health"
        in container["livenessProbe"]["exec"]["command"][2]
    )
    assert advisor["spec"]["strategy"] == {"type": "Recreate"}


def test_bootstrap_git_credentials_are_not_exposed_in_process_arguments():
    for role in ("advisor", "student"):
        deployment = load_kubernetes_template(f"{role}-deployment.yaml")
        container = container_for(deployment)
        bootstrap = container["args"][0]

        assert "${GITHUB_TOKEN}@github.com" not in bootstrap
        assert "GIT_ASKPASS" in bootstrap
        assert "mkdir -p /workspace" in bootstrap

    for role in ("advisor", "student"):
        container = container_for(load_kubernetes_template(f"{role}-deployment.yaml"))
        assert (
            "senpai_agent.supervisor health"
            in container["startupProbe"]["exec"]["command"][2]
        )
        assert container["startupProbe"]["failureThreshold"] == 60


def test_runtime_git_auth_uses_ephemeral_askpass_not_a_credential_store():
    guard = (ROOT / "plugins" / "senpai" / "scripts" / "git-guard.sh").read_text(
        encoding="utf-8"
    )
    assert "GIT_ASKPASS" in guard
    assert "GIT_TERMINAL_PROMPT" in guard
    assert ".git-credentials" not in guard
    assert 'credential.helper "store' not in guard
    assert "x-access-token:%s@github.com" not in guard

    for role in ("advisor", "student"):
        entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
            encoding="utf-8"
        )
        assert 'GIT_ASKPASS_FILE="/tmp/senpai-git-askpass"' in entrypoint
        assert ".git-credentials" not in entrypoint
        assert 'credential.helper "store' not in entrypoint


def test_github_is_the_only_cross_node_communication_dependency():
    deployment = load_kubernetes_template("advisor-deployment.yaml")
    student = load_kubernetes_template("student-deployment.yaml")

    advisor_container = container_for(deployment)
    advisor_env = named_items(advisor_container["env"])
    student_env = named_items(container_for(student)["env"])
    assert "ports" not in advisor_container
    assert not {
        "SENPAI_ADVISOR_EVENT_TOKEN",
        "SENPAI_ADVISOR_NOTIFY_URL",
        "SENPAI_ADVISOR_NOTIFY_TOKEN",
    } & (set(advisor_env) | set(student_env))
    assert not (ROOT / "k8s" / "advisor-service.yaml").exists()


def test_launch_configuration_names_only_the_two_role_images():
    config = yaml.safe_load((ROOT / "senpai.yaml").read_text(encoding="utf-8"))

    assert "advisor_image" in config
    assert "student_image" in config
    assert "control_image" not in config
    assert "image" not in config
