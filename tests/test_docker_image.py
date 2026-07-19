"""End-to-end test for the senpai Docker image on a k8s cluster.

Deploys a test pod, verifies the image runtime, then tears down the pod.

Usage:
    uv run pytest tests/test_docker_image.py -v -s
"""

import json
import subprocess
import time
from pathlib import Path

import pytest

ENTITY = "wandb-applied-ai-team"
PROJECT = "senpai-v1"
POD_NAME = "senpai-image-test"
IMAGE = "ghcr.io/wandb/senpai:pr-3467-fdcfbaf668"
REPO_URL = "https://github.com/wandb/senpai.git"
REPO_BRANCH = "main"
POD_TEMPLATE = Path(__file__).parent / "test-pod.yaml"
STARTUP_TIMEOUT = 120
TAG = "test"


def kubectl(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True, text=True, timeout=timeout, input=input,
    )
    return result.stdout.strip()


def kubectl_check(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True, text=True, timeout=timeout, input=input,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def wait_for_pod(name: str, timeout: int = STARTUP_TIMEOUT):
    """Poll until pod is Running or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = kubectl("get", "pod", name, "-o", "jsonpath={.status.phase}")
        if out == "Running":
            return
        time.sleep(5)
    raise TimeoutError(f"Pod {name} not running after {timeout}s")


def _build_configmap() -> str:
    """Generate the senpai-config-test ConfigMap YAML."""
    return "\n".join([
        "apiVersion: v1", "kind: ConfigMap", "metadata:",
        "  name: senpai-config-test",
        "  labels:",
        "    app: senpai",
        "    role: test",
        f"    research-tag: {TAG}",
        "data:",
        f'  REPO_URL: "{REPO_URL}"',
        f'  REPO_BRANCH: "{REPO_BRANCH}"',
        f'  RESEARCH_TAG: "{TAG}"',
        f'  WANDB_ENTITY: "{ENTITY}"',
        f'  WANDB_PROJECT: "{PROJECT}"',
    ])


def _render_pod_template() -> str:
    """Render the test-pod.yaml template with IMAGE and RESEARCH_TAG."""
    text = POD_TEMPLATE.read_text()
    return text.replace("{{IMAGE}}", IMAGE).replace("{{RESEARCH_TAG}}", TAG)


@pytest.fixture(scope="module")
def test_pod():
    """Create configmap + test pod, wait for it, yield, then clean up."""
    kubectl("delete", "pod,configmap", "-l", f"research-tag={TAG}", "--ignore-not-found", timeout=120)
    time.sleep(2)

    # Apply configmap then pod
    kubectl_check("apply", "-f", "-", input=_build_configmap())
    kubectl_check("apply", "-f", "-", input=_render_pod_template())
    wait_for_pod(POD_NAME)
    yield POD_NAME

    kubectl("delete", "pod,configmap", "-l", f"research-tag={TAG}", "--ignore-not-found", timeout=120)


def test_tools_installed(test_pod):
    """All baked-in tools are available."""
    for cmd in ["claude --version", "gh --version", "uv --version", "yq --version"]:
        out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=15)
        assert out, f"`{cmd}` returned empty output"


def test_legacy_weave_claude_plugin_removed(test_pod):
    """The retired Claude Code tracing plugin is absent."""
    cmd = "! command -v weave-claude-plugin && test ! -e ~/.weave_claude_plugin && echo ok"
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=15)
    assert out == "ok"


def test_python_deps_and_icml_target_import(test_pod):
    """The image has the Python deps needed by the new ICML target."""
    cmd = (
        "python - <<'PY'\n"
        "import importlib.metadata\n"
        "import sys\n"
        "import numpy\n"
        "import openhands.sdk\n"
        "import torch\n"
        "import torch_geometric\n"
        "import weave_openhands\n"
        "import yaml\n"
        "assert sys.version_info[:2] == (3, 13)\n"
        "assert torch.__version__.startswith('2.13.')\n"
        "assert torch.version.cuda.startswith('13.')\n"
        "assert importlib.metadata.version('openhands-sdk') == '1.36.1'\n"
        "assert importlib.metadata.version('weave-openhands') == '0.1.0'\n"
        "print('ok')\n"
        "PY"
    )
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=20)
    assert "ok" in out


def test_cuda_runtime_on_a_real_gpu(test_pod):
    """PyTorch can execute a kernel through the host NVIDIA runtime."""
    out = kubectl_check(
        "exec", test_pod, "--", "senpai-gpu-smoke-test", timeout=30
    )
    result = json.loads(out)

    assert result["status"] == "ok"
    assert result["devices"]


def test_openhands_plugin_loads_workflow_skills_and_exa(test_pod):
    """The native OpenHands plugin carries the workflow skills and Exa."""
    cmd = (
        "python - <<'PY'\n"
        "from openhands.sdk.plugin import Plugin\n"
        "plugin = Plugin.load('/workspaces/senpai/plugins/senpai')\n"
        "skills = {skill.name for skill in plugin.skills}\n"
        "assert {'senpai-gh', 'survey-prs'} <= skills\n"
        "assert 'exa' in plugin.mcp_config\n"
        "print('ok')\n"
        "PY"
    )
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=20)
    assert "ok" in out
