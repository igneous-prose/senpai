import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k8s"))

import launch_helpers
import launch
from launch_helpers import (
    existing_student_names,
    is_immutable_image_reference,
    kubectl_apply,
    pod_template_hash,
    preflight_check_exa_api_key,
    preflight_check_target_repo_access,
    preflight_check_wandb_api_key,
    render_launch_secret,
    source_revision_for_image,
)

ROOT = Path(__file__).resolve().parents[1]


def run_launch(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "k8s" / "launch.py"),
            "--dry_run",
            "--tag",
            "image-split",
            "--target_repo_url",
            "https://github.com/example/problem.git",
            "--n_students",
            "1",
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "image",
    [
        f"ghcr.io/wandb/senpai:sha-{'a' * 40}",
        f"ghcr.io/wandb/senpai@sha256:{'b' * 64}",
        f"registry.example:5000/team/senpai@sha256:{'c' * 64}",
    ],
)
def test_immutable_image_references(image):
    assert is_immutable_image_reference(image)


@pytest.mark.parametrize(
    "image",
    [
        "",
        "ghcr.io/wandb/senpai:latest",
        "ghcr.io/wandb/senpai:sha-deadbeef",
        f"ghcr.io/wandb/senpai:sha-{'A' * 40}",
        f"ghcr.io/wandb/senpai@sha256:{'b' * 63}",
    ],
)
def test_mutable_or_incomplete_image_references(image):
    assert not is_immutable_image_reference(image)


def test_source_revision_is_derived_from_full_sha_tag():
    revision = "a" * 40

    assert source_revision_for_image(f"ghcr.io/wandb/senpai:sha-{revision}") == revision


def test_digest_requires_an_explicit_source_revision():
    image = f"ghcr.io/wandb/senpai@sha256:{'b' * 64}"
    revision = "c" * 40

    assert source_revision_for_image(image, revision) == revision
    with pytest.raises(ValueError, match="require an explicit repo_revision"):
        source_revision_for_image(image)


def test_explicit_revision_must_match_sha_tag():
    image = f"ghcr.io/wandb/senpai:sha-{'a' * 40}"

    with pytest.raises(ValueError, match="does not match"):
        source_revision_for_image(image, "b" * 40)


def test_launch_renders_each_role_with_its_matching_image():
    revision = "a" * 40
    advisor_image = f"ghcr.io/wandb/senpai-advisor:sha-{revision}"
    student_image = f"ghcr.io/wandb/senpai-student:sha-{revision}"

    result = run_launch(
        "--advisor",
        "--advisor_image",
        advisor_image,
        "--student_image",
        student_image,
    )

    assert result.returncode == 0, result.stderr
    assert f"image: {advisor_image}" in result.stdout
    assert f"image: {student_image}" in result.stdout
    assert "kind: Service" not in result.stdout
    assert "advisor-event-token:" not in result.stdout
    assert f'REPO_REVISION: "{revision}"' in result.stdout


def test_launch_needs_no_cross_node_service_or_rbac():
    revision = "a" * 40

    result = run_launch(
        "--advisor",
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
    )

    assert result.returncode == 0, result.stderr
    assert "kind: Service" not in result.stdout
    assert "kind: ServiceAccount" not in result.stdout
    assert "kind: Role" not in result.stdout
    assert "serviceAccountName:" not in result.stdout


@pytest.mark.parametrize("gate", ["relative/start-gate", "/tmp/start-gate"])
def test_launch_rejects_start_gate_outside_the_shared_pvc(gate: str):
    revision = "a" * 40

    result = run_launch(
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
        "--pvc_mount_path",
        "/mnt/shared",
        "--start_gate_path",
        gate,
    )

    assert result.returncode != 0
    assert "start_gate_path" in result.stderr
    assert "shared PVC" in result.stderr


def test_launch_accepts_start_gate_beneath_the_shared_pvc():
    revision = "a" * 40

    result = run_launch(
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
        "--pvc_mount_path",
        "/mnt/shared",
        "--start_gate_path",
        "/mnt/shared/gates/start",
    )

    assert result.returncode == 0, result.stderr
    assert 'SENPAI_START_GATE_PATH: "/mnt/shared/gates/start"' in result.stdout


def test_launch_secret_is_self_contained_and_both_roles_reference_it():
    revision = "a" * 40

    result = run_launch(
        "--advisor",
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
    )

    assert result.returncode == 0, result.stderr
    assert "wandb-api-key:" in result.stdout
    assert result.stdout.count("name: senpai-launch-secrets-image-split") >= 3
    assert result.stdout.count("key: wandb-api-key") == 2
    assert "senpai-secrets" not in result.stdout


def test_pod_template_hash_is_deterministic_and_covers_config_and_secrets():
    config = "kind: ConfigMap\ndata:\n  POLL_INTERVAL: '60'\n"
    secret = render_launch_secret(
        "track",
        "github",
        "anthropic",
        "exa",
        "wandb",
    )

    first = pod_template_hash(config, secret)

    assert first == pod_template_hash(config, secret)
    assert first != pod_template_hash(
        config.replace("'60'", "'120'"),
        secret,
    )
    assert first != pod_template_hash(
        config,
        secret.replace(
            base64.b64encode(b"wandb").decode(),
            base64.b64encode(b"new-wandb").decode(),
        ),
    )


def test_rendered_deployments_include_their_effective_content_hash():
    revision = "a" * 40

    first = run_launch(
        "--advisor",
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
        "--poll_interval_s",
        "60",
    )
    changed = run_launch(
        "--advisor",
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{revision}",
        "--poll_interval_s",
        "120",
    )

    assert first.returncode == changed.returncode == 0
    first_hashes = re.findall(
        r'senpai\.wandb\.com/content-hash: "([0-9a-f]{64})"',
        first.stdout,
    )
    changed_hashes = re.findall(
        r'senpai\.wandb\.com/content-hash: "([0-9a-f]{64})"',
        changed.stdout,
    )
    assert len(first_hashes) == len(changed_hashes) == 2
    assert first_hashes != changed_hashes


def test_launch_rejects_role_images_built_from_different_source_commits():
    result = run_launch(
        "--advisor_image",
        f"ghcr.io/wandb/senpai-advisor:sha-{'a' * 40}",
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{'b' * 40}",
    )

    assert result.returncode != 0
    assert "same source revision" in result.stderr


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_launch_rejects_mutable_role_images(role):
    revision = "a" * 40
    images = {
        "advisor": f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        "student": f"ghcr.io/wandb/senpai-student:sha-{revision}",
    }
    images[role] = f"ghcr.io/wandb/senpai-{role}:latest"

    result = run_launch(
        "--advisor_image",
        images["advisor"],
        "--student_image",
        images["student"],
    )

    assert result.returncode != 0
    assert f"--{role}_image must be an immutable digest" in result.stderr


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_deployments_checkout_and_verify_the_image_source_revision(role):
    template = (
        Path(__file__).resolve().parents[1] / "k8s" / f"{role}-deployment.yaml"
    ).read_text(encoding="utf-8")

    assert (
        'git -C /workspace/senpai fetch --depth 1 "$REPO_URL" "$REPO_REVISION"'
        in template
    )
    assert 'GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0' in template
    assert "checkout --detach FETCH_HEAD" in template
    assert 'test "$(git rev-parse HEAD)" = "$REPO_REVISION"' in template
    assert 'test "$SENPAI_IMAGE_REVISION" = "$REPO_REVISION"' in template
    assert "REPO_BRANCH" not in template


def test_exa_preflight_exercises_instant_publication_search(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"results":[{"id":"publication"}]}'

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    preflight_check_exa_api_key("exa-secret")

    request = captured["request"]
    payload = json.loads(request.data)
    assert request.full_url == "https://api.exa.ai/search"
    assert request.method == "POST"
    assert request.headers["X-api-key"] == "exa-secret"
    assert payload == {
        "query": "api credential preflight",
        "type": "instant",
        "category": "publication",
        "numResults": 1,
    }
    assert "contents" not in payload
    assert "summary" not in payload
    assert captured["timeout"] == 15


def test_exa_preflight_rejects_a_non_search_success_response(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"status":"ok"}'

    monkeypatch.setattr(
        launch_helpers.urllib.request,
        "urlopen",
        lambda request, timeout: Response(),
    )

    with pytest.raises(SystemExit, match="invalid search response"):
        preflight_check_exa_api_key("exa-secret")


def test_wandb_preflight_authenticates_with_a_minimal_viewer_query(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"data":{"viewer":{"id":"user"}}}'

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(launch_helpers.urllib.request, "urlopen", urlopen)

    preflight_check_wandb_api_key("wandb-secret")

    request = captured["request"]
    assert request.full_url == "https://api.wandb.ai/graphql"
    assert request.method == "POST"
    assert request.headers["Authorization"] == (
        "Basic " + base64.b64encode(b"api:wandb-secret").decode()
    )
    assert json.loads(request.data) == {
        "query": "query SenpaiPreflight { viewer { id } }"
    }
    assert captured["timeout"] == 10


def test_repo_access_accepts_push_permission_without_read_org_scope(monkeypatch):
    class Headers:
        def get(self, name):
            assert name == "X-OAuth-Scopes"
            return "repo"

    class Response:
        headers = Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"permissions":{"push":true}}'

    monkeypatch.setattr(
        launch_helpers.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(),
    )

    preflight_check_target_repo_access(
        "https://github.com/example/problem.git",
        "github-secret",
    )


def test_kubectl_apply_fails_the_launch_on_an_apply_error(monkeypatch):
    monkeypatch.setattr(
        launch_helpers.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["kubectl"],
            returncode=1,
            stdout="",
            stderr="forbidden",
        ),
    )

    with pytest.raises(RuntimeError, match="advisor service.*forbidden"):
        kubectl_apply("kind: Service", "advisor service")


def test_kubectl_helpers_scope_commands_to_context_and_namespace(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=("fern\nfrieren\n" if "get" in argv else "applied\n"),
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    names = existing_student_names(
        "track-a",
        kube_context="gpu-cluster",
        namespace="research",
    )
    kubectl_apply(
        "kind: ConfigMap",
        "track config",
        kube_context="gpu-cluster",
        namespace="research",
    )

    assert names == ["fern", "frieren"]
    assert calls[0][0] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research",
        "get",
        "deployments",
        "-l",
        "app=senpai,role=student,research-tag=track-a",
        "-o",
        'jsonpath={range .items[*]}{.metadata.labels.student}{"\\n"}{end}',
    ]
    assert calls[1][0] == [
        "kubectl",
        "--context",
        "gpu-cluster",
        "--namespace",
        "research",
        "apply",
        "-f",
        "-",
    ]
    assert calls[1][1]["input"] == "kind: ConfigMap"


def test_kubectl_default_scope_omits_an_empty_context(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="applied\n",
            stderr="",
        )

    monkeypatch.setattr(launch_helpers.subprocess, "run", run)

    kubectl_apply("kind: ConfigMap", "track config")

    assert captured["argv"] == [
        "kubectl",
        "--namespace",
        "default",
        "apply",
        "-f",
        "-",
    ]


def test_launch_uses_one_kubernetes_scope_for_apply_discovery_and_handoff_commands(
    monkeypatch,
    capsys,
):
    revision = "a" * 40
    args = launch.Args(
        tag="scope-test",
        target_repo_url="https://github.com/example/problem.git",
        names="fern",
        advisor=True,
        advisor_image=f"ghcr.io/wandb/senpai-advisor:sha-{revision}",
        student_image=f"ghcr.io/wandb/senpai-student:sha-{revision}",
        kube_context="gpu-cluster",
        namespace="research",
    )
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(launch, "resolve_github_token", lambda _path: "github")
    monkeypatch.setattr(launch, "resolve_anthropic_api_key", lambda _path: "anthropic")
    monkeypatch.setattr(launch, "resolve_exa_api_key", lambda _path: "exa")
    monkeypatch.setattr(launch, "resolve_wandb_api_key", lambda _path: "wandb")
    monkeypatch.setattr(launch, "preflight_check_target_repo_access", lambda *_: None)
    monkeypatch.setattr(
        launch,
        "preflight_check_target_repo_branch",
        lambda *_: "main",
    )
    monkeypatch.setattr(launch, "preflight_check_anthropic_api_key", lambda *_: None)
    monkeypatch.setattr(launch, "preflight_check_exa_api_key", lambda *_: None)
    monkeypatch.setattr(launch, "preflight_check_wandb_api_key", lambda *_: None)
    monkeypatch.setattr(launch, "ensure_advisor_branch", lambda *_: None)
    monkeypatch.setattr(launch, "ensure_target_repo_labels", lambda *_: None)
    discovery = []

    def existing(tag, *, kube_context, namespace):
        discovery.append((tag, kube_context, namespace))
        return []

    monkeypatch.setattr(launch, "existing_student_names", existing)
    applies = []

    def apply(manifest, name, *, kube_context, namespace):
        applies.append((name, kube_context, namespace))

    monkeypatch.setattr(launch, "kubectl_apply", apply)

    launch.main()

    assert discovery == [("scope-test", "gpu-cluster", "research")]
    assert applies == [
        ("secret senpai-launch-secrets-scope-test", "gpu-cluster", "research"),
        ("student fern", "gpu-cluster", "research"),
        ("advisor", "gpu-cluster", "research"),
    ]
    output = capsys.readouterr().out
    prefix = "kubectl --context gpu-cluster --namespace research"
    assert f"{prefix} get deployments -l research-tag=scope-test" in output
    assert f"{prefix} get deployment senpai-advisor-scope-test" in output
    assert f"{prefix} logs -f deployment/senpai-scope-test-fern" in output
    assert (
        f"{prefix} delete deployments,configmaps,secrets "
        "-l research-tag=scope-test"
    ) in output
