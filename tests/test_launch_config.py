import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "k8s"))

import launch_helpers
from launch_helpers import (
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
