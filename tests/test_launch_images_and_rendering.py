import base64
import re

import pytest
import yaml

from launch_test_support import (
    ADVISOR_IMAGE,
    REVISION,
    STUDENT_IMAGE,
    launch,
    launch_args,
    launch_helpers,
    render_role,
    run_launch,
)


@pytest.mark.parametrize(
    "image",
    [
        f"ghcr.io/wandb/senpai:sha-{'a' * 40}",
        f"ghcr.io/wandb/senpai@sha256:{'b' * 64}",
        f"registry.example:5000/team/senpai@sha256:{'c' * 64}",
    ],
)
def test_image_reference_accepts_only_full_source_sha_tags_or_digests(image):
    assert launch_helpers.is_immutable_image_reference(image)


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
def test_image_reference_rejects_mutable_or_incomplete_pins(image):
    assert not launch_helpers.is_immutable_image_reference(image)


def test_source_revision_is_derived_from_a_full_sha_tag():
    image = f"ghcr.io/wandb/senpai:sha-{REVISION}"

    assert launch_helpers.source_revision_for_image(image) == REVISION


def test_digest_image_requires_an_explicit_source_revision():
    image = f"ghcr.io/wandb/senpai@sha256:{'b' * 64}"

    assert launch_helpers.source_revision_for_image(image, REVISION) == REVISION
    with pytest.raises(ValueError, match="require an explicit repo_revision"):
        launch_helpers.source_revision_for_image(image)


def test_explicit_revision_must_match_the_source_sha_tag():
    image = f"ghcr.io/wandb/senpai:sha-{REVISION}"

    with pytest.raises(ValueError, match="does not match"):
        launch_helpers.source_revision_for_image(image, "b" * 40)


def test_dry_run_binds_each_role_image_to_the_derived_source_revision():
    result = run_launch(
        "--advisor",
        "--advisor_image",
        ADVISOR_IMAGE,
        "--student_image",
        STUDENT_IMAGE,
    )

    assert result.returncode == 0, result.stderr
    rendered_yaml = re.sub(r"^--- .+ ---$", "---", result.stdout, flags=re.MULTILINE)
    documents = [
        document
        for document in yaml.safe_load_all(rendered_yaml)
        if isinstance(document, dict)
    ]
    deployments = {
        document["metadata"]["labels"]["role"]: document
        for document in documents
        if document.get("kind") == "Deployment"
    }
    assert {
        role: deployment["spec"]["template"]["spec"]["containers"][0]["image"]
        for role, deployment in deployments.items()
    } == {"advisor": ADVISOR_IMAGE, "student": STUDENT_IMAGE}
    assert {
        document["data"]["REPO_REVISION"]
        for document in documents
        if document.get("kind") == "ConfigMap"
    } == {REVISION}


def test_launch_rejects_role_images_from_different_source_revisions():
    result = run_launch(
        "--advisor_image",
        ADVISOR_IMAGE,
        "--student_image",
        f"ghcr.io/wandb/senpai-student:sha-{'b' * 40}",
    )

    assert result.returncode != 0
    assert "same source revision" in result.stderr


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_launch_rejects_a_mutable_image_for_either_role(role):
    images = {"advisor": ADVISOR_IMAGE, "student": STUDENT_IMAGE}
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
def test_role_bootstrap_verifies_both_checkout_and_image_source_revision(role):
    _configmap, deployment, _secret = render_role(role)
    command = yaml.safe_load(deployment)["spec"]["template"]["spec"]["containers"][
        0
    ]["args"][0]

    assert 'fetch --depth 1 "$REPO_URL" "$REPO_REVISION"' in command
    assert 'test "$(git rev-parse HEAD)" = "$REPO_REVISION"' in command
    assert 'test "$SENPAI_IMAGE_REVISION" = "$REPO_REVISION"' in command


@pytest.mark.parametrize(
    "gate",
    ["relative/start-gate", "/tmp/start-gate", "/mnt/shared/../escape"],
)
def test_start_gate_must_be_a_normalized_path_beneath_the_shared_pvc(gate):
    args = launch_args(pvc_mount_path="/mnt/shared", start_gate_path=gate)

    with pytest.raises(SystemExit, match="shared PVC"):
        launch.validate_timing_args(args)


def test_start_gate_is_rendered_when_it_is_beneath_the_shared_pvc():
    args = launch_args(
        pvc_mount_path="/mnt/shared",
        start_gate_path="/mnt/shared/gates/start",
    )

    launch.validate_timing_args(args)
    configmap, _deployment, _secret = render_role("student", args)

    assert yaml.safe_load(configmap)["data"]["SENPAI_START_GATE_PATH"] == (
        "/mnt/shared/gates/start"
    )


def test_launch_secret_contains_each_credential_and_both_roles_reference_it():
    expected_values = {
        "github-token": "github",
        "anthropic-api-key": "anthropic",
        "exa-api-key": "exa",
        "wandb-api-key": "wandb",
    }

    _configmap, _deployment, secret = render_role("advisor")
    secret_document = yaml.safe_load(secret)
    assert {
        key: base64.b64decode(value).decode()
        for key, value in secret_document["data"].items()
    } == expected_values

    for role in ("advisor", "student"):
        _configmap, deployment, _secret = render_role(role)
        container = yaml.safe_load(deployment)["spec"]["template"]["spec"][
            "containers"
        ][0]
        references = {
            item["valueFrom"]["secretKeyRef"]["key"]: item["valueFrom"][
                "secretKeyRef"
            ]["name"]
            for item in container["env"]
        }
        assert references == {
            key: "senpai-launch-secrets-test-track" for key in expected_values
        }


def test_pod_template_hash_covers_complete_config_and_secret_content():
    config = "kind: ConfigMap\ndata:\n  POLL_INTERVAL: '60'\n"
    secret = launch_helpers.render_launch_secret(
        "track",
        "github",
        "anthropic",
        "exa",
        "wandb",
    )

    first = launch_helpers.pod_template_hash(config, secret)

    assert first == launch_helpers.pod_template_hash(config, secret)
    assert first != launch_helpers.pod_template_hash(
        config.replace("'60'", "'120'"), secret
    )
    assert first != launch_helpers.pod_template_hash(
        config, secret.replace("d2FuZGI=", "bmV3LXdhbmRi")
    )


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_rendered_role_annotation_matches_its_effective_content_hash(role):
    configmap, deployment, secret = render_role(role)

    annotation = yaml.safe_load(deployment)["spec"]["template"]["metadata"][
        "annotations"
    ]["senpai.wandb.com/content-hash"]

    assert annotation == launch_helpers.pod_template_hash(configmap, secret)
