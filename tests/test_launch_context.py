import base64

import pytest
import yaml

from launch_test_support import launch, launch_args, render_role


@pytest.mark.parametrize("backend", ["kubernetes", "docker", "aws"])
def test_launch_context_records_resolved_runtime_facts(backend):
    args = launch_args(
        gpus_per_student=3,
        timeout_minutes=12.5,
        max_epochs=7,
    )

    context = launch.build_extra_instructions(
        args,
        args.tag,
        ["fern"],
        backend=backend,
    )

    assert "resolved by the Senpai launcher" in context
    assert "override conflicting compute or run-limit claims" in context
    assert f"Compute backend: `{backend}`" in context
    assert "Visible GPUs per student: `3`" in context
    assert (
        "Hard limits for each training run: `12.5` minutes wall-clock\n"
        "  and `7` epochs"
    ) in context


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_authoritative_launch_context(role):
    args = launch_args(
        gpus_per_student=2,
        timeout_minutes=20,
        max_epochs=9,
        extra_instructions="Prefer small, measurable experiments.",
    )

    configmap, _deployment, _secret = render_role(role, args)
    encoded = yaml.safe_load(configmap)["data"]["EXTRA_INSTRUCTIONS_B64"]
    context = base64.b64decode(encoded, validate=True).decode()

    assert "Compute backend: `kubernetes`" in context
    assert "Visible GPUs per student: `2`" in context
    assert (
        "Hard limits for each training run: `20` minutes wall-clock\n"
        "  and `9` epochs"
    ) in context
    assert context.endswith(
        "# Additional operator instructions\n\n"
        "Prefer small, measurable experiments."
    )
