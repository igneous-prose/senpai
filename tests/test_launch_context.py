import base64

import pytest
import yaml

from launch_test_support import launch, launch_args, render_role


def test_default_fleet_is_four_students_with_one_gpu_each():
    args = launch.Args(
        tag="defaults",
        target_repo_url="https://github.com/example/problem.git",
    )

    assert args.n_students == 4
    assert args.gpus_per_student == 1
    assert args.program_path == ""


@pytest.mark.parametrize("backend", ["kubernetes", "docker", "aws"])
def test_launch_context_records_resolved_runtime_facts(backend):
    args = launch_args(
        tag="foil-run",
        advisor_branch="research-v2",
        target_repo_branch="main",
        gpus_per_student=3,
        timeout_minutes=12.5,
        max_epochs=7,
    )

    context = launch.build_launch_context(
        args,
        args.tag,
        ["fern", "frieren"],
        backend=backend,
    )

    assert "resolved by the Senpai launcher" in context
    assert "override conflicting compute or run-limit claims" in context
    assert f"Compute backend: `{backend}`" in context
    assert "Visible GPUs per student: `3`" in context
    assert (
        "Hard limits for each training run: `12.5` minutes wall-clock and `7` epochs"
        in context
    )
    assert "research tag `foil-run`" in context
    assert "advisor branch `research-v2`" in context
    assert "base branch `main`" in context
    assert "fern, frieren" in context
    assert "{{" not in context


def test_launch_context_limits_each_role_to_its_assigned_students():
    args = launch_args(tag="bounded", advisor_branch="research")

    advisor = launch.build_launch_context(
        args,
        args.tag,
        ["fern", "stark"],
        backend="kubernetes",
    )
    student = launch.build_launch_context(
        args,
        args.tag,
        ["stark"],
        backend="kubernetes",
    )

    assert "fern, stark" in advisor
    assert "fern" not in student
    assert "stark" in student


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_authoritative_launch_context(role):
    args = launch_args(
        gpus_per_student=2,
        timeout_minutes=20,
        max_epochs=9,
        extra_instructions="Prefer small, measurable experiments.",
    )

    configmap, _deployment, _secret = render_role(role, args)
    data = yaml.safe_load(configmap)["data"]
    context = base64.b64decode(
        data[launch.LAUNCH_CONTEXT_ENV], validate=True
    ).decode()
    operator = base64.b64decode(
        data["EXTRA_INSTRUCTIONS_B64"], validate=True
    ).decode()

    assert "Compute backend: `kubernetes`" in context
    assert "Visible GPUs per student: `2`" in context
    assert (
        "Hard limits for each training run: `20` minutes wall-clock and `9` epochs"
        in context
    )
    assert "Prefer small, measurable experiments." not in context
    assert operator == "Prefer small, measurable experiments."


def test_launch_context_source_is_combined():
    root = launch.ROOT / "system_instructions"

    assert (root / "SENPAI-LAUNCH-CONTEXT.md").is_file()
    assert not (root / "SENPAI-LAUNCH-RUNTIME.md").exists()
    assert not (root / "SENPAI-LAUNCH-ISOLATION.md").exists()


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_each_role_receives_the_configured_program_path(role):
    configmap, _deployment, _secret = render_role(
        role,
        launch_args(program_path="senpai/program.md"),
    )

    assert yaml.safe_load(configmap)["data"]["SENPAI_PROGRAM_PATH"] == (
        "senpai/program.md"
    )
