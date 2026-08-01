#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import posixpath
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
from launch_helpers import (
    ensure_advisor_branch,
    ensure_target_repo_labels,
    existing_student_names,
    expand_student_names,
    is_immutable_image_reference,
    kubectl_apply,
    kubectl_command,
    pod_template_hash,
    preflight_check_anthropic_api_key,
    preflight_check_exa_api_key,
    preflight_check_student_name_availability,
    preflight_check_target_repo_access,
    preflight_check_target_repo_branch,
    preflight_check_wandb_api_key,
    render_configmap,
    render_launch_secret,
    render_template,
    resolve_anthropic_api_key,
    resolve_exa_api_key,
    resolve_github_token,
    resolve_wandb_api_key,
    routing_labels,
    source_revision_for_image,
    target_repo_slug,
)

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"


@dataclass
class Args:
    """Launch senpai advisor and/or student agents on Kubernetes."""

    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo (entrypoint clones this into $PROBLEM_DIR; agent commits/PRs land here) — REQUIRED, no default
    target_repo_branch: str = ""  # target repo branch used as the base when creating advisor_branch; empty = target repo default branch
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 8  # GPUs requested by each student pod
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    repo_url: str = (
        "https://github.com/wandb/senpai.git"  # git repo URL (senpai runner)
    )
    repo_revision: str = (
        ""  # exact runner commit; derived from :sha-<commit> image tags
    )
    advisor_image: str = ""  # advisor source-SHA tag or image digest — REQUIRED
    student_image: str = ""  # student source-SHA tag or image digest — REQUIRED
    kube_context: str = ""  # kubectl context; empty uses the current context
    namespace: str = "default"  # Kubernetes namespace for all launch resources
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    human_issues: bool = (
        True  # allow human GitHub issue triage; disable for isolated launches
    )
    advisor_branch: str = "schmidhuber"  # branch the advisor works on inside the problem-package repo (students PR into it; created from target_repo_branch if missing)
    gh_history_scope: str = "branch"  # branch=normal track memory, fresh=clean ablation, repo=whole-repo memory
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into pods
    pvc_mount_path: str = (
        "/mnt/new-pvc"  # mount path for the dataset PVC inside the containers
    )
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    extra_instructions: str = (
        ""  # extra prompt text for the advisor: a .md file path or a literal string
    )
    timeout_minutes: float = (
        30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    )
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = (
        600  # default advisor/student outer-loop sleep between GitHub polls
    )
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    start_gate_path: str = ""  # optional shared file path that must exist before advisor/student loops begin
    dry_run: bool = (
        False  # render manifests only: do not apply them or validate credentials
    )
    preflight_only: bool = (
        False  # validate credentials/access only: do not render or apply manifests
    )


def validate_timing_args(args: Args) -> None:
    if args.timeout_minutes <= 0:
        sys.exit("ERROR: --timeout_minutes must be positive")
    if args.max_epochs < 1:
        sys.exit("ERROR: --max_epochs must be at least 1")
    positive = ["poll_interval_s"]
    non_negative = [
        "poll_jitter_s",
        "stale_wip_seconds",
    ]
    for name in positive:
        if getattr(args, name) < 1:
            sys.exit(f"ERROR: --{name} must be at least 1")
    for name in non_negative:
        if getattr(args, name) < 0:
            sys.exit(f"ERROR: --{name} must be non-negative")
    if args.start_gate_path:
        gate = posixpath.normpath(args.start_gate_path)
        mount = posixpath.normpath(args.pvc_mount_path)
        inside_mount = gate.startswith(f"{mount.rstrip('/')}/")
        if (
            not posixpath.isabs(gate)
            or gate != args.start_gate_path
            or not posixpath.isabs(mount)
            or not inside_mount
        ):
            sys.exit(
                "ERROR: --start_gate_path must be an absolute normalized file "
                "path beneath the shared PVC --pvc_mount_path"
            )


def load_extra_instructions(extra_instructions: str) -> str:
    if not extra_instructions:
        return ""
    p = Path(extra_instructions)
    return p.read_text() if p.exists() else extra_instructions


def build_extra_instructions(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
) -> str:
    students = ", ".join(student_list)
    target_base = args.target_repo_branch or "<default>"
    runtime = f"""# Authoritative launch context

These values were resolved by the Senpai launcher and describe the actual runtime.
They override conflicting compute or run-limit claims in the target repository's
`program.md` and role prompts.

- Compute backend: `{backend}`.
- Visible GPUs per student: `{args.gpus_per_student}`.
- Hard limits for each training run: `{args.timeout_minutes:g}` minutes wall-clock
  and `{args.max_epochs}` epochs.
- Use tools and operational commands that work with `{backend}`. Do not follow
  target-repository instructions written for another backend.
- Do not assume additional GPUs or bypass, extend, or continue past the hard training limits.
"""
    isolation = f"""# Launch isolation and run-limit rules

- This launch is scoped to research tag `{tag}`, advisor branch `{args.advisor_branch}`, and target base branch `{target_base}`.
- Only inspect, modify, or reason from `{args.advisor_branch}` plus PR branches assigned to these students in this launch: {students}.
- Do not inspect, compare, summarize, cherry-pick, borrow from, or base decisions on any PR or branch outside `{args.advisor_branch}` and the assigned student PR branches for this launch.
- Do not use unrelated experiment runs or historical results unless the human explicitly names them during this launch.
- Students branch from `{args.advisor_branch}`. Do not rebase or retarget work onto unrelated branches.
- Treat `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` as hard per-training-run bounds. Do not override them or continue a run past them.
"""
    user_extra = load_extra_instructions(args.extra_instructions)
    return (
        runtime + "\n" + isolation
        if not user_extra
        else runtime
        + "\n"
        + isolation
        + "\n# Additional operator instructions\n\n"
        + user_extra
    )


def encoded_extra_instructions(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
) -> str:
    return base64.b64encode(
        build_extra_instructions(
            args,
            tag,
            student_list,
            backend=backend,
        ).encode()
    ).decode()


def render_student(
    template: str,
    student_name: str,
    tag: str,
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    student_configmap_name = f"senpai-config-student-{tag}-{student_name}"
    student_deployment_name = f"senpai-{tag}-{student_name}"
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_REVISION": args.repo_revision,
            "TARGET_REPO_URL": args.target_repo_url,
            "TARGET_REPO_BRANCH": args.target_repo_branch,
            "GH_REPO": target_repo_slug(args.target_repo_url),
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_MODE": "online",
            "ADVISOR_BRANCH": args.advisor_branch,
            "GH_HISTORY_SCOPE": args.gh_history_scope,
            "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
            "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
            "EXTRA_INSTRUCTIONS_B64": encoded_extra_instructions(
                args,
                tag,
                [student_name],
                backend="kubernetes",
            ),
            "PROBLEM_DIR": args.problem_dir,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "SENPAI_START_GATE_PATH": args.start_gate_path,
        },
    )
    deployment = render_template(
        template,
        {
            "STUDENT_DEPLOYMENT_NAME": student_deployment_name,
            "STUDENT_CONFIGMAP_NAME": student_configmap_name,
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "STUDENT_IMAGE": args.student_image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "STUDENT_CPU": str(student_cpu),
            "STUDENT_MEMORY": f"{student_memory_gi}Gi",
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
        },
    )
    return configmap + "\n---\n" + deployment


def render_advisor(
    template: str,
    tag: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    advisor_configmap_name = f"senpai-config-advisor-{tag}"
    advisor_deployment_name = f"senpai-advisor-{tag}"
    data = {
        "REPO_URL": args.repo_url,
        "REPO_REVISION": args.repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "STUDENT_NAMES": ",".join(student_list),
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    data["EXTRA_INSTRUCTIONS_B64"] = encoded_extra_instructions(
        args,
        tag,
        student_list,
        backend="kubernetes",
    )
    configmap = render_configmap(
        name=advisor_configmap_name,
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=data,
    )
    deployment = render_template(
        template,
        {
            "ADVISOR_DEPLOYMENT_NAME": advisor_deployment_name,
            "ADVISOR_CONFIGMAP_NAME": advisor_configmap_name,
            "RESEARCH_TAG": tag,
            "ADVISOR_IMAGE": args.advisor_image,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
        },
    )
    return configmap + "\n---\n" + deployment


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if min(args.gpus_per_student, args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit(
            "ERROR: --gpus_per_student, --cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1"
        )
    validate_timing_args(args)
    if not args.preflight_only:
        for role, image in (
            ("advisor", args.advisor_image),
            ("student", args.student_image),
        ):
            if not is_immutable_image_reference(image):
                sys.exit(
                    f"ERROR: --{role}_image must be an immutable digest or "
                    "a :sha-<40-character-commit> tag"
                )
        try:
            advisor_revision = source_revision_for_image(
                args.advisor_image, args.repo_revision
            )
            student_revision = source_revision_for_image(
                args.student_image, args.repo_revision
            )
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
        if advisor_revision != student_revision:
            sys.exit(
                "ERROR: --advisor_image and --student_image must use the "
                "same source revision"
            )
        args.repo_revision = advisor_revision
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(args.repo_url):
        sys.exit("ERROR: --target_repo_url must be a different repo from --repo_url")

    # Resolve student list before backend-independent GitHub preflight.
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        student_list = expand_student_names(args.n_students)
    if args.student_prefix:
        student_list = [f"{args.student_prefix}-{name}" for name in student_list]

    github_token = anthropic_api_key = exa_api_key = wandb_api_key = ""
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        anthropic_api_key = resolve_anthropic_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        wandb_api_key = resolve_wandb_api_key(DOTENV_PATH)
        preflight_check_target_repo_access(args.target_repo_url, github_token)
        args.target_repo_branch = preflight_check_target_repo_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
        )
        preflight_check_student_name_availability(
            args.target_repo_url,
            github_token,
            student_list,
            args.advisor_branch,
        )
        preflight_check_anthropic_api_key(anthropic_api_key)
        preflight_check_exa_api_key(exa_api_key)
        preflight_check_wandb_api_key(wandb_api_key)
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    if not args.dry_run:
        ensure_advisor_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        ensure_target_repo_labels(
            args.target_repo_url,
            github_token,
            routing_labels(args.advisor_branch, student_list),
        )

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"
    launch_secret = render_launch_secret(
        args.tag,
        github_token if not args.dry_run else "<REDACTED_GITHUB_TOKEN>",
        (anthropic_api_key if not args.dry_run else "<REDACTED_ANTHROPIC_API_KEY>"),
        exa_api_key if not args.dry_run else "<REDACTED_EXA_API_KEY>",
        wandb_api_key if not args.dry_run else "<REDACTED_WANDB_API_KEY>",
    )

    # --- Apply per-launch secret first (pods reference it on startup) ---
    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(launch_secret)
        print()
    else:
        kubectl_apply(
            launch_secret,
            f"secret {secret_name}",
            kube_context=args.kube_context,
            namespace=args.namespace,
        )

    # --- Deploy students ---
    for name in student_list:
        manifest = render_student(
            student_template,
            name,
            args.tag,
            secret_name,
            launch_secret,
            args,
        )
        if args.dry_run:
            print(f"--- Student: {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(
                manifest,
                f"student {name}",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )

    advisor_student_list = student_list
    if args.advisor and not args.dry_run:
        advisor_student_list = list(
            dict.fromkeys(
                existing_student_names(
                    args.tag,
                    kube_context=args.kube_context,
                    namespace=args.namespace,
                )
                + student_list
            )
        )

    # --- Deploy advisor ---
    if args.advisor:
        manifest = render_advisor(
            advisor_template,
            args.tag,
            advisor_student_list,
            secret_name,
            launch_secret,
            args,
        )
        if args.dry_run:
            print("--- Advisor ---")
            print(manifest)
            print()
        else:
            kubectl_apply(
                manifest,
                "advisor",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )

    if not args.dry_run:
        print(f"\nLaunched {len(student_list)} students: {', '.join(student_list)}")
        if args.advisor:
            print("Launched advisor pod")
        kubectl = shlex.join(
            kubectl_command(
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        )
        print("\nMonitor:")
        print(f"  {kubectl} get deployments -l research-tag={args.tag}")
        if args.advisor:
            print(f"  {kubectl} get deployment senpai-advisor-{args.tag}")
        if student_list:
            print(
                f"  {kubectl} logs -f "
                f"deployment/senpai-{args.tag}-{student_list[0]}"
            )
        print("\nStop:")
        print(
            f"  {kubectl} delete deployments,configmaps,secrets "
            f"-l research-tag={args.tag}"
        )


if __name__ == "__main__":
    main()
