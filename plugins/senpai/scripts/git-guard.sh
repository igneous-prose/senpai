#!/bin/bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Bootstrap-only Git guards. Model-facing GitHub writes use typed tools.

install_senpai_git_guard() {
    local workdir="$1" target_workdir="$2" askpass_file="$3"
    if [ -z "$workdir" ] || [ -z "$target_workdir" ] || [ -z "$askpass_file" ]; then
        echo "install_senpai_git_guard: usage: <workdir> <target-workdir> <askpass-file>" >&2
        return 2
    fi

    git remote set-url --push origin DISABLED
    git config remote.origin.pushurl DISABLED
    git config --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf 2>/dev/null || true

    export TARGET_WORKDIR="$target_workdir"
    export SENPAI_REAL_GIT="${SENPAI_REAL_GIT:-$(command -v git)}"

    mkdir -p .git/hooks "$workdir/git-guard-bin"
    cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
echo "ERROR: refusing to push from the senpai runner repo; use the cloned target repo instead." >&2
exit 1
EOF
    chmod +x .git/hooks/pre-push

    cat > "$workdir/git-guard-bin/git" <<'EOF'
#!/bin/sh
real_git="${SENPAI_REAL_GIT:-/usr/bin/git}"
if [ "$1" = "push" ]; then
    top="$("$real_git" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${TARGET_WORKDIR:-}" ] && [ "$top" != "${TARGET_WORKDIR%/}" ]; then
        echo "ERROR: refusing git push outside target repo; cwd=$(pwd), top=${top:-none}, target=$TARGET_WORKDIR" >&2
        exit 2
    fi
fi
exec "$real_git" "$@"
EOF
    chmod +x "$workdir/git-guard-bin/git"
    export PATH="$workdir/git-guard-bin:$PATH"

    cat > "$askpass_file" <<'EOF'
#!/bin/sh
case "$1" in
    *Username*) printf '%s\n' x-access-token ;;
    *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
EOF
    chmod 700 "$askpass_file"
    export GIT_ASKPASS="$askpass_file"
    export GIT_TERMINAL_PROMPT=0
    git config --global --unset-all credential.helper 2>/dev/null || true
}

install_senpai_target_git_guard() {
    local target_workdir="$1"
    if [ -z "$target_workdir" ] || [ ! -d "$target_workdir/.git" ]; then
        echo "install_senpai_target_git_guard: target repo not found: ${target_workdir:-<missing>}" >&2
        return 2
    fi

    mkdir -p "$target_workdir/.git/hooks"
    cat > "$target_workdir/.git/hooks/pre-push" <<'EOF'
#!/bin/sh
while read -r local_ref _ remote_ref _; do
    [ "$remote_ref" = "refs/heads/$ADVISOR_BRANCH" ] || continue
    [ "$SENPAI_ROLE" != "student" ] || {
        echo "SENPAI-GIT-GUARD: students must not push $ADVISOR_BRANCH" >&2
        exit 2
    }
    [ "$SENPAI_ROLE" != "advisor" ] || [ "$local_ref" = "refs/heads/$ADVISOR_BRANCH" ] || {
        echo "SENPAI-GIT-GUARD: advisor must push $ADVISOR_BRANCH from local $ADVISOR_BRANCH, not ${local_ref:-<unknown>}" >&2
        exit 2
    }
done
EOF
    chmod +x "$target_workdir/.git/hooks/pre-push"
}
