# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# Shared OpenHands invocation for Senpai advisor/student entrypoints.
# Each loop iteration must set LOGFILE before calling.
#
# Usage: run_senpai_claude <max_turns> <user_prompt> [extra runtime argv, e.g. -c]

run_senpai_claude() {
    local max_turns=$1 user_prompt=$2
    shift 2

    local openhands_cmd=(python -m senpai_agent.openhands_runner "$@" --max-turns "$max_turns")
    if [ -n "${SENPAI_CLAUDE_TIMEOUT_SECONDS:-}" ] && command -v timeout >/dev/null 2>&1; then
        local kill_after="${SENPAI_CLAUDE_TIMEOUT_KILL_AFTER_SECONDS:-30}"
        openhands_cmd=(timeout -k "$kill_after" "$SENPAI_CLAUDE_TIMEOUT_SECONDS" "${openhands_cmd[@]}")
    fi

    printf '%s' "$user_prompt" | "${openhands_cmd[@]}" >> "$LOGFILE" 2>&1
}
