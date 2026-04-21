#!/bin/bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# PreToolUse hook: check for new PRs, issues, or idle students and inject
# them as additionalContext into the running CC session.
#
# Throttled by NOTIFY_INTERVAL_S (default 120s) — exits silently when the
# cache is fresh, so the hook adds near-zero overhead on most tool calls.
# Only runs inside pods (IS_SANDBOX=1).

# --- Guard: only run in pods ---
[ "${IS_SANDBOX:-}" = "1" ] || exit 0

WORKDIR="${WORKDIR:-/workspace/senpai}"
CACHE_DIR="${WORKDIR}/advisor_logs"
[ -n "${STUDENT_NAME:-}" ] && CACHE_DIR="${WORKDIR}/student_logs"
CACHE_FILE="${CACHE_DIR}/.notify_cache_ts"
INTERVAL="${NOTIFY_INTERVAL_S:-120}"

# --- Throttle: skip if checked recently ---
if [ -f "$CACHE_FILE" ]; then
    last=$(cat "$CACHE_FILE")
    now=$(date +%s)
    elapsed=$(( now - last ))
    [ "$elapsed" -lt "$INTERVAL" ] && exit 0
fi

# --- Source helpers ---
source "${WORKDIR}/plugins/senpai/scripts/senpai-gh.sh" 2>/dev/null || exit 0

# --- Run role-appropriate checks ---
lines=()

if [ -n "${STUDENT_NAME:-}" ]; then
    # Student: check for assigned PRs and issues
    assigned=$(student_poll_for_work "$STUDENT_NAME" 2>/dev/null)
    assigned_n=$(printf '%s' "$assigned" | json_len 2>/dev/null || echo 0)
    issues=$(check_gh_issues "student:$STUDENT_NAME" 2>/dev/null)
    issues_n=$(printf '%s' "$issues" | json_len 2>/dev/null || echo 0)

    [ "$assigned_n" -gt 0 ] && lines+=("Assigned PRs ($assigned_n): $(printf '%s' "$assigned" | json_numbers)")
    [ "$issues_n" -gt 0 ]   && lines+=("GitHub issues ($issues_n): $(printf '%s' "$issues" | json_numbers)")
else
    # Advisor: check for review PRs, issues, and idle students
    since=""
    last_check="${WORKDIR}/advisor_logs/.last_check_ts"
    [ -f "$last_check" ] && since=$(cat "$last_check")

    reviews=$(list_ready_for_review_prs "${ADVISOR_BRANCH:-}" "$since" 2>/dev/null)
    reviews_n=$(printf '%s' "$reviews" | json_len 2>/dev/null || echo 0)
    issues=$(check_gh_issues "${ADVISOR_BRANCH:-}" "$since" 2>/dev/null)
    issues_n=$(printf '%s' "$issues" | json_len 2>/dev/null || echo 0)
    idle=$(list_idle_students "${STUDENT_NAMES:-}" "${ADVISOR_BRANCH:-}" 2>/dev/null)
    idle_n=$(printf '%s' "$idle" | json_len 2>/dev/null || echo 0)

    [ "$reviews_n" -gt 0 ] && lines+=("PRs ready for review ($reviews_n): $(printf '%s' "$reviews" | json_numbers)")
    [ "$issues_n" -gt 0 ]  && lines+=("GitHub issues ($issues_n): $(printf '%s' "$issues" | json_numbers)")
    [ "$idle_n" -gt 0 ]    && lines+=("Idle students ($idle_n): $(printf '%s' "$idle" | json_join)")
fi

# --- Update throttle timestamp ---
date +%s > "$CACHE_FILE"

# --- Exit silently if nothing actionable ---
[ ${#lines[@]} -eq 0 ] && exit 0

# --- Build additionalContext ---
msg="[NOTIFICATION] New items detected since last check:"
for line in "${lines[@]}"; do
    msg+=$'\n'"- ${line}"
done
msg+=$'\n'"Act on these when you reach a natural stopping point."

# --- Output hook JSON ---
python3 -c "
import json, sys
print(json.dumps({'hookSpecificOutput': {'additionalContext': sys.argv[1]}}))
" "$msg"
