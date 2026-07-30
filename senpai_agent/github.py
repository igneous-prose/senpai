"""Complete, context-bounded GitHub pull-request retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from pydantic import SecretStr

from senpai_agent.github_http import GitHubReader, GitHubReadError

_ARTIFACT_MAX_AGE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class PRManifestEntry:
    """Compact identity for one rendered pull request."""

    number: int
    title: str
    head_sha: str
    url: str


@dataclass(frozen=True)
class PRRetrievalResult:
    """Markdown returned inline or written to one external artifact."""

    manifest: tuple[PRManifestEntry, ...]
    markdown: str | None
    path: Path | None


@dataclass(frozen=True)
class _PR:
    details: dict[str, Any]
    issue_comments: tuple[dict[str, Any], ...]
    reviews: tuple[dict[str, Any], ...]
    inline_comments: tuple[dict[str, Any], ...]

    @property
    def number(self) -> int:
        return int(self.details["number"])

    @property
    def manifest_entry(self) -> PRManifestEntry:
        return PRManifestEntry(
            number=self.number,
            title=str(self.details.get("title") or ""),
            head_sha=str((self.details.get("head") or {}).get("sha") or ""),
            url=str(self.details.get("html_url") or ""),
        )


def get_prs(
    repo: str,
    *,
    numbers: Sequence[int] = (),
    date_range: tuple[str | date, str | date] | None = None,
    search: str | None = None,
    max_inline_prs: int = 5,
    artifact_dir: str | Path | None = None,
    target_workspace: str | Path | None = None,
    token: SecretStr | None = None,
) -> PRRetrievalResult:
    """Retrieve selected pull requests as complete Markdown.

    Selectors are unified: explicit ``numbers`` are combined with results from
    ``search`` and/or the inclusive PR creation ``date_range``. Every selected
    PR includes its full body, all issue comments, all review submissions, and
    all inline review comments across every GitHub API page.

    Up to ``max_inline_prs`` PRs are returned in ``markdown``. Larger selections
    are written to one deterministic ``.md`` artifact outside
    ``target_workspace`` and returned in ``path`` with a compact ``manifest``.

    ``max_inline_prs`` defaults to 5. Raising it above 5 risks polluting agent context;
    callers that deliberately raise it receive a warning.

    Args:
        repo: GitHub repository in ``owner/name`` form.
        numbers: Explicit PR numbers to include.
        date_range: Inclusive ``(start, end)`` creation-date selector.
        search: Additional GitHub issue-search qualifiers or terms.
        max_inline_prs: Inline selection limit. Raising this above 5 risks polluting
            agent context.
        artifact_dir: External directory for oversized Markdown results. Defaults
            to ``$SENPAI_OPENHANDS_STATE_DIR/github`` or a temporary directory.
        target_workspace: Target checkout that must not contain artifacts.
        token: Optional typed GitHub credential passed only to the shared HTTP
            reader. Ambient GitHub token variables are ignored.

    Returns:
        A manifest plus either inline Markdown or one Markdown artifact path.
    """
    _validate_repo(repo)
    explicit_numbers = _normalize_numbers(numbers)
    normalized_range = _normalize_date_range(date_range)
    normalized_search = search.strip() if search and search.strip() else None
    if not explicit_numbers and normalized_range is None and normalized_search is None:
        raise ValueError("get_prs requires at least one selector")
    if isinstance(max_inline_prs, bool) or max_inline_prs < 0:
        raise ValueError("max_inline_prs must be a non-negative integer")
    if max_inline_prs > 5:
        warnings.warn(
            "Raising max_inline_prs above 5 risks polluting agent context.",
            UserWarning,
            stacklevel=2,
        )

    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    reader = GitHubReader(token)
    selected_numbers = set(explicit_numbers)
    if normalized_range is not None or normalized_search is not None:
        selected_numbers.update(
            _search_pr_numbers(
                reader,
                repo,
                normalized_range,
                normalized_search,
            )
        )

    pull_requests = tuple(
        _fetch_pr(reader, repo, number) for number in sorted(selected_numbers)
    )
    markdown = _render_markdown(repo, pull_requests)
    manifest = tuple(pr.manifest_entry for pr in pull_requests)

    if len(pull_requests) <= max_inline_prs:
        return PRRetrievalResult(manifest=manifest, markdown=markdown, path=None)

    output_dir = _external_artifact_dir(artifact_dir, target_workspace)
    path = output_dir / _artifact_name(
        repo=repo,
        numbers=explicit_numbers,
        date_range=normalized_range,
        search=normalized_search,
        manifest=manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_expired_artifacts(output_dir)
    if not path.exists() or path.read_text(encoding="utf-8") != markdown:
        path.write_text(markdown, encoding="utf-8")
    path.chmod(0o600)
    return PRRetrievalResult(manifest=manifest, markdown=None, path=path)


def _validate_repo(repo: str) -> None:
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo must use owner/name form")


def _normalize_numbers(numbers: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError("PR numbers must be positive integers")
        normalized.add(number)
    return tuple(sorted(normalized))


def _normalize_date_range(
    value: tuple[str | date, str | date] | None,
) -> tuple[str, str] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("date_range must contain exactly a start and end date")
    start, end = (_iso_date(item) for item in value)
    if start > end:
        raise ValueError("date_range start must not be after its end")
    return start, end


def _iso_date(value: str | date) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid ISO date: {value!r}") from error


def _search_pr_numbers(
    reader: GitHubReader,
    repo: str,
    date_range: tuple[str, str] | None,
    search: str | None,
) -> tuple[int, ...]:
    query = [f"repo:{repo}", "is:pr"]
    if search is not None:
        query.append(search)
    if date_range is not None:
        query.append(f"created:{date_range[0]}..{date_range[1]}")
    endpoint = "/search/issues?" + urlencode({"q": " ".join(query), "per_page": 100})
    numbers: set[int] = set()
    for page in reader.pages(endpoint):
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise GitHubReadError("GitHub returned invalid issue search results")
        numbers.update(
            int(item["number"]) for item in page["items"] if isinstance(item, dict)
        )
    return tuple(sorted(numbers))


def _fetch_pr(
    reader: GitHubReader,
    repo: str,
    number: int,
) -> _PR:
    root = f"/repos/{repo}"
    details = reader.get(f"{root}/pulls/{number}")
    if not isinstance(details, dict):
        raise TypeError(f"GitHub returned an invalid PR #{number} response")
    return _PR(
        details=details,
        issue_comments=_paginated_items(
            reader,
            f"{root}/issues/{number}/comments?per_page=100",
        ),
        reviews=_paginated_items(
            reader,
            f"{root}/pulls/{number}/reviews?per_page=100",
        ),
        inline_comments=_paginated_items(
            reader,
            f"{root}/pulls/{number}/comments?per_page=100",
        ),
    )


def _paginated_items(
    reader: GitHubReader,
    endpoint: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(reader.objects(endpoint))


def _render_markdown(repo: str, pull_requests: tuple[_PR, ...]) -> str:
    parts = [
        "# GitHub pull requests",
        "",
        f"Repository: `{repo}`",
        "",
        f"Selected pull requests: {len(pull_requests)}",
    ]
    if not pull_requests:
        parts.extend(("", "_No pull requests selected._"))
    for pr in pull_requests:
        parts.extend(("", "---", "", _render_pr(pr)))
    return "\n".join(parts) + "\n"


def _render_pr(pr: _PR) -> str:
    details = pr.details
    base = details.get("base") or {}
    head = details.get("head") or {}
    author = details.get("user") or {}
    sections = [
        f"## PR #{pr.number} — {details.get('title') or ''}",
        "",
        f"- URL: {details.get('html_url') or ''}",
        f"- Author: @{author.get('login') or 'unknown'}",
        f"- State: {details.get('state') or 'unknown'}",
        f"- Draft: {'yes' if details.get('draft') else 'no'}",
        f"- Created: {details.get('created_at') or 'unknown'}",
        f"- Updated: {details.get('updated_at') or 'unknown'}",
        f"- Base: `{base.get('ref') or ''}` (`{base.get('sha') or ''}`)",
        f"- Head: `{head.get('ref') or ''}` (`{head.get('sha') or ''}`)",
        "",
        "### PR body",
        "",
        _body(details.get("body")),
        "",
        f"### Issue comments ({len(pr.issue_comments)})",
        "",
        _render_entries(
            sorted(pr.issue_comments, key=_created_key),
            _render_issue_comment,
            "issue comments",
        ),
        "",
        f"### Review submissions ({len(pr.reviews)})",
        "",
        _render_entries(
            sorted(pr.reviews, key=_review_key),
            _render_review,
            "review submissions",
        ),
        "",
        f"### Inline review comments ({len(pr.inline_comments)})",
        "",
        _render_entries(
            sorted(pr.inline_comments, key=_created_key),
            _render_inline_comment,
            "inline review comments",
        ),
    ]
    return "\n".join(sections)


def _render_entries(entries: list[dict], render, label: str) -> str:
    if not entries:
        return f"_No {label}._"
    return "\n\n".join(render(entry, index) for index, entry in enumerate(entries, 1))


def _render_issue_comment(comment: dict[str, Any], index: int) -> str:
    user = comment.get("user") or {}
    return "\n".join(
        (
            f"#### Issue comment {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- Created: {comment.get('created_at') or 'unknown'}",
            f"- Updated: {comment.get('updated_at') or 'unknown'}",
            f"- URL: {comment.get('html_url') or ''}",
            "",
            _body(comment.get("body")),
        )
    )


def _render_review(review: dict[str, Any], index: int) -> str:
    user = review.get("user") or {}
    return "\n".join(
        (
            f"#### Review {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- State: {review.get('state') or 'unknown'}",
            f"- Submitted: {review.get('submitted_at') or 'unknown'}",
            f"- Commit: `{review.get('commit_id') or ''}`",
            f"- URL: {review.get('html_url') or ''}",
            "",
            _body(review.get("body")),
        )
    )


def _render_inline_comment(comment: dict[str, Any], index: int) -> str:
    user = comment.get("user") or {}
    line = comment.get("line")
    if line is None:
        line = comment.get("original_line")
    location = str(comment.get("path") or "")
    if line is not None:
        location += f":{line}"
    reply_to = comment.get("in_reply_to_id")
    return "\n".join(
        (
            f"#### Inline comment {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- Location: `{location}`",
            f"- Side: {comment.get('side') or 'unknown'}",
            f"- Created: {comment.get('created_at') or 'unknown'}",
            f"- Updated: {comment.get('updated_at') or 'unknown'}",
            f"- Commit: `{comment.get('commit_id') or ''}`",
            f"- Reply to: {reply_to if reply_to is not None else 'none'}",
            f"- URL: {comment.get('html_url') or ''}",
            "",
            _body(comment.get("body")),
        )
    )


def _body(value: Any) -> str:
    return str(value) if value else "_No body._"


def _created_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item.get("created_at") or ""), int(item.get("id") or 0)


def _review_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item.get("submitted_at") or ""), int(item.get("id") or 0)


def _external_artifact_dir(
    artifact_dir: str | Path | None,
    target_workspace: str | Path | None,
) -> Path:
    target = (
        Path(
            target_workspace
            or os.environ.get("SENPAI_OPENHANDS_WORKSPACE")
            or Path.cwd()
        )
        .expanduser()
        .resolve()
    )
    if artifact_dir is None:
        state_dir = os.environ.get("SENPAI_OPENHANDS_STATE_DIR")
        artifact_dir = (
            Path(state_dir) / "github"
            if state_dir
            else Path(tempfile.gettempdir()) / "senpai-github"
        )
    output = Path(artifact_dir).expanduser().resolve()
    if output == target or output.is_relative_to(target):
        raise ValueError("GitHub artifacts must be outside the target workspace")
    return output


def _remove_expired_artifacts(output_dir: Path) -> None:
    cutoff = time.time() - _ARTIFACT_MAX_AGE_SECONDS
    for path in output_dir.glob("pull-requests-*.md"):
        if path.stat().st_mtime < cutoff:
            path.unlink()


def _artifact_name(
    *,
    repo: str,
    numbers: tuple[int, ...],
    date_range: tuple[str, str] | None,
    search: str | None,
    manifest: tuple[PRManifestEntry, ...],
) -> str:
    identity = {
        "repo": repo,
        "numbers": numbers,
        "date_range": date_range,
        "search": search,
        "heads": [(entry.number, entry.head_sha) for entry in manifest],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"pull-requests-{digest}.md"
