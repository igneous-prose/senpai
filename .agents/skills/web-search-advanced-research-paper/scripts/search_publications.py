# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run one bounded Exa publication search with the Python standard library."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EXA_SEARCH_URL = "https://api.exa.ai/search"
MAX_RESULTS = 25
SEARCH_TYPES = ("auto", "fast", "instant", "deep")


def _result_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= MAX_RESULTS:
        raise argparse.ArgumentTypeError(
            f"num-results must be between 1 and {MAX_RESULTS}"
        )
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Exa for academic publications and print compact JSON."
    )
    parser.add_argument("query", help="Precise publication-search query.")
    parser.add_argument(
        "--num-results",
        type=_result_count,
        default=8,
        help=f"Number of results (1-{MAX_RESULTS}; default: 8).",
    )
    parser.add_argument(
        "--search-type",
        choices=SEARCH_TYPES,
        default="auto",
        help="Exa search strategy (default: auto).",
    )
    parser.add_argument(
        "--start-published-date",
        metavar="ISO-8601",
        help="Only include publications on or after this timestamp.",
    )
    parser.add_argument(
        "--end-published-date",
        metavar="ISO-8601",
        help="Only include publications on or before this date.",
    )
    parser.add_argument(
        "--include-domain",
        action="append",
        default=[],
        help="Restrict results to this domain; repeat for multiple domains.",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Exclude this domain; repeat for multiple domains.",
    )
    parser.add_argument(
        "--include-text",
        help="Require this text; Exa accepts one publication text filter.",
    )
    parser.add_argument(
        "--exclude-text",
        help="Exclude this text; Exa accepts one publication text filter.",
    )
    return parser


def _compact_result(result: dict[str, object]) -> dict[str, object]:
    fields = {
        key: result[key]
        for key in ("id", "title", "url", "publishedDate", "author")
        if result.get(key) is not None
    }
    return fields


def _error_detail(error: HTTPError | URLError, api_key: str) -> str:
    if isinstance(error, HTTPError):
        body = error.read().decode(errors="replace").strip()
        detail = f"HTTP {error.code}" + (f": {body[:500]}" if body else "")
    else:
        detail = str(error.reason)
    return detail.replace(api_key, "<redacted>")


def search_publications(
    query: str,
    *,
    api_key: str,
    num_results: int,
    search_type: str = "auto",
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    include_domains: Sequence[str] = (),
    exclude_domains: Sequence[str] = (),
    include_text: str | None = None,
    exclude_text: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "query": query,
        "type": search_type,
        "category": "publication",
        "numResults": num_results,
    }
    if start_published_date:
        payload["startPublishedDate"] = start_published_date
    if end_published_date:
        payload["endPublishedDate"] = end_published_date
    if include_domains:
        payload["includeDomains"] = list(dict.fromkeys(include_domains))
    if exclude_domains:
        payload["excludeDomains"] = list(dict.fromkeys(exclude_domains))
    if include_text:
        payload["includeText"] = [include_text]
    if exclude_text:
        payload["excludeText"] = [exclude_text]

    request = Request(
        EXA_SEARCH_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "senpai-publication-search",
            "x-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = json.load(response)
    except (HTTPError, URLError) as error:
        raise RuntimeError(
            f"Exa publication search failed: {_error_detail(error, api_key)}"
        ) from error

    results = body["results"]
    if not isinstance(results, list):
        raise TypeError("Exa returned an invalid results payload")
    if any(not isinstance(result, dict) for result in results):
        raise RuntimeError("Exa returned an invalid publication result")
    return {
        "query": query,
        "results": [_compact_result(result) for result in results],
    }


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    args = _parser().parse_args(argv)
    api_key = env.get("EXA_API_KEY")
    if not api_key:
        raise SystemExit("EXA_API_KEY is required")
    result = search_publications(
        args.query,
        api_key=api_key,
        num_results=args.num_results,
        search_type=args.search_type,
        start_published_date=args.start_published_date,
        end_published_date=args.end_published_date,
        include_domains=args.include_domain,
        exclude_domains=args.exclude_domain,
        include_text=args.include_text,
        exclude_text=args.exclude_text,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
