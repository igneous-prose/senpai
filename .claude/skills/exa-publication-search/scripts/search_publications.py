#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Search Exa's publication index and emit compact, agent-ready JSON."""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any, Sequence

from exa_py import Exa

DEFAULT_NUM_RESULTS = 30
DEFAULT_HIGHLIGHTS_MAX_CHARACTERS = 1200
SEARCH_TYPES = ("auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning")
DEEP_SEARCH_TYPES = {"deep-lite", "deep", "deep-reasoning"}


def bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        number = int(value)
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"expected an integer from {minimum} to {maximum}"
            )
        return number

    return parse


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Exa's dedicated scholarly publication index."
    )
    parser.add_argument("query", help="Natural-language publication search query")
    parser.add_argument(
        "-n",
        "--num-results",
        type=bounded_int(1, 100),
        default=DEFAULT_NUM_RESULTS,
        help=f"number of publications to return (default: {DEFAULT_NUM_RESULTS})",
    )
    parser.add_argument(
        "--search-type",
        choices=SEARCH_TYPES,
        default="auto",
        help="Exa search quality/latency mode (default: auto)",
    )
    parser.add_argument("--start-published-date", help="ISO lower publication date")
    parser.add_argument("--end-published-date", help="ISO upper publication date")
    parser.add_argument(
        "--include-domain",
        action="append",
        dest="include_domains",
        help="only return this domain; repeat to add domains",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        dest="exclude_domains",
        help="exclude this domain; repeat to add domains",
    )
    parser.add_argument("--include-text", help="required exact text constraint")
    parser.add_argument("--exclude-text", help="excluded exact text constraint")
    parser.add_argument(
        "--additional-query",
        action="append",
        dest="additional_queries",
        help="additional query for deep search; repeat to add queries",
    )
    parser.add_argument(
        "--highlights-max-characters",
        type=bounded_int(1, 10_000),
        default=DEFAULT_HIGHLIGHTS_MAX_CHARACTERS,
        help=(
            "maximum highlight characters per publication "
            f"(default: {DEFAULT_HIGHLIGHTS_MAX_CHARACTERS})"
        ),
    )
    parser.add_argument(
        "--summary-query",
        help="request a per-result summary focused on this question",
    )
    parser.add_argument(
        "--no-highlights",
        action="store_true",
        help="return publication metadata without query highlights",
    )

    args = parser.parse_args(argv)
    if args.additional_queries and args.search_type not in DEEP_SEARCH_TYPES:
        parser.error("--additional-query requires a deep search type")
    return args


def build_contents(args: argparse.Namespace) -> dict[str, Any] | bool:
    contents: dict[str, Any] = {}
    if not args.no_highlights:
        contents["highlights"] = {
            "max_characters": args.highlights_max_characters,
        }
    if args.summary_query:
        contents["summary"] = {"query": args.summary_query}
    return contents or False


def without_empty(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            key: without_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [without_empty(item) for item in value]
    return value


def serialize_result(rank: int, result: Any) -> dict[str, Any]:
    return without_empty(
        {
            "rank": rank,
            "title": result.title,
            "url": result.url,
            "id": result.id,
            "published_date": result.published_date,
            "author": result.author,
            "score": result.score,
            "highlights": result.highlights,
            "summary": result.summary,
        }
    )


def search_publications(
    args: argparse.Namespace,
    client: Exa | None = None,
) -> dict[str, Any]:
    exa = client or Exa()
    options: dict[str, Any] = {
        "category": "publication",
        "num_results": args.num_results,
        "type": args.search_type,
        "contents": build_contents(args),
    }
    optional = {
        "start_published_date": args.start_published_date,
        "end_published_date": args.end_published_date,
        "include_domains": args.include_domains,
        "exclude_domains": args.exclude_domains,
        "include_text": [args.include_text] if args.include_text else None,
        "exclude_text": [args.exclude_text] if args.exclude_text else None,
        "additional_queries": args.additional_queries,
    }
    options.update({key: value for key, value in optional.items() if value is not None})

    response = exa.search(args.query, **options)
    return without_empty(
        {
            "query": args.query,
            "category": "publication",
            "search_type": args.search_type,
            "requested_results": args.num_results,
            "result_count": len(response.results),
            "search_time_ms": response.search_time,
            "cost_dollars": response.cost_dollars,
            "results": [
                serialize_result(rank, result)
                for rank, result in enumerate(response.results, start=1)
            ],
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    payload = search_publications(parse_args(argv))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
