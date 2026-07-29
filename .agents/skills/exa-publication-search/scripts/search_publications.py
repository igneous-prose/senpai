#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Search Exa's publication index and emit compact, agent-ready Markdown."""

# ruff: noqa: RUF009 - simple_parsing stores CLI metadata in dataclass fields.

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from exa_py import Exa
from simple_parsing import ArgumentParser, DashVariant
from simple_parsing.helpers import field

DEFAULT_NUM_RESULTS = 30
DEFAULT_HIGHLIGHTS_MAX_CHARACTERS = 2000
SearchType = Literal[
    "auto",
    "fast",
    "instant",
    "deep-lite",
    "deep",
    "deep-reasoning",
]
DEEP_SEARCH_TYPES: set[SearchType] = {"deep-lite", "deep", "deep-reasoning"}
MARKDOWN_ESCAPES = str.maketrans(
    {character: f"\\{character}" for character in "\\`*_[]<>#"}
)


@dataclass
class SearchArguments:
    """Search Exa's dedicated scholarly publication index."""

    query: str = field(
        positional=True,
        metavar="QUERY",
        help="natural-language publication search query",
    )
    num_results: int = field(
        default=DEFAULT_NUM_RESULTS,
        alias="-n",
        metavar="N",
        help="number of publications to return (1-100)",
    )
    search_type: SearchType = field(
        default="deep",
        help="Exa search quality/latency mode",
    )
    start_published_date: str | None = field(
        default=None,
        nargs=None,
        metavar="DATE",
        help="ISO lower publication date",
    )
    end_published_date: str | None = field(
        default=None,
        nargs=None,
        metavar="DATE",
        help="ISO upper publication date",
    )
    exclude_domains: list[str] = field(
        default_factory=list,
        metavar="DOMAIN [DOMAIN ...]",
        help="exclude these domains",
    )
    include_text: str | None = field(
        default=None,
        nargs=None,
        metavar="TEXT",
        help="required exact text constraint",
    )
    exclude_text: str | None = field(
        default=None,
        nargs=None,
        metavar="TEXT",
        help="excluded exact text constraint",
    )
    additional_queries: list[str] = field(
        default_factory=list,
        metavar="QUERY [QUERY ...]",
        help="additional queries for deep search",
    )
    highlights_max_characters: int = field(
        default=DEFAULT_HIGHLIGHTS_MAX_CHARACTERS,
        metavar="N",
        help="maximum highlight characters per publication (1-10000)",
    )
    summary_query: str | None = field(
        default=None,
        nargs=None,
        metavar="QUESTION",
        help="request a per-result summary focused on this question",
    )
    no_highlights: bool = field(
        default=False,
        action="store_true",
        help="return publication metadata without query highlights",
    )

    def validate(self) -> None:
        if not 1 <= self.num_results <= 100:
            raise ValueError("--num-results must be between 1 and 100")
        if not 1 <= self.highlights_max_characters <= 10_000:
            raise ValueError("--highlights-max-characters must be between 1 and 10000")
        if self.additional_queries and self.search_type not in DEEP_SEARCH_TYPES:
            raise ValueError("--additional-queries requires a deep search type")
        if len(self.additional_queries) > 10:
            raise ValueError("--additional-queries accepts at most 10 queries")


def parse_args(argv: Sequence[str] | None = None) -> SearchArguments:
    parser = ArgumentParser(
        add_option_string_dash_variants=DashVariant.DASH,
        description="Search Exa's dedicated scholarly publication index.",
    )
    parser.add_arguments(SearchArguments, dest="options")
    args: SearchArguments = parser.parse_args(argv).options
    try:
        args.validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def build_contents(args: SearchArguments) -> dict[str, Any] | bool:
    contents: dict[str, Any] = {}
    if not args.no_highlights:
        contents["highlights"] = {
            "max_characters": args.highlights_max_characters,
        }
    if args.summary_query:
        contents["summary"] = {"query": args.summary_query}
    return contents or False


def without_empty(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            key: without_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [without_empty(item) for item in value]
    return value


def markdown_text(value: Any) -> str:
    return " ".join(str(value).split()).translate(MARKDOWN_ESCAPES)


def markdown_url(value: Any) -> str:
    return (
        str(value).strip().replace("<", "%3C").replace(">", "%3E").replace(" ", "%20")
    )


def render_mapping(mapping: dict[str, Any], indent: int = 2) -> list[str]:
    lines = []
    for key, value in mapping.items():
        label = key.replace("_", " ").title()
        prefix = f"{' ' * indent}- **{label}:**"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.extend(render_mapping(value, indent + 2))
        else:
            lines.append(f"{prefix} {markdown_text(value)}")
    return lines


def render_summary(value: Any) -> list[str]:
    parts = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not parts:
        return []
    if len(parts) == 1 and " - " in parts[0]:
        parts = parts[0].split(" - ")
    parts = [part.removeprefix("- ").strip() for part in parts]
    if parts[0].rstrip(":").casefold() == "summary":
        parts = parts[1:]
    if not parts:
        return []
    lines = [f"- **Summary:** {markdown_text(parts[0])}"]
    lines.extend(f"  - {markdown_text(part)}" for part in parts[1:])
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exa Publication Search",
        "",
        f"- **Query:** {markdown_text(payload['query'])}",
        f"- **Category:** {markdown_text(payload['category'])}",
        f"- **Search type:** {markdown_text(payload['search_type'])}",
        (
            f"- **Results:** {payload['result_count']} returned / "
            f"{payload['requested_results']} requested"
        ),
    ]
    if "search_time_ms" in payload:
        lines.append(f"- **Search time:** {payload['search_time_ms']} ms")
    if cost := payload.get("cost_dollars"):
        lines.append("- **Cost (USD):**")
        lines.extend(render_mapping(cost))

    results = payload.get("results", [])
    for result in results:
        title = markdown_text(result.get("title") or "Untitled publication")
        lines.extend(["", f"## {result['rank']}. {title}", ""])
        if url := result.get("url"):
            lines.append(f"- **URL:** <{markdown_url(url)}>")
        for key, label in (
            ("author", "Authors"),
            ("published_date", "Published"),
            ("id", "Exa ID"),
            ("score", "Score"),
        ):
            if key in result:
                lines.append(f"- **{label}:** {markdown_text(result[key])}")
        if summary := result.get("summary"):
            lines.extend(render_summary(summary))
        if highlights := result.get("highlights"):
            lines.append("- **Highlights:**")
            lines.extend(f"  - {markdown_text(highlight)}" for highlight in highlights)

    if not results:
        lines.extend(["", "No publications were returned."])
    return "\n".join(lines)


def create_exa_client() -> Exa:
    load_dotenv()
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set; add it to .env or the environment")
    return Exa(api_key)


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
    args: SearchArguments,
    client: Exa | None = None,
) -> dict[str, Any]:
    exa = client if client is not None else create_exa_client()
    options: dict[str, Any] = {
        "category": "publication",
        "num_results": args.num_results,
        "type": args.search_type,
        "contents": build_contents(args),
    }
    optional = {
        "start_published_date": args.start_published_date,
        "end_published_date": args.end_published_date,
        "exclude_domains": args.exclude_domains,
        "include_text": [args.include_text] if args.include_text else None,
        "exclude_text": [args.exclude_text] if args.exclude_text else None,
        "additional_queries": args.additional_queries,
    }
    options.update({key: value for key, value in optional.items() if value})

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
    print(render_markdown(payload))


if __name__ == "__main__":
    main()
