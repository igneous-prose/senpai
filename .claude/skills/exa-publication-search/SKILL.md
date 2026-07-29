---
name: exa-publication-search
description: Search scholarly publications through Exa's dedicated publication index. Use when finding research papers, preprints, journal articles, or literature for scientific research.
context: fork
model: claude-opus-4-8
effort: high
---

# Exa Publication Search

Use the bundled `search_publications.py` tool. It calls the official `exa_py`
library with `category="publication"`. It uses `python-dotenv` to load the
nearest `.env`, while preserving an `EXA_API_KEY` already set in the
environment.

## Search

```bash
python "$CLAUDE_SKILL_DIR/scripts/search_publications.py" \
  "mesh-based neural operators for aerodynamic surrogate modeling"
```

The tool returns JSON with ranked publications, query-relevant highlights,
search latency, and Exa's reported cost.

### Defaults

| Parameter | Default | Rationale |
| --- | --- | --- |
| `--num-results` | `30` | Broad enough for literature discovery without flooding context. |
| `--search-type` | `auto` | Exa's balanced quality/latency mode. |
| `--highlights-max-characters` | `1200` | Useful evidence for triage without returning full papers. |
| Published-date filter | none | Preserve both seminal and recent work. |
| Domain filter | none | Use Exa's full publication index rather than biasing toward one host. |

All defaults are adjustable:

```bash
python "$CLAUDE_SKILL_DIR/scripts/search_publications.py" \
  "equivariant mesh neural operators" \
  --num-results 50 \
  --start-published-date 2023-01-01 \
  --include-domains arxiv.org openreview.net \
  --highlights-max-characters 1800
```

Useful options:

- `--num-results N`: return 1–100 publications.
- `--search-type`: `auto`, `fast`, `instant`, `deep-lite`, `deep`, or
  `deep-reasoning`. Keep `auto` unless the task clearly needs a different
  quality/latency tradeoff.
- `--start-published-date` / `--end-published-date`: ISO dates for genuinely
  time-bounded questions. Do not add a recency filter to searches for
  foundational or transferable mechanisms.
- `--include-domains` / `--exclude-domains`: space-separated source filters.
  Use only when provenance matters; do not default to arXiv.
- `--include-text` / `--exclude-text`: one exact text constraint each.
- `--additional-queries`: up to 10 space-separated, quoted query variants for
  deep search modes.
- `--summary-query`: request an additional per-result summary when highlights
  are insufficient. Summaries add latency and cost.
- `--no-highlights`: metadata-only search.

## Search strategy

Write natural-language queries that describe the mechanism, setting, or result
you want—not just a bag of keywords. For broad literature work, run two or
three distinct query angles at 20–30 results each, then merge and deduplicate
by canonical URL and normalized title.

Use the first pass for discovery. Read promising papers from their original
source, traverse citations with Semantic Scholar, inspect discussion through
AlphaXiv, and find implementations on GitHub. Exa is the discovery index, not
a substitute for validating claims against the paper.

## Output

Return:

1. Results with title, URL, author, publication date, and relevant highlights.
2. A short synthesis of the mechanisms and disagreements that matter.
3. Notes about missing metadata, uncertain matches, and search coverage.

Do not paste the raw JSON into the final research report. Distill it into the
decision-useful researcher-agent format.
