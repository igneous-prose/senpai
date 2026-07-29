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

The tool returns Markdown—not raw JSON—with a search summary, one heading per
publication, bold metadata, and nested bullets for summaries, highlights, and
costs.

### Defaults

| Parameter | Default | Rationale |
| --- | --- | --- |
| `--num-results` | `30` | Broad enough for literature discovery without flooding context. |
| `--search-type` | `deep` | Comprehensive multi-step research and synthesis for discovering connections across publications. |
| `--highlights-max-characters` | `2000` | Enough evidence to compare mechanisms without returning full papers. |
| Published-date filter | none | Preserve both seminal and recent work. |
| Domain filter | none | Use Exa's full publication index rather than biasing toward one host. |

All defaults are adjustable:

```bash
python "$CLAUDE_SKILL_DIR/scripts/search_publications.py" \
  "equivariant mesh neural operators" \
  --num-results 50 \
  --start-published-date 2023-01-01 \
  --exclude-domains example.com \
  --highlights-max-characters 2500
```

Useful options:

- `--num-results N`: return 1–100 publications.
- `--search-type`: `auto`, `fast`, `instant`, `deep-lite`, `deep`, or
  `deep-reasoning`. Use the default `deep` search to find non-obvious,
  insightful connections across publications. Choose a faster mode only when
  latency matters, or `deep-reasoning` for especially complex synthesis.
- `--start-published-date` / `--end-published-date`: ISO dates for genuinely
  time-bounded questions. Do not add a recency filter to searches for
  foundational or transferable mechanisms.
- `--exclude-domains`: space-separated sources to exclude. Exa's publication
  category rejects include-domain filters, so this tool does not expose one.
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
