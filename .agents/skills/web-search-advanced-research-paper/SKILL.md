---
name: web-search-advanced-research-paper
description: Search for research papers and academic publications with a bounded Exa query. Use for literature searches, arXiv discovery, or finding evidence for an experiment hypothesis.
context: fork
model: Codex-opus-4-8
effort: high
---

# Publication search

Use the checked-in stdlib client:

```bash
python "$HOME/.agents/skills/web-search-advanced-research-paper/scripts/search_publications.py" \
  "precise research question"
```

The script always searches Exa's publication category and returns metadata
only. It requires `EXA_API_KEY`, which the runtime supplies as a masked command
secret.

Useful bounds:

```bash
# Recent work
python "$HOME/.agents/skills/web-search-advanced-research-paper/scripts/search_publications.py" \
  --start-published-date 2025-01-01 --num-results 8 \
  "neural CFD surrogate conservation loss"

# Selected publication domains
python "$HOME/.agents/skills/web-search-advanced-research-paper/scripts/search_publications.py" \
  --include-domain arxiv.org --include-domain openreview.net \
  "operator learning irregular meshes"
```

Keep the search progressive:

1. **Candidates:** start with one precise metadata-only query and five to eight
   results.
2. **Refine:** if needed, make one smaller domain/date/text-filtered query and
   deduplicate candidates by ID or URL.
3. **Read:** open only the one to three papers that can change the current
   hypothesis, using the browser or AlphaXiv, then return a cited synthesis of
   the finding, assumptions, and concrete experiment implication.

Do not paste raw API responses or full papers into the conversation.
