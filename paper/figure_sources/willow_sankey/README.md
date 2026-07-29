# Willow Sankey Figure

Use `paper/figures/willow_simplified_sankey.png` for LaTeX inclusion. The build script also writes PDF and SVG versions using the shared paper palette: Okabe-print family colors, green reserved for merged outcomes, muted greys for closed/open outcomes, a white background, and regular-weight Times-style serif labels.

Regenerate from the repository root:

```bash
python paper/figure_sources/willow_sankey/build_willow_sankey.py
sips -s format png -z 3280 5200 paper/figures/willow_simplified_sankey.svg --out paper/figures/willow_simplified_sankey.png
```

Background variants can be generated with environment variables:

```bash
WILLOW_SANKEY_BG="#F6FAFC" WILLOW_SANKEY_SUFFIX="_cool_mist" \
  python paper/figure_sources/willow_sankey/build_willow_sankey.py
```

Inputs:

- GitHub PR metadata from `morganmcg1/TandemFoilSet-Balanced`, base branches `icml-appendix-willow-pai2e-r1` through `icml-appendix-willow-pai2e-r5`.
- Local result categories from `experiment_log/tandemfoil_balanced_pr_results_2026-05-06_07-27-38.md`.

Generated supporting files:

- `willow_prs_classified.csv`
- `willow_sankey_summary.md`
