# Willow Sankey Figure

Use `willow_simplified_sankey.pdf` for LaTeX inclusion. It is the lavender-pearl variant (`#F8F6FB`) with deep-navy physics/features (`#355C7D`) and regular-weight Times-style serif labels. The build script writes this PDF directly as vector artwork; do not regenerate it with `sips`, which rasterizes the SVG.

Regenerate from the repository root:

```bash
python papers/icml2026/figures/willow_sankey/build_willow_sankey.py
sips -s format png -z 1640 2600 papers/icml2026/figures/willow_sankey/willow_simplified_sankey.svg --out papers/icml2026/figures/willow_sankey/willow_simplified_sankey.png
```

Background variants can be generated with environment variables:

```bash
WILLOW_SANKEY_BG="#F6FAFC" WILLOW_SANKEY_SUFFIX="_cool_mist" \
  python papers/icml2026/figures/willow_sankey/build_willow_sankey.py
```

Inputs:

- GitHub PR metadata from `morganmcg1/TandemFoilSet-Balanced`, base branches `icml-appendix-willow-pai2e-r1` through `icml-appendix-willow-pai2e-r5`.
- Local result categories from `experiment_log/tandemfoil_balanced_pr_results_2026-05-06_07-27-38.md`.

Generated supporting files:

- `willow_prs_classified.csv`
- `willow_sankey_summary.md`
