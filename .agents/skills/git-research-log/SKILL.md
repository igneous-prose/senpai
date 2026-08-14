---
name: git-research-log
description: >
  How to document and publish ML experiment results to GitHub as a pull request.
  Use this skill whenever a student agent has finished running experiments and
  needs to create or update a PR — even if the instruction is just "wrap up" or
  "log your results" or "open a PR". Also use it mid-session to update an
  in-progress PR as trials complete.
---

# Experiment PR Skill

The master agent creates an empty PR and hands it to you. Your job is to fill it out as you run trials and finalize it when you're done. The master reads these PRs to decide what to explore next — completeness and honest analysis matter more than polish.

## PR Title Format

```
<experiment-name> <best-wandb-run-id-if-there-is-a-positive-result>
```

- `experiment-name`: the idea you explored (e.g. `multi-scale-attn`, `surface-loss-reweight`)
- `best-wandb-run-id`: W&B run ID of your best result (or most recent if all crashed)

## PR Description Template

Fill this in as you go. Write the TL;DR last, once all trials are done.

```markdown
# <agent-name>-<experiment-name>-<best-wandb-run-id>

## TL;DR - Result summary
<2-4 sentences: what you tried, what happened overall, and the bottom line verdict>

<!-- Fill in the block below only if a trial beat the baseline. Otherwise delete it. -->
**Main Hypothesis:** <what you believed would work and why>
**Key Result:** val_loss=X.XX | surf_Ux=X.XX | surf_p=XX.X (vs baseline: val_loss=X.XX)
**wandb:** <url to the best run>

## Trials run

### <YYYY-MM-DD_HH-MM> <wandb_run_id> <trial-name>
**Hypothesis:** <what you expected this specific run to show>
**Background research:** <relevant context — papers, similar approaches, why you thought this might work>
**Result:** val_loss=X.XX | surf_Ux=X.XX | surf_p=XX.X | memory=XX.XGB | status=keep/discard/crash
**wandb:** <wandb run url>
**Conclusion:** <what this result tells you — did the hypothesis hold? what would you try next?>

### <YYYY-MM-DD_HH-MM> <wandb_run_id> <trial-name>
...
```

Trials are ordered chronologically by start time, oldest first.

## Workflow

1. **Start of session**: read the PR the master created to understand your assignment
2. **After first trial completes**: edit the PR body to add the first trial entry; set the title
3. **As each subsequent trial finishes**: add a comment with quick metrics, then update the description to add the full trial entry
4. **When all trials are done**: write the TL;DR, set labels, promote to ready for review

Starting the PR early lets the master see in-progress results and potentially redirect you before you've finished all planned trials.
