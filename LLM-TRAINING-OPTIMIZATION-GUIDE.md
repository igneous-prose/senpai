<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Large Language Model Training Optimization Guide

This guide distills training-optimization lessons from the Modded-NanoGPT
Senpai autoresearch run. The source material is the GitHub pull request corpus
for `morganmcg1/modded-nanogpt-senpai`: 2,443 pull requests, numbered from
[#1](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1) through
[#2473](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2473), plus
16,665 issue and pull request comments. Most pull requests were opened between
May 15 and June 13, 2026. 2,275 pull requests contained a `SENPAI-RESULT`
marker, and 3,387 structured result blocks were parseable from comments.

The case study focused on the Modded-NanoGPT track 3 optimization benchmark:
keep the dataset, model architecture, and batch size fixed, then reduce the
number of optimizer steps needed to reach FineWeb validation cross-entropy below
3.28. The primary metric was `speedrun/final_first_step_to_target`, where lower
is better and `-1` means the run did not reach the target. The benchmark rule
requires statistical evidence:

```text
(3.28 - mean_validation_loss) * sqrt(number_of_runs) >= 0.004
```

This is a training guide, not an inference guide. It is about optimizer design,
schedules, initialization, validation discipline, and experimental control.

## Terms Used In This Guide

- Large language model training optimization: reducing the amount of training
  needed to reach a target validation loss without changing the benchmark
  contract.
- Step count: the number of optimizer updates required to reach the validation
  target. In this benchmark, step count mattered more than wall-clock time.
- Validation loss: FineWeb cross-entropy on the benchmark validation set. Lower
  is better.
- First step to target: the earliest validation checkpoint whose validation loss
  reaches the target.
- Muon: an optimizer family that uses matrix-shaped updates and Newton-Schulz
  orthogonalization.
- Aux AdamW: AdamW applied to non-body parameters such as embeddings, output
  head parameters, scalar parameters, and other groups not handled by Muon.
- Cooldown: the late-training learning-rate decay region where many schedules
  and moment buffers changed behavior.
- Preconditioner: a transformation that rescales or rotates gradients before the
  optimizer update.
- Exponential moving average: a smoothed copy of parameters or optimizer state.
  It can be used during optimization or only for evaluation.
- Paired comparison: running related variants in a way that controls for seed or
  hardware variance enough to compare small effects.

## Executive Summary

Large language model training optimization is not only "find a better
optimizer." In this case study, the most useful work came from treating the
training run as a controlled system:

1. Keep the benchmark contract fixed.
2. Separate optimizer mechanisms from retuning work.
3. Measure validation loss with enough seeds to avoid chasing noise.
4. Track parameter groups separately.
5. Treat cooldown as a distinct optimization phase.
6. Promote only changes that survive cleanup and retesting.
7. Kill low-value sweeps early.

The Modded-NanoGPT Senpai run produced real evidence about optimizer schedules,
preconditioners, averaging, initialization, and parameter-group specialization.
It also produced a warning: an autoresearch system can become too patient with
tiny effects. Many pull requests spent substantial effort on edge-case
interactions, small scalar sweeps, and speculative compositions after the
expected value had become low. The best lesson is not to copy the full search
shape. The best lesson is to keep the fixed benchmark, clean result contract,
mechanism-level hypotheses, statistical gates, and explicit negative results,
while adding stronger stop rules.

## Modded-NanoGPT Case Study: What It Showed

The run is an imperfect example progression, not an optimal research path. It
shows how an agent fleet can explore a training optimizer landscape, and it also
shows how easily that fleet can drift into excessive local hill-climbing.

Example phases:

- Baseline reproduction and optimizer-family orientation:
  [pull request 1](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1)
  through
  [pull request 10](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/10)
  reproduced and isolated Muon, NorMuon, Muon-squared, MuonH, SOAP-Muon,
  Contra-Muon, and related public optimizer records.
- Early mechanism search:
  [pull request 20](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/20)
  tested cautious update masking,
  [pull request 50](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/50)
  tested Polyak-style tail averaging, and
  [pull request 250](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/250)
  explored Newton-Schulz polynomial coefficients.
- Schedule, cooldown, and parameter-group work:
  [pull request 321](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/321)
  swept learning-rate cooldown fraction,
  [pull request 737](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/737)
  explored cooldown-aware parameter averaging,
  [pull request 864](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/864)
  retuned exponential moving average warmup, and
  [pull request 1381](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1381)
  tested cooldown decay shape.
- Preconditioner and geometry work:
  [pull request 1036](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1036)
  ablated SOAP refresh cadence,
  [pull request 1138](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1138)
  tested Newton-Muon activation-covariance preconditioning,
  [pull request 1240](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1240)
  extended Newton-Muon coverage and update period, and
  [pull request 1543](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1543)
  added Tikhonov shrinkage to the Newton-Muon buffer.
- Averaging and evaluation-wrapper work:
  [pull request 1378](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1378)
  isolated parameter exponential moving average refresh,
  [pull request 1429](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1429)
  confirmed it with another seed, and
  [pull request 1533](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1533)
  evaluated validation loss on averaged weights.
- Aux AdamW moment control:
  [pull request 1532](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1532)
  found a beta2 pulse at cooldown onset,
  [pull request 1614](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1614)
  made that pulse canonical,
  [pull request 2393](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2393)
  moved the pulse earlier,
  [pull request 2403](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2403)
  tried a two-pulse staircase rule, and
  [pull request 2405](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2405)
  swept pulse amplitude.
- Late composition and cleanup:
  [pull request 1966](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1966)
  scheduled Muon momentum during cooldown,
  [pull request 2071](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2071)
  baked the winner as a default,
  [pull request 2298](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2298)
  corrected Arbor Muon,
  [pull request 2317](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2317)
  composed Cautious-Muon with Arbor and reference interpolation, and
  [pull request 2368](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2368)
  pinned a torch version to avoid Blackwell bfloat16 instability.

The case study improved the known step-count frontier, but it also accumulated
too much research debt. A better autoresearch system would have moved promising
mechanisms through screen, retune, multi-seed confirmation, cleanup, and
baseline promotion faster, while cutting off speculative micro-sweeps once their
expected value fell below the cost of more experiments.

## Core Principles

### Optimize The Actual Training Contract

This benchmark was not a wall-clock speedrun. It optimized step count under a
fixed dataset, fixed architecture, and fixed batch size. Slow optimizer code was
allowed if it reduced the number of optimizer updates. That distinction matters.
A systems optimization that makes each step faster is useful for cost, but it is
not the primary scientific claim unless it also changes step count or makes more
experiments affordable.

For a new training-optimization project, write the contract before running
experiments:

- what data is fixed,
- what architecture is fixed,
- what batch size and token budget are fixed,
- whether wall-clock time or step count is the metric,
- how many seeds are required for a claim,
- how validation checkpoints are selected,
- which optimizer code must be included for reproducibility.

### Use Statistical Gates, Not Best Single Runs

The benchmark rule required a margin below 3.28 that grows with the number of
runs. Single-run screens were useful for exploration, but a guide or baseline
should not promote a fragile one-seed win as a new principle.

This case study repeatedly showed why:

- many variants crossed the target at one seed and failed to generalize cleanly,
- small loss differences near 3.28 were comparable to validation noise,
- several "wins" needed another seed before promotion,
- merged defaults needed cleanup smoke tests after the result-producing PR.

A practical rule is:

- use one-seed runs for cheap screening,
- use paired or matched runs for small local comparisons,
- require multiple seeds for promoted baselines,
- report all non-cherry-picked seeds,
- choose the target step before the confirmation batch.

### Separate Mechanism Search From Retuning

Retuning learning rate and weight decay is necessary, especially after changing
an optimizer mechanism. But a research program that only sweeps scalar
hyperparameters becomes narrow very quickly.

The useful pattern is:

1. Propose a mechanism with a reason it might change optimization dynamics.
2. Run a small screen.
3. Retune the main sensitive knobs around that mechanism.
4. Compare against a retuned baseline, not a stale baseline.
5. Clean up the implementation if it wins.

The run often did this well for mechanisms such as Newton-Muon, SOAP refresh,
parameter averaging, beta2 pulses, and Arbor. It also sometimes slipped into
too many adjacent scalar sweeps after the mechanism-level question had been
answered.

### Treat Parameter Groups As Different Systems

The model was not optimized by one homogeneous optimizer. Body matrices, output
head, embeddings, scalar parameters, attention projections, and feed-forward
matrices often wanted different treatment.

Useful examples:

- body matrices used Muon-family updates,
- auxiliary parameters used AdamW,
- output head and embeddings often carried different sensitivity from the body,
- attention and multilayer perceptron blocks often responded differently to
  learning-rate and preconditioning changes,
- scalar parameters had their own learning-rate and epsilon behavior.

This suggests a general principle: do not assume one optimizer schedule is right
for all parameter groups. First split the groups by role, then test whether a
group-specific schedule is truly worth the added complexity.

### Cooldown Is A Separate Phase

Many of the strongest observed changes in this case study concerned late
training behavior: learning-rate decay shape, Muon momentum schedule, AdamW
beta2 pulses, exponential moving average refresh, and reference interpolation.

The training run near the target is not just "early training with smaller
learning rate." Gradients are smaller, validation noise matters more, moment
buffers can become stale, and smoothing can turn a ragged validation curve into
an earlier target crossing.

For a new benchmark, explicitly map the phases:

- warmup,
- main descent,
- cooldown entry,
- target-crossing region,
- final validation window.

Then ask which optimizer states should stay stable, which should become more
reactive, and which should be smoothed.

### Averaging Can Be A Real Method, But It Must Be Declared

Parameter averaging and validation on averaged weights can reduce noise and move
the first target crossing. That is useful, but it changes what "the model at
step N" means. It must be part of the declared training method, not an
after-the-fact validation trick.

Good averaging experiments report:

- when averaging starts,
- which parameters are averaged,
- whether averaging affects training or only evaluation,
- whether the first-step-to-target metric uses raw or averaged weights,
- whether the same rule is used for every seed.

### Preconditioners Need Geometry And Maintenance

Preconditioners were a major research direction: SOAP, PMuon, Newton-Muon,
Tikhonov shrinkage, Arbor/Sinkhorn equilibration, and related covariance buffers.
The important lesson is not that one named preconditioner always wins. The
lesson is that preconditioners have maintenance costs and failure modes:

- stale bases,
- unstable eigenspectra,
- refresh cadence,
- buffer initialization,
- excessive damping,
- coverage gaps across parameter shapes,
- interactions with cooldown and momentum.

The best preconditioner experiments measured not only final loss, but also what
the preconditioner was doing: coverage, refresh period, damping, variance,
matrix shape, and interaction with Muon orthogonalization.

### Promote Winners Into Simpler Baselines

A winning PR is not finished until the code path is cleaned up and made the new
baseline. Otherwise every follow-up has to remember sentinel flags, branchy
logic, and stale defaults.

The run had cleanup pull requests such as
[pull request 1614](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1614),
[pull request 2071](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2071),
[pull request 2325](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2325),
and
[pull request 2363](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2363).
That habit is worth keeping. The cleaner the baseline, the easier it is for the
next experiment to ask a real question.

### Avoid The Infinite Micro-Sweep

This case study also shows the danger of autoresearch without enough stop
pressure. The PR count was enormous, and many late experiments were tiny
variants around already narrow neighborhoods.

Useful stop rules:

- stop when the expected step improvement is below the seed-noise equivalent,
- stop when a mechanism has produced multiple nulls after fair retuning,
- stop when a proposed sweep changes only a scalar without a mechanism-level
  reason,
- stop when the implementation complexity exceeds the plausible gain,
- stop when the result would not change the next baseline or portfolio decision.

Negative results are useful. Low-value loops are not.

## Training-Optimization Research Categories

### Optimizer Mechanisms

Promising experiments:

- Muon-family update changes,
- Newton-Schulz polynomial and iteration changes,
- cautious or sign-aligned update masks,
- outer optimizers such as Nesterov wrappers,
- optimizer-state resets or pulses,
- new moment transforms before orthogonalization.

Common failure modes:

- NaNs from aggressive transformations,
- changes that help one parameter group and hurt another,
- mechanisms that only work after retuning but are compared to an untuned
  baseline,
- added complexity that is not justified by step-count improvement.

### Schedules And Cooldown Control

Promising experiments:

- learning-rate cooldown fraction,
- cooldown decay shape,
- momentum schedules,
- AdamW beta1 and beta2 schedules,
- epsilon schedules or pulses,
- phase-specific weight decay,
- warmup length interactions with later pulses.

Common failure modes:

- overfitting schedule timing to one seed,
- forgetting that target crossing may happen before the nominal final step,
- retuning one group while leaving the body or auxiliary optimizer stale,
- assuming a schedule that helps raw weights also helps averaged weights.

### Parameter-Group Specialization

Promising experiments:

- separate body, embedding, output-head, and scalar schedules,
- attention versus feed-forward learning-rate splits,
- per-block or depth-aware multipliers,
- group-specific clipping,
- group-specific optimizer states.

Common failure modes:

- too many flags and group-specific branches,
- group interactions that are not tested after composition,
- wins that disappear when the promoted baseline is cleaned up.

### Preconditioning And Geometry

Promising experiments:

- SOAP or Shampoo-style covariance preconditioning,
- Newton-Muon activation covariance,
- PMuon bilateral covariance,
- Tikhonov damping,
- Sinkhorn or spectrum equilibration,
- refresh cadence and buffer warm-starts.

Common failure modes:

- expensive machinery that does not improve step count,
- stale or noisy covariance estimates,
- fragile eigendecompositions,
- incorrect coverage assumptions,
- improvements that are really from retuning rather than the preconditioner.

### Averaging And Validation Wrappers

Promising experiments:

- parameter exponential moving averages,
- Polyak-Ruppert or stochastic weight averaging,
- phase-specific averaging,
- evaluation-only averaging,
- reference interpolation when declared before validation.

Common failure modes:

- validation peeking,
- comparing averaged and raw weights without saying so,
- using smoothing to hide a worse underlying optimizer,
- promoting a wrapper without proving it composes with the next baseline.

### Initialization And Parameterization

Promising experiments:

- embedding initialization scale,
- residual projection initialization,
- orthogonal initialization,
- muP-style depth-aware scaling,
- scalar and normalization parameter initialization,
- output-head initialization and tying choices.

Common failure modes:

- effects that vanish after optimizer retuning,
- initialization wins that consume stability margin,
- confusing faster early descent with better final target crossing.

### Infrastructure And Reproducibility

Promising experiments and safeguards:

- pinning framework versions when hardware behavior changes,
- smoke tests after cleanup,
- fixed logging fields,
- explicit seed accounting,
- avoiding per-run early stopping,
- preserving full code needed to reproduce a result.

Common failure modes:

- framework upgrades that change numerical stability,
- hidden benchmark-contract changes,
- untracked data or environment differences,
- result comments that omit seed count or statistical margin.

## Warm-Start Playbook For A New Training Optimization Run

### Phase 1: Reproduce The Benchmark

Deliverables:

- exact dataset and data-loading path,
- fixed architecture and batch size,
- baseline validation loss curve,
- baseline first step to target,
- seed and significance rule,
- known run-to-run variance,
- minimal command to reproduce.

Do not test new optimizers until the baseline reproduces.

### Phase 2: Build A Result Ledger

Every experiment should emit structured metrics:

- step count,
- validation loss,
- number of seeds,
- statistical margin,
- optimizer changes,
- schedule changes,
- initialization changes,
- known caveats,
- result status.

The ledger should distinguish screens, candidates, confirmations, cleanups, and
negative results.

### Phase 3: Screen Mechanisms Cheaply

Use short or one-seed runs to ask whether a mechanism has any signal. Do not
over-interpret them.

Good screens:

- compare against a current baseline,
- include the main sensitive hyperparameters,
- report whether the run reached target,
- include gradient and parameter health if the mechanism is risky.

### Phase 4: Retune Around A Mechanism

If a mechanism shows signal, retune learning rate, weight decay, cooldown, and
key moment parameters around it. This is support work. It should not become an
endless sweep.

### Phase 5: Confirm With Multiple Seeds

Promote only after a fair confirmation batch. Use the benchmark's statistical
rule, not a best single run.

### Phase 6: Clean Up The Baseline

When a change wins, remove dead flags, bake the new default, run a smoke test,
and update the result ledger. Future experiments should build on the clean
baseline, not on a fragile pile of flags.

### Phase 7: Manage The Portfolio

Keep a balanced portfolio:

- mechanism search,
- retuning of promising mechanisms,
- ablations and cleanup,
- skeptical replication,
- infrastructure stability.

If the portfolio becomes mostly scalar sweeps, add more mechanism search. If it
becomes mostly new mechanisms without retuning, add more exploitation. If it
becomes mostly confirmation of tiny effects, stop and re-rank the backlog.

## Experiment Template For An Autoresearch System

```text
Hypothesis:
  The optimizer, schedule, initialization, or validation mechanism being tested.

Benchmark contract:
  Dataset, architecture, batch size, maximum step count, validation cadence, and
  statistical rule.

Current baseline:
  Commit or PR, step count, validation loss, seed count, and known caveats.

Mechanism:
  Why the change should reduce steps to target.

Retuning plan:
  Learning rate, weight decay, cooldown, momentum, beta, epsilon, and group
  schedules that may need retuning.

Measurement:
  One-seed screen, paired comparison, multi-seed confirmation, or cleanup smoke.

Promotion gate:
  The exact step count, mean validation loss, seed count, and statistical margin
  required to make this the new baseline.

Stop rule:
  When to kill the idea, when to retune, and when to promote.

Result:
  Structured metrics, run IDs, code path, and a plain-language conclusion.
```

## High-Value Defaults For The Next Training Run

Start with these defaults:

- Fix the benchmark contract before the search starts.
- Use step count and statistical margin as the main claims.
- Keep one-seed screens cheap and multi-seed claims strict.
- Retune learning rate and weight decay after mechanism changes.
- Treat cooldown as its own phase.
- Split body, embedding, output head, and scalar parameter groups.
- Record every promoted baseline and every killed mechanism.
- Prefer simple baselines after cleanup over flag-heavy stacks.
- Use paired comparisons for small effects.
- Re-rank the portfolio whenever the search becomes mostly micro-sweeps.

Avoid these habits:

- claiming a universal optimizer lesson from one small benchmark,
- promoting a one-seed result as a stable baseline,
- comparing a tuned new method to an untuned old method,
- selecting the best validation step after seeing all seeds,
- adding optimizer machinery without an ablation path,
- keeping unused flags after a result is promoted,
- spending many PRs on effects smaller than the validation noise floor.

## Evidence Appendix

Selected pull request evidence:

| Pull request | Lesson |
| --- | --- |
| [#1](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1) | The first requirement was to find the baseline's actual shortest statistically valid step count in this repo. |
| [#5](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/5) | NorMuonH framed row and column variance preconditioning plus hyperball constraints as a strong Muon-family baseline. |
| [#8](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/8) | SOAP-style preconditioning for multilayer perceptron weights became an early mechanism to isolate. |
| [#50](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/50) | Tail averaging reached a 3304.2 first-step-to-target result, showing averaging as a real but not automatically dominant lever. |
| [#250](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/250) | Newton-Schulz coefficient scans showed that optimizer math changes need validity constraints, not arbitrary coefficient sweeps. |
| [#321](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/321) | Cooldown fraction was important enough to become a baseline-shaping schedule axis. |
| [#571](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/571) | Scalar parameters needed their own AdamW learning-rate treatment. |
| [#699](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/699) | Depth-aware initialization produced a strong validation-loss result and kept initialization in the research portfolio. |
| [#737](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/737) | Cooldown-aware parameter averaging reached the 2925-step region, showing late-phase smoothing mattered. |
| [#864](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/864) | Exponential moving average warmup retuning kept averaging effects from being treated as a one-off trick. |
| [#925](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/925) | Muon momentum drops at cooldown onset reached 2975 steps and helped motivate later momentum schedules. |
| [#1036](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1036) | SOAP refresh cadence ablations showed that preconditioner maintenance is itself a tunable mechanism. |
| [#1138](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1138) | Newton-Muon activation-covariance preconditioning reached 3175 steps but needed coverage and period tuning. |
| [#1240](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1240) | Extending Newton-Muon coverage and period reached 3150 steps with validation loss 3.26339. |
| [#1378](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1378) | Parameter exponential moving average refresh isolated a late-phase averaging signal at 2875 steps. |
| [#1429](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1429) | Confirmation work mattered: the parameter exponential moving average refresh needed a second seed. |
| [#1532](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1532) | A transient aux AdamW beta2 increase near cooldown reached 2875 steps and became a canonical default. |
| [#1533](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1533) | Evaluation on averaged weights reached 2925 and 2912.5 step results, but had to be treated as part of the method. |
| [#1543](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1543) | Tikhonov shrinkage for Newton-Muon buffers produced a paired validation-loss improvement. |
| [#1614](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1614) | Cleanup pull requests converted winning flags into a simpler default path. |
| [#1966](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/1966) | Scheduling Muon momentum down during cooldown produced a 2875-step candidate and became a default after cleanup. |
| [#2298](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2298) | Corrected Arbor Muon reached a 2856.25 step result and re-opened the late-stage composition track. |
| [#2317](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2317) | Cautious-Muon plus Arbor and reference interpolation improved validation loss to 3.276193 on the merged Arbor base. |
| [#2349](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2349) | AdamW epsilon still mattered late in the run, but only after the larger stack was fixed. |
| [#2393](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2393) | Moving the aux AdamW beta2 pulse earlier improved reference-interpolated validation loss to 3.274835. |
| [#2403](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2403) | A two-pulse beta2 staircase produced 2825 and 2837.5 first-step-to-target results, but it needed careful interpretation as partial versus complete evidence. |
| [#2429](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2429) | Extending Muon momentum warmup to interact with the beta2 pulse produced an n=4 mean loss of 3.2777 at step 2850 and became one of the late winners. |
| [#2368](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2368) | Framework pinning was part of optimizer research because numerical instability can invalidate the entire experiment stream. |

Corpus statistics:

- Pull requests reviewed: 2,443.
- Pull requests with comments: 2,427.
- Pull requests with a `SENPAI-RESULT` marker: 2,275.
- Structured result blocks parsed: 3,387.
- Open pull requests at review time:
  [#2466](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2466),
  [#2467](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2467),
  [#2468](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2468),
  [#2469](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2469),
  [#2470](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2470),
  [#2471](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2471),
  [#2472](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2472),
  [#2473](https://github.com/morganmcg1/modded-nanogpt-senpai/pull/2473).
- Main recurring themes: Muon-family optimizer changes, AdamW auxiliary
  schedules, cooldown control, exponential moving averages, SOAP and
  Newton-Muon preconditioners, initialization, parameter-group specialization,
  statistical confirmation, cleanup, and infrastructure stability.
