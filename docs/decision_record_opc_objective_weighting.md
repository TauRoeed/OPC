# Decision record: OPC training objective and importance weighting (2026-09-27)

Status: the development work on this component is closed. Working development defaults were set on
2026-09-28 (see the update below). The final paper choice is left to the scientific reassessment.
All evidence comes from development runs; the frozen protocol will be evaluated on fresh confirmatory
seeds and conditions.

## Question

Two problems motivated this work:
- The OPC loss divided its correction by the minibatch mean weight, so the objective depended on the
  batch size that Optuna searches.
- Under the log trick, a weight transform changes the objective being optimized.

The questions were which objective to train, how to differentiate it, and with which importance
weights. The aim was to make sure the objective named in the paper is the one actually optimized, not
to find a new weighting method.

## What was considered

| Option | What it optimizes (docs/training_losses.md 3.4) | Where it stands |
|---|---|---|
| Legacy minibatch SNDR (`sndr`, `--sn-scope batch`) | a per-batch, stop-gradient ratio surrogate; the objective depends on the batch size | CLI default (provisional); baseline / diagnostic |
| Global SNDR (`--sn-scope global`) | DR's direction, with the correction divided by a full-data mean weight that is fixed per epoch (stop-gradient, stale within the epoch); not exact SNDR | baseline / diagnostic |
| Exact SNDR (ratio gradient, refreshed baseline) | — | not implemented (closed) |
| `dr` (no self-normalization) | DM + g(w)(r − q̂), per-example additive | candidate objective |
| Log trick with a transform g | DM + H(w)(r − q̂), with H = ∫ g(t)/t dt (arctan-saturated for `shrink:λ`) | reproducibility / appendix |
| Direct gradient (`--opc-gradient direct`) | the named estimate DM + g(w)(r − q̂) itself | candidate gradient form |
| `none` | raw DR | unregularized baseline |
| `clip:10`, `shrink:10000` as training weights | — | reproducibility / appendix |
| `shrink:100` (Su et al. 2020) | optimistic shrinkage, non-monotone above √λ | smooth-weighting candidate |
| `harmonic:λ` (Metelli et al. 2021) | w/(1 − λ + λw): monotone, bounded by 1/λ, differentiable | smooth-weighting candidate; λ ∈ {0.05, 0.1, 0.2} |

## Evidence (development runs, paired on identical configurations)

The numbers are in docs/training_losses.md, section 9, and the runs in
artifacts/full_study/run_registry.csv.
- **Objective:** `dr` beats legacy and global SNDR on nearly every identical configuration, by 0.02–0.32
  points of true CTR per trial. Global and legacy SNDR are within 0.05 points of each other.
- **Gradient form:** with raw weights, the log trick and the direct gradient are identical, as they must
  be. With `clip:10` the gradient form makes no consistent difference. With `shrink:100` the direct
  gradient is equal or slightly better (up to +0.14 per trial at 25k), and it keeps policies closer to
  the logger.
- **Weights:** `shrink:100` beats raw and `clip:10` weights from 25k on, under either gradient form. At
  5k the weighting doesn't matter.

- **Final bounded comparison (`dr`, direct gradient, λ fixed at 0.05, 0.1, 0.2):**
  - `shrink:100` and every harmonic λ improve on raw DR from 25k up.
  - `harmonic:0.05` matches `shrink:100`.
  - `harmonic:0.1` and `harmonic:0.2` are ahead of `shrink:100` by 0.09–0.49 points per trial, with every
    CI excluding 0, and by 0.09–0.57 on the selected policy. The effect of λ is monotone, with its best
    value at the edge of the set.
  - ESS, heavy-weight shares and selection-estimate errors are similar across methods. Harmonic policies
    sharpen more than `shrink:100` policies and less than raw DR.

## Decisions

1. **SNDR work is closed.** Legacy and global SNDR remain reproducible development baselines and
   diagnostics. Exact SNDR is not implemented.
2. **The gradient control (`--opc-gradient`) and the harmonic transform (`harmonic:λ`) are added.**
   Both are tested and recorded in all run metadata. The study refuses harmonic training weights under
   the log trick.
3. **Kept out of the main future grid, unless a specific scientific reason appears:** the log-trick
   (arctan) surrogate for transformed weights, `clip:10` and `shrink:10000` training weights, and legacy
   and global SNDR.
4. **Raw DR remains the unregularized baseline.**
5. **Working development defaults (updated 2026-09-28, see below).** The full-study runners default to
   `dr`, the direct gradient and `harmonic:0.1`, with selection kept at `clip:10`. Before 2026-09-28 the
   defaults were legacy `sndr`, the log trick and `shrink:100`.

## Left open for the reassessment

- **The main method's weighting.** The candidates are the direct gradient of a smooth correction:
  `shrink:100` (Su et al.) or `harmonic:λ` (Metelli et al.). Raw DR is the baseline.
  - On these development conditions, harmonic at λ = 0.1 and 0.2 was ahead of `shrink:100`, and it has
    the cleaner OPL interpretation: monotone, bounded and differentiable, and designed for gradient-based
    learning.
  - The effect of λ is monotone, with its best value at the edge of the prespecified set. The paper's own
    rule chooses λ from n, δ and the 2-Rényi divergence, and would give a smaller λ here.
  - Whether to adopt harmonic, and at which λ, is not decided here.
- **The objective and gradient form** are the working defaults since 2026-09-28 (`dr`, direct). Freezing
  them for the paper is part of the protocol freeze before confirmatory runs.

## Update 2026-09-28: working development defaults

On the user's decision, the full-study runners (`run_full_study`, `run_full_study_parallel`) use these
working defaults for development runs:
- **OPC objective:** `dr`.
- **Gradient:** `direct`.
- **Training weights:** `harmonic:0.1`.
- **Selection weights:** `clip:10`, unchanged.

These are the working method, not the final paper choice. `harmonic:0.1` was picked because it is an
established smooth, monotone, differentiable OPL correction, and its asymptotic cap is 10 (1/λ).
`harmonic:0.2` was not picked merely because the best development result sat at the edge of the small λ
set.

The alternatives keep defined roles:
- **Standard comparison:** Su et al. `shrink:100` with the direct gradient, the prespecified smooth-weight
  comparison.
- **Reference:** raw DR (`none`), the unregularized reference.
- **Reproducibility only, not main candidates:** legacy and global SNDR, log-trick variants of transformed
  weights, `clip:10` training weights and `shrink:10000`.

The previous defaults are reproduced by `--policy-losses sndr --sn-scope batch --opc-gradient log-trick
--train-weights shrink:100`. The H1 runner (`run_h1_study`) keeps those settings explicitly; aligning H1
is not part of this change.

## Provenance

- **Fixed code:** b5efc7d (short final batch weighted by rows), 7bc3fb7 (replay sampler), 3bed8de
  (`--opc-gradient`) and d40aaef (harmonic).
- **Pre-fix code:** b584edc (the TPE re-tunes). The b5efc7d fix changes `dr`, global SNDR, DM-only and
  no-propensity only in their final short minibatch; legacy SNDR is unchanged.
- **Reproducibility:** the reused raw-DR and `shrink:100` runs (3bed8de) reproduce bit for bit on d40aaef
  (one condition each, all 60 trials).
