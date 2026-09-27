# Representation repair: development results (2026-09-27)

Status: these are development runs only, on seeds 100/101 with the paired random sampler (no TPE warm
start). They are not the confirmatory protocol. The method is the working development default since
f5cade9: OPC = DR, differentiated directly, harmonic:0.1 training weights, clip:10 selection weights.
It is not the final paper choice (see `docs/decision_record_opc_objective_weighting.md`). The tables
are in `artifacts/full_study/summaries_20260927/` and the runs in `artifacts/full_study/run_registry.csv`.

## Question

Can logged-feedback learning undo a logger's representation bias, and when does the propensity
correction matter? The results bear on this development proposition:

> Representation bias is correctable from logged feedback when the distortion has recoverable shared
> structure and the log contains sufficient counterfactual information; propensity correction becomes
> particularly valuable when reward-model extrapolation is weak.

The work comes in three stages:
- **Stage 1** asks how much of each bias the learner's own policy class can repair in principle (the
  oracle repair).
- **Stage 2** asks how much of that repair logged learning achieves, and with which ingredients.
- **Stage 3** asks how logging support (logger sharpness) changes it.

## Setup

- **Worlds** ([representation_bias.md](representation_bias.md)). Clean BPR vectors define the true
  click model. The logger, the reward model and the policy see biased vectors. The bias types:
  - **warp** mixes each side with one random linear map of itself;
  - **group** mixes it with one offset per k-means cluster;
  - **vector** mixes it with one offset per user or item.

  "high" on one type keeps about 88% of the taste signal; the combined levels (all three types)
  keep 75% (medium) and 50% (high). The logger is the softmax over the biased vectors, sharpened
  so its CTR is 80% of its own greedy CTR.
- **Datasets and seeds:** ml, kuairand and anime × seeds 100 and 101.
- **Arms** (Stage 2). All train the same linear repair of both sides, `(I + D)x + b`, with a
  learnable logit scale:
  - **OPC:** DR with harmonic:0.1 weights, direct gradient;
  - **DM-only:** the reward model's value, no propensities;
  - **no-propensity:** the naive logged reward;
  - **tempered logger:** the logger's ranking, only sharpened by a chosen logit scale.

  OPC, DM-only and no-propensity share each trial's configuration and seed (paired random
  sampler, 20 trials). The reward model is fit on each train size's own rows and cross-fitted by
  user in 5 folds. Selection uses DR with clip:10 on 20,000 validation rows.
- **Oracle repair** (Stage 1). The same policy class, fit to the true click probabilities: Adam,
  3,000 steps, the best of 3 learning rates, 20,000 users weighted by the user prior. Two fits:
  logit scale fixed and logit scale learned. The learner's class bound is the better of the two.
- **Measures:**
  - true values V (exact, over the user prior) of the stochastic policy and of its greedy
    ranking;
  - loss = V_clean − V_logger;
  - gain = V − V_logger;
  - structural recoverability = oracle gain / loss;
  - fraction of oracle repair = learned gain / oracle gain.

  The greedy (ranking) versions are the primary measures. The stochastic ones also count
  sharpening, which the repair class can do by scaling the vectors.

## Stage 1: structural recoverability

Means over 3 datasets × 2 seeds, in CTR points:

| Bias | loss (stoch. / greedy) | oracle gain (stoch. / greedy) | recoverability, greedy (range) | ml / kuairand / anime |
|---|---|---|---|---|
| warp only (high) | 5.79 / 7.27 | 11.72 / 7.22 | 0.994 (0.986–0.999) | 0.994 / 0.994 / 0.993 |
| group only (high) | 4.71 / 5.89 | 8.79 / 3.99 | 0.680 (0.538–0.792) | 0.775 / 0.701 / 0.564 |
| vector only (high) | 5.54 / 6.94 | 8.02 / 3.46 | 0.499 (0.305–0.622) | 0.612 / 0.576 / 0.308 |
| combined medium | 8.49 / 10.55 | 10.89 / 7.00 | 0.669 (0.546–0.745) | 0.737 / 0.713 / 0.556 |
| combined high | 14.14 / 17.60 | 14.58 / 12.09 | 0.689 (0.590–0.773) | 0.720 / 0.752 / 0.596 |

- **Ordering.** warp > group > vector holds in all 6 dataset × seed pairs. With no bias, the
  oracle stays within 3e-5 of the ceiling (29.94%).
- **Stochastic ratios exceed 1.** The stochastic recoverability runs from 1.04 to 2.10: the class
  can sharpen beyond the clean logger's exploration level (V_clean = 23.97%). Sharpening alone
  (the scale-only class) recovers 0.85–1.10 of the stochastic loss for single-type biases.
- **Caveat on the bound.** 86% of the linear fits flattened, and the winning learning rate was
  the largest of the three in 35 of 36 fits. The group and vector bounds may therefore be
  slightly conservative.

## Stage 2: learned recovery

Grid: 6 bias settings × 5k / 25k / 100k × 3 datasets × 2 seeds × 4 arms (n = 6 paired conditions
per cell).

Greedy fraction of the oracle repair, 5k → 25k → 100k:

| Bias | OPC | DM-only | no-propensity |
|---|---|---|---|
| warp | 0.27 → 0.43 → 0.50 | 0.08 → 0.28 → 0.34 | ≤ 0.11 |
| group | 0.14 → 0.28 → 0.39 | −0.08 → 0.14 → 0.27 | ≤ 0.06 |
| vector | 0.07 → 0.24 → 0.37 | −0.16 → 0.16 → 0.29 | ≤ 0.07 |
| combined medium | 0.25 → 0.43 → 0.51 | 0.13 → 0.32 → 0.39 | ≤ 0.13 |
| combined high | 0.27 → 0.42 → 0.49 | 0.32 → 0.39 → 0.45 | ≤ 0.11 |

The tempered logger is 0 by construction. As a share of the whole ranking loss, OPC recovers at
100k: warp 0.50, group 0.26, vector 0.19, combined 0.34.

OPC minus DM-only, true CTR points, mean and 95% t-interval over the paired conditions:

| Bias | 5k | 25k | 100k |
|---|---|---|---|
| warp | +1.09 [+0.06, +2.13] | +0.95 [+0.52, +1.38] | +1.08 [+0.75, +1.42] |
| group | +0.76 [+0.04, +1.48] | +0.52 [+0.21, +0.83] | +0.45 [+0.32, +0.57] |
| vector | +0.56 [−0.08, +1.20] | +0.26 [+0.08, +0.44] | +0.25 [+0.11, +0.40] |
| combined medium | +0.68 [−0.16, +1.52] | +0.76 [+0.33, +1.19] | +0.78 [+0.55, +1.00] |
| combined high | −0.70 [−2.28, +0.87] | +0.39 [−0.16, +0.93] | +0.46 [+0.19, +0.74] |
| no bias | +0.36 [−0.31, +1.03] | −0.13 [−0.27, +0.02] | −0.25 [−0.33, −0.18] |

**OPC against the other two arms:**
- **no-propensity:**
  - combined: +0.9 to +4.7;
  - warp: +1.4 to +2.8;
  - group and vector: the interval includes 0 at 5k, +0.8 to +1.4 from 25k;
  - no bias: −0.74 at 5k, +0.01 at 100k.
- **tempered logger:**
  - combined: +1.7 to +5.9;
  - warp: +1.7 to +3.6;
  - group and vector: the interval includes 0 at 5k, +1.0 to +1.6 from 25k;
  - no bias: OPC trails by 1.46 at 5k and 0.39 at 100k.

**Support and selection:**
- **Weights.** OPC's selected policies have a raw-weight ESS of ~2,160 (5k) to ~970 (100k) of
  20,000 validation rows. Weights above 10 are 1.7–2.4% of rows, and the cell-mean max weight is
  125–434.
- **Estimate errors.**
  - OPC: its DR point estimate is optimistic by +0.4 to +1.1 points and its lower bound
    pessimistic by −0.2 to −0.8.
  - DM-only: its own estimate is optimistic by +3.7 (5k), +1.3 (25k) and +0.5 (100k).
  - The no-propensity arm's naive estimate sits about 15 points below the truth. It only ranks
    its trials, so its regret stays small.
- **Regret.** True selection regret is ≤ 0.08 points for OPC and ≤ 0.32 for DM-only.

**Tentative, at small data on anime:**
- OPC's vector fraction is −0.05 at 5k and 0.08 at 25k.
- Under combined high at 5k, OPC trails DM-only in both seeds (−0.55, −1.30).
- DM-only ranks worse than the logger for group and vector at 5k.

### Robustness: shrink:100 training weights (Su et al. 2020)

OPC alone, single-type highs × 25k × ml and kuairand, paired trial by trial with the main runs
(identical configurations in 12 of 12 conditions). Harmonic:0.1 minus shrink:100:
- **Selected policy:** +0.07 [−0.08, +0.22] (warp), +0.04 [−0.10, +0.17] (group) and +0.04
  [−0.19, +0.26] (vector) points.
- **Per trial:** +0.061 [+0.044, +0.079] points overall. Harmonic was higher in 81–92% of paired
  trials.
- **Overlap.** Harmonic's selected policies have lower raw-weight ESS, e.g. vector 983 vs 1,659.

The story does not change.

## Stage 3: logging support

Grid: ml and kuairand × seeds 100/101 × 25k × logger greedy share {0.6, 0.8, 0.95} × warp / group /
vector at high level × the four arms (n = 4 per cell). Each share has its own oracle bound. The 0.8
cells reproduce the Stage 2 25k cells bit for bit (48 of 48 trial tables).

- **The oracle doesn't depend on support.** The logger's stochastic value rises with sharpness
  (13.9–14.8% at 0.6, 22.0–23.4% at 0.95), but its ranking does not change. The oracle's greedy
  gain is essentially the same at every share: warp 6.4, group 4.1, vector 4.0–4.2 points.

Greedy fraction of the oracle repair, shares 0.6 / 0.8 / 0.95:

| Bias | OPC | DM-only | no-propensity |
|---|---|---|---|
| warp | 0.39 / 0.46 / 0.46 | 0.25 / 0.28 / 0.26 | ≤ 0.10 |
| group | 0.17 / 0.27 / 0.26 | 0.05 / 0.12 / 0.15 | ≤ 0.05 |
| vector | 0.29 / 0.32 / 0.32 | 0.18 / 0.22 / 0.23 | ≤ 0.06 |

OPC minus DM-only, true CTR points, 95% t-interval:

| Bias | 0.6 | 0.8 | 0.95 |
|---|---|---|---|
| warp | +0.83 [+0.33, +1.34] | +1.04 [+0.24, +1.84] | +1.09 [+0.38, +1.80] |
| group | +0.51 [+0.06, +0.96] | +0.52 [+0.03, +1.01] | +0.45 [+0.32, +0.58] |
| vector | +0.40 [+0.02, +0.77] | +0.32 [+0.04, +0.60] | +0.34 [−0.08, +0.76] |

**OPC against the other two arms:**
- no-propensity: warp +2.1 to +2.6; group +0.8 to +1.0 (intervals include 0); vector +1.1.
- tempered logger: warp +2.6 to +3.0; group +0.8 to +1.1; vector +1.2 to +1.5.

**Support diagnostics.** The raw-weight ESS of OPC's selected policies rises with logger sharpness
(warp 290 → 536 → 743, vector 534 → 983 → 1,825). The share of weights above 10 falls from 2.9–3.9%
to about 1.5%.

**Reading.**
- In this range, a repairable error does not become unrecoverable as the logger sharpens. From 0.8
  to 0.95, OPC's ranking recovery is flat and its lead over DM-only holds.
- The most exploratory logger (0.6) gave the lowest recovery and the heaviest weights against the
  sharp learned policies. Two candidate reasons, not tested: it logs fewer clicks (V_logger ≈ 14% vs
  22%), and its weights against sharp targets are heavier.
- Every share tested keeps full softmax support, so a near-deterministic logger is untested.
- The stochastic fractions fall with sharpness for every arm, mostly because less room for
  sharpening is left (the tempered logger falls from 0.60 to 0.16).

## Reading against the proposition (development evidence only)

- **"Correctable when the distortion has recoverable shared structure": partly supported.**
  Structurally, warp (one shared linear map) is almost fully repairable by the linear class (0.99),
  group (one offset per cluster) partly (0.68), and vector (one offset per item or user) least
  (0.50), in every dataset and seed. With logged data, recovery follows the same order. But even for
  warp, OPC reaches only about half of the oracle's ranking repair at 100k. "Correctable" therefore
  holds partially at these data sizes, not fully.
- **"When the log contains sufficient counterfactual information": not contradicted within logger
  shares 0.6–0.95, but its limit is untested.** Recovery did not fall as the logger sharpened. The
  most exploratory logger in the range gave the least recovery at 25k. A logger without full support
  was not tested.
- **"Propensity correction particularly valuable when reward-model extrapolation is weak":
  consistent, but not tested directly here.** The reward model in these runs is well specified and
  budget-fair.
  - With it, OPC beats DM-only by 0.3–1.1 points for single-type biases from 25k up. The margin is
    largest for warp. Under combined high bias, OPC and DM-only tie below 100k.
  - The earlier budget and misspecification runs (at the previous defaults: legacy SNDR, log
    trick, shrink:100) showed DM-only losing 1.4–1.5 points with a budget-fair q̂ at 5k, and 2–9
    points with a misspecified q̂, while OPC stayed within noise.
  - Weak extrapolation was not varied within Stages 1–3.

## Reproduction

The runs used the parallel runner with, at each size:
- 20 trials;
- `--slim --learn-logit-scale --sampler random --stage development`;
- the default budget: train-mode q̂ with 5 folds, validation 20,000;
- `--log-select-weights none clip:1 clip:3 clip:10 clip:30 clip:100 clip:300 clip:1000 shrink:10 shrink:100 shrink:1000 shrink:10000 shrink:100000 dm`.

The commands are:

```bash
# Stage 1 (oracle repair bound)
python -m training.oracle_repair --datasets ml kuairand anime --seeds 100 101 \
  --emb-dir BPR/embeddings --out artifacts/full_study/run_oracle_repair_20260927/<name>

# Stage 2 (per dataset; the worker planner sets the concurrency, --max-workers is the cap)
python -m training.run_full_study_parallel --ctr-levels 0.05 --seeds 100 101 \
  --train-sizes 5000 25000 100000 --n-trials 20 --slim --learn-logit-scale --sampler random \
  --stage development --log-select-weights <as above> --datasets <ml|kuairand|anime> \
  --bias-configs none medium high high/none/none none/high/none none/none/high \
  --methods opc no_propensity dm tempered_logger --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study --run-tag stage2_<dataset> --max-workers 4 --num-gpus 1

# Robustness slice (OPC with shrink:100 at 25k)
python -m training.run_full_study_parallel <as Stage 2> --datasets ml kuairand \
  --bias-configs high/none/none none/high/none none/none/high --train-sizes 25000 \
  --methods opc --train-weights shrink:100 --run-tag stage2_su_shrink_100

# Tables
python -m training.analyze_recoverability stage1 artifacts/full_study/run_oracle_repair_20260927 \
  --out artifacts/full_study/summaries_20260927
python -m training.analyze_recoverability stage2 artifacts/full_study/run_oracle_repair_20260927 \
  --runs artifacts/full_study/run_stage2_ml artifacts/full_study/run_stage2_kuairand artifacts/full_study/run_stage2_anime \
  --su artifacts/full_study/run_stage2_su_shrink_100 --out artifacts/full_study/summaries_20260927

# Stage 3: oracle bounds for the new logger shares (0.8 is Stage 1)
python -m training.oracle_repair --datasets ml kuairand --seeds 100 101 \
  --bias-configs high/none/none none/high/none none/none/high --logger-greedy-share <0.6|0.95> \
  --emb-dir BPR/embeddings --out artifacts/full_study/run_oracle_repair_stage3_lgs_<0_6|0_95>
# learned sweep per share
python -m training.run_full_study_parallel <as Stage 2> --datasets ml kuairand --train-sizes 25000 \
  --bias-configs high/none/none none/high/none none/none/high --logger-greedy-share <0.6|0.8|0.95> \
  --run-tag stage3_lgs_<0_6|0_8|0_95>
# tables per share (oracle root for 0.8: run_oracle_repair_20260927)
python -m training.analyze_recoverability stage2 <oracle root> --runs artifacts/full_study/run_stage3_lgs_<tag> \
  --out artifacts/full_study/summaries_20260927/stage3_lgs_<tag>
```

The code commits are in the registry. The Stage 2 ml and kuairand runs are at 5b66e18, and the
anime relaunch and the slice at f2e03b5. Later commits changed scheduling, analysis and tests only.
