# Why OPC's training gap is larger, and where OPC wins (pre-registration, 2026-10-09)

*Development stage. Branch `representation-mismatch-research-next` from `eb9dfd3`. §0–§12 are written
and committed before any experiment of this study has run. Results follow in later sections. Nothing here is
confirmatory.*

## 0. Question and scope

The shared-objective study (`docs/shared_objective_study.md`) decomposed, on the 24 biased development worlds at
N = 25,000, each arm's distance from the class's value optimum θ_value* (greedy CTR points):

| | objective mismatch | training gap | selection gap |
|---|---|---|---|
| penalized likelihood | 1.42 | 2.11 | 0.12 |
| OPC (harmonic DR) | 0 (by the convention of that study) | 3.81 | 0.15 |

This study asks **why OPC's training gap is larger**, and **in which regimes** OPC's better population target
outweighs its larger learning and selection cost. The deliverable is a regime map with a mechanism, not one more
mean comparison. It is not a search for a setup in which OPC wins.

Distinguished throughout: (A) the true population value gradient; (B) the raw IPS gradient; (C) the raw DR
gradient; (D) the current harmonic DR gradient. Also distinguished: the population objective of each estimator and
its finite-sample version.

Not in this phase: hierarchical (group, regional, individual) correction, new corruption families, RecoGym, new
BLOB or CausE tuning, fresh confirmatory worlds, new estimators (trust regions, natural gradients, elaborate control
variates, learned or adaptive transforms).

## 1. The live OPC training path (audited from the code at eb9dfd3)

### 1.1 Policy class (`models/shared_objectives.py` `SharedCorrectionModel`, the `shared_opc` arm)

- Frozen source vectors: the logger's biased vectors x_u (`our_x`) and a_j (`our_a`), K = 32.
- Corrections: u′ = (I + D_u)x + b_u, a′ = (I + D_a)a + b_a (`GlobalLinearCorrection`, D and b start at 0);
  g_θ(u, j) = ⟨u′_u, a′_j⟩.
- Learned scale s = exp(30 θ_s) (`LOGIT_SCALE_SPEED` = 30, θ_s starts at 0); T = the logger's temperature
  (`policy_temperature`).
- Policy: π_θ(j | u) = softmax_j(s · g_θ(u, j) / T) over the whole catalog; no ε-greedy mixing (`eps_greedy` = 0).
- θ = (D_u, b_u, D_a, b_a, θ_s): 2K² + 2K + 1 = 2,113 parameters. At θ = 0, π_θ is the logger's softmax exactly.

### 1.2 Loss (`models/custom_losses.py` `DRPolicyLoss`, weights `harmonic:0.1`, `--opc-gradient direct`)

For a minibatch B of logged rows (x_i, a_i, r_i, p_i):

    L_B(θ) = −(1/|B|) Σ_{i∈B} [ Σ_j π_θ(j|x_i) q̂(x_i, j) + h(w_i) (r_i − q̂(x_i, a_i)) ]  + λ R(θ)
    w_i = π_θ(a_i|x_i) / max(p_i, 1e-10),     h(w) = w / (0.9 + 0.1 w)

- h is the harmonic transform of Metelli, Russo and Restelli (2021) with λ_h = 0.1: increasing, h(w) ≤ 10,
  h′(w) = 0.9 / (0.9 + 0.1 w)², h(1) = 1, h′(1) = 0.9.
- **Direct gradient:** π_θ is attached everywhere. The gradient is ∇ of the transformed estimate itself:
  ∇DM + h′(w) ∇w (r − q̂). It is not the log-trick surrogate.
- **No self-normalization** (`normalization='none'`): a mean of per-row terms. The short final batch is scaled by
  rows / nominal batch (`training_utils.minibatch_loss`), so an epoch's minibatch gradients sum to (n / batch) times
  the full-data gradient at fixed θ.
- q̂ enters as a constant (no gradient through it). λR(θ) is the source anchor of the shared study (§1.5).

### 1.3 Reward model (`trainer_trials.fit_shared_regression_bundle`, `run_full_study._crossfit_bundles`)

- **Model:** sklearn `LogisticRegression(random_state=12345)` with its defaults: L2 penalty C = 1, lbfgs, 100
  iterations. Features: `interaction` = [x, a, x ⊙ a] on the biased vectors.
- **Data** (`--reward-data train`): q̂ is fit on the same N training rows as the policy.
- **Cross-fitting** (`--crossfit-folds 5`): users are split into 5 folds by `derive_seed(seed, "crossfit",
  "user_fold")`, fixed per world.
  - The training loss reads `CrossFitScoresLookup`: user u's q̂ row comes from the model fit without the training rows
    of u's fold.
  - Selection and scoring use the full-data model on all N rows, applied to the validation rows, which are never
    training rows.

### 1.4 Data, optimizer and search

- **Data:** one simulation per (world, N) of 50,000 reg rows (unused with `--reward-data train`, but drawn), N
  training rows and 20,000 validation rows, randomly partitioned (`_build_regression_logged_split`; the seed
  depends on N).
  - Users are drawn from the user prior, actions from the logger, rewards from Bernoulli(q).
  - The stored propensity is the exact logging probability, in float64.
- **Optimizer:** Adam with no weight decay; the trial's lr is multiplied by its decay factor every epoch; the gradient
  norm is clipped at 1 every step; a fixed number of epochs per trial.
  - The DataLoader shuffles with the trial's seed.
  - The batch size is drawn per trial: {512, 1024, 2048} for N ≤ 25k, {2048, 4096, 8192} for N ≤ 100k
    (`batch_schedule`).
- **Search:** a paired seeded random search, 20 trials, `optuna_sampler(seed, label, N)`. Every world with the same
  seed, label and N trains the same configurations. Ranges:
  - lr 1e-4–2e-3, log-uniform; epochs 5–30; lr decay 0.8–1;
  - λ from {0, 0.001, 0.01, 0.1, 1}; the batch size from the schedule above.
- **Selection:**
  - native: OPC's 95% DR lower bound of π_θ on the validation rows, clip:10 weights, the full-data q̂;
  - common: the same bound for the trial's greedy policy;
  - best of 20: by true value, diagnosis only.
- **Penalized likelihood** (`shared_likelihood`): the same model read as a click model σ(s g / T + c). Its objective
  is the mean Bernoulli NLL at the logged pairs plus λR; its head (θ_s, c) starts at its fitted minimizer; it is
  selected by validation NLL.

### 1.5 Regularization

R(θ) = E_u‖u′ − x‖² / E_u‖x‖² + E_j‖a′ − a‖² / E_j‖a‖², uniform over the catalog (closed form); s and c are not
penalized. In the shared study it neither helped nor hurt OPC (shared OPC − historical OPC: −0.03).

### 1.6 Logging-support controls in the simulator (`utils/representation_bias.build_world`)

- **`--logger-greedy-share`** (default 0.8): the biased logger's temperature T_log is lowered from the spread
  temperature T until its CTR is that share of its own greedy CTR (`sharpen_logger`). A lower share gives a flatter
  logger, more exploration and better support; a higher share gives a sharper one.
  - The click model is calibrated beforehand, on the spread-temperature reference logger, and does not depend on the
    share (module docstring step 5). §5 verifies this.
- **`--logging-uniform-mix`** α: p = (1 − α) softmax + α / P. It is a support floor, but the policy's start point is
  then not the logger (an open risk in the Atlas).
- `--logging-spread` recalibrates the click model, so it changes the reward world: excluded.

## 2. The gradient estimators

Notation:
- θ is fixed. D = {(x_i, a_i, r_i, p_i)}, i = 1..n, is a logged dataset: x_i ~ prior, a_i ~ π0(·|x_i),
  r_i ~ Bernoulli(q(x_i, a_i)), p_i = π0(a_i|x_i).
- q is the true click probability; q̂ the cross-fitted reward model (§1.3), whose row for x_i comes from the fold model
  fit without x_i's fold.
- π = π_θ, w_i = π(a_i|x_i) / p_i, h the harmonic transform.
- Every estimate is a full-data gradient over the n training rows (λ = 0; R's gradient is deterministic and common to
  all).

| | estimator of V(θ) whose gradient is taken | fixed inputs |
|---|---|---|
| **G0** | V(θ) = Σ_u prior(u) Σ_j π(j\|u) q(u, j): the exact population value | q (simulator) |
| **G1** raw IPS | (1/n) Σ_i w_i r_i | p |
| **G2** raw DR, oracle q | (1/n) Σ_i [ Σ_j π(j\|x_i) q(x_i, j) + w_i (r_i − q(x_i, a_i)) ] | p, q |
| **G3** raw DR, q̂ | the same with q̂ | p, q̂ |
| **G4** harmonic DR, oracle q | (1/n) Σ_i [ Σ_j π(j\|x_i) q(x_i, j) + h(w_i) (r_i − q(x_i, a_i)) ] | p, q |
| **G5** harmonic DR, q̂ | the same with q̂: the current OPC training estimator | p, q̂ |

They are computed by `DRPolicyLoss` itself:
- weights `none` for G1–G3, `harmonic:0.1` for G4–G5;
- q̂ = 0 for G1, the true q rows for G2 and G4, the cross-fitted lookup for G3 and G5;
- autograd at the same θ; G0 by autograd of the exact value, chunked over every user.

**Analytical expectations.** Assumptions:
- (i) the propensities are exact and π0 > 0 wherever π_θ > 0 (a softmax logger has full support);
- (ii) the rows are i.i.d. from the world;
- (iii) q̂'s row for x_i does not depend on row i. Cross-fitting by user folds gives this, since fold k's model never
  sees a fold-k row.

Conditioning on x and q̂, E[w (r − c) | x] = Σ_j π(j|x)(q(x, j) − c(x, j)) for any fixed c, and the gradient passes
through the sum. Hence:
- **G1, G2, G3 are unbiased for g* = ∇V(θ)** for any q̂ satisfying (iii). G3's expectation is over datasets,
  including q̂'s variation.
- **G4 is also unbiased.** With the oracle q, the correction's conditional mean E[h(w)(r − q) | x, a] is 0 whatever
  the transform, and so is its gradient h′(w)∇w E[r − q | x, a]. With the oracle q the harmonic transform only damps
  noise.
- **G5 is biased** unless q̂ = q:
  E[G5] − g* = E_x Σ_j π0(j|x) (h′(w_j) − 1) ∇w_j (q(x, j) − q̂(x, j)), with w_j = π(j|x)/π0(j|x).
  The bias grows with the reward model's error and with the shrinkage h′ − 1 (−0.1 at w = 1, −1 for large w). It is
  computable exactly for a given q̂, over users and items (the *conditional bias*, §4).
- **The population objectives.** Raw DR's is V(θ) for any q̂ (given (i)). Harmonic DR's, with q̂_∞ (the population
  limit of the reward model), is
  V_h(θ) = Σ_u prior Σ_j [ π q̂_∞ + π0 h(π/π0)(q − q̂_∞) ],
  whose optimum θ_harm* need not be θ_value*. V(θ_value*) − V(θ_harm*) is the harmonic transform's surrogate bias at
  infinite data.
- Tests (§13) check these statements by enumeration on a tiny world.

## 3. Where the gradient is measured (policy states)

All in the shared parameterization θ = (D_u, b_u, D_a, b_a, θ_s), and every estimator at the same θ.

1. **Source:** θ = 0, the logger. Every w_i = 1.
2. **θ_value\*:** the population value optimum.
   - Fit from the source by Adam on the exact value: exact expectations over items and the true q, 2,048
     prior-sampled users per step, 3,000 steps with a cosine schedule.
   - Three learning rates {0.003, 0.01, 0.03}; the best by exact value is kept.
   - This is the class-oracle procedure of `training/class_oracles.py`, in this parameterization. It is validated
     against the existing affine-bilinear value oracles (`run_class_oracles_20261005`) by greedy and stochastic
     value.
3. **Mid-trajectory:** along the kept θ_value* run, the checkpoint (every 25 steps) whose exact value is closest to
   (V(source) + V(θ_value*)) / 2. It is deterministic, an actual optimization-path point, not a parameter average.
4. **θ_likelihood\*:** the population likelihood optimum in this parameterization.
   - Fit the same way: the logging-weighted Bernoulli NLL of σ(s g / T + c), exact expectations, the head started at
     its population fit.
   - Its scale s is a click model's, about 0.17 in the shared study, far flatter than any policy. The checkpoint is
     therefore θ_log*'s maps with θ_s set to the value-maximizing scale by an exact one-dimensional search: the
     likelihood optimum's ranking, deployed at its best sharpness.
   - The click-scale state is reported as a secondary point.
   - Validated against the existing affine-bilinear likelihood oracles by greedy value.

At θ_value*, g* ≈ 0, so relative measures there use ‖g*(source)‖ of the same world as the reference norm.

## 4. Gradient-quality metrics

For each world, policy state and estimator, R independent logged datasets of size N are drawn from the world with
the same logger.
- Seeds: `derive_seed(seed, "gradient_benchmark", N, support, r)`, disjoint from the study's splits.
- q̂ is refit with cross-fitting on every dataset, as in training.

Reported, each with a 95% interval (a bootstrap over datasets per world, a t-interval over worlds when pooled):

1. **Relative bias** ‖ḡ − g*‖ / ‖g*‖, ḡ the mean over datasets. The Monte Carlo floor is removed: ‖bias‖² is
   estimated as ‖ḡ − g*‖² − tr(Ĉ)/R. The unbiasedness check is the ratio ‖ḡ − g*‖² / (tr(Ĉ)/R), about 1 under no
   bias, together with a t-test of the bias along g*.
   - **G5's bias** is measured two more ways: paired, as the mean of G5 − G3 on the same datasets (G3 being unbiased),
     and semi-analytically, as the exact conditional bias of §2 averaged over each dataset's q̂.
2. **SNR** ‖E ĝ‖² / E‖ĝ − E ĝ‖² (unbiased plug-in estimates).
3. **Cosine** cos(ĝ, g*): its mean and median, and P(⟨ĝ, g*⟩ > 0) with a Wilson interval.
4. **Norm ratio** ‖ĝ‖ / ‖g*‖, and the total variance tr(Ĉ) (also as tr(Ĉ)/‖g*‖²).
5. **Weights** at θ: the mean, sd, quantiles (50/90/99/99.9%), maximum and ESS (Σw)²/Σw² of the raw ratios
   w_i = π_θ(a_i|x_i)/p_i on the logged rows. Also the population ESS share 1 / E_π0[w²].
6. **Minibatch noise** (component d of the question). On each dataset, M = 4 minibatch gradients of the study's
   default batch for N (1,024 at 25k) give the within-dataset minibatch variance. Comparing the full-data error
   E‖ĝ − g*‖² with the minibatch error E‖ĝ_B − g*‖² = E‖ĝ − g*‖² + E‖ĝ_B − ĝ‖² splits a single training step's
   gradient error into the logged-sample part and the minibatch part (G3 and G5).
7. **Reward-model error** on each dataset: the logging-weighted and the target-weighted RMSE of q̂ against q, exact
   over items.

**Replicates.** A timing pilot on three worlds, one per dataset, measures the cost per dataset. R is then fixed in an
addendum (§12.1) before any comparison is read:
- the largest R ≥ 100 that keeps Stage 8A within about 8 GPU-hours at 3 workers, capped at 400;
- the achieved Monte Carlo precision (the bias floor sqrt(tr Ĉ / R) / ‖g*‖) is reported, not tuned.

## 5. Logging support levels

The control is `--logger-greedy-share` (§1.6), the logging temperature.
- **Candidates:** shares {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98}.
- **Diagnostics, computed on the Stage 8A worlds before any learning:**
  - the logger's effective number of items per user, exp(entropy), prior-weighted;
  - the quantiles of the propensities of logged actions;
  - the population ESS share of the target/logging ratio π_target/π0 for two fixed targets that do not depend on the
    logger: the θ_value* policy and the mid-trajectory policy (as distributions over items);
  - the share of the target's mass on items with π0 < 1e-4.
- **Rule (fixed now):**
  - *current* is 0.8;
  - *poor support* is the candidate whose geometric-mean effective items (over the 15 worlds) is closest to 1/4 of
    the current value;
  - *better support* is the candidate closest to 4×.
  - These are propensity diagnostics only; no learner is run to choose them.
  - If the target-ratio ESS does not move in the same direction as the effective items, that is reported, and the
    levels stay as the rule gives them.
- **Same reward world:** for every chosen share, the click model (`env.scale`, `env.offset`, q on a fixed 2,000-user
  × all-item block) must equal the default world's bit for bit. Otherwise the study stops (§11).

## 6. Decomposition of the 25k training gap (Phase 5)

Arms: penalized likelihood (L), raw-DR OPC (O_raw, a new arm `shared_opc_raw`: `shared_opc` with weights `none`)
and harmonic-DR OPC (O_h, `shared_opc`). Per world, all in greedy value (primary) and stochastic value:

- **Population optimum of the arm's objective,** θ_obj*:
  - θ_log* for L;
  - θ_value* for O_raw (its population objective is V);
  - θ_harm* for O_h, the optimum of V_h with q̂_∞ (§2). q̂_∞ is the reward model's population fit: the
    logging-weighted logistic regression on the same features, exact over items, unpenalized, since sklearn's L2
    term vanishes as n grows.
- **Empirical-objective reference,** θ_emp: the arm's objective on the fixed 25k training set of the world, with the
  study's cross-fitted q̂ and λ = 0.
  - Optimized by full-batch Adam from the source: 3 learning rates {3e-4, 1e-3, 3e-3}, 3,000 steps each, cosine to 0.
  - The restart with the best empirical objective is kept; true value is never used to choose it.
  - It is called a reference, not a global optimum. Its final gradient norm is reported.
  - The true value along the kept path is recorded: the oracle-stopped best is a diagnostic of early stopping.
- **Standard training:** the 20 trials. **Best candidate:** the best of the 20 by true value (diagnostic).
  **Selected:** the native rule, and the common rule.

The exact identity, per world:

    V(θ_value*) − V(selected) = [V(θ_value*) − V(θ_obj*)]     objective (surrogate) mismatch, M
                              + [V(θ_obj*)  − V(θ_emp)]       finite-sample empirical-objective gap, E
                              + [V(θ_emp)   − V(best)]        optimization / minibatch / search gap, O
                              + [V(best)    − V(selected)]    selection gap, S

Each bracket is a difference of measured values. The sum is exact, but the parts are not orthogonal causes. E and O
can be negative: early stopping and λ can beat the converged empirical solution. E mixes finite-sample sampling
error with reward-model error; §7A separates them by refitting with the oracle q. "Finite search" is read from the
expected best of k trials, k = 1..20, resampled.

## 7. Interventions (Phase 6), at 25k, current support

Each adds an arm with the paired configurations (seed label `shared`), the same rows, the same λ grid and the same
selection diagnostics.

- **A. Reward model.**
  - `shared_opc_oq` and `shared_opc_raw_oq` train with the oracle q in the loss. This is diagnostic only: selection
    keeps the budget-fair full-data q̂, so the selection gap is comparable.
  - Their empirical-objective references are refit with the oracle q.
  - The existing misspecified reward model (`concat` features) is not run: the gradient benchmark's G2-against-G3
    contrast already brackets reward-model quality.
- **B. Minibatch noise.** The `shared_opc` and `shared_opc_raw` configurations are replayed with the same number of
  optimizer steps and the same per-step lr schedule (trial epochs × ⌈n/b⌉ steps, decayed at the same steps), only
  with larger batches:
  - a batch of 8,192 rows (`_b8192`), sampled without replacement within passes;
  - the full batch of 25,000 rows (`_bfull`).
  - The replay at the trial's own batch size must reproduce the standard trial (a test).
  - If the full-batch replay exceeds 12 GPU-hours on all 15 worlds, it runs on the ml and kuairand worlds only;
    the 8,192 replay runs on all 15.
- **C. Raw against harmonic DR.** `shared_opc_raw` against `shared_opc`, everything else equal.

## 8. The regime map (Phases 7–9)

- **Axes:**
  - N ∈ {5,000, 25,000, 100,000};
  - support ∈ {poor, current, better} (§5);
  - corruption ∈ {none (control), warp (near well specified), group, vector, combined}, all at high;
  - datasets ml, kuairand, anime;
  - the existing, budget-fair learned q̂ only.
- **Arms:** `shared_likelihood`, `shared_opc_raw`, `shared_opc`. Same model, regularizer, rows, split, 20 paired
  trials, and the native, common and best-of-20 selections.
- **Stage 8A** (gradient diagnosis): the first development seed for each dataset × corruption type, seed 100, 15
  worlds. N = 25k, current support, G0–G5 at the 4 states.
  - **Internal stop:** if the diagnostics do not behave as §2 predicts, the study stops there (§11).
- **Stage 8B** (small map): the 9 (N, support) cells on the same 15 worlds.
  - The 25k-current cell is rerun from scratch and must reproduce `run_shared_main_25k` for the two existing arms
    (§11).
  - The gradient benchmark runs in every cell, for G0, G2, G3, G4 and G5 at the source and mid-trajectory states,
    with R_regime replicates (fixed with R in §12.1).
- **Stage 8C:** the other 15 development worlds (seed 101), only if 8B shows a transition between likelihood and OPC
  along an axis.
- **Search space:** that of the shared study for every arm and cell.
  - The new arm `shared_opc_raw` gets one tuning round on the 18 tuning worlds (seeds 200/201 × ml, kuairand, anime ×
    warp, vector, combined high, 25k, current support), under the shared study's edge rule with at most one
    extension (`docs/shared_objective_study.md` §8).
  - No arm is tuned per cell.
  - Edge statistics of the development cells are reported as diagnostics, without extending.
- **Population optima per support level:** θ_log*, and θ_harm* with that level's q̂_∞. θ_value* does not depend on
  the logger.

## 9. The OPC-favorability margin (Phase 10)

Per world and cell, with V* = V(θ_value*):
- M_L = V* − V(θ_log*);
- T_L = V(θ_log*) − V(best_L);
- T_O = V* − V(best_O), which counts harmonic OPC's surrogate bias inside its training gap, as the directive does;
- S = V(best) − V(selected).

    F = M_L − (T_O − T_L) − (S_O − S_L)

F equals V(OPC selected) − V(likelihood selected) **exactly**, by construction:
V(sel_O) = V* − T_O − S_O and V(sel_L) = V* − M_L − T_L − S_L. The analysis checks this numerically per world (it
must hold to rounding). F's value is therefore its parts, not a prediction:
- it says which term decides the sign in each cell;
- the parts are related to gradient quality (§10).

F is reported by N, support, corruption and dataset, for raw and harmonic OPC, with native and common selection.

## 10. Relating OPC's success to gradient quality (Phase 11)

- **Per cell and world:** F and its parts against
  - the gradient SNR, mean cosine and P(⟨ĝ, g*⟩ > 0) of G3 and G5 at the source and mid states;
  - G5's bias;
  - the ESS and maximum ratio;
  - the reward-model error;
  - M_L.
- **Methods:** paired plots, Spearman correlations and stratified summaries only. No predictive model is fit.

## 11. Stop conditions (from the directive)

The study stops and reports, without repairing and continuing, if:
- the raw DR gradient (G2 or G3) is biased beyond its Monte Carlo floor;
- the exact population gradient disagrees with finite differences;
- the 25k-current cell fails to reproduce `run_shared_main_25k` materially, beyond exact equality up to the GPU's
  non-determinism;
- a bug would invalidate prior results;
- a support level changes the reward world (§5);
- any result would need a protected or held-out set.

## 12. Expectations, recorded before the results (not desired outcomes)

- E1. G1–G4 are unbiased; G5's bias is nonzero and grows from the source (h′ ≈ 0.9) toward sharper states.
- E2. With q̂, G3's noise exceeds G2's; G1's exceeds both. G4's noise is below G2's.
- E3. A single minibatch gradient's error is dominated by minibatch noise at 25k (about n/b times the logged-sample
  variance). Whether that matters for training is §7B's question, not this one.
- E4. Better support lowers the weights' tails at the mid and θ_value* states. More data raises SNR about linearly
  in N for the oracle-q estimators.

### 12.1 Replicate count and two implementation details (addendum, after the timing pilot, before any comparison)

**Timing pilot** (2026-10-09): the warp worlds of ml, kuairand and anime, seed 100, 2–3 datasets each. Only timings
were read.
- One replicate costs about 6–8 s (ml), 12–15 s (kuairand) and 8–11 s (anime) with two pilots running at once.
  4–8 s of that are the five reward-model fits (sklearn, CPU); the gradients of all five estimators at five states,
  the conditional bias and the minibatch gradients take the rest.
- The world and state fits take 1–7 minutes per world.

**R = 300 for Stage 8A.** With one process per dataset and 4 CPU threads each, the longest chain (kuairand) takes
about 5 × 300 × 14 s ≈ 5.8 h plus the world and state fits. That is inside the rule's ~8 h; R = 400 would bring it to
about 8.3 h.

**R_regime = 40 for the gradient cells of Stage 8B,** at the source, mid and mid_greedy states, all five estimators.
The 100k cells' reward-model fits take about four times longer, so 8 cells × 15 worlds × 40 replicates is about 25
worker-hours, run beside the training runs.

**An added state, `mid_greedy`.** On the value path the softmax value passes halfway within the first 3–6 Adam steps,
mostly by sharpening (the learned scale moves fastest). `mid_greedy` is the path's checkpoint halfway in **greedy**
value, where the ranking itself is half repaired. Both are reported; `mid` stays the pre-registered state.
- The path keeps a checkpoint at every step for its first 200 steps, then every 25: the pre-registered grid of 25 steps
  would have skipped the halfway point.
- Checkpoints are valued on a fixed 20,000-user prior sample to locate the halfway points; the chosen states are then
  valued exactly.

**The matched-steps replay (§7B)** is a batch sampler that draws full-size batches without replacement within passes,
so there is no short final batch.
- With full-size batches, a replay at the trial's own batch size is not the DataLoader's shuffle. The planned test
  "the replay at the trial's own batch size reproduces a standard trial" is therefore replaced by three checks:
  - the replay keeps the trial's configuration and its step count, epochs × ⌈n/b⌉ (tested);
  - the standard path is unchanged (its existing reproducibility tests);
  - the 25k cell reproduces the main grid (§11).
- The oracle-q training loss reads `fit_shared_regression_bundle(dataset, None, reward_model="oracle")`, tested equal
  to the simulator's q.

### 12.2 Calibrating the unbiasedness check (addendum, 2026-10-09, after reading five complete 8A worlds)

**Why it was added.** This addendum was written after the bias tables of the first five complete Stage 8A worlds had
been read: ml none, warp and group; kuairand none; anime none. In anime none, at the source, G2 and G3 had a bias
ratio of 4.1. §4 says only that the ratio is "about 1 under no bias". It does not give a threshold.

**The ratio's null distribution.**
- Under no bias, R‖ḡ − g*‖² is distributed as Σ_i λ_i χ²_1, where the λ_i are the eigenvalues of C.
- The ratio is therefore about χ²_ν / ν, with ν = (tr C)² / tr C² the noise's effective rank (Satterthwaite).
- At the source and mid states, ν ≈ 1.1–2.4: the noise has one dominant direction. With one dominant direction, a
  ratio of 4 has p ≈ 0.03.

**The check as now applied.**
- Every (world, state) cell of G1–G4 gets two p-values:
  - the Satterthwaite p-value of the ratio;
  - a split-sample Hotelling T² on 10 principal directions. The directions come from the odd replicates, and the
    test uses the even ones. When the directions are estimated from the same sample (P = 2,113 > R = 300), their
    variances are inflated and the test never rejects (tested).
- Holm's adjustment is applied over the cells of each estimator.
- The §11 stop fires if G2 or G3 has a Holm-adjusted p < 0.05 in either test.
- The pre-registered t-test along g* is still reported.
- Tests: with one dominant noise direction and no bias, both p-values hold their level (7.7% and about 5% at the 5%
  level over 300 simulations), and both detect a bias of 0.6 noise sd at R = 200.

**On the five worlds read.** The smallest Holm-adjusted p-values (Satterthwaite / Hotelling) were:

| Estimator | Smallest Holm-adjusted p |
|---|---|
| G1 | 1 / 0.077 |
| G2 | 0.84 / 1 |
| G3 | 0.82 / 1 |
| G4 | 0.18 / 0.45 |

The anime source ratio is shared by G1–G5 on the same datasets (2.5–4.1), consistent with one chance excursion
along the dominant noise direction.

## 13. Tests, written before results are trusted

1. the exact population gradient against central finite differences on selected coordinates, and against
   `calc_reward`'s value;
2. raw IPS's expectation by enumeration on a tiny world;
3. raw DR's with the oracle q;
4. raw DR's with an arbitrary fixed q̂;
5. harmonic DR: a nonzero bias with a wrong q̂, zero with the oracle q;
6. identical policy parameters across arms;
7. identical logged rows across paired arms;
8. full-batch against minibatch loss and gradient agreement, and the matched-steps replay reproducing a standard
   trial;
9. gradient-benchmark reproducibility;
10. resume and idempotence of the regime runs;
11. the decomposition and F accounting.

Targeted tests first, then the full CPU and GPU suites before every push.

## 14. Provenance

- Artifacts: `artifacts/full_study/opc_gradient_regime/`. Earlier studies' artifacts are not touched.
- Long runs use pinned worktrees, and every run goes into `artifacts/full_study/run_registry.csv` with:
  - the commit and the configuration;
  - the worlds, the support level, N and the q̂ condition;
  - the objective, the seeds and the artifact path.
