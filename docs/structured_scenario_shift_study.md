# Structured scenario shift with a matched low-rank adapter

Pre-registration, 2026-10-10. Committed before any method is trained on these worlds. The completed OPC gradient and
regime study (`docs/opc_gradient_regime_study.md`, frozen at fc83193) is referenced, never modified.

## 0. Question

The earlier simulators corrupted the learner's representation with global warp, group offsets and per-vector noise,
while the learner had one global affine correction. That deliberately mixed two things: the representation the learner
cannot express, and the target it must reach.

This study asks a cleaner question. Suppose the decision-relevant source-to-target shift is realistically structured
and lies inside the learner's correction family. Can predictive likelihood and direct counterfactual value optimization
still prefer different adaptations, because the target setting also changes response or calibration behavior?

Two kinds of shift are separated:
1. **Decision-relevant representation shift.** It changes which item is best: a low-rank change of the score.
2. **Response or calibration shift.** It changes absolute click probabilities through user-level sharpness and
   baseline engagement, but never the best item.

Both likelihood and OPC use exactly the same representation-adaptation family. The design does not aim at an OPC win.
Negative and null results are kept.

Not in this phase: adaptive harmonic shrinkage; importance-weight or support regularization; marginalized importance
sampling; new control variates or reward models; KL constraints; per-vector or group correction; the public
cross-scenario datasets; changes to the harmonic transform; reopening the gradient and regime study; protected or
confirmation worlds.

## 1. Source

**Embeddings.**
- BPR user and item factors of ml, kuairand and anime: K = 32; 6,038 × 3,533, 27,111 × 7,579 and 73,417 × 10,803 users
  × items.
- Used as they are: no centering, no popularity term.
- The user prior is the legacy one: exponential draws normalized, seeded by the world seed.

**The source score** is s_A(u, i) = x_uᵀ a_i. It is the learner's representation: the learner sees x and a, and the
logger ranks by s_A.

## 2. The target decision score: a matched low-rank shift

**The target score** is s_B(u, i) = x_uᵀ (I + Δ*) a_i, with Δ* = U* D* V*ᵀ of rank r = 4. There is no rank sweep.

**Directions (data aligned).**
- P_x: the top 8 eigenvectors of the prior-weighted user covariance of x.
- P_a: the top 8 eigenvectors of the item covariance of a, uniform over the catalog.
- U* = P_x Q_u and V* = P_a Q_v. Q_u and Q_v are 8 × 4 with orthonormal columns: the Q factor of a Gaussian 8 × 4
  draw, seeded by `derive_seed(seed, "structured_shift", "directions", side)`.
- U* and V* therefore have orthonormal columns inside the dominant 8-dimensional source subspaces.
- D* = γ diag(ε_1, …, ε_4): equal magnitudes γ and signs ε_k = ±1 drawn from the same seed.
- The construction is identical for every method in a world, and for every severity level of a dataset and seed. Only
  γ changes with the level.

**Severity.**
- The statistic is ρ(γ): the mean, over calibration users drawn from the prior, of the Pearson correlation across all
  items (uniform) between s_A(u, ·) and s_B(u, ·).
- The calibration users are the legacy calibration's 1,000 prior draws. A prior-drawn sample makes the mean
  prior-weighted.
- Targets: none ρ = 1.00 (γ = 0), moderate ρ = 0.90, strong ρ = 0.75.
- γ is the smallest γ ≥ 0 with ρ(γ) = target (a grid on log γ, then brentq), to ±0.005.
- If a target is unreachable, the world fails calibration and the study stops (§11).
- Calibration depends only on the source embeddings and the statistic, never on a method.

**Recorded per world:**
- ρ, the user-level score correlation, and its distribution;
- top-1 agreement: the prior-weighted share of users whose source and target top items coincide;
- top-10 overlap;
- the singular values of Δ*;
- ‖Δ*‖_F;
- the true gain available by adapting: the target greedy value of argmax s_B minus that of argmax s_A, under q_B.

## 3. Response heterogeneity that preserves the ranking

**The target click model.** q_B(u, i) = σ(κ α_u s̃_B(u, i) + c + β_u), where:
- s̃_B = (s_B − μ_B) / σ_B, with μ_B and σ_B the exact mean and sd of s_B over all user–item pairs (uniform), as the
  legacy standardization;
- α_u > 0 is user sharpness;
- β_u is user baseline engagement;
- κ and c are global.

Because α_u > 0 and β_u does not depend on the item, argmax_i q_B(u, i) = argmax_i s_B(u, i). Heterogeneity changes
prediction and calibration, never the correct ranking.

**Construction (smooth functions of the source representation).**
- **Directions.** w_α = P_x[:, :4] c_α, with c_α a random unit vector (seeded by
  `derive_seed(seed, "structured_shift", "response", "alpha")`). w_β = P_x[:, :4] c_β, with c_β drawn the same way
  under label "beta" and made orthogonal to c_α in the metric of the top-4 eigenvalues. The projections x_uᵀ w_α and
  x_uᵀ w_β are then uncorrelated under the prior.
- **Standardized projections.** z_α,u and z_β,u: prior-weighted mean 0 and sd 1, clipped at ±3, then re-centered and
  re-scaled to prior mean 0 and sd 1.
- **log α_u = σ_α z_α,u − log E_prior[exp(σ_α z_α)],** so the prior mean of α is 1 and sd_prior(log α) = σ_α.
- **β_u = σ_β z_β,u,** with prior mean 0.

**Levels** (pre-specified starting targets; **amended in §13.1 to half these values**):

| Level | sd(log α) = σ_α | sd(β) = σ_β |
|---|---|---|
| none | 0 | 0 |
| moderate | 0.25 | 0.5 |
| strong | 0.5 | 1.0 |

**Click calibration.** For every world, i.e. every (shift, response) pair, κ and c solve the legacy targets:
- the reference policy's CTR is 0.05. The reference policy is the spread-temperature softmax logger on the source
  vectors, covering half the catalog (`logging_spread` 0.5), with 2,048 sampled items per calibration user;
- the best item's click probability, averaged over the calibration users, is 0.30.

This is the legacy procedure with the truth replaced by q_B (brentq on κ, Newton on c).

**The one-time sanity adjustment.** A world's click distribution is pathological if any of these holds:
- more than 1% of users have a best-item q above 0.95;
- more than 5% of users have a logger-average q below 0.001;
- the calibration fails.

If any level is pathological before any method is trained, one adjustment is made: the projections are clipped at ±2
instead of ±3, for every level. It is recorded in an addendum (§13) and frozen. If the click distribution is still
pathological, the study stops.

**Recorded per world:**
- the distributions of α_u and β_u;
- the user-average click probability under the logger;
- the best-item click probability;
- κ and c;
- the logger's statistics;
- the source–target ranking agreement.

**The logger.**
- Softmax over s_A / T_log.
- The spread temperature T comes from the source scores, as the legacy one.
- T is sharpened until the logger earns 0.8 of its own greedy CTR under q_B (`logger_greedy_share` 0.8, the current
  support).

**Legacy equivalence.** With shift none and response none, q_B is the legacy click model of the clean world with
representation bias none and reference bias none:
- s_B = s_A;
- κ and c solve the legacy α and b;
- the logger is the legacy logger.

This is tested (§12).

**Implementation.** The truth is carried by `SyntheticBanditEnv` with augmented vectors. Users are
[(κ α_u / σ_B) (I + Δ*)ᵀ x_u, c + β_u − κ α_u μ_B / σ_B], items are [a_i, 1], with scale 1 and offset 0. Every
existing consumer of the click model (simulation, oracle reward model, greedy and stochastic values, population tools)
then reads q_B unchanged. The learner and the logger see only x and a.

## 4. The learner: a matched rank-4 adapter

**Adapter.** M_θ = I + U_θ V_θᵀ with U_θ, V_θ ∈ ℝ^{32×4}, and ŝ_θ(u, i) = x_uᵀ M_θ a_i.
- It is applied on the item side: a′_i = a_i + U_θ (V_θᵀ a_i), and the user side is the identity.
- U_θ V_θᵀ ranges over all matrices of rank ≤ 4. This is the family of U D Vᵀ, since D is absorbed into U.
- The truth is representable exactly: U_θ = U* D* and V_θ = V* give M_θ = I + Δ*. This is tested.

**Initialization** is the identity, i.e. the source policy: U_θ = 0. V_θ = V_0 is a random 32 × 4 matrix with
orthonormal columns, seeded by `derive_seed(seed, "lowrank_adapter", "init")`. It is the same for every arm and trial
of a world and is not aligned with V*.

**Policy and response.**
- The learned logit scale s = exp(30 θ_s) and the logger's temperature T are kept.
- The policy is π_θ(i|u) = softmax_i(s ŝ_θ / T), the existing `CFModel` form.

**Regularizer.**
- R(θ) = E_i ‖U_θ V_θᵀ a_i‖² / E_i ‖a_i‖², uniform over the catalog: the scale-normalized item displacement, the
  low-rank analogue of the shared-objective study's `SourceAnchor`.
- λ comes from the same grid {0, 0.001, 0.01, 0.1, 1} for every arm.

**Response heads.**
- **Ordinary likelihood:** q̂ = σ(s ŝ_θ / T + c), one global head. It is started at `fit_click_head`'s fit with the
  adapter at the identity, as the shared-objective study does.
  - With response none it is correctly specified: q_B = σ((κ/σ_B) s_B − κ μ_B/σ_B + c).
  - With response heterogeneity the adapter stays correctly specified but this head cannot represent α_u and β_u.
    That is **response-model (calibration) misspecification, not representation misspecification.**
- **Calibration-aware likelihood (diagnostic):**
  q̂ = σ(exp(w_αᵀ x_u) (s ŝ_θ / T + γ) + w_βᵀ x_u + c), with w_α, w_β ∈ ℝ³² and γ ∈ ℝ starting at 0, so it starts as
  the ordinary head.
  - It represents the truth exactly up to the clipped tail of z: log α_u is linear in x up to a constant (absorbed by
    s), and β_u − κ α_u μ_B / σ_B = w_βᵀ x + α_u · γ (absorbed by w_β, γ and c).
  - It shares the adapter, its initialization, the optimizer, the search and R(θ). R does not penalize the nuisance
    parameters.

## 5. Methods (arms)

All arms are new low-rank arms (`lr_*`), on the same rows, folds and paired configurations. The estimators are frozen
(§0): `harmonic:0.1`, the cross-fitted sklearn logistic q̂ on [x, a, x ⊙ a] with 5 user folds, the propensities and the
selection rules as in the current code. Only mechanical changes for the adapter are made, and each is tested against an
equivalent old configuration.

| Arm | Objective | Role |
|---|---|---|
| `lr_likelihood` | the click NLL, ordinary head | ordinary likelihood |
| `lr_likelihood_calib` | the click NLL, calibration-aware head | response-model diagnostic |
| `lr_opc` | harmonic DR, q̂ | current OPC |
| `lr_opc_raw` | raw DR, q̂ | diagnostic |
| `lr_opc_oq` | harmonic DR, the simulator's q | diagnostic ceiling |

## 6. Population objectives (computed before any finite-sample run)

Per world, exact over users (prior) and items, in the adapter class:
- **V(θ):** the stochastic value. Its optimum θ_value* is fitted by the existing value-path procedure (Adam, 3,000
  steps, 2,048 prior-drawn users per step, learning rates {0.003, 0.01, 0.03}, the best kept).
- **The logging-weighted Bernoulli NLL with the ordinary head:** θ_lik*. Its ranking is then deployed at the
  value-optimal scale, as in the regime study §3.
- **The same with the calibration-aware head:** θ_calib*.
- **Harmonic DR's population objective with q̂_∞,** the population fit of the reward model's logistic features under
  the logger: θ_harm*.
- **Raw DR's population objective, which is V itself.**
- **The truth adapter** (U* D*, V*): exact greedy value, the representability reference.

**Mismatches** (greedy CTR points; greedy value = each user's top item under q_B):
- M_L = V_g(θ_value*) − V_g(θ_lik*);
- M_calib = V_g(θ_value*) − V_g(θ_calib*);
- M_harm = V_g(θ_value*) − V_g(θ_harm*);
- also V_g(truth) − V_g(θ_value*), the class optimizer's gap.

**Sanity checks** (stop and diagnose, §11; 0.25 points is the warning threshold):
- **A.** With response none, M_L < 0.25 at every shift level.
- **B.** With response present, M_calib < 0.25.
- **C.** V_g(truth adapter) equals the target greedy value (representability, by construction and tested), and
  V_g(θ_value*) is within 0.25 of it (the optimizer reaches it).

**The first report.** As soon as the population optima are in: whether increasing response heterogeneity creates
prediction-optimal ≠ decision-optimal (M_L growing with response while M_calib ≈ 0) with the rank-4 adapter correctly
specified.

## 7. Finite-sample grids

**Tuning (only if the adapter needs a search-space adjustment).**
- Seeds 200 and 201 × 3 datasets × {(moderate, moderate), (strong, strong)} shift × response: 12 worlds.
- Arms `lr_likelihood`, `lr_likelihood_calib`, `lr_opc`, `lr_opc_raw`; 20 paired trials; 25k; the current support.
- The search is the shared-objective study's: paired random search (seed label "shared"), lr log-uniform 1e-4–2e-3,
  epochs 5–30, the batch schedule by N, λ from the grid.
- Its pre-registered edge rule decides any range extension, applied to every arm alike.
- Seeds 100 and 101 are never used for tuning.

**Stage 1 (the primary factorial).**
- 3 datasets × seeds {100, 101} × shift {none, moderate, strong} × response {none, moderate, strong}: 54 worlds.
- N = 25,000; logger greedy share 0.8; 20 paired trials; validation 20,000 rows; the study's split, cross-fitting and
  CTR level (0.05) as in the regime study; the five arms of §5.
- **Before the grid:** unit tests; a tiny smoke world; a few ml cells; the population identities; data identity across
  arms; training stability; measured runtime and an ETA.

**Stage 2 (user-gated shift; only after Stage 1 is validated).**
- s_B(u, i) = x_uᵀ a_i + g(x_u) x_uᵀ U* D* V*ᵀ a_i, with g(x) = σ(2 z_g(x)). z_g is the standardized prior projection
  on a direction w_g = P_x[:, :4] c_g (label "gate"), clipped at ±3.
- γ is recalibrated with the gate to the moderate target ρ = 0.90.
- The learner gets the matched gated family: ŝ_θ = xᵀ a + g_θ(x) xᵀ U_θ V_θᵀ a, with g_θ(x) = σ(v_θᵀ x + d_θ), v_θ = 0
  and d_θ = 0 at the start.
- Grid: moderate shift × response {none, moderate, strong} × 3 datasets × seeds {100, 101} × N = 25k × the current
  support: 18 worlds, with the population analysis first.

**Expansion gate (§12 of the directive).**
- Expand only selected cells. The criteria, any of:
  - (a) pooled M_L − M_calib ≥ 0.5 points in a (shift, response) cell;
  - (b) the sign of OPC − likelihood (native) changes across response levels at a fixed shift;
  - (c) M_harm < M_L in the pooled cell while OPC still loses at 25k, i.e. the right qualitative target with a
    finite-sample, support or selection problem;
  - (d) a clear standard vs calibration-aware likelihood difference at 25k: CI excluding 0.
- The expansion is N ∈ {5k, 25k, 100k} × logger greedy share ∈ {0.6, 0.8, 0.9} for those cells only, all datasets and
  both seeds.
- If instead fixed harmonic shrinkage is again the dominant obstacle (M_harm ≥ M_L in most worlds and the gap largest
  for harmonic OPC), the study ends after the population and 25k characterization.

## 8. Selection and reporting

**Selections.** Each arm's native rule:
- the likelihood arms use their own validation NLL;
- the OPC arms use the DR value of the current rule;
- also the common DR rule (the 95% DR lower bound of the greedy policy on the validation rows, clip 10), the oracle
  best of the 20 trials, and the mean over the trials.

**Metrics.** The primary metric is the true greedy value (CTR); the stochastic value is secondary.

**Per world, at least:**
- dataset, seed, shift and response levels, rank, N, support;
- ρ, top-1 agreement, top-10 overlap, ‖Δ*‖ and its singular values;
- the α and β statistics;
- M_L, M_harm and M_calib;
- each arm's gain under every rule;
- OPC − likelihood under every rule;
- the reward model's error under the logger and the target;
- the weight and ESS diagnostics;
- the correction's size;
- the training gap where the population optima allow.

Pooled summaries, per dataset and pooled, are secondary.

## 9. Hypotheses (not desired outcomes)

- **H1** (matched shift, homogeneous response): with response none, likelihood is hard to beat and M_L ≈ 0.
- **H2** (prediction nuisance): with action-independent response heterogeneity, M_L grows with the response level even
  though the correct ranking stays representable.
- **H3** (correctly specified nuisance): M_calib ≈ 0, removing most of that mismatch.
- **H4** (OPC opportunity): OPC − likelihood rises with M_L while overlap stays adequate.
- **H5** (the current harmonic limitation): harmonic OPC may not exploit it if M_harm stays large. If so, this is
  quantified, not fixed, in this study.
- **H6** (no outcome engineering): if likelihood stays strongest, that is the result. No level is changed afterwards.

## 10. Primary analyses

- **Population:** M_L, M_calib and M_harm per world and pooled, by shift × response.
- **25k:** OPC − likelihood under every rule, per world, per dataset and pooled, by shift × response. The decomposition
  M + (finite sample + optimization) + S per arm, as in the regime study §6.
- **Mechanism:** M_L against the response level; OPC − likelihood against M_L and M_harm; the support diagnostics.

## 11. Stop conditions

Stop and report before proceeding if:
- a sanity check of §6 fails (A, B or C);
- a world cannot be calibrated (an unreachable shift target or click target), or the click distribution stays
  pathological after the one adjustment;
- shift none with response none does not reproduce the legacy baseline (§12);
- a mechanical change fails its equivalence test, or the arms do not see identical rows;
- more than 5% of a cell's trials diverge;
- any result would need a protected or held-out set.

## 12. Tests (before results are trusted)

1. The low-rank truth and the learner's adapter are equivalent: the truth adapter reproduces s_B and its ranking
   exactly.
2. Ranking invariance: argmax q_B = argmax s_B for positive α_u and item-independent β_u.
3. Click calibration: the reference CTR and the best-item CTR hit their targets per level.
4. Shift none with response none reproduces the legacy world (no representation bias, reference bias none): q, the
   logger temperature and the vectors.
5. The augmented-vector environment equals the formula of §3.
6. The adapter starts at the source policy (identity), and its regularizer is 0 there.
7. The calibration-aware head starts as the ordinary head, and represents the truth when the nuisance is exact.
8. Population value and gradient: autograd against finite differences in the adapter class.
9. The arms see identical rows and configurations.
10. The legacy world, model and arms are unchanged: old commands still produce old worlds.

## 13. Addenda

### 13.1 The response levels and the click calibration (2026-10-10, before any method was trained; the user's decision)

**What failed.** World statistics only, no method trained:
- **Calibration.** With the reference CTR held at 5%, users with low β click rarely on every item, so the mean
  best-item CTR saturates as κ grows. On ml seed 100 the supremum is 72% / 49% / 29.3% for response none / moderate /
  strong, and on kuairand 46% / 30.0% / 20.8%. The 30% target is unreachable at strong response, and on kuairand even at
  moderate.
- **Pathology.** The §3 pathology rule (more than 1% of users with a best item above 0.95) also flagged worlds without
  heterogeneity: kuairand's homogeneous world, the legacy calibration itself, has 1.4%.
- **The one pre-registered adjustment** (clipping at ±2) fixed neither. A scan of 3 datasets × seeds 100, 101, 200, 201
  × 9 cells at both clips found:
  - strong response: pathological or failing calibration in most worlds (up to 13% near-deterministic best items, or
    up to 67% of users with a logger click probability below 0.001);
  - moderate: pathological in about a third of worlds.

  By §3 and §11 this was a stop.

**The options put to the user**, evaluated on the same worlds:
1. keep the σ targets and fix κ at the homogeneous world's value;
2. halve the σ targets and keep both CTR targets;
3. halve the σ targets and fix κ;
4. stop and redesign.

The user chose option 3.

**The definition as amended and frozen:**
- **Response levels:**

  | Level | sd(log α) | sd(β) |
  |---|---|---|
  | none | 0 | 0 |
  | moderate | 0.125 | 0.25 |
  | strong | 0.25 | 0.5 |

  The strong level equals the first moderate one. The construction of §3 is unchanged, including the clipping at ±3.
- **Click calibration.**
  - κ is the homogeneous world's κ for the same dataset, seed and shift: the legacy targets of 5% reference CTR and
    30% best-item CTR, as before.
  - With heterogeneity, c is re-solved so the reference CTR stays at 5%.
  - The best item's CTR becomes an outcome and is reported. With response none, κ and c are exactly the legacy ones.
- **Pathology, judged relative to the homogeneous world** (§3's role). A world is pathological if any of these holds:
  - its share of users with a best item above 0.95 exceeds the homogeneous world's by more than 10 points;
  - that share exceeds 15%;
  - more than 5% of users have a logger click probability below 0.001.

  A pathological world raises and stops the study (`utils/structured_shift.py`).

**The worlds under the amended definition.** 108 worlds (3 datasets × seeds 100, 101, 200, 201 × 9 cells): all
calibrate, none pathological.

| | Best-item CTR | Near-deterministic share | Excess over the homogeneous world |
|---|---|---|---|
| response none | 30.0% | at most 1.4% | — |
| response moderate | 27.0–30.6% | at most 4.8% | at most 3.5 points |
| response strong | 23.3–30.1% | at most 6.7% | at most 5.6 points |

At most 0.1% of users have a logger click probability below 0.001.

The shift levels hit their correlation targets exactly:

| Shift | Score correlation | Top-1 agreement | Available gain |
|---|---|---|---|
| moderate | 0.90 | 32% | 3.7–8.1 points |
| strong | 0.75 | 15% | 7.0–15.0 points |

**Frozen from here.** No level is changed after any method result.

### 13.2 Tuning outcome and the main grid's spaces (2026-10-10, before the main grid)

`run_ss_tune_s200` (d7fd60f, pinned worktree, 21:14–23:09, 2 workers): seeds 200 and 201 × 3 datasets ×
{(moderate, moderate), (strong, strong)}, the four tuned arms, 20 paired trials each. 960 trials, none diverged. In every
world the arms saw identical training and validation rows (row hashes) and trial k trained the same configuration in
every arm. Tables: `artifacts/full_study/structured_scenario_shift/tuning/`.

**The edge rule** (the shared-objective study's §8, as §7 pre-registers): mean gap of a trial's greedy gain below its
world's best trial of the same arm, in points, per learning-rate bin (configurations: 15 / 17 / 8).

| arm | 1e-4–3.2e-4 | 3.2e-4–1e-3 | 1e-3–2e-3 | margin of the top bin | extend |
|---|---|---|---|---|---|
| `lr_likelihood` | −2.83 | −1.67 | −0.66 | 1.01 | yes |
| `lr_likelihood_calib` | −4.11 | −2.67 | −1.08 | 1.59 | yes |
| `lr_opc` | −2.38 | −1.11 | −0.29 | 0.82 | yes |
| `lr_opc_raw` | −1.90 | −0.89 | −0.30 | 0.60 | yes |

- The rule fires for all four arms, on the learning rate at the top edge.
- Not extended: every arm's best epoch bin (18–24) and best λ (0.1 for the likelihood, 0.001 for the others) are
  interior. Batch size 512 (the low edge) and lr decay 0.9–1.0 (the high edge) are edges the protocol does not extend.

**Decision.**
- The main grid's learning rate for all five arms: **1e-4–6.3246e-3**, one half-decade past 2e-3. `lr_opc_oq`, not
  tuned, takes `lr_opc`'s space. One shared space keeps trial k's configuration identical across the arms. Every other
  range is as in §7.
- **Supplementary round** `run_ss_tune_s200_supp`, after the main grid (by the protocol an extension changes only the
  main grid's range and there is no second one, so the round cannot change the grid): the four arms on the same 12
  worlds, 20 new paired trials (seed tag `shared_supplement`), lr 1e-4–6.3246e-3. The rule is applied again to both
  rounds pooled. A second firing is reported and the boundary stays.

The tuning worlds' native gains over the logger (one replicate per world; a diagnostic, not a result): ordinary
likelihood 3.47, calibration-aware likelihood 4.98, harmonic OPC 3.08, raw DR 2.54 points, mean over the 12 worlds.

**Order of the main grid.** One dataset at a time (ml, kuairand, anime). Each starts only after its 18 population worlds
are finished with every check of §6 passing, so the population optima of a world always precede its finite-sample runs.
The tuning round ran beside the population analysis, on other seeds (200 and 201 against 100 and 101).

**Pre-launch checks (§7).**
- Unit tests: `tests/test_structured_shift.py` (12, including the gated truth adapter of Stage 2) and
  `tests/test_structured_shift_analysis.py` (4) pass.
- Smoke world (ml, seed 200, the five arms, 2 trials, `--save-policies`): each arm's saved policy reproduces its recorded
  greedy value exactly.
- Population identities: the population source's greedy value equals the trainer's logger greedy value (0.25004889 on
  the smoke world), so population and trial values can be subtracted. The 25k reward model rebuilt by
  `ss_population --finite-qhat` sees the trainer's rows (25,000 rows, 5,021 clicks).
- Data identity and training stability: the tuning round above.
- Runtime: 14–18 minutes per pair of worlds in tuning (four arms, two workers, beside the population jobs).

**A metadata note.** A structured world's `run_meta.json` carries the legacy `noise_mode`, `noise_axis` and
`noise_component` fields at their defaults (`representation_bias`, `both`, `combined`). `world_family`, `shift_level`
and `response_level` identify the world; the analysis reads only those.

### 13.3 Stage 1's expansion gate and the expansion's design (2026-10-11, after Stage 1, before the expansion)

**The gate** (§7), on the pooled cells (6 worlds each; `artifacts/full_study/structured_scenario_shift/compare/expansion_gate.csv`):

| cell (shift / response) | M_L − M_calib | (a) | OPC − likelihood, native | (b) | (c) | calibration-aware − ordinary likelihood, native [95% CI] | (d) | expand |
|---|---|---|---|---|---|---|---|---|
| none / none | −0.00 | no | −0.18 | yes | no | +0.00 [−0.00, +0.00] | no | **yes** |
| none / moderate | +0.64 | yes | +0.58 | yes | no | +0.71 [+0.44, +0.97] | yes | **yes** |
| none / strong | +1.46 | yes | +1.61 | yes | no | +1.80 [+0.98, +2.63] | yes | **yes** |
| moderate / none | −0.00 | no | −1.65 | yes | no | +0.08 [−0.04, +0.19] | no | **yes** |
| moderate / moderate | +0.93 | yes | −0.61 | yes | no | +0.82 [+0.70, +0.94] | yes | **yes** |
| moderate / strong | +2.00 | yes | +0.70 | yes | no | +1.98 [+1.27, +2.69] | yes | **yes** |
| strong / none | −0.00 | no | −2.95 | no | no | +0.56 [−0.15, +1.27] | no | no |
| strong / moderate | +0.93 | yes | −2.37 | no | no | +0.85 [−0.08, +1.78] | no | **yes** |
| strong / strong | +2.37 | yes | −0.75 | no | no | +2.14 [+1.18, +3.10] | yes | **yes** |

- **(b)** is read as a property of a shift level: every cell of a level whose OPC − likelihood changes sign across the
  response levels (none: −, +, +; moderate: −, −, +; strong: −, −, −). The reading is the analysis code's
  (`expansion_gate`, fea48d5), committed before Stage 1's first result.
- **The stop criterion** (fixed harmonic shrinkage dominant) does not hold. M_harm ≥ M_L holds in 36 of 54 worlds (28 of
  the 46 where the two differ by 0.25 points or more). But the largest pooled 25k gap to θ_value* (native) is raw DR's,
  3.98 points, against harmonic OPC 3.38, ordinary likelihood 2.75 and calibration-aware likelihood 1.76.
- **8 of the 9 cells are expanded**, all but strong shift with homogeneous response.

**The expansion** (fixed before any of it runs):
- 8 cells × 3 datasets × seeds 100 and 101: 48 worlds.
- Settings: N ∈ {5k, 25k, 100k} × logger greedy share ∈ {0.6, 0.8, 0.9}, without Stage 1's 25k at 0.8: 8 per world.
- The five arms, 20 paired trials, Stage 1's spaces (lr 1e-4–6.3246e-3, epochs 5–30, the λ grid), validation 20,000
  rows, selected policies saved. Run tag `run_ss_expand`. One dataset at a time (ml, kuairand, anime), share 0.8 first.
- **Population optima at shares 0.6 and 0.9.** θ_lik*, θ_calib* and θ_harm* depend on the logger; θ_value* and the
  truth do not. 96 world runs beside the grids. They split the gap to θ_value* into M and E+O at those supports and
  gate nothing; the gap itself needs only θ_value*. The anime grid waits for the anime population jobs (at most 3 anime
  processes).
- **Estimated cost:** 38,400 trials, about 20–25 hours, from Stage 1's trial times and the regime study's 5k / 25k / 100k
  time ratios (0.3 / 1 / 1.6–2.3).
- **Analysis:** OPC − likelihood and the other pairs under every rule per (N, share) and cell, per dataset and pooled;
  each arm's gap to θ_value*, decomposed where the population optima exist.
- **Order with the rest:** after Stage 2 and the supplementary tuning round of §13.2.

**Stage 2's pre-launch checks** (on tuning seed 200, before Stage 2's worlds):
- The gated arms (ml, `s-moderate.r-moderate.gated`, 2 trials, `--save-policies`): the 36-dimensional policies they
  export (the user vector [s x, s g(x) xU], the item vector [a, aV]) reproduce each recorded greedy value exactly.
- The gated population (the same world): checks B and C pass; the class optimizer's gap is 0.0007 points. On ml's
  Stage 2 worlds, the gate's clip touches at most 0.02% of users, so the gated truth adapter is exact for the rest.
- The 18 gated worlds calibrate: ρ = 0.900 at every response level, best-item CTR 24–30%, available gain 3.8–8.3
  points, none pathological (`scripts/ss_world_scan.py --gated`, `world_scan/world_scan_gated.csv`).

### 13.4 The expansion is held for review (2026-10-11 05:15, before any of it ran)

§13.3's mechanical reading of the gate selects 8 of the 9 cells: in effect the full N × support grid. The directive's
gate (its §12) asks for the opposite: not to repeat the full grid, not to expand every cell because compute is
available, and to expand only selected, mechanistically informative conditions; or, if fixed harmonic shrinkage is
again the dominant obstacle, to finish the population and 25k characterization and stop for review. Whether Stage 1
shows that obstacle is a judgment the §7 operationalization does not settle:
- **For it:** harmonic OPC's population target misses the value optimum by 1.3–1.5 points under moderate shift and
  4.4–4.9 under strong shift, at every response level. That miss is the largest part of its 25k gap under strong
  shift (4.6 of 6.7 points).
- **Against it:** the arm without shrinkage, raw DR, has the largest 25k gap (variance, not bias), so the criterion
  §13.3 tested does not hold.

None of the expansion has run. The supplementary tuning round of §13.2 and Stage 2 continue as planned. The scope of
the expansion (none, a selected subset, or §13.3's eight cells) is left to the review.

### 13.5 The supplementary tuning round's outcome (recorded 2026-10-13; the run finished 2026-10-11 07:09)

`run_ss_tune_s200_supp` (518dba7, pinned worktree, 06:42–07:09, 3 workers): the four tuned arms on the 12 tuning worlds,
20 new paired trials (seed tag `shared_supplement`), lr 1e-4–6.3246e-3. 960 trials, none diverged. Tables:
`tuning_round2/` (both rounds pooled, the decision) and `tuning_round2_supplement_only/` (the check).

| arm | best lr bin (configurations) | margin over 1e-3–3.2e-3 | best epoch bin | margin over 18–24 |
|---|---|---|---|---|
| `lr_likelihood` | 3.2e-3–6.3e-3 (6) | 0.17 | 25–30 | 0.22 |
| `lr_likelihood_calib` | 3.2e-3–6.3e-3 (6) | **0.31** | 25–30 | **0.62** |
| `lr_opc` | 3.2e-3–6.3e-3 (6) | 0.09 | 25–30 | **0.41** |
| `lr_opc_raw` | 3.2e-3–6.3e-3 (6) | 0.13 | 25–30 | **0.34** |

- The extended learning-rate bin is the best for every arm. Its margin exceeds 0.25 only for the calibration-aware
  likelihood. That is a second firing: by the protocol (§13.2) it is reported and the boundary stays.
- The epoch dimension now fires at the top (30) for three arms. It fires only in the check round, which by the
  protocol changes nothing.
- The main grid's ranges therefore stand. All arms share the same range, and all lean toward its upper edge in
  learning rate and epochs: every arm, the calibration-aware likelihood most, may be somewhat under-trained at 25k.
  The comparisons are paired within this common budget.

## 14. Provenance

- Artifacts: `artifacts/full_study/structured_scenario_shift/`, with a README.
- Every run goes into `artifacts/full_study/run_registry.csv`.
- Long runs use pinned worktrees.
- World configurations and fingerprints are recorded per world.

## 15. Results: Stage 1 (54 worlds; 2026-10-11)

Runs: `run_ss_population` (the population optima), `run_ss_main_25k` (the 25k grid). Tables and figures:
`artifacts/full_study/structured_scenario_shift/population/` and `compare/`. Greedy CTR points throughout; intervals are
95% CIs over worlds, paired by world. A cell pools 6 worlds (3 datasets × 2 seeds), so per-dataset cells (2 worlds) give
directions, not intervals.

### 15.1 Validation

- Every world passes checks A–C (§6). The truth adapter reaches the target greedy value. The value path reaches the
  truth within 0.004 points in every world, so the rank-4 class is correctly specified and its optimum is found.
- In every world the five arms saw identical training and validation rows (row hashes) and trial k trained the same
  configuration in every arm. None of the 5,400 trials diverged.
- The population source's greedy value equals the trainer's logger greedy value to 2e-7 points in all 54 worlds, so
  population and trial values can be subtracted (the decomposition below).
- The 25k cross-fitted reward model's error under θ_value*'s policy (`qhat25k_*`) is about 0.03–0.04 without shift and
  0.09–0.20 with shift, close to q̂_∞'s population error: under shift, the reward model's error is misspecification of
  its features, not sample size.

### 15.2 Population optima (the first report of §6)

| shift / response | M_L | M_calib | M_harm | available gain |
|---|---|---|---|---|
| none / none | 0.00 | 0.00 | 0.00 | 0.00 |
| none / moderate | 0.64 [0.53, 0.76] | 0.00 | 0.00 [0.00, 0.01] | 0.00 |
| none / strong | 1.47 [1.03, 1.90] | 0.00 | 0.01 [0.01, 0.02] | 0.00 |
| moderate / none | 0.00 | 0.00 | 1.52 [1.26, 1.79] | 5.99 |
| moderate / moderate | 0.93 [0.65, 1.21] | 0.00 | 1.44 [1.29, 1.59] | 5.59 |
| moderate / strong | 2.01 [1.44, 2.57] | 0.00 | 1.29 [1.12, 1.46] | 4.99 |
| strong / none | 0.00 | 0.00 | 4.86 [3.90, 5.81] | 12.02 |
| strong / moderate | 0.93 [0.73, 1.12] | 0.00 | 4.63 [3.83, 5.44] | 11.48 |
| strong / strong | 2.37 [1.53, 3.21] | 0.00 | 4.37 [3.73, 5.00] | 10.61 |

Per dataset (mean of the two seeds):

| shift / response | ml M_L | kuairand M_L | anime M_L | ml M_harm | kuairand M_harm | anime M_harm |
|---|---|---|---|---|---|---|
| none / moderate | 0.67 | 0.57 | 0.68 | 0.00 | 0.01 | 0.00 |
| none / strong | 1.59 | 1.12 | 1.69 | 0.01 | 0.01 | 0.01 |
| moderate / none | 0.00 | 0.00 | 0.00 | 1.53 | 1.55 | 1.48 |
| moderate / moderate | 1.09 | 0.89 | 0.82 | 1.50 | 1.35 | 1.45 |
| moderate / strong | 2.18 | 1.78 | 2.05 | 1.43 | 1.13 | 1.33 |
| strong / none | 0.00 | 0.00 | 0.00 | 5.10 | 4.54 | 4.93 |
| strong / moderate | 0.94 | 0.95 | 0.90 | 5.15 | 4.05 | 4.70 |
| strong / strong | 2.73 | 2.05 | 2.33 | 4.90 | 3.63 | 4.57 |

- **H1 holds in the population:** with homogeneous response, the likelihood's optimum is decision-optimal at every
  shift level (M_L = 0.00 in all 18 worlds).
- **H2:** response heterogeneity makes the prediction-optimal adapter differ from the decision-optimal one, with the
  rank-4 correction correctly specified. M_L grows with the response level at every shift level and on every dataset.
  Its size varies by seed (on ml, seed 101's M_L is about half of seed 100's).
- **H3:** the calibration-aware head removes it: |M_calib| < 0.01 in every world.
- **Harmonic DR's population target** (with q̂_∞) is on the value optimum without shift (M_harm ≤ 0.02) and misses it
  under shift, by 1.3–1.5 points (moderate) and 4.4–4.9 (strong), almost independently of the response level. Its miss
  is smaller than the likelihood's only without shift (with heterogeneity) and at moderate shift with strong response.

### 15.3 The 25k grid

**Harmonic OPC − ordinary likelihood, native rule** (per dataset: seed 100, seed 101):

| shift / response | ml | kuairand | anime | pooled (OPC higher / 6) |
|---|---|---|---|---|
| none / none | −0.19, −0.27 | −0.04, −0.09 | −0.23, −0.25 | −0.18 [−0.28, −0.08] (0/6) |
| none / moderate | +0.97, +0.59 | +0.49, +0.38 | +0.53, +0.53 | +0.58 [+0.37, +0.79] (6/6) |
| none / strong | +2.54, +1.26 | +0.93, +1.24 | +2.42, +1.25 | +1.61 [+0.89, +2.33] (6/6) |
| moderate / none | −2.28, −1.08 | −2.42, −0.98 | −2.05, −1.12 | −1.65 [−2.35, −0.96] (0/6) |
| moderate / moderate | −1.39, −0.59 | +0.22, −0.13 | −1.34, −0.40 | −0.61 [−1.29, +0.08] (1/6) |
| moderate / strong | +0.16, +0.25 | +1.25, +0.79 | +1.18, +0.54 | +0.70 [+0.21, +1.18] (6/6) |
| strong / none | −5.73, −1.54 | −3.58, −1.27 | −3.63, −1.94 | −2.95 [−4.73, −1.17] (0/6) |
| strong / moderate | −4.01, −0.73 | −1.93, −1.14 | −5.57, −0.82 | −2.37 [−4.45, −0.28] (0/6) |
| strong / strong | −1.45, −0.25 | −0.82, −0.41 | −1.71, +0.13 | −0.75 [−1.50, +0.00] (1/6) |

**Native gains over the logger, pooled** (the value optimum's gain is the ceiling):

| shift / response | value optimum | likelihood | calibration-aware | harmonic OPC | raw DR | harmonic OPC, oracle q |
|---|---|---|---|---|---|---|
| none / none | 0.00 | 0.00 | 0.00 | −0.18 | −0.15 | −0.16 |
| none / moderate | 0.00 | −0.72 | −0.02 | −0.14 | −0.16 | −0.15 |
| none / strong | 0.00 | −1.82 | −0.02 | −0.21 | −0.15 | −0.23 |
| moderate / none | 5.99 | +4.08 | +4.15 | +2.42 | +2.05 | +2.86 |
| moderate / moderate | 5.59 | +2.92 | +3.74 | +2.31 | +1.73 | +2.29 |
| moderate / strong | 4.99 | +1.31 | +3.28 | +2.00 | +1.44 | +2.40 |
| strong / none | 12.02 | +8.00 | +8.56 | +5.05 | +3.51 | +5.90 |
| strong / moderate | 11.48 | +6.91 | +7.76 | +4.54 | +3.48 | +6.34 |
| strong / strong | 10.61 | +5.23 | +7.37 | +4.48 | +3.10 | +5.62 |

**All 54 worlds, every rule** (a − b: mean [95% CI] (a higher / 54)):

| pair | native | best of 20 | mean of 20 |
|---|---|---|---|
| harmonic OPC − likelihood | −0.62 [−1.09, −0.16] (20) | −0.89 [−1.29, −0.49] (13) | +0.12 [−0.05, +0.29] (26) |
| calibration-aware − likelihood | +0.99 [+0.74, +1.25] (49) | +0.67 [+0.44, +0.90] (50) | +0.38 [+0.26, +0.50] (45) |
| harmonic OPC − calibration-aware | −1.62 [−2.05, −1.18] (1) | −1.56 [−2.02, −1.11] (2) | −0.26 [−0.33, −0.19] (9) |
| oracle q − harmonic OPC | +0.51 [+0.24, +0.78] (39) | +0.89 [+0.59, +1.18] (49) | +0.36 [+0.26, +0.47] (50) |
| raw DR − harmonic OPC | −0.60 [−0.80, −0.41] (9) | −0.55 [−0.74, −0.36] (15) | −0.33 [−0.42, −0.24] (0) |

Per dataset, harmonic OPC − likelihood pooled over the 18 cells (native): ml −0.76 (6/18), kuairand −0.42 (7/18),
anime −0.69 (7/18) (`compare/tables.md` §3).

- **H4 is supported in direction.** At every shift level OPC − likelihood rises with the response level, pooled and on
  every dataset but one row: anime under strong shift goes −2.79, −3.20, −0.79 (none, moderate, strong; 2 worlds
  each). OPC wins every world of three cells: no shift with moderate or strong heterogeneity, and moderate shift with
  strong heterogeneity.
- **Across the 54 worlds, OPC − likelihood follows the difference of the population mismatches.** Against M_L − M_harm:
  Spearman 0.87; OLS slope 0.61 [0.46, 0.76], intercept 0.04 (`compare/mechanism.csv`; best rule 0.83, mean rule 0.70).
  It is an association across designed worlds; the slope below 1 means the 25k differences are smaller than the
  population ones.
- **H5 holds:** under strong shift, harmonic OPC loses at every response level, its population target being 4.4–4.9
  points from the value optimum.
- **H6:** the strongest arm is the calibration-aware likelihood, the correctly specified response model. It beats
  harmonic OPC in 53 of 54 worlds.
- **Without shift**, the ordinary likelihood's 25k loss (0.7–1.8 points) is mostly selection: the best of its 20 trials
  loses 0.02, while the trial its validation NLL picks bends the ranking to fit the heterogeneity (§15.4).

### 15.4 Decomposition and diagnostics

Gap to θ_value* at 25k, native rule = M (the arm's population objective) + E+O (finite sample and optimization, to the
best of 20) + S (selection), mean over the 18 worlds of each shift level:

| shift | arm | gap | M | E+O | S |
|---|---|---|---|---|---|
| none | likelihood | 0.85 | 0.70 | −0.69 | 0.84 |
| none | calibration-aware | 0.01 | 0.00 | 0.00 | 0.01 |
| none | harmonic OPC | 0.18 | 0.01 | 0.01 | 0.16 |
| none | raw DR | 0.15 | 0.00 | 0.01 | 0.14 |
| none | oracle q | 0.18 | 0.00 | 0.02 | 0.16 |
| moderate | likelihood | 2.76 | 0.98 | 1.54 | 0.24 |
| moderate | calibration-aware | 1.80 | 0.00 | 1.66 | 0.14 |
| moderate | harmonic OPC | 3.28 | 1.42 | 1.75 | 0.11 |
| moderate | raw DR | 3.79 | 0.00 | 3.56 | 0.23 |
| moderate | oracle q | 3.01 | 0.00 | 2.65 | 0.36 |
| strong | likelihood | 4.66 | 1.10 | 3.46 | 0.10 |
| strong | calibration-aware | 3.47 | 0.00 | 3.41 | 0.06 |
| strong | harmonic OPC | 6.68 | 4.62 | 1.96 | 0.10 |
| strong | raw DR | 8.01 | 0.00 | 7.84 | 0.16 |
| strong | oracle q | 5.42 | 0.00 | 4.44 | 0.98 |

- **Harmonic OPC under shift: the target, not the learning.** Its E+O is the smallest of the five arms under strong
  shift (1.96) and level with the likelihoods' under moderate shift (1.75 against 1.54–1.66). Its objective misses by
  M = 1.4 (moderate) and 4.6 (strong). Its selection gap is small (≈ 0.1).
- **Removing the target's bias exposes variance.** Raw DR (no shrinkage) and harmonic DR with the oracle q both have
  M = 0, but larger E+O: 3.6 / 2.7 (moderate) and 7.8 / 4.4 (strong). The oracle q recovers part of harmonic OPC's
  loss (+0.51 pooled, native), not all of it.
- **The likelihoods' gap under shift is mostly E+O** (1.5–3.5), the finite-sample and optimization cost of fitting the
  click model, plus M_L where responses differ.
- **Support** (logger greedy share 0.8): the selected OPC policies keep a validation ESS of about 2,200–3,400 of 20,000
  rows (median by shift level) and change 16–57% of users' greedy picks against the logger's; the likelihood arms change
  31–82%. Support was not varied in Stage 1; that is what the held expansion would vary (§13.4).

### 15.5 Stage 2: the user-gated shift (18 worlds)

Runs: the `.gated` worlds of `run_ss_population` and `run_ss_stage2_25k`. Tables: `stage2_population/`,
`stage2_compare/`. Moderate shift (ρ = 0.90 with the gate), the matched gated adapter in every arm.

**Validation.** Every population check passes; the class optimizer's gap is at most 0.009 points and |M_calib| at most
0.008. Identical rows and paired configurations in all 18 worlds; none of the 1,800 trials diverged.

**Population** (pooled, 6 worlds per response level; the ungated moderate-shift row of §15.2 for comparison):

| response | M_L | M_calib | M_harm | available gain | q̂_∞ RMSE under θ_value* | ungated M_L | ungated M_harm |
|---|---|---|---|---|---|---|---|
| none | 0.00 | 0.00 | 2.38 [1.72, 3.04] | 6.38 | 0.149 | 0.00 | 1.52 |
| moderate | 1.05 [0.49, 1.60] | 0.00 | 2.37 [1.88, 2.85] | 6.33 | 0.151 | 0.93 | 1.44 |
| strong | 2.54 [1.08, 4.01] | 0.00 | 2.22 [1.77, 2.67] | 5.96 | 0.150 | 2.01 | 1.29 |

**Harmonic OPC − ordinary likelihood, native rule** (per dataset: seed 100, seed 101):

| response | ml | kuairand | anime | pooled, gated | pooled, ungated (§15.3) |
|---|---|---|---|---|---|
| none | −2.63, −1.16 | −3.80, −1.19 | −0.86, −1.05 | −1.78 [−3.02, −0.55] (0/6) | −1.65 [−2.35, −0.96] (0/6) |
| moderate | −2.04, −1.03 | −2.00, −0.21 | −0.92, −0.46 | −1.11 [−1.92, −0.31] (0/6) | −0.61 [−1.29, +0.08] (1/6) |
| strong | −0.45, −0.26 | −1.06, +1.99 | +1.88, +0.46 | +0.42 [−0.90, +1.75] (3/6) | +0.70 [+0.21, +1.18] (6/6) |

Best of 20: −1.87, −1.01, +0.34 (0, 1, 3 of 6); mean of 20: −0.42, +0.04, +0.78 (1, 4, 6 of 6).

Native gains over the logger, pooled (value optimum, likelihood, calibration-aware, harmonic OPC, raw DR, oracle q):
none 6.38 / +3.68 / +3.92 / +1.90 / +1.62 / +2.24; moderate 6.33 / +3.12 / +4.07 / +2.00 / +1.55 / +2.34; strong
5.96 / +1.77 / +3.75 / +2.19 / +1.57 / +2.51.

All 18 worlds, native (gated; ungated moderate shift in brackets): calibration-aware − likelihood +1.06 [+0.59, +1.53],
17/18 (+0.96, 16/18); harmonic OPC − calibration-aware −1.88 [−2.36, −1.41], 0/18 (−1.48, 0/18); oracle q − harmonic
OPC +0.33 [+0.18, +0.49], 16/18 (+0.27, 14/18); raw DR − harmonic OPC −0.45 [−0.64, −0.26], 1/18 (−0.51, 0/18).
Decomposition (all 18 worlds): harmonic OPC gap 4.19 = M 2.32 + E+O 1.80 + S 0.08; likelihood 3.37 = 1.20 + 2.08 +
0.10; calibration-aware 2.31 = 0.00 + 2.28 + 0.03; raw DR 4.64 = 0 + 4.53 + 0.11; oracle q 3.86 = 0 + 3.26 + 0.60.

- **The gate leaves the likelihood's mismatch about where it was and moves harmonic DR's target further away.** M_harm
  rises from 1.3–1.5 to 2.2–2.4 points. The reward model's logistic features on [x, a, x ⊙ a] describe a user-gated
  score worse than an ungated one (q̂_∞'s error under θ_value*'s policy 0.15 against 0.10).
- **Same direction, lower level.** OPC − likelihood still rises with the response level, but less of it is above 0:
  at strong heterogeneity OPC wins 3 of 6 worlds (two anime, one kuairand), against 6 of 6 ungated.
- **The mechanism across worlds holds:** OPC − likelihood against M_L − M_harm, Spearman 0.82, slope 0.78
  [0.51, 1.04], intercept 0.05 (`stage2_compare/mechanism.csv`).
- **The calibration-aware likelihood is again the strongest arm** (ahead of harmonic OPC in all 18 worlds).
