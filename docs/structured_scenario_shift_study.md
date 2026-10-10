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

## 14. Provenance

- Artifacts: `artifacts/full_study/structured_scenario_shift/`, with a README.
- Every run goes into `artifacts/full_study/run_registry.csv`.
- Long runs use pinned worktrees.
- World configurations and fingerprints are recorded per world.
