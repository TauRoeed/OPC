# Three training objectives on one global correction model (pre-registration 2026-10-06, results 2026-10-07)

*Development stage. Branch `representation-mismatch-research-next`, from the tested handoff commit `9445a12`.
§0–§10 were written and committed (2ce3385) before any tuning or main result of this study existed; §10.1 and §11
before the main grid ran. Results follow from §11. Nothing here is confirmatory.*

**Results in brief** (2026-10-07; development worlds, N = 25,000; §11–§20):
- **Population (§13).** Under warp the logging-likelihood optimum is the value optimum (−0.04 greedy points). Under
  group, vector and combined bias it ranks 1.23, 1.12 and 3.36 points below it, in 18 of 18 worlds. Weighting the
  likelihood toward the uniform action distribution moves its optimum further from value: 0.63 points below the
  logging-likelihood optimum on the biased worlds (clip 10: 0.41).
- **25k (§14).** Greedy gain over the logger on the 24 biased worlds, native selection:
  - penalized likelihood +2.98, OPC's objective +2.66, uniform-weighted likelihood +0.38 (clip 10) and −0.67 (raw);
  - the likelihood leads OPC by 0.32 [0.10, 0.54], mostly under warp (0.98, 6 of 6) and combined bias (0.32, 5 of 6);
    group and vector bias are ties.
- **Selection (§15)** does not explain the lead: under one common DR selector it is 0.22, and among the best of 20
  trials 0.28. OPC's candidates are better on average under group, vector and combined bias (18 of 18 worlds), but not
  at the top.
- **Decomposition (§16)**, points below the value optimum: likelihood 1.42 objective mismatch + 2.11 training gap +
  0.12 selection = 3.65; OPC 0 + 3.81 + 0.15 = 3.97. The global class itself stops 2.80 points short of the
  target-best ceiling (1.93, 3.51 and 5.64 under group, vector and combined bias).
- **Weights (§17).** The uniform-reference weights have an effective sample size of about 39 of the 25,000 rows. Raw
  weighting is unusable; clipping at 10 helps (+1.05 over raw) but stays 2.60 points below the plain likelihood.
- **Decision gate:** §19.

## 0. Question and scope

The CausE study compared a penalized likelihood learner with OPC, but the two arms also differed in selection rule,
optimizer, regularizer and search space (`docs/training_objectives_audit.md` §5). This study isolates the **training
objective**. The source representation, the data budget, the correction family, the policy parameterization, the
regularizer, the optimizer and the search budget are held fixed. Three objectives are compared:

1. penalized outcome likelihood under the logging distribution;
2. propensity-weighted penalized outcome likelihood toward a fixed uniform action distribution;
3. direct off-policy value optimization (OPC's objective).

They are compared at three levels:
- **Population optima** (infinite data, exact simulator expectations): does the objective point at the right solution?
- **Finite samples at N = 25,000**: does the objective produce better candidates?
- **Selection**: does a realistic rule pick them?

**Not in this phase**, by instruction:
- group, regional or per-user/per-item correction;
- new corruption mechanisms;
- RecoGym, new BLOB development, 100k studies.

The historical OPC arm and every earlier result stay as they are. The new arms are new methods next to it.

## 1. The shared correction model (read from the live code)

OPC's policy is `models/models.py` `CFModel`, built in `training/trainer_trials.py` `regression_trainer_trial` with
`make_policy_transform("linear", K)` for both sides, `temperature = dataset["policy_temperature"]` and
`learn_logit_scale=True`, as in the revalidated study configuration.

| component | live code | the shared model |
|---|---|---|
| source user vectors | `our_x` (the logger's biased vectors, frozen `nn.Embedding`) | x_u ∈ R^K, frozen |
| source item vectors | `our_a` (frozen) | a_j ∈ R^K, frozen |
| user correction | `GlobalLinearCorrection`: u′ = x + D_u x + b_u, D_u (K×K) and b_u (K) starting at 0 | the same |
| item correction | `GlobalLinearCorrection`: a′ = a + D_a a + b_a, starting at 0 | the same |
| score | `user_embedding @ actions_embedding.T` | g_θ(u, j) = ⟨u′_u, a′_j⟩ |
| logit scale | s = exp(30 θ_s) (`LOGIT_SCALE_SPEED` = 30), θ_s starting at 0 | the same |
| temperature | T = the logger's temperature (`_policy_temperature`) | the same, fixed |
| policy | softmax_j(s · g_θ(u, j) / T) (`CFModel.forward`) | π_θ(j \| u), the same |
| greedy policy | argmax_j of the exported vectors' dot product | argmax_j g_θ(u, j) |
| click model | none (OPC has no click model) | q_θ(u, j) = σ(s · g_θ(u, j) / T + c), one scalar intercept c, likelihood arms only |

- **Parameters:** θ = (D_u, b_u, D_a, b_a, θ_s) and, for the likelihood arms, c. That is 2K² + 2K + 1 = 2,113 (+1)
  at K = 32.
- **Same model for all objectives.** c is constant across items, so it never changes a ranking or the softmax policy;
  OPC does not use it.
- **One stochastic policy definition.** Every arm's stochastic policy is softmax(s · g / T). For the likelihood arms s
  is the click model's scale, which is flatter than a policy's, so their stochastic value is a convention. The
  primary metric is the greedy value.
- **Function class.** The click logit s·g/T + c spans the affine-bilinear class xᵀMa + wᵀa + vᵀx + c of the class
  oracles (`training/class_oracles.py`): M = (s/T)(I + D_u)ᵀ(I + D_a), w = (s/T)(I + D_a)ᵀb_u,
  v = (s/T)(I + D_u)ᵀb_a, for any invertible M. Rankings use M and w only.
- **Implementation.** `SharedCorrectionModel` subclasses `CFModel`. In policy mode it is `CFModel` exactly; in click
  mode it returns the click logits s·g/T + c. A unit test checks that identical correction parameters give identical
  logits, rankings and policies to `CFModel`, and identical OPC-loss gradients.
- **No popularity column:** these worlds have `pop_strength` 0.

## 2. One source-anchoring regularizer for every objective

    R(θ) = E_u ‖u′_u − x_u‖² / E_u ‖x_u‖²  +  E_j ‖a′_j − a_j‖² / E_j ‖a_j‖²

- **The expectations** are uniform over the catalog's users and items: source quantities only, no target data.
- **In closed form**, with the second moments S_x = E[x xᵀ] and means μ_x:
  E‖D x + b‖² = tr(D S_x Dᵀ) + 2 bᵀ D μ_x + ‖b‖². The same holds for items.
- **Why this form:**
  - R is the relative mean squared displacement of the user and item representations, 0 at the source.
  - It does not change when the source vectors are rescaled, so one λ grid means the same on every dataset.
  - It measures the actual movement whatever the split between D and b.
  - It leaves s and c free: neither moves the representation.
- **Every objective uses the same R and the same candidate strengths**, searched as one dimension of the paired
  search:

      λ ∈ {0, 0.001, 0.01, 0.1, 1}

  λ = 0 is unregularized; at λ = 0 the OPC arm is the current OPC objective exactly.
- **Scale check** (from existing results, not this study's): the historical OPC selections at 25k moved the
  representation by R ≈ 0.01–0.25. Their value gains over the logger are 1–6 CTR points (0.01–0.06), and the
  likelihood's gains in NLL are of the order of 0.01 nats per row. λR then ranges from negligible (λ = 0.001) to
  dominant (λ = 1) for every objective.
- **Fairness:** the grid is the same for all objectives, and each arm picks its λ by its own selection rule. The
  per-row data terms differ in scale (nats for the likelihoods, CTR for OPC), so equal λ does not mean equal anchoring.
  The grid spans two decades on either side of where λR matches either objective's gains, so each objective can reach
  its own useful range.

## 3. The objectives (four arms)

The logged training rows are (x_t, a_t, r_t, p_t), t = 1..n, with p_t = π0(a_t | x_t) the exact logging propensity
(the stored `pscore`; no uniform mix in these worlds). ℓ(r, q) = −r log q − (1 − r) log(1 − q) is the Bernoulli
negative log-likelihood, computed from the logit. P is the number of items.

| arm | training objective (minimized) | native selection |
|---|---|---|
| `shared_likelihood` | (1/n) Σ_t ℓ(r_t, q_θ(x_t, a_t)) + λ R(θ) | validation NLL |
| `shared_iw_likelihood` | (1/n) Σ_t w_t ℓ(r_t, q_θ(x_t, a_t)) + λ R(θ), w_t = 1 / (P p_t) | validation IW-NLL, the same weights |
| `shared_iw_likelihood_clip10` | the same with w_t = min(1 / (P p_t), 10) | validation IW-NLL, clipped weights |
| `shared_opc` | −(1/n) Σ_t [ Σ_j π_θ(j\|x_t) q̂⁻ᵏ⁽ᵗ⁾(x_t, j) + h(π_θ(a_t\|x_t) / p_t) (r_t − q̂⁻ᵏ⁽ᵗ⁾(x_t, a_t)) ] + λ R(θ) | 95% DR lower bound of π_θ |

- **Likelihood:** ordinary conditional click-model fitting under the logging distribution. It is not CausE.
- **IW likelihood:** still outcome-model learning, toward the fixed uniform action distribution μ(j|x) = 1/P:
  w_t = μ(a_t|x_t) / p_t. Under full support E_{π0}[w] = 1. It is a mean over n, not self-normalized; the target
  policy never enters.
  - The raw version is primary.
  - Clipping at 10 is the one pre-specified robustness variant. It is not tuned, per world or otherwise.
- **OPC:** the current validated objective (`DRPolicyLoss`, normalizer none, the direct gradient).
  - Weights: harmonic, h(w) = w / (0.9 + 0.1 w).
  - q̂⁻ᵏ: the logistic reward model on [x, a, x⊙a], cross-fitted over 5 user folds on the same n rows.
  - At λ = 0 it is the historical OPC arm's objective, parameterization and optimizer exactly; a test runs one trial
    both ways. Unlike the historical arm it also searches λ, so its trials are not the historical trials.
- **Minibatches:**
  - Each objective is a mean of per-row terms plus λR, so a minibatch gradient is unbiased for the full objective.
  - The short final batch is scaled by its share of a full batch (`training_utils.minibatch_loss`), R included.

**Training, identical for every arm:**
- The optimizer and schedule:
  - Adam without weight decay; the learning rate is multiplied by the decay factor each epoch.
  - The gradient norm is clipped at 1; the number of epochs is fixed per trial.
- The search:
  - Seeded random search over lr, epochs, lr decay, batch size and λ, 20 trials per world.
  - The four arms draw from one seed label, `shared`. Trial k is therefore the same configuration with the same
    seed and batch order in all four arms: a paired design.
- **Initialization:** the correction maps start at the identity (the logger's ranking).
  - OPC starts at s = 1, which is exactly the logger π0.
  - Each likelihood arm starts with its head (θ_s, c) at the minimizer of its own objective with the maps at the
    identity: a two-parameter weighted logistic regression of r_t on g₀(x_t, a_t)/T over the n training rows, with
    weights 1, raw or clipped. The slope is floored at 1e-3 if it comes out non-positive.
  - So every arm starts at the logger's ranking with its objective's best head, and the maps then move only to
    re-rank or re-score.
  - The head fit uses only the n training rows.

## 4. Data and budget fairness

- **One condition, one invocation.** All four arms run in the same condition and invocation of the study runner, so
  they share:
  - the world, the source vectors, and the N = 25,000 training rows;
  - the 20,000 validation rows and the exact propensities.
- **No arm gets more target data.**
- **The reward model** q̂ is fit on the same N rows (`--reward-data train`, cross-fitted for OPC's training loss, the
  full-data model for scoring).
  - The likelihood arms do not train with it.
  - They meet it only in the common DR selection diagnostic (§5), which every arm shares.
- **Data identity is checked in the outputs:**
  - Every shared arm's summary row records the training and validation row counts, click sums, the propensity sum
    and a hash of the logged (user, item, click) rows.
  - The analysis asserts that they are identical across the four arms of a world, and match the BLOB rows' click
    sums from the earlier runs of the same world.

## 5. Training objective and model selection, separated

For each arm and world, three trials are reported:

| selection | rule | deployable |
|---|---|---|
| native | the arm's own rule (§3: validation NLL, validation IW-NLL, or the DR lower bound of π_θ) | yes |
| common | the 95% DR lower bound of the trial's **greedy** policy on the validation rows, clip:10 weights, the full-data q̂ (`cause_trials._greedy_dr`, as in the CausE study) | yes |
| oracle-best | the highest true greedy value among the 20 trials | no: diagnosis only |

- **Why the greedy policy for the common rule:**
  - The primary metric is the greedy value.
  - The greedy policy is defined the same way for every arm, whatever its scale.
  - The rule uses the same rows and the same budget-fair q̂ for every arm.

## 6. Population objective oracles

The function class is the affine-bilinear class (§1). Every optimum uses exact expectations over items and clicks, and
a 20,000-user prior-weighted sample over users. The fitting is that of `training/class_oracles.py`:
- Adam with a cosine schedule, 3,000 steps of 2,048 users with every item in each step;
- three learning rates, keeping the best exact objective;
- the affine class warm-started from the bilinear fit of the same objective.

| optimum | objective | status |
|---|---|---|
| θ_log* | min Σ_u prior(u) Σ_j π0(j\|u) CE(q(u, j), σ(f(u, j))) | exists (`likelihood`, run_class_oracles_20261005) |
| θ_uniform* | min Σ_u prior(u) (1/P) Σ_j CE(q(u, j), σ(f(u, j))) | new objective `uniform_likelihood` |
| θ_clip10* | min Σ_u prior(u) Σ_j min(1/P, 10 π0(j\|u)) CE(…) (the clipped arm's population objective) | new objective `clip10_likelihood` |
| θ_value* | max Σ_u prior(u) Σ_j softmax(f(u, ·))_j q(u, j) | exists (`value`) |

For each optimum the analysis reports:
- the true greedy value, and the true stochastic value of the value oracle;
- L_log and L_uniform: for the value oracle, after its best two-parameter recalibration under L_log;
- correction size:
  - the relative distance of M from its best multiple of I, ‖M − (tr M/K) I‖ / ‖(tr M/K) I‖;
  - the item term's share, RMS(wᵀa) / RMS(xᵀMa);
- distances between the optima: the cosine distance between the normalized M, and the prior-weighted share of users
  whose greedy item agrees.

**Objective mismatch** is V(θ_value*) − V(θ_obj*) in greedy value. For OPC it is 0 by construction (its population
objective is the value). The harmonic transform and a misspecified q̂ make OPC's finite-sample objective a biased value
estimate; that bias lands in its training gap below.

## 7. Development experiment

- **Main grid:** N = 25,000; the 30 development worlds (ml, kuairand, anime × no bias, warp, group, vector, combined
  high × seeds 100, 101). Labelled DEVELOPMENT.
- **Settings:** the revalidated study configuration:
  - the paired random sampler, 20 trials, `--slim`;
  - validation 20,000, `--reward-data train --crossfit-folds 5`;
  - OPC's `dr`, `harmonic:0.1`, the direct gradient, `clip:10` selection weights and a learned logit scale.
- **The search space** (all four arms):
  - lr 1e-4–2e-3 (log-uniform), epochs 5–30, lr decay 0.8–1;
  - batch size from {512, 1024, 2048} (the 25k schedule);
  - λ from the grid of §2.
- **Reused comparator rows** (not rerun), from `artifacts/full_study/blob_prior_calibration/compare/table_conditions.csv`:
  - the logger, DM-only, historical OPC, CausE-cap-C at ρ = 0, and BLOB-Pnorm-NQ (P₀ = 10);
  - the existing class oracles.

## 8. Tuning protocol and edge rule (pre-registered)

- **Tuning worlds** (never the main worlds): the 18 tuning worlds of the BLOB calibration.
  - ml, kuairand, anime × warp, vector, combined high × seeds 200, 201.
  - All four arms, the search space of §7, 20 trials per world.
- **Edge rule** (the BLOB study's rule, `training/analyze_blob.py` `edge_check`), per arm and searched dimension:
  - Each trial's gap is its true greedy gain over the logger's greedy value minus that of its world's best trial of
    the same arm, in CTR points.
  - Compute the mean gap per value: lr per half-decade, epochs in 4 bins, lr decay in 2 bins; λ and batch size per
    value.
  - If the best value lies at an edge of the range and beats its neighbour by more than 0.25 points, that arm's range
    is extended one step past the edge. Steps: a half-decade for lr, 30 → 60 epochs at the top, λ = 10 at the top.
    λ = 0 is the bottom and cannot be extended.
  - An extension runs as a supplementary 20-trial tuning run of that arm on the same worlds, and the rule is applied
    again.
  - **No second extension:** a second firing is reported, and the boundary stays.
- **What an extension changes:** only that arm's range in the main grid. Its draws then map the same uniform numbers
  onto the extended range.
- **Nothing else is tuned.** The weights, the clip at 10, the λ grid itself, the trial budget and the selection rules
  are fixed here.

## 9. Hypotheses (recorded before the tuning and main results)

These are hypotheses, not desired outcomes.

- **H1. Global warp (well-specified for a global map).** Penalized likelihood is hard to beat there: propensity
  correction adds variance without much objective-mismatch benefit.
  - Expected: V(θ_value*) − V(θ_log*) small under warp, below 0.25 points pooled.
  - Expected: `shared_likelihood` not below `shared_opc` in the paired greedy contrast over the warp worlds.
- **H2. Misspecification.** Under group, vector and combined corruption with only global correction capacity, the
  logging-distribution likelihood optimum may differ from the value optimum.
  - Measured by V(θ_value*) − V(θ_log*) per bias, with a 95% CI over worlds.
- **H3. Middle ground.** Propensity-weighted likelihood may recover part of the gap.
  - In the population: V(θ_log*) ≤ V(θ_uniform*) ≤ V(θ_value*).
  - At 25k: `shared_iw_likelihood` between `shared_likelihood` and `shared_opc`.
- **H4. Variance.** Raw importance-weighted likelihood may suffer where logging support is weak (low ESS of w).
  Clipping may improve finite-sample behaviour at the cost of objective bias.
  - Measured by the training and selection gaps of raw vs clip10 against their population optima, and their relation
    to the ESS.
- **H5. No bias.** Every adaptation method is checked for unnecessary degradation when the source representation is
  already right: the greedy gain over the logger in the no-bias worlds.

## 10. Analysis plan

- **Primary metric:** the true greedy CTR. Its gain over the logger's greedy CTR is in CTR points.
- **Per world and arm:**
  - the stochastic value (OPC) or its convention (likelihood arms);
  - the native-, common- and oracle-selected trials' values, and the selection regrets;
  - the fraction of the value oracle's gap recovered, (V − V_logger) / (V(θ_value*) − V_logger);
  - the validation NLL; the correction size R(θ), the norms, s and c;
  - the share of users whose greedy item changes from the logger's, and the value gained and lost on those changes;
  - weight diagnostics on the training and validation rows: mean, variance, maximum, quantiles, ESS, and the clipped
    share at 10;
  - the data-identity checks.
- **Pooled:** means and 95% t-intervals over worlds per bias; paired differences between arms, with the count of
  worlds where each is ahead.
- **Decomposition**, per arm and bias, in greedy CTR points:

      V(θ_value*) − V(selected) = [V(θ_value*) − V(θ_obj*)] + [V(θ_obj*) − V(best trial)] + [V(best trial) − V(selected)]
                                    objective mismatch          finite-sample training gap    selection gap (native or common)

- **Decision gate** (§19): the eight questions of the phase directive, answered from these tables.

### 10.1 Reporting details fixed after the first tuning worlds, before the main grid (2026-10-07)

These were added while the tuning grid ran, after its first worlds had been read and before any main-grid result
existed. None changes a training objective, a search space or a selection rule.

- **One set of configurations per seed.** The paired sampler is seeded by (seed, seed label, train size), not by the
  world. Every world with the same seed therefore trains the same 20 configurations, in every arm. The historical OPC
  arm was searched the same way. The tuning grid thus holds 40 distinct configurations, each on 9 worlds, and the main
  grid 40, each on 15. The tuning tables report the distinct configurations behind every bin (`configs`).
- **The supplementary round, if the edge rule fires.** It draws new configurations with the seed tag
  `shared_supplement` (`--shared-seed-tag`), still paired across the shared arms, as BLOB's supplementary round did
  (`docs/blob_controlled_integration.md` §3.1). The rule is applied again to the first and supplementary trials pooled
  (the gap to each world's best trial over both rounds), and the supplementary trials alone are analyzed the same way
  as a check.
- **Candidates as a whole.** Besides the best of the 20 trials, the mean greedy gain of all 20 trials is reported per
  arm and world. The maximum of 20 favours an arm whose trials spread more.
- **The stochastic value.** It is reported for OPC (its deployable softmax policy) and for the reused comparators: OPC
  and DM-only with their own softmax, CausE-cap and BLOB-Pnorm with their DR-tempered softmax (as in their studies).
  The likelihood arms' softmax at the click model's scale is not reported as a policy value; their policy is the greedy
  one (§1).
- **The same world as the comparators.** Per world, the analysis checks that the logger's greedy value is the one in
  the reused comparator rows.
- **Population profile.** Per bias and optimum: the greedy gain, L_log, L_uniform and L_clip10 (the value optimum after
  its recalibration under L_log, §6), the correction size, and per pair of optima the two distances.

## 11. Tuning, first round (`run_shared_tune_s200`)

**Run.**
- Code 052462c from the main checkout. Its workers loaded the code at launch, and every condition records 052462c,
  not dirty.
- 2026-10-06 23:55 – 2026-10-07 01:38, 3 workers, beside the population-oracle jobs (§12).
- 18 tuning worlds × 4 arms × 20 paired trials = 1,440 trials; none diverged.
- Tables: `artifacts/full_study/shared_objective_study/tuning/` (`tuning_summary.csv`, `tuning_selected.csv`,
  `tuning_marginals.csv`, `tuning_edges.csv`, `tuning_weights.csv`, `tuning_trials_long.csv.gz`).

**Greedy gain over the logger on the tuning worlds** (CTR points, mean over the 18 worlds, 95% CI for the native
selection). These are tuning worlds, used only to fix the search spaces:

| arm | native | common DR | best of 20 | mean of 20 | native regret | common regret |
|---|---|---|---|---|---|---|
| `shared_likelihood` | +3.82 [+2.46, +5.17] | +3.61 | +3.86 | +2.58 | 0.05 | 0.25 |
| `shared_iw_likelihood` | −0.51 [−1.50, +0.49] | +0.38 | +0.53 | −0.43 | 1.04 | 0.15 |
| `shared_iw_likelihood_clip10` | +0.53 [+0.14, +0.92] | +0.62 | +0.77 | −0.47 | 0.24 | 0.15 |
| `shared_opc` | +3.27 [+2.01, +4.53] | +3.22 | +3.45 | +2.74 | 0.18 | 0.23 |

Native selection per bias (6 worlds each):

| arm | warp | vector | combined |
|---|---|---|---|
| `shared_likelihood` | +4.00 | +0.75 | +6.71 |
| `shared_iw_likelihood` | +0.28 | −1.82 | +0.02 |
| `shared_iw_likelihood_clip10` | +0.73 | −0.09 | +0.94 |
| `shared_opc` | +2.70 | +0.86 | +6.25 |

- The penalized likelihood leads here, mostly under warp (+4.00 against OPC's +2.70); under vector bias OPC is level.
- Both weighted likelihoods stay near the logger: even their best trials reach only +0.53 (raw) and +0.77 (clip 10).
  The raw arm's native selection (validation IW-NLL) loses 1.04 points to its best trial.
- On the training rows the uniform-reference weights have an ESS of about 0.2% of n (`tuning_weights.csv`).

**The edge rule** (§8). Per arm and dimension: the bin with the smallest mean gap to each world's best trial of the
arm, how many distinct configurations it rests on (§10.1), and its margin over the neighbouring bin, in CTR points:

| arm | dimension | best bin (configurations) | edge | margin over the neighbour | extend |
|---|---|---|---|---|---|
| `shared_likelihood` | lr | [0.001, 0.0032) (8) | high | 0.23 | no |
| `shared_likelihood` | epochs | 25–30 (7) | high | 0.08 | no |
| `shared_likelihood` | λ | 0.01 (9) | interior | — | no |
| `shared_likelihood` | batch size | 512 (10) | low | 0.58 | no (not extendable) |
| `shared_likelihood` | lr decay | [0.9, 1.0] (16) | high | 0.33 | no (not extendable) |
| `shared_iw_likelihood` | lr | [0.0001, 0.00032) (15) | low | 0.39 | **yes** |
| `shared_iw_likelihood` | epochs | 5–11 (11) | low | 0.02 | no (not extendable) |
| `shared_iw_likelihood` | λ | 0.1 (14) | interior | — | no |
| `shared_iw_likelihood` | batch size | 2,048 (14) | high | 0.61 | no (not extendable) |
| `shared_iw_likelihood` | lr decay | [0.8, 0.9) (24) | low | 1.01 | no (not extendable) |
| `shared_iw_likelihood_clip10` | lr | [0.0001, 0.00032) (15) | low | 0.50 | **yes** |
| `shared_iw_likelihood_clip10` | epochs | 5–11 (11) | low | 0.11 | no (not extendable) |
| `shared_iw_likelihood_clip10` | λ | 0.1 (14) | interior | — | no |
| `shared_iw_likelihood_clip10` | batch size | 2,048 (14) | high | 0.82 | no (not extendable) |
| `shared_iw_likelihood_clip10` | lr decay | [0.8, 0.9) (24) | low | 1.16 | no (not extendable) |
| `shared_opc` | lr | [0.001, 0.0032) (8) | high | 0.18 | no |
| `shared_opc` | epochs | 25–30 (7) | high | 0.07 | no |
| `shared_opc` | λ | 0.01 (9) | interior | — | no |
| `shared_opc` | batch size | 512 (10) | low | 0.22 | no (not extendable) |
| `shared_opc` | lr decay | [0.9, 1.0] (16) | high | 0.15 | no (not extendable) |

**Decision.**
- The rule fires for the two weighted likelihoods only, on the learning rate at the bottom. Their best bin,
  1e-4–3.2e-4, beats the next by 0.39 (raw) and 0.50 (clip 10) points.
- Their lr range is extended one half-decade down: **3.16e-5–2e-3**. All other ranges stay as in §7.
- Not extended:
  - The likelihood's and OPC's top lr bins lead by 0.23 and 0.18, inside the 0.25 margin; their top epoch bins by 0.08
    and 0.07.
  - Every arm's best λ is interior: 0.01 for the likelihood and OPC, 0.1 for the weighted likelihoods.
  - The weighted likelihoods' best epochs (5–11), batch size (2,048) and lr decay (0.8–0.9) lie at edges the protocol
    does not extend (§8).
- The only arm-specific range is thus the weighted likelihoods' learning rate. Their gradients carry the heavy-tailed
  weights, and in tuning their smallest learning rates did best.

### 11.1 Supplementary round and main-grid spaces (fixed before either runs)

- **Supplementary round** `run_shared_tune_s200_supp`:
  - the two weighted-likelihood arms only, on the same 18 worlds;
  - 20 new paired trials per world (seed tag `shared_supplement`), lr 3.16e-5–2e-3, every other dimension as in the
    first round.
  - The rule is applied again to the first and supplementary trials pooled, with the supplement alone as a check.
  - No second extension: a second firing is reported, and the boundary stays.
- **Order.** The round cannot change the main grid: by §8 an extension changes only that arm's main-grid range, and
  there is no second one. The main grid therefore runs first and the supplementary round after it.
- **Main-grid spaces** (§7 otherwise):
  - `shared_iw_likelihood` and `shared_iw_likelihood_clip10`: lr 3.16e-5–2e-3;
  - `shared_likelihood` and `shared_opc`: lr 1e-4–2e-3.
  - Trial k's draws map the same uniform numbers onto each arm's range (`--shared-arm-space`), so the four arms stay
    paired.

### 11.2 Supplementary round: the extension is not at an edge again

`run_shared_tune_s200_supp` (42522d1, pinned worktree, 2026-10-07 02:46–03:08, 3 workers, after the main grid): the two
weighted likelihoods on the 18 tuning worlds, 20 new paired trials each, lr 3.16e-5–2e-3; 720 trials, none diverged.
Tables: `tuning_round2/` (both rounds pooled, the decision) and `tuning_round2_supplement_only/` (the check).

| arm | lr bin | first and supplementary rounds pooled: mean gap (configurations) | supplement alone |
|---|---|---|---|
| `shared_iw_likelihood` | 3.2e-5–1e-4 (new) | −0.353 (10) | −0.328 (10) |
| | 1e-4–3.2e-4 | −0.357 (27) | −0.412 (12) |
| | 3.2e-4–1e-3 | −0.725 (29) | −0.761 (12) |
| | 1e-3–2e-3 | −2.223 (14) | −1.214 (6) |
| `shared_iw_likelihood_clip10` | 3.2e-5–1e-4 (new) | −0.500 (10) | −0.467 (10) |
| | 1e-4–3.2e-4 | −0.521 (27) | −0.659 (12) |
| | 3.2e-4–1e-3 | −0.922 (29) | −0.944 (12) |
| | 1e-3–2e-3 | −2.819 (14) | −1.602 (6) |

- **The rule does not fire again.** The new bottom bin leads the next by 0.004 (raw) and 0.020 (clip 10) points pooled,
  and by 0.08 and 0.19 in the supplement alone: inside the 0.25 margin. No other dimension fires.
- The lower learning rates did not help either arm: the supplementary trials' best of 20 is +0.53 (raw) and +0.76
  (clip 10) over the logger, as in the first round (+0.53, +0.77). Native selection gives −0.84 and +0.52.
- The main grid's spaces (§11.1) stand as run.

## 12. Runs and provenance

| run | code | what | when (2026-10-06/07) |
|---|---|---|---|
| `run_shared_tune_s200` | 052462c, main checkout | §11: 18 tuning worlds × 4 arms × 20 trials | 23:55–01:38, 3 workers |
| `run_shared_oracles_20261006` | 052462c, main checkout | the population optima of `uniform_likelihood` and `clip10_likelihood`, bilinear and affine-bilinear, on the 30 development worlds, policies saved; ml, kuairand and anime as separate jobs | 23:52–02:13; ml 1 h 2 min, kuairand 1 h 48 min, anime 2 h 21 min |
| `run_shared_main_25k` | 42522d1, pinned worktree | §7 and §11.1: 30 development worlds × 4 arms × 20 trials, selected policies saved | 01:41–02:46, 3 workers; none diverged |
| `run_shared_tune_s200_supp` | 42522d1, pinned worktree | §11.1–§11.2: the weighted likelihoods' supplementary tuning round, 18 worlds × 2 arms × 20 trials | 02:46–03:08, 3 workers; none diverged |

- The value and logging-likelihood optima are the existing `run_class_oracles_20261005` (d0a8a77), fitted the same way.
  The analysis checks every optimum's greedy value against its oracle run (to 1e-7).
- The comparator rows are reused, not rerun: `artifacts/full_study/blob_prior_calibration/compare/table_conditions.csv`.
- Tables and figures: `artifacts/full_study/shared_objective_study/` (`population/`, `tuning/`, `compare/`,
  `tuning_round2/`), built by `python -m training.analyze_shared_objectives` (§21).

## 13. Population optima: does each objective point at the right solution?

The four optima of §6 on the affine-bilinear class, per world: exact expectations over items and clicks, 20,000
prior-weighted users. Tables: `artifacts/full_study/shared_objective_study/population/` (`oracles.csv`,
`oracle_summary.csv`, `oracle_profile.csv`, `oracle_distances.csv`, `oracle_distance_summary.csv`).

**Objective mismatch** (greedy CTR points; mean [95% CI] over the 6 worlds of each bias; worlds where the first optimum
ranks higher):

| quantity | no bias | warp | group | vector | combined | biased (24) |
|---|---|---|---|---|---|---|
| V(θ_value*) − V(θ_log*) | +0.01 [−0.02, +0.03] (2/6) | −0.04 [−0.11, +0.03] (1/6) | **+1.23** [+0.82, +1.64] (6/6) | **+1.12** [+0.38, +1.87] (6/6) | **+3.36** [+2.06, +4.66] (6/6) | **+1.42** [+0.81, +2.02] (19/24) |
| V(θ_value*) − V(θ_uniform*) | −0.01 [−0.02, −0.01] (0/6) | −0.11 [−0.17, −0.05] (0/6) | +1.89 [+1.11, +2.67] (6/6) | +2.14 [+1.20, +3.09] (6/6) | +4.25 [+2.79, +5.70] (6/6) | +2.04 [+1.29, +2.80] (18/24) |
| V(θ_value*) − V(θ_clip10*) | −0.01 [−0.02, −0.01] (0/6) | −0.11 [−0.17, −0.05] (0/6) | +1.44 [+0.90, +1.98] (6/6) | +2.00 [+1.13, +2.87] (6/6) | +3.98 [+2.60, +5.36] (6/6) | +1.83 [+1.12, +2.54] (18/24) |
| V(θ_uniform*) − V(θ_log*) | +0.02 [−0.01, +0.05] (5/6) | +0.07 [+0.01, +0.12] (6/6) | **−0.66** [−1.23, −0.08] (1/6) | **−1.02** [−1.30, −0.74] (0/6) | **−0.89** [−1.23, −0.55] (0/6) | **−0.63** [−0.85, −0.40] (7/24) |
| V(θ_clip10*) − V(θ_log*) | +0.02 [−0.01, +0.05] (4/6) | +0.06 [+0.01, +0.12] (6/6) | −0.20 [−0.49, +0.08] (2/6) | −0.88 [−1.09, −0.66] (0/6) | −0.62 [−0.85, −0.40] (0/6) | −0.41 [−0.59, −0.23] (8/24) |

**Each optimum** (biased worlds pooled unless stated; the value optimum's losses after its best recalibration α f + β
under L_log, §6):

| optimum | greedy gain: warp / group / vector / combined | L_log | L_uniform | L_clip10 | M's deviation from a multiple of I | item term's spread / user term's |
|---|---|---|---|---|---|---|
| θ_log* | +7.19 / +2.72 / +2.31 / +8.61 | **0.3934** | 0.1173 | 0.0466 | 1.42 | 0.22 |
| θ_uniform* | +7.26 / +2.06 / +1.29 / +7.72 | 0.3963 | **0.1167** | 0.0464 | 1.21 | 0.24 |
| θ_clip10* | +7.26 / +2.52 / +1.43 / +7.99 | 0.3955 | 0.1167 | **0.0464** | 1.22 | 0.25 |
| θ_value* | +7.15 / +3.95 / +3.43 / +11.97 | 0.3978 | 0.1204 | 0.0472 | 1.55 | 0.08 |

- Without bias every optimum stays at the logger (greedy gains between −0.02 and 0). The value optimum's softmax gains
  +5.93 stochastic points there, all of it sharpening; +10.64 on the biased worlds.
- **Distances** (biased pooled): the cosine distance between the normalized M of θ_value* and θ_log* is 0.29; they give
  the same user the same top item for 54% of users (prior-weighted), θ_uniform* and θ_value* for 51%. Under warp the
  optima agree on 83–86% of users while their values differ by at most 0.11 points, so the items they disagree on are
  near-ties in value.
  Under combined bias they agree on 25–29%.

**What this says.**
- **Warp: no objective mismatch.** All four optima rank within 0.11 points of each other, and the value optimum is
  0.12 points [0.03, 0.21] below the target-best ceiling. The global class is well specified for a warp, so every
  objective's optimum finds the same repair.
- **Group, vector, combined: the likelihood optimum is not the value optimum.** With global capacity only, the
  logging-distribution likelihood optimum ranks 1.12–3.36 points below the value optimum, in all 18 of these worlds.
  The likelihood barely separates the two solutions: the value optimum is 0.0044 nats per row worse in L_log than
  θ_log* (1.1%), while it ranks 1.42 points better.
- **Uniform weighting moves the likelihood optimum away from value, not toward it.** θ_uniform* ranks 0.63 points
  below θ_log* on the biased worlds (lower in 17 of 24), and below it under each misspecified bias. Clipping at 10 keeps
  the weighted objective closer to the logging distribution and loses less (−0.41). The direction H3 expected holds only
  under warp and no bias, by at most 0.07 points.
- **A description, not a tested mechanism.** The logger's rows concentrate near the top of each user's biased ranking,
  where the greedy decision is made; the uniform distribution spreads the fit over the whole catalog. The optima also
  spend the class differently: the value optimum's item term varies 0.08 times as much as its user-specific scores
  (the likelihood optima: 0.22–0.25), and its K × K part deviates more from a multiple of I (1.55 against 1.21–1.42).
- **What a global map cannot reach.** The value optimum's own gap to the target-best ceiling is 0.12 (warp), 1.93
  (group), 3.51 (vector) and 5.64 (combined) points: it recovers 98%, 67%, 49% and 68% of the logger's gap to the
  ceiling (7.27, 5.89, 6.94 and 17.60 points).

## 14. Finite samples at 25k

`run_shared_main_25k`: 30 development worlds × 4 arms × 20 paired trials, none diverged. Tables and figures:
`artifacts/full_study/shared_objective_study/compare/` (`tables.md`, `table_summary.csv`, `table_conditions.csv`,
`table_paired.csv`, `table_per_dataset.csv`, `fig1_native_gain`, `fig1b_common_gain`, `fig2_decomposition`,
`fig3_population_mismatch`).

**Data identity** (§4), checked in all 30 worlds: the four arms trained and validated on the same rows (the hash of
the logged (user, item, click) rows, the click sums, the propensity sums, the row counts); those rows have the same
training and validation click sums as the BLOB run of the world; and the world's logger has the greedy value of the
reused comparator rows.

**Native-selected greedy gain over the logger** (CTR points; per bias the mean of 6 worlds, pooled the mean and 95% CI
over the 24 biased worlds):

| arm | no bias | warp | group | vector | combined | biased (24) |
|---|---|---|---|---|---|---|
| `shared_likelihood` | −0.02 | +4.02 | +1.41 | +0.84 | +5.64 | **+2.98** [+2.05, +3.90] |
| `shared_iw_likelihood` | −3.68 | +0.38 | −1.19 | −2.28 | +0.40 | **−0.67** [−1.61, +0.27] |
| `shared_iw_likelihood_clip10` | −0.11 | +0.76 | +0.04 | −0.05 | +0.75 | **+0.38** [+0.10, +0.65] |
| `shared_opc` | −0.57 | +3.04 | +1.35 | +0.91 | +5.32 | **+2.66** [+1.82, +3.49] |
| OPC (historical arm, reused) | −0.61 | +3.03 | +1.32 | +0.96 | +5.42 | +2.69 [+1.84, +3.53] |
| DM-only (reused) | −0.55 | +2.09 | +0.81 | +0.64 | +5.07 | +2.15 [+1.30, +3.01] |
| CausE-cap-C, ρ = 0 (reused) | −0.55 | +4.23 | +1.22 | +1.03 | +5.89 | +3.09 [+2.10, +4.08] |
| BLOB-Pnorm-NQ (reused) | −0.21 | +2.99 | +1.18 | +1.06 | +6.03 | +2.82 [+1.84, +3.79] |

**Paired contrasts** (a − b by world; mean [95% CI]; worlds where a is higher):

| a − b | warp | group | vector | combined | biased (24) |
|---|---|---|---|---|---|
| `shared_opc` − `shared_likelihood` | −0.98 [−1.49, −0.47] (0/6) | −0.06 [−0.18, +0.05] (1/6) | +0.07 [−0.23, +0.38] (4/6) | −0.32 [−0.67, +0.03] (1/6) | **−0.32** [−0.54, −0.10] (6/24) |
| `shared_iw_likelihood` − `shared_likelihood` | −3.64 (0/6) | −2.60 (0/6) | −3.12 (0/6) | −5.24 (0/6) | −3.65 [−4.70, −2.60] (0/24) |
| `shared_iw_likelihood_clip10` − `shared_likelihood` | −3.26 (0/6) | −1.37 (0/6) | −0.89 (0/6) | −4.88 (0/6) | −2.60 [−3.41, −1.79] (0/24) |
| `shared_iw_likelihood_clip10` − `shared_iw_likelihood` | +0.39 (6/6) | +1.23 (6/6) | +2.23 (5/6) | +0.36 (4/6) | +1.05 [+0.17, +1.93] (21/24) |
| `shared_opc` − `shared_iw_likelihood_clip10` | +2.28 (6/6) | +1.31 (6/6) | +0.96 (6/6) | +4.57 (6/6) | +2.28 [+1.54, +3.02] (24/24) |
| `shared_opc` − OPC (historical) | +0.01 (3/6) | +0.03 (3/6) | −0.05 (2/6) | −0.10 (2/6) | −0.03 [−0.09, +0.03] (10/24) |
| `shared_likelihood` − CausE-cap-C | −0.21 (2/6) | +0.19 (5/6) | −0.19 (1/6) | −0.25 (1/6) | −0.12 [−0.27, +0.04] (9/24) |
| `shared_opc` − CausE-cap-C | −1.18 (0/6) | +0.13 (5/6) | −0.12 (2/6) | −0.57 (1/6) | −0.44 [−0.69, −0.19] (8/24) |

- **The penalized likelihood is the strongest shared arm**, by 0.32 points over OPC on the biased worlds. Its lead is
  the warp worlds' (0.98, 6 of 6); under group and vector bias the two are level, under combined bias the likelihood
  leads by 0.32 (5 of 6).
- **The comparison isolates the objective.** CausE-cap at ρ = 0 led the historical OPC arm by 0.41 points
  (`docs/cause_fair_comparison_25k.md`), with a different objective form, selection rule, optimizer and search. With
  all of these held fixed, the plain likelihood leads the OPC objective by 0.32 [0.10, 0.54]. The shared likelihood is
  level with CausE-cap (−0.12 [−0.27, +0.04]), and the shared OPC arm, with λ searched, equals the historical OPC arm
  (−0.03 [−0.09, +0.03]).
- **Both weighted likelihoods fail at 25k.** The raw arm loses 0.67 points to the logger (24 of 24 worlds below the
  plain likelihood); clipping at 10 makes it better than raw in 21 of 24 worlds, but it still recovers only +0.38.
- **Per dataset** (biased worlds; `table_per_dataset.csv`) the order likelihood ≥ OPC ≫ clip 10 > raw holds on ml
  (+3.35, +3.03, +0.74, −1.27), kuairand (+3.11, +2.72, +0.16, −0.65) and anime (+2.48, +2.21, +0.23, −0.10).
- **Stochastic value.** OPC's selected softmax gains +6.71 [+6.13, +7.28] points over the logger's stochastic value,
  level with the historical arm (+6.75) and below CausE-cap's DR-tempered softmax (+7.13). The likelihood arms'
  softmax is not a deployable policy (§10.1).
- **Decisions changed** (biased, native): the likelihood moves 55% of users to another top item (value gained there
  +3.69, lost −0.71 points), OPC 56% (+3.49, −0.83), clip 10 15% (+0.72, −0.34), raw 27% (+0.73, −1.40).

## 15. Native selection against common selection

Greedy gain over the logger on the 24 biased worlds, per selection rule, and the regret against the best of the 20
trials:

| arm | native | common DR | best of 20 | mean of 20 | native regret | common regret |
|---|---|---|---|---|---|---|
| `shared_likelihood` | +2.98 | +2.89 | +3.09 [+2.18, +4.01] | +2.04 | **0.12** | 0.20 |
| `shared_iw_likelihood` | −0.67 | +0.26 | +0.41 [+0.19, +0.63] | −0.26 | **1.08** | 0.15 |
| `shared_iw_likelihood_clip10` | +0.38 | +0.60 | +0.69 [+0.36, +1.03] | −0.13 | 0.32 | 0.09 |
| `shared_opc` | +2.66 | +2.67 | +2.81 [+2.00, +3.62] | +2.25 | 0.15 | 0.14 |

`shared_opc` − `shared_likelihood`, by rule (biased; worlds where OPC is higher):

| rule | warp | group | vector | combined | biased (24) |
|---|---|---|---|---|---|
| native | −0.98 (0/6) | −0.06 (1/6) | +0.07 (4/6) | −0.32 (1/6) | −0.32 [−0.54, −0.10] (6/24) |
| common DR | −0.87 (0/6) | −0.14 (1/6) | +0.27 (6/6) | −0.15 (3/6) | −0.22 [−0.43, −0.01] (10/24) |
| best of 20 | −1.03 (0/6) | +0.00 (4/6) | +0.16 (5/6) | −0.27 (1/6) | −0.28 [−0.50, −0.07] (10/24) |
| mean of 20 | −0.22 (1/6) | +0.13 (6/6) | +0.22 (6/6) | +0.69 (6/6) | +0.21 [+0.05, +0.36] (19/24) |

- **Selection is not what separates the likelihood from OPC.** Under one common rule the likelihood still leads by
  0.22 points; its best trials lead by 0.28. NLL selection costs the likelihood 0.12 points and the DR lower bound
  costs OPC 0.15: both native rules pick close to the best of 20.
- **OPC's candidates are better on average where the objectives differ, not at the top.** Under group, vector and
  combined bias the mean of OPC's 20 trials beats the likelihood's in 18 of 18 worlds (by 0.13–0.69 points), but the
  best of 20 does not (+0.00, +0.16, −0.27). The likelihood's trials spread more (its mean of 20 is 1.05 points below
  its best; OPC's 0.56), and its own rule finds its good trials.
- **Native selection fails the raw weighted likelihood.** Its validation IW-NLL rests on about 30 effective rows
  (§17). It loses 1.08 points to the best of 20, and in one no-bias world (anime, seed 100) it picked a trial that
  moved 96% of users and lost 19.0 points, while the best trial lost 0.001. The common DR rule cuts its regret to 0.15
  and turns −0.67 into +0.26.
- Without bias the common rule costs the likelihood 0.20 points (−0.22 against −0.02 native): the DR lower bound of a
  greedy policy prefers trials that change decisions, which no-bias worlds do not reward.

## 16. Where each objective loses value

V(θ_value*) − V(native) = objective mismatch + training gap + selection gap (greedy CTR points; mean [95% CI] over
worlds; the parts add up per world):

| arm | part | warp | group | vector | combined | biased (24) |
|---|---|---|---|---|---|---|
| `shared_likelihood` | objective mismatch | −0.04 | +1.23 | +1.12 | +3.36 | +1.42 [+0.81, +2.02] |
| | training gap | +2.97 | +1.24 | +1.29 | +2.94 | +2.11 [+1.62, +2.61] |
| | selection gap | +0.20 | +0.06 | +0.18 | +0.03 | +0.12 [+0.02, +0.21] |
| | **total** | +3.13 | +2.54 | +2.59 | +6.33 | **+3.65** [+2.78, +4.51] |
| `shared_iw_likelihood` | objective mismatch | −0.11 | +1.89 | +2.14 | +4.25 | +2.04 [+1.29, +2.80] |
| | training gap | +6.65 | +1.95 | +1.22 | +6.87 | +4.17 [+2.96, +5.38] |
| | selection gap | +0.23 | +1.30 | +2.35 | +0.45 | +1.08 [+0.19, +1.98] |
| | **total** | +6.77 | +5.14 | +5.71 | +11.57 | **+7.30** [+5.70, +8.90] |
| `shared_iw_likelihood_clip10` | objective mismatch | −0.11 | +1.44 | +2.00 | +3.98 | +1.83 [+1.12, +2.54] |
| | training gap | +6.20 | +2.34 | +1.35 | +6.52 | +4.10 [+3.00, +5.21] |
| | selection gap | +0.29 | +0.14 | +0.13 | +0.72 | +0.32 [+0.14, +0.49] |
| | **total** | +6.39 | +3.91 | +3.48 | +11.21 | **+6.25** [+4.79, +7.70] |
| `shared_opc` | objective mismatch | 0 | 0 | 0 | 0 | 0 |
| | training gap | +3.96 | +2.48 | +2.26 | +6.57 | +3.81 [+2.90, +4.73] |
| | selection gap | +0.15 | +0.13 | +0.26 | +0.08 | +0.15 [+0.09, +0.22] |
| | **total** | +4.11 | +2.60 | +2.52 | +6.65 | **+3.97** [+3.07, +4.87] |

Beyond the global class, the value optimum is itself 0.12 (warp), 1.93 (group), 3.51 (vector) and 5.64 (combined)
points below the target-best ceiling: 2.80 [1.82, 3.78] on the biased worlds (§13).

- **OPC trades a population advantage for a training gap.** It has no objective mismatch by construction, but its
  best trial stays 3.81 points below the value optimum, against the likelihood's 2.11 below its own optimum. The
  likelihood's mismatch (1.42) and OPC's extra training gap (1.70) nearly cancel; the totals are 3.65 and 3.97.
- **The balance depends on the bias.** Under warp there is no mismatch to exploit and OPC's training gap is 1 point
  larger: the likelihood wins. Under group and vector bias the likelihood's mismatch (1.2, 1.1) is about what OPC
  loses in training (1.2, 1.0 more): a tie. Under combined bias the mismatch is largest (3.4), but so is OPC's training
  gap (6.6 against 2.9).
- **The likelihood's own optimum caps it.** Its best trials reach 2.11 points below θ_log*, so 3.5 points below θ_value*
  on average. Under combined bias half of its shortfall (3.36 of 6.33 points) is objective mismatch.
- **The weighted likelihoods lose everywhere.** Their population optima are further from the value than θ_log* (§13),
  their training gaps are about twice the likelihood's (4.1–4.2), and the raw arm's native selection adds 1.1 points.
- **Share of the value optimum's gain recovered** (biased, native): likelihood 0.41, OPC 0.36, CausE-cap 0.42,
  BLOB-Pnorm 0.39, historical OPC 0.37, DM-only 0.26, clip 10 0.04, raw −0.17.

## 17. The uniform-reference weights

On the 25,000 training rows of each world (`tables.md` §13, per dataset and bias):

- w = 1/(P p) has a median of 0.008 and a mean of 0.81 (1 in expectation under full support; 0.31–2.95 across worlds,
  a heavy-tailed sample mean). The 99.9% quantile is 83 on average and the maximum 5,682 (342–53,837).
- **Effective sample size** (Σw)²/Σw²: 39 rows on average (0.16% of n; 2 to 156 rows). On the 20,000 validation rows
  the IW-NLL that selects the raw arm rests on 0.15% of them.
- **Clipping at 10** touches 0.74% of the rows but removes 67% of the weight mass; the clipped weights' ESS is 1,193
  rows (4.8%). The clipped objective weights the rarely logged pairs as the logging distribution does (×10) and the
  others uniformly.
- **The ESS does not sort the worlds.** Across the 24 biased worlds the raw arm's training gap is not clearly related
  to its ESS (Spearman ρ = −0.21, p = 0.32); every world's ESS is tiny. For clip 10 the native gain rises somewhat
  with the clipped ESS (ρ = +0.42, p = 0.04).
- The likelihood arms' starting heads (§3) put the click scale at s = 0.17 (plain), 0.25 (raw) and 0.18 (clip 10). In
  one world (kuairand, warp, seed 101; ESS 11 rows) the raw arm's weighted head fit gave a non-positive slope, which
  the floor held at 1e-3; that arm stayed at the logger there (−0.003).
- **The arms barely move with clipped weights.** The native correction size R(θ) is 0.005 for clip 10, against 0.095
  (likelihood) and 0.066 (OPC). The raw arm moves 0.059: it changes 27% of decisions, gaining 0.73 points on some and
  losing 1.40 on others.

## 18. The hypotheses (§9), against the results

- **H1. Supported.** Under warp, V(θ_value*) − V(θ_log*) = −0.04 [−0.11, +0.03], below the 0.25-point bar, and
  `shared_likelihood` is not below `shared_opc`: it is 0.98 points above, in 6 of 6 worlds.
- **H2. Supported.** The logging-likelihood and value optima differ under group (+1.23), vector (+1.12) and combined
  (+3.36) bias, each in 6 of 6 worlds with the 95% CI above zero.
- **H3. Not supported, in either part.** In the population θ_uniform* ranks below θ_log* (−0.63, lower under each
  misspecified bias), not between θ_log* and θ_value*. At 25k `shared_iw_likelihood` is below both other arms in every
  biased world.
- **H4. Supported for raw weights; clipping does not rescue the objective.** The raw weights have an ESS of 0.16% of
  the rows, and the raw arm has the largest selection gap (1.08). Clipping improves the finite-sample result in 21 of
  24 worlds (+1.05). Unlike H4's trade-off it costs no value: θ_clip10* is closer to θ_value* than θ_uniform* is
  (1.83 against 2.04 points below). But its training gap is as large as raw's (4.10 against 4.17), and its gain stays
  at +0.38.
- **H5. Partly supported.** Without bias the likelihood stays at the logger (−0.02; 6% of decisions change), clip 10
  near it (−0.11), OPC loses 0.57 [0.18, 0.96] (30% of decisions change; its native rule costs 0.43 of it), and the
  raw arm loses 3.68, mostly in two worlds where its selection failed (−19.0 and −2.8; §15).

## 19. Decision gate

1. **In the well-specified warp setting, does likelihood remain strongest?** Yes. With the model, regularizer, data,
   optimizer and search held fixed, the penalized likelihood beats the OPC objective under warp by 0.98 points
   [0.47, 1.49] in 6 of 6 worlds, and both weighted likelihoods by more than 3 points. In the population there is
   nothing to gain from a value objective there (−0.04).
2. **In the misspecified group / vector / combined settings, do the population likelihood and value optima differ?**
   Yes. By 1.23, 1.12 and 3.36 greedy points, in 18 of 18 worlds; the likelihood barely separates the two solutions
   (0.0044 nats per row).
3. **Does propensity-weighted likelihood move the solution toward higher policy value?** No, not with the uniform
   reference. It moves the population optimum away from value: θ_uniform* is 0.63 points below θ_log* on the biased
   worlds, θ_clip10* 0.41 below.
4. **Does that theoretical benefit survive finite-sample variance at 25k?** There is no benefit to survive, and the
   variance makes it worse: an ESS of about 39 rows, native gains of −0.67 (raw) and +0.38 (clip 10) against the plain
   likelihood's +2.98, in 0 of 24 biased worlds above it.
5. **Does OPC generate better candidates than likelihood, or is any difference mainly model selection?** Not
   selection. Under one common rule the likelihood leads by 0.22, and its best trials by 0.28. OPC's candidates are
   better on average under group, vector and combined bias (by 0.13–0.69 in 18 of 18 worlds), but not at the top,
   and not under warp. OPC's population advantage (1.42) is spent by its larger training gap (3.81 against 2.11).
6. **Is raw weighting usable, or does clip10 materially improve the bias-variance tradeoff?** Raw weighting is not
   usable: below the logger on average, a 1-point selection regret and a 19-point selection failure. Clipping helps
   (+1.05, 21 of 24 worlds) but not materially: +0.38 over the logger, 2.60 points below the plain likelihood in 24 of
   24 worlds.
7. **Does the evidence justify moving to the hierarchical correction-capacity study?** Yes, with one condition.
   - Under group, vector and combined bias the global class itself stops 1.93, 3.51 and 5.64 points short of the
     target-best ceiling, more than the likelihood's objective mismatch in each (1.23, 1.12, 3.36).
   - The condition: at 25k no objective reaches even the global class's value. The best arms recover about 40% of it,
     and OPC's training gap grows with the bias (6.6 points under combined).
   - More capacity therefore needs the regularization the study plans (partial pooling toward the global map), and
     must be compared at the same budgets and with the same selection protocol.
   - With enough capacity the true click model is in the class, so the likelihood's optimum ranks perfectly. The
     objective mismatch measured here should shrink as capacity grows, and that is a prediction to test.
8. **Which three objectives should be carried forward?**
   - **Penalized likelihood:** (1/n)Σ ℓ(r, σ(s g/T + c)) + λR(θ), λ searched over {0, 0.001, 0.01, 0.1, 1}, validation
     NLL selection. It is the strongest arm here, and its native selection is near-perfect.
   - **OPC's value objective:** the DR loss with harmonic:0.1 weights and a cross-fitted q̂, the direct gradient,
     + λR(θ) with the same grid, the 95% DR lower bound (clip:10) for selection. It is the only objective without
     objective mismatch, and its candidates are better on average where the mismatch exists.
   - **Clipped uniform-reference likelihood** (w = min(1/(P p), 10), validation IW-NLL selection), kept as the
     propensity-weighted outcome model, but as a reference arm and not a candidate. This study gives no reason to
     expect it to win at higher capacity. Raw weighting should be dropped.
   - **Selection, for every arm:** report the common DR selector (the 95% lower bound of the greedy policy) beside the
     native rule. It costs the likelihood and OPC at most 0.1 points and repairs the weighted likelihood's selection.

## 20. Null and negative results, and limitations

- **Null:**
  - the source anchor λR(θ) changes nothing for OPC: the shared OPC arm equals the historical arm, −0.03 [−0.09, +0.03];
  - the OPC objective and the likelihood tie under group and vector bias, natively and under the common rule;
  - the shared likelihood is level with CausE-cap (−0.12 [−0.27, +0.04]).
- **Negative:**
  - weighting the likelihood toward the uniform action distribution lowers its population optimum's value under every
    misspecified bias;
  - at 25k both weighted arms stay near the logger;
  - the raw arm's own selection rule fails badly once (−19.0 points without bias).
- **The tuning protocol's one extension** (the weighted likelihoods' lower learning rates, §11) did not make them
  competitive. The supplementary round's check is in §11.2.
- **Limitations:**
  - Development worlds only (seeds 100/101), 25k, one logger sharpness (80% of its greedy CTR), K = 32.
  - The 20 configurations of a seed are shared by all its worlds (§10.1), so arm differences rest on 40 distinct
    configurations.
  - The uniform reference is the only weighting tested. A weighting toward a policy, or toward the decision region,
    is outside this phase.
  - The population optima are numerical fits: a fixed optimizer and budget, and the best of three learning rates.
    Warp's −0.04 to −0.11-point differences between optima are within their fitting error.
  - The likelihood arms' stochastic policies were not tempered, so their stochastic value is not compared.

## 21. Reproduction

From the repository root, with the BPR embeddings in `BPR/embeddings`. The runs used `--max-workers 3` on one 48 GB
GPU; results do not depend on the worker count.

```bash
COMMON="--emb-dir BPR/embeddings --out-dir artifacts/full_study --datasets ml kuairand anime --ctr-levels 0.05 --train-sizes 25000 --n-trials 20 --sampler random --stage development --slim --learn-logit-scale --lr-range 1e-4 2e-3 --epochs-range 5 30"
SHARED="--methods shared_likelihood shared_iw_likelihood shared_iw_likelihood_clip10 shared_opc"
IWSPACE="--shared-arm-space shared_iw_likelihood:lr=3.1623e-5,2e-3 shared_iw_likelihood_clip10:lr=3.1623e-5,2e-3"
# tuning, first round (§11; the code at 052462c)
python -m training.run_full_study_parallel --run-tag shared_tune_s200 $COMMON --bias-configs high/none/none none/none/high high --seeds 200 201 $SHARED
# the population optima of the weighted likelihoods (§6), one job per dataset
python -m training.class_oracles --datasets ml --seeds 100 101 --bias-configs none high/none/none none/high/none none/none/high high --classes bilinear affine_bilinear --objectives uniform_likelihood clip10_likelihood --save-policies --out artifacts/full_study/run_shared_oracles_20261006/ml
# the main grid (§7, §11.1)
python -m training.run_full_study_parallel --run-tag shared_main_25k $COMMON --bias-configs none high/none/none none/high/none none/none/high high --seeds 100 101 $SHARED $IWSPACE --save-policies
# the supplementary tuning round (§11.1)
python -m training.run_full_study_parallel --run-tag shared_tune_s200_supp $COMMON --bias-configs high/none/none none/none/high high --seeds 200 201 --methods shared_iw_likelihood shared_iw_likelihood_clip10 $IWSPACE --shared-seed-tag shared_supplement
# tables and figures
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200 --out artifacts/full_study/shared_objective_study/tuning
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200 artifacts/full_study/run_shared_tune_s200_supp --out artifacts/full_study/shared_objective_study/tuning_round2
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200_supp --out artifacts/full_study/shared_objective_study/tuning_round2_supplement_only
python -m training.analyze_shared_objectives oracles --out artifacts/full_study/shared_objective_study/population
python -m training.analyze_shared_objectives compare --runs artifacts/full_study/run_shared_main_25k --oracles artifacts/full_study/shared_objective_study/population --out artifacts/full_study/shared_objective_study/compare
```
