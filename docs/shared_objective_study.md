# Three training objectives on one global correction model (design and pre-registration, 2026-10-06)

*Development stage. Branch `representation-mismatch-research-next`, from the tested handoff commit `9445a12`.
§1–§9 were written and committed before any tuning or main result of this study existed. Results follow in later
sections. Nothing here is confirmatory.*

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
