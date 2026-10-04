# Simulator logging fix and OPC revalidation (2026-10-04)

**Status: in progress.** This document is written phase by phase. Phase 0 (freeze and inventory) and Phase 1
(regression tests) are complete; the OPC re-tuning (Phase 2), the reruns (Phase 3), the old-vs-new comparison
(Phase 4) and the CausE assessment (Phase 5) follow. CausE is paused: no CausE experiment runs during this work.

Artifacts: `artifacts/full_study/opc_revalidation_20261004/`. Every new run records its source commit
(`code_commit` in `run_meta.json` and `run_manifest.json`).

---

## Phase 0. The bug, the fix and the inventory

### 0.1 The bug

**Where.** `_simulate_from_embedding_policy` (training/trainer_trials.py) builds the logger as a `Policy` whose
generator is `default_rng(random_state % (2**31 − 1))` (since 433583f, 2026-05-05), and passes it to
`create_simulation_data_from_policy` (utils/simulation_utils.py), which draws everything else from
`default_rng(random_state)`. Every split seed of the study is below 2³¹ − 1 (`seed + 1009·train_size + 17·run +
7`, e.g. 25,225,124), so the two generators were the same PCG64 stream.

**When it became a bug.** 69fffab (2026-09-24 10:08 +0300, "Make logged-data sampling identical on CPU and GPU")
replaced the samplers by one inverse-CDF sampler that draws **one uniform per row**, `self.rng.random(n)`. The
simulation draws the users first, `rng.choice(n_users, size=n, p=user_prior)`, which also consumes one uniform per
row, in the same order. From 69fffab on, row i's action used exactly the uniform U_i that had drawn row i's user.
(The chunking at 100,000 rows does not break this: the policy draws its uniforms chunk by chunk, in the same
positions.) Before 69fffab:
- the CPU sampler (Gumbel-max) consumed |A| uniforms per row, so row i's own user uniform was reused only by row 0;
  the other overlaps tie the noise of the first ⌈n/|A|⌉ rows (at most a few dozen) to other rows' users;
- the GPU sampler, the default device, used a torch generator seeded by one draw from the policy's stream.

The coupling was therefore negligible before 69fffab.

**Range.**
- Buggy: every commit from 69fffab (2026-09-24) to c11b2b3 (CRM head before the fix, 2026-10-04): 59 commits. Every
  study run since 2026-09-26 used code in this range (verified with `git merge-base --is-ancestor` for each run's
  code commit in the run registry).
- Fixed: 5c011a9 (local branch `cause-baseline`, 2026-10-03 22:45), and the same change on `CRM` as dbc401b
  (cherry-picked, without any CausE code). The logger's generator is now seeded with
  `derive_seed(random_state, "logging_policy_actions")`. `create_simulation_data_from_policy` now refuses a
  policy whose generator is in the simulation's state.

**The coupling, exactly.** Let C be the normalized cumulative user prior in user-index order, and F_u the
cumulative π0(·|u) over the items in item-index order. One uniform U gave both the user and the action:

```text
user(U)   = u  such that  C(u−1) ≤ U < C(u)
action(U) = a  such that  F_u(a−1) ≤ U < F_u(a)
```

The logs were therefore drawn from the joint and conditional

```text
P(u, a)      = | [C(u−1), C(u)) ∩ [F_u(a−1), F_u(a)) |
P_eff(a | u) = P(u, a) / prior(u)
```

The prior gives each user a slice of [0, 1) of width prior(u), about 1/|U|. A user's action is the item whose
CDF interval covers that slice, so it is nearly a fixed function of the user. Because the slice sits at
position ≈ C(u), users with small indices receive items early in the item order, and the user and action indices
correlate.

**Which propensity was wrong.**
- **Stored:** the pscore, π0(a|u) (the logger's softmax, or its uniform mixture), for the logged pair. That number
  is the policy's probability, and the older tests checked it.
- **Actual:** the row was generated with probability P_eff(a|u), not π0(a|u).
- **Consequences:**
  - Every importance weight w = π(a|u)/pscore had the wrong denominator.
  - Within a user the logs contained essentially no action variation.
  - Across users, the action was a deterministic function of the user's position in the index order.

**Rewards were unaffected.** They were drawn after all users, from the simulation stream (positions n to 2n−1),
which the policy never reached, so r ~ Bernoulli(q(u, a)) given the logged pair.

**What looked right, and why the bug was missed.**
- The pscore of each logged pair was the logger's probability.
- The mean logged click rate matched V(π0).
- The CPU and GPU samplers agreed draw for draw, the property 69fffab was written to guarantee.

The existing tests checked only these marginal properties.

### 0.2 What the logs actually were

`training/logging_coupling_diagnostic.py` computes P_eff exactly over every user of a world and draws the old and
the fixed logs of the 25k split (95,000 rows: 50,000 regression + 25,000 training + 20,000 validation). Its
reconstruction of the old sampler reproduces the code at c11b2b3 bit for bit (users, actions, pscores and rewards
of ml/none and kuairand/high, 95,000 rows each). Full table:
`artifacts/full_study/opc_revalidation_20261004/phase0/coupling_exact_lgs0.8.csv`.

Seed 100, logger share 0.8, the six Stage 2 bias settings. Ranges over the six settings per dataset:

| quantity (prior-weighted over users) | ml | kuairand | anime |
|---|---|---|---|
| actions with P_eff > 0, per user | 2.00–2.29 | 1.50–1.63 | 1.29–1.37 |
| largest P_eff(a\|u) | 0.945–0.975 | 0.965–0.974 | 0.975–0.992 |
| total variation between P_eff(·\|u) and π0(·\|u) | 0.88–0.92 | 0.83–0.88 | 0.91–0.96 |
| mean stored pscore of a generated row | 0.08–0.12 | 0.12–0.17 | 0.04–0.09 |
| mean true generating probability of a generated row | 0.93–0.97 | 0.95–0.97 | 0.97–0.99 |
| V under P_eff − V(π0) (points) | −0.14 to +0.13 | −0.09 to +0.03 | −0.03 to +0.04 |
| E_gen[w] for the logger ×2 / ×0.5 / uniform (1 if correct) | 0.97–1.01 / 0.95–1.04 / 0.81–1.11 | 0.98–1.01 / 0.98–1.02 / 0.92–1.04 | 1.00–1.02 / 0.99–1.02 / 0.98–1.15 |
| IPS limit − truth, logger ×2 (points) | −0.38 to +0.23 | −0.28 to +0.05 | −0.13 to +0.27 |
| IPS limit − truth, uniform (points) | −0.41 to +0.19 | −0.19 to +0.13 | −0.05 to +0.30 |
| total variation of the action marginal | 0.20–0.28 | 0.17–0.20 | 0.11–0.16 |
| users (≥ 2 rows) who always got the same action: old / fixed | 0.91–0.95 / 0.01 | 0.94–0.96 / 0.06–0.10 | 0.96–0.99 / 0.02–0.05 |
| corr(user index, action index): old / fixed | +0.72 to +0.84 / −0.04 to 0.00 | +0.82 to +0.89 / −0.00 | +0.51 to +0.72 / −0.03 to −0.02 |
| distinct logged actions (95,000 rows): old / fixed | 1,275–2,223 / 2,260–3,395 | 4,544–5,202 / 6,600–6,959 | 3,425–6,823 / 4,047–8,279 |

**Reading.**
- The old logs were drawn from a nearly deterministic per-user logger: one action per user, carrying 93–99% of
  that user's probability.
- The stored pscores understated the generating probability 6–25 fold.
- Aggregates were nearly preserved: the logger's own value to within 0.14 points.
- The "IPS limit" is the expectation of the importance-weighted estimate under the generated distribution, with
  the stored pscores. For the three targets checked (the logger sharpened ×2, flattened ×0.5, uniform), it lies
  within about ±0.4 points of the truth, with mean weights 0.81–1.15.
- These population-level checks bound only the infinite-data bias of a few fixed estimates. They say nothing about
  learning. The learned policies, their selection and the variance of the estimates all saw logs without
  within-user randomization, so no learned result can be assumed to carry over.

### 0.3 What is preserved

- **Old run folders.** Every `artifacts/full_study/run_*` folder is untouched (local, excluded from git), as are the
  old summaries (`summaries_20260927/`, `analysis_20260927/`) and the old report figures (`report_20260928/`).
- **Old documents.** These stay as historical records:
  - `representation_repair_dev_20260927.md`, `representation_repair_followup_20260927.md`;
  - `representation_repair_experimental_report_20260928.md`;
  - `decision_record_opc_objective_weighting.md`;
  - `training_losses.md` §9;
  - `roee_handoff_20260928.md`.

  Each gets a banner naming the buggy simulator (Phase 4), with no other change.
- **CausE.** Everything is untouched:
  - the branch `cause-baseline` (d31a5dd, with the CausE code and its own copy of the fix, 5c011a9);
  - the M5 runs (`run_cause_dev_25k_opc_20261004`, `run_cause_dev_25k_cause_20261004`);
  - the pushed CausE documents and results (`docs/cause_baseline.md`, `docs/cause_dev_report_20261004.md`,
    `artifacts/cause_repro/`, `artifacts/full_study/cause_dev_25k_20261004/`).

  No CausE code is on `CRM`.
- **The stash** (`stash@{0}`, "issue1: reward features") is untouched.

### 0.4 Inventory

Classes:
- **U**, unaffected;
- **R**, affected and must be rerun;
- **S**, affected but supporting or diagnostic only;
- **M**, a mathematical result unaffected, although its empirical experiment is affected.

A run is affected if it trained, tuned, selected or estimated on logged rows produced by
`_simulate_from_embedding_policy` with code in the buggy range. "Plausible-looking" was not a criterion: every
run's code commit was checked for 69fffab.

| experiment | runs (run registry) | what used the logs | class | action |
|---|---|---|---|---|
| Stage 1 oracle repair bound | `run_oracle_repair_20260927`; `run_oracle_repair_stage3_lgs_0_6`, `_0_95` | nothing: Adam on the exact true value from the logger's vectors; the logger's temperature from `sharpen_logger` (exact over the catalog); world calibration from its own derived streams (`derive_seed(seed, "world", …)`), users and items drawn in sequence from one generator | U | reused unchanged |
| Oracle validation | `run_oracle_validation_20260927` | nothing (true-reward oracle) | U | reused unchanged |
| Exact values | every run's V(π), greedy values, V_target_best, logger value and ranking loss | none (`calc_reward` is exact; `policy_reward_mode` = exact in every run) | U | the denominators of the fractions and the structural gap stand |
| Stage 2 learned recovery: OPC, DM-only, no-propensity, tempered logger | `run_stage2_ml`, `_kuairand`, `_anime` | training rows, q̂ fit (budget-fair, cross-fitted by user), the validation rows used for selection | R | Phase 3A, all arms, same worlds and dev seeds |
| Stage 2 fractions, OPC − baselines, gap decomposition | `summaries_20260927/stage2_*`, `followup/` | derived from Stage 2 | R (the structural gap U) | recomputed in 3A with the unchanged oracle |
| Su robustness slice | `run_stage2_su_shrink_100` | as Stage 2 | R | the weighting study (2C) and the robustness arm (3A) |
| Stage 3 logging support | `run_stage3_lgs_0_6`, `_0_8`, `_0_95` | as Stage 2 | R (oracle bounds U) | Phase 3D |
| Objective comparison: DR vs legacy and global SNDR | `run_gradcmp_dr_shrink_100_log_trick`, `run_replay_sndr_batch_shrink_100`, `run_replay_sndr_global_shrink_100`; TPE `run_retune_*` | as Stage 2 | R for the DR family; S for the SNDR arms | Phase 2A |
| Legacy and global SNDR characterization | (analysis) | none: the batch-size dependence of legacy SNDR and the stop-gradient, epoch-stale normalizer of global SNDR are properties of the objectives | M; their empirical deficit vs DR is S | stay reproduction-only |
| DR as the working objective | f5cade9, from the runs above | as above | R | Phase 2A |
| Direct gradient vs log trick | `run_gradcmp_*` | the empirical comparison | M: with raw weights the two gradients are identical; with a transform g the log trick ascends DM + H(w)(r − q̂), H = ∫ g(t)/t dt, not the named estimate. The empirical "direct ≥ log trick" is S | direct kept; a small confirmation (2B) |
| Training weights raw / clip / Su shrink / Metelli harmonic | `run_tune_weights_*` (2026-09-26, spread logger), `run_retune_dr_train_*`, `run_gradcmp_*`, `run_final_dr_harmonic_*`, `run_final_check_*` | as Stage 2 | R | Phase 2C |
| Selection weights clip:10 | `run_tune_weights_*` (`weight_tuning_picks.csv`), d7a434c | the logged selection scores against the true values of the trials | R | Phase 2C/2D (post hoc from the logged per-trial scores) |
| Reward-model budget (external 50k vs budget-fair q̂) | `run_logger_explore`, `run_logger_explore_budget` | as Stage 2; also on the older pipeline (legacy SNDR, log trick, shrink:100, TPE, before the short-batch fix) | R | Phase 3B on the current pipeline |
| Budget-fair q̂, 5-fold cross-fitting, validation 20,000 | 66fd303, from the runs above | the evidence (the DR standard error at 5k vs 20k validation rows) | S: a budget and fairness design choice; the standard error argument is generic | kept as a design choice, not re-tuned |
| Reward-model misspecification (interaction vs concat q̂) | `run_qhat_concat` (also the older pipeline) | as Stage 2 | R | Phase 3C |
| Interaction q̂ features | b953d89 | the fit-quality evidence (RMSE, rank correlation against the truth) was measured on affected logs | M for the structural argument (the true score is a dot product; concat cannot rank items per user); the fit numbers are S | kept; 3C reruns the comparison |
| No-propensity arm | Stage 2 and 3 | its training rows | R | 3A, 3D |
| DM-only arm | Stage 2 and 3 | q̂'s training rows | R | 3A, 3D |
| Tempered logger | Stage 2 and 3 | its scale is chosen by the DR score on validation rows | R | 3A, 3D |
| Selection-estimation diagnostics (estimate error, optimism, regret, Spearman) | Stage 2 and 3, the weighting study | validation rows | R | recomputed from the reruns |
| ESS and heavy-weight diagnostics | Stage 2 and 3, the weighting study | validation rows | R | recomputed from the reruns |
| Logger greedy share 0.8 | e7115f5 (Test 2 = `run_logger_explore_budget`); `representation_bias.md` | Test 1 (the room the logs can evaluate, from a policy trained on the truth under an ESS constraint) is exact; Test 2 (learning) is affected | S: a world-design choice; it should not be chosen by a method's performance | kept at 0.8 so that old and new runs pair; Stage 3 reports the dependence |
| Post-hoc tempering (off) | 63a3cc1 | its motivating observation (learned scale ~7 vs ~60) came from affected logs | S | reviewed in Phase 2D |
| Learnable logit scale (on in Stage 2) | 2a29056 | design; no tuning on logs | S | reviewed in Phase 2D |
| Linear policy transform | b8450c9 | design: starts exactly at the logger, can invert the warp | U (not chosen from logs) | kept |
| Policy search ranges: lr 1e-4–1e-3, epochs 5–25, lr decay 0.8–1, batch schedule | 6a7a39c (2026-05-27), f684df3 (2026-09-20) | set before the bug, on older protocols, never validated on the current pipeline | U as to the bug. In old Stage 2, 92% of selected OPC trials at 5k and 72% at 25k had lr ≥ 6e-4, against 20–25% of the trials: the optimum sat at the upper edge | Phase 2D: widened and re-tuned |
| Selection rule (DR lower bound) | b865449 (2026-07-19) | set before the bug; never tuned on the current pipeline | U as to the bug | reviewed post hoc in Phase 2D |
| Baseline before weight tuning | `run_baseline_step4` | as Stage 2 | S (superseded) | none |
| KL check | `run_tune_kl_clip_100` | as Stage 2 | S (diagnostic) | none |
| Reproducibility checks | `run_final_check_*` | as Stage 2 | S; their conclusion, bit-identical reruns, is a property of the code (U) | none |
| Early runs | `run_20260503_192442`, `run_demo_ml` | before 69fffab | U (older protocol; not used) | none |
| H1 study | design in `h1_experiment.md` | no H1 runs in this repository since 2026-09-24 (any made elsewhere since then are affected) | — | none |
| CausE port validation (ML-10M reproduction against TensorFlow) | `artifacts/cause_repro/` | not the simulator | U | none |
| CausE M5 bounded comparison | `run_cause_dev_25k_opc_20261004`, `run_cause_dev_25k_cause_20261004` | logs generated **with** the fix (5c011a9) | the logs are U. The OPC rows used the defaults chosen on buggy logs (dr, direct, harmonic:0.1, selection clip:10, the old search ranges); the CausE rows used CausE's own search | Phase 5 (assessment only) |

### 0.5 Defaults that rest on affected evidence

| default | set in | evidence | status |
|---|---|---|---|
| OPC objective `dr`, direct gradient | f5cade9 | affected runs; the gradient argument is analytic | 2A / 2B |
| training weights `harmonic:0.1` | f5cade9 | `run_final_dr_harmonic_*` (affected); λ was best at the edge of {0.05, 0.1, 0.2} | 2C |
| selection weights `clip:10` | d7a434c | `run_tune_weights_*` (affected, spread logger) | 2C / 2D |
| logger share 0.8 | e7115f5 | partly affected | kept (world design) |
| budget-fair cross-fitted q̂, validation 20,000 | 66fd303 | partly affected; design | kept (design) |
| interaction q̂ features | b953d89 | structural argument plus affected fit numbers | kept; 3C |
| post-tempering off; learnable logit scale on | 63a3cc1, 2a29056 | affected / design | 2D |
| search ranges, selection rule | before the bug | never validated on the current pipeline; lr at the edge | 2D |

---

## Phase 1. Regression tests

`tests/test_logging_propensity_calibration.py` (15 tests) and `tests/test_logging_rng_independence.py` (4 tests,
with the fix). They test the production path `_simulate_from_embedding_policy`.

| # | test | what it checks |
|---|---|---|
| 1 | `test_each_fixed_user_gets_actions_with_the_recorded_probabilities` (6 cases) | four fixed users, ~10,000 draws each, at temperatures 0.3 / 1 / 3, with and without a 0.3 uniform mixture: chi-square of P_emp(a\|u) against the recorded π0(a\|u); the most frequent action's frequency equals its pscore |
| 2 | `test_actions_are_independent_of_the_user_when_the_logger_is_the_same_for_all` | identical user vectors, so π0(·\|u) is one distribution: users × actions contingency test of independence; rank correlation |
| 2 | `test_users_follow_the_prior_and_rewards_follow_q_given_the_logged_pair` | users against the prior; the click rate in every logged (u, a) cell against q(u, a); residuals uncorrelated with users and actions |
| 3 | `test_the_joint_of_users_and_actions_is_prior_times_pscore` | P_emp(u, a) against prior(u)·π0(a\|u) over all cells; E[π_e/pscore] = 1 and E[π_e/pscore · r] = V(π_e) for a sharper, a flatter and the uniform target |
| 4 | `test_stored_pscore_is_the_generating_policys_probability` (2 cases) | the stored pscore equals the logger's probability (formula, rtol 1e-12) and each fixed user's realized frequency of each logged action |
| 4 | `test_uniform_rows_store_exactly_one_over_the_catalog` | uniform loggers (mixture weight 1; constant logits): pscore exactly 1/\|A\|; actions uniform and independent of the user |
| 4 | `test_softmax_rows_store_the_exact_softmax_in_float64` | row by row, the float64 softmax |
| 5 | `test_a_sharper_logger_is_more_concentrated_in_the_population_and_in_the_logs` | logger shares 0.6 / 0.8 / 0.95: exact entropy falls, collision probability and value rise, the uniform target's ESS falls; in the logs the per-user repeat rate equals Σπ0², the mean reward equals V(π0), the logged action entropy falls |
| 6 | `test_minimal_case_two_users_two_actions` | the smallest failing case, documented in the module docstring: two users with equal prior and π0 = (½, ½) for both. With the coupled streams user 0 always got action 0 and user 1 always action 1 (pscore 0.5 recorded) |

**On the old code** (c11b2b3, in a temporary worktree), 13 of the 15 fail. Two pass, because they check what the
old code did correctly:
- the user and reward draws;
- the float64 softmax value of the stored pscore.

Of the four tests in `test_logging_rng_independence.py`, two fail on the old code (the repeat rate and the
shared-seed guard). The other two pass there: the pscore formula and per-seed determinism. On the fixed code all
pass, with and without the GPU.

(Phases 2–5 follow.)
