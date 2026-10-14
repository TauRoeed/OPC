# OPC with adaptive estimators and statistical reliability under structured scenario shifts

Pre-registration, 2026-10-13. Committed before any new method is trained, and before Stage A's evaluators are run at
scale. The completed studies (`docs/structured_scenario_shift_study.md`, `docs/opc_gradient_regime_study.md`,
`docs/shared_objective_study.md`) are referenced, never modified, except for recording runs they already
specified.

## 0. Question, starting point and scope

**Starting point.**
- Branch `representation-mismatch-research-next` at 19b9fa3 (pushed), plus f6c54de. f6c54de records the structured
  study's supplementary tuning round, a run that study specified and that finished on 2026-10-11.
- The structured study's N × support expansion was held (its §13.4). This study supersedes it and does not launch it.

**What the structured study established** (its §15, 54 + 18 worlds):
- OPC beats ordinary likelihood where response heterogeneity is strong and the shift is small enough for its target
  to stay close.
- Fixed harmonic shrinkage (λ = 0.1) with the logistic reward model misses the value optimum under shift: by 1.3–1.5
  points at moderate shift and 4.4–4.9 at strong shift. That miss is the largest part of its gap under strong shift.
- Raw DR has the right target but loses more to variance.
- The calibration-aware likelihood is the strongest arm.

**The goal.** A practical OPC learner that:
- (A) improves the true value over the unadapted source;
- (B) beats ordinary likelihood under realistic response heterogeneity;
- (C) closes ground on the calibration-aware likelihood;
- (D) controls bias, variance and poor support;
- (E) explains its gains through independently measured estimator error.

**Not in this study:** the public cross-scenario datasets; new simulator families; any change of the response
coefficients, the calibration or the shift construction; the held N × support expansion; protected confirmation
worlds before Stage F.

## 1. Audit: what exists and is reused

| component | where | role here |
|---|---|---|
| structured worlds | `utils/structured_shift.py` (`build_structured_world`, `calibrate_gamma`, `SHIFT_LEVELS`) | every world; the new small level (§2) |
| rank-4 adapter | `models/lowrank_adapter.py` (`SharedLowRankModel`, `LowRankAnchor`) | the one representation class of every arm |
| arms and trainer | `training/trainer_trials.py` `regression_trainer_trial` (`shared_objective`, `shared_adapter`, `shared_head_kind`, `train_weights`, `train_reward`); `training/run_full_study.py` `SHARED_OBJECTIVE_ARMS` | the baselines; the new arms extend the same trial |
| DR loss | `models/custom_losses.py` `dr_sndr_surrogate` / `DRPolicyLoss` (direct gradient, normalizer `none`); `utils/importance_weights.py` (`harmonic:λ` = w / (1 − λ + λw)) | families A–C |
| reward model | `trainer_trials.fit_shared_regression_bundle` (sklearn logistic on [x, a, x ⊙ a]); `run_full_study._crossfit_bundles`; `CrossFitScoresLookup` (5 user folds) | D0; the cross-fitting scheme of D1–D4 |
| exact population tools | `training/opc_gradients.py` (`WorldTensors`, `exact_value`, `fit_value_path`, `estimator_gradients`, the conditional bias); `training/opc_empirical_objective.py` (`population_reward_model` q̂_∞, `fit_harmonic_population`); `training/ss_population.py` | population objectives (§9); gradient diagnostics (§10) |
| selectors | `trainer_trials._shared_trial_diagnostics`, `cause_trials._greedy_dr` | native and common selection |
| policy export and value | `get_params`, `--save-policies` (exported ux, ia), `utils.policies.Policy`, `calc_greedy_reward` | every evaluation |
| weight / ESS statistics | the summary's `train_w_*`, `val_w_*`, the trial's `ess` | support reporting |
| analysis | `training/analyze_structured_shift.py` (`population`, `compare`, `expand`), `training/analyze_shared_objectives.py` (selections, the edge rule) | tables and figures |
| **new: fixed-policy oracle evaluator** | `training/oracle_eval.py`, `tests/test_oracle_eval.py` | Stage A and every later oracle diagnostic (§8) |

**Audit facts that shape the design.**
- The trainer's harmonic transform is the directive's h_λ(w) = w / (1 − λ + λw).
- OPC trains on the direct gradient of the transformed DR estimate, with no self-normalization (regime study §1.2).
- **OPC selects with a different estimator than it trains on.** Native selection is the 95% lower bound of DR with
  clip-10 weights and the full-data q̂, on the 20,000 validation rows; training uses the cross-fitted q̂ and h₀.₁.
- The likelihood arms' softmax policies sit at the click model's scale: nearly uniform, with S2 of 10⁷–10⁸ on the
  first world measured. They are evaluated through their greedy policies. DR evaluation of those greedy policies has
  almost no support (S2 of the ordinary likelihood's greedy policy: 18,186 on that world).
- **Legacy reproduction:** a Stage-1 world (ml, seed 100, moderate shift, strong response; five arms × 20 trials)
  rerun from 19b9fa3 reproduces `run_ss_main_25k` bit for bit (all 100 trials: true values, objective values,
  selections).
- **Training budget:** the structured study's supplementary tuning round (its §13.5) found every arm leaning toward the
  top of the common learning-rate and epoch ranges. The ranges stay as they are (§6), shared by every arm, and this is
  a stated limitation.

## 2. Worlds

- Datasets ml, kuairand and anime, with the BPR source vectors and the matched rank-4 adapter.
- Reward generation as the structured study (its §2–§4 and the §13.1 levels): q_B = σ(κ α_u s̃_B + c + β_u), with the
  shift and response levels unchanged.
- **New shift level `small`.** Target mean user score correlation 0.95. Same directions, same `calibrate_gamma`, same
  click calibration and pathology rule as moderate and strong.
  - A world scan (statistics only: ρ, top-1 agreement, available gain, best-item CTR, pathology) is committed before
    any method is trained on it.
  - If a small world cannot be calibrated, that is a stop (§13), not a re-tune.

| regime | shift (ρ) | response | role |
|---|---|---|---|
| **R1** | small (0.95) | strong | primary |
| **R2** | moderate (0.90) | strong | primary |
| **R3** | moderate (0.90) | moderate | primary |
| **C1** | moderate (0.90) | none | control: homogeneous response |
| **C2** | none (1.00) | strong | control: the source is optimal (a win measures resistance to harmful adaptation) |
| **S1** | strong (0.75) | strong | stress; never used for tuning or method choice |

**Per world:**
- AvailableGain = V_g(target-optimal) − V_g(source). The target-optimal greedy policy is each user's best item under
  q_B. The truth adapter reaches it, since the ranking is preserved (structured study §15.1).
- TrueGain = V_g(π) − V_g(source).
- RecoveryFraction = TrueGain / AvailableGain, reported where AvailableGain ≥ 0.5 points.
- Stochastic values are reported beside the greedy ones.

## 3. Seeds and data splits

| use | seeds | status |
|---|---|---|
| Stage A | 100, 101 | existing worlds and selected policies (`run_ss_main_25k`) |
| Stage B pilots | 300, 301 | new; method development only |
| Stage C (and D, E) | 100, 101 | development seeds; baselines reused where their definitions match exactly (§4) |
| Stage F confirmation | 900, 901, 902, 903 | never built, scanned or inspected before Stage F |

- **Per world and N:** the study's split. N training rows and 20,000 validation rows, drawn by the study's simulator
  with the existing seeds. 5 user folds for cross-fitting.
- **Fresh evaluation logs (§8):** R logs of 25,000 rows, seeded `derive_seed(seed, dataset, label, "stage_a", r)`
  (another label per stage). They are never used for training, selection or reward-model fitting.

## 4. Baselines

| | arm | notes |
|---|---|---|
| B0 | the unadapted source policy | the logger's own vectors (π = μ for the softmax) |
| B1 | `shared_lr_likelihood` | ordinary likelihood, rank-4 adapter |
| B2 | `shared_lr_likelihood_calib` | calibration-aware likelihood |
| B3 | `shared_lr_opc` | fixed harmonic λ = 0.1 (= A0) |
| B4 | `shared_lr_opc_raw` | raw DR (= A1 at λ = 0, = C0) |
| B5 | `shared_lr_opc_oq` | harmonic λ = 0.1 with the simulator's q: diagnostic only |
| B6 | θ_value* | the value optimum of the adapter class (population fit): oracle reference |

- **Reuse.** The structured study's results are reused for Stage C on seeds 100 and 101, where the world, the rows,
  the 20-trial paired search and the definitions are identical (regimes R2, R3, C1, C2 and S1). R1 and every pilot
  seed need new baseline runs.
- **Parameters.** Representation parameters are the same in every arm: 2·32·4 adapter + 1 scale. Nuisance
  parameters are reported per arm: the click head; the calibration head (2·32 + 2); the reward models of §5.4.

## 5. The new methods

**Notation.**
- Logged rows (x_i, a_i, r_i, μ_i), i = 1..n.
- π_θ: the adapter's softmax policy. w_i = π_θ(a_i|x_i) / μ_i.
- q̂: the cross-fitted reward model in training; the full-data model in selection.
- Per-row DR contribution with weight transform g:

      f_i^g(θ) = Σ_j π_θ(j|x_i) q̂(x_i, j) + g(w_i) (r_i − q̂(x_i, a_i)),     V̂^g(θ) = (1/n) Σ_i f_i^g(θ).

**Shared training rules.**
- Every OPC arm maximizes V̂^g − λ_anchor R(θ) [+ the method's own term].
- **The base estimator per family** (fixed here, before any pilot):
  - A changes g itself.
  - B (support) and C (CRM) keep raw DR (g = w), the unbiased estimator, and act on the policy (B) or on the
    estimate's variance (C). The gated C2 alone uses h₀.₁. This separates "shrink the weights" (A) from "keep the
    policy where the logger has support" (B) and "penalize the estimate's uncertainty" (C).
  - D keeps the current harmonic λ = 0.1, so that it isolates the reward model's effect on the existing method.
- It uses the direct gradient, the current minibatch loop, optimizer and paired search (§6).
- Method parameters (λ, β, C, γ) are not searched inside a trial budget. Each value is its own arm in Stage B, and
  Stage C freezes one per family (§11).

### 5.1 Family A: adaptive importance-weight shrinkage

- **A0:** g = h₀.₁ (= B3).
- **A1:** g = h_λ for fixed λ ∈ {0.005, 0.01, 0.03, 0.2}. With B4 (λ = 0) and B3 (0.1), this traces the bias–variance
  curve over {0, 0.005, 0.01, 0.03, 0.1, 0.2}.
- **A2, sample-information-adaptive.** Metelli, Russo and Restelli (NeurIPS 2021, Theorem 5.1) give the
  concentration-optimal parameter of the harmonic correction:

      λ*₂ = sqrt( 2 log(1/δ) / (3 I₂ n) ),     I₂ = E_μ[w²] = S2(π).

  - λ shrinks as n grows. It also shrinks as S2 grows, because their bias bound grows with S2. "Usable information"
    therefore enters through n mainly.
  - **Our variant:**

        λ_t = min(1, sqrt( 2 log(1/δ) / (3 n Ŝ2(π_θt)) )),    δ = 0.05,
        Ŝ2(π) = (1/n) Σ_i Σ_j π(j|x_i)² / μ(j|x_i).

    Ŝ2 is enumerated over every item at the training rows' users, since the logger is known. It has none of the
    empirical w²'s heavy tail. The paper's own data-driven rule (its eq. (3)) uses the corrected weights' second
    moment, because it cannot enumerate.
  - λ_t is computed from θ at the start of each epoch and held fixed during the epoch (no gradient through λ).
  - At n = 25,000 and S2 ≈ 8 (the current OPC policies), λ ≈ 0.003.
- **A3, bias-aware.** At the start of each epoch, at the current θ, on the training rows (cross-fitted q̂), for each λ
  in Λ = {0, 0.005, 0.01, 0.03, 0.1, 0.2}:

      d_i^λ = f_i^{h_λ} − f_i^{raw},
      MSÊ(λ) = max(0, mean(d^λ)² − var(d^λ)/n) + var(f^{h_λ})/n.

  - λ_t = argmin MSÊ is held fixed during the epoch.
  - The bias term uses raw DR's unbiasedness. Its own sampling variance is subtracted (and the result floored at 0)
    rather than treating a noisy difference as the bias. The uncertainty of λ_t is reported: its distribution across
    epochs and across Stage A's replicate logs.
  - A3 targets the MSE of the value estimate, not of the gradient. A gradient-MSE variant is exploratory and not run
    unless §10's diagnostics motivate it.
- **Population objectives.** A fixed λ has one: V_{h_λ} with q̂_∞, computed by `fit_harmonic_population`. A2 and A3
  have none. Their analysis compares the λ they choose with the fixed-λ curve and does not force a decomposition.

### 5.2 Family B: squared-importance-weight support control

- **The support statistic:** S2(π) = E_x Σ_a π(a|x)² / μ(a|x), the exponentiated Rényi-2 divergence.
  - In training it is estimated on each minibatch by enumerating all items at the batch's users.
  - In evaluation it is computed exactly, over the prior and the catalog.
  - The greedy counterpart's S2 = E_x 1 / μ(a*(x)|x) is reported beside it. Constraining the softmax's S2 does not
    bound the greedy policy's support, and no such claim is made.
- **B-soft:** maximize V̂ − β (S2(π) − 1), β ∈ {0.0003, 0.001, 0.003, 0.01}, in CTR per unit of S2. At the current OPC
  policies' S2 ≈ 8, the penalty spans 0.002–0.07 CTR, against available gains of 0.05–0.12.
- **B-hard:** maximize V̂ subject to S2(π) ≤ C, C ∈ {5, 10, 20, 50}, a nominal ESS fraction of 1/C. It is solved by an
  augmented Lagrangian:

      L = V̂ − ν (S2 − C) − (ρ/2) max(0, S2 − C)²,    ρ = 1,

  with ν ← max(0, ν + ρ (S2_epoch − C)) at the end of each epoch, S2_epoch over the training users. The selected
  policy's exact S2 must satisfy S2 ≤ 1.05 C (§13).
- **Comparators.** The existing source anchor (λ_anchor, in every arm's search). A policy-to-logger divergence,
  KL(π‖μ) = E_x Σ_a π log(π/μ) enumerated the same way: B-KL with β ∈ {0.001, 0.003} per nat (exploratory).
- **Reported:** the true value lost to conservatism, V_g(B3) − V_g(B-variant) where positive, beside the variance
  reduction.

### 5.2a Amendment: the hard constraint's scale (2026-10-13, before any family-B run)

**What failed.**
- The pilot worlds' population fits (§9) solved the S2-capped value problem with the augmented Lagrangian of §5.2,
  ρ = 1 on the raw violation S2 − C.
- On the first world (ml, R1, seed 300) the capped optima were not monotone in C: C = 10 lost 2.4 points and C = 5
  lost 0.4. Every multiplier ended at 0 with S2 well inside its cap.
- **The cause:** a violation of a few S2 units costs ρ/2 · (S2 − C)², tens of times the value's whole range. Its
  gradients inflate Adam's second-moment estimates, and the value gradients then barely move the parameters for
  hundreds of steps: the fit stalls where the penalty left it.
- The B-hard training arms used the same form. None of them had run.

**The fix** (applied identically to every cap, before any family-B pilot):
- **Training (B-hard):** the augmented Lagrangian on the normalized violation g = S2/C − 1, with ρ = 0.1 in value
  units: ν g_B + (ρ/2) max(0, g_B)², and ν ← max(0, ν + ρ g) at each epoch's end. The 5% tolerance check (§13) is
  unchanged.
- **Population (§9):** the value–S2 frontier by Lagrangian relaxation. θ*(ν) = argmax V − ν (S2 − 1) for
  ν ∈ {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}, by continuation from θ_value*, 1,500 Adam steps each at θ_value*'s
  learning rate.
  - The grid contains B-soft's β values, so θ*(β) is B-soft's population target.
  - The cap-C optimum is the best greedy value among θ_value* and the path points with exact S2 ≤ C: a monotone
    inner approximation of the capped optimum.
- On the same world the frontier is smooth: S2 from 27 (θ_value*) down to 1.1, the loss from 0 up to 1.7 points.
  Caps 5 / 10 / 20 / 50 cost 0.38 / 0.19 / 0.07 / 0.00 points.

### 5.3 Family C: counterfactual risk minimization (CRM)

- **Variance and dependence.** Rows are i.i.d. draws (user from the prior, action from the logger), so f_i are i.i.d.
  given q̂ even when a user repeats. The cross-fitted q̂ couples the rows of a fold weakly. The training penalty uses
  the i.i.d. form; at selection the user-clustered standard error is reported beside it.
- **C0:** raw DR, γ = 0 (= B4).
- **C1:** J = V̂^{raw}(θ) − γ sqrt( var_B(f^{raw}(θ)) / N ), γ ∈ {0.5, 1, 2}.
  - var_B is the minibatch variance of the per-row contributions, the attached estimate of the per-row variance.
  - N is the full training size, so the penalty is γ times the standard error of the full-data estimate.
  - The square root is taken of var + 1e-12. A check confirms the gradient against finite differences (§14).
- **C2:** the same with g = h₀.₁. **Gated:** launched only if C1 is numerically stable in the pilot (≤ 5% diverged
  trials, no non-finite penalty).
- **C3, delta-CRM:** f_i^Δ = f_i(π_θ) − f_i(π₀), π₀ the source softmax (= μ, so its w ≡ 1), and
  J = mean(f^Δ) − γ sqrt(var_B(f^Δ)/N). Its mean has raw DR's gradient (V̂(π₀) is constant). Only the penalty
  differs: the variance of the improvement, where noise common to π_θ and π₀ cancels. **Gated** on Stage A's
  measurement of that cancellation (§8) and on C1's stability.
- CRM does not remove shrinkage bias, and no claim is made that it does. Minimizing variance alone is not an
  objective here, because γ ≤ 2 keeps the value term.

### 5.4 Family D: reward models

All are fit on the logs only, with the same 5 user folds of cross-fitting as D0. The full-data model is used for
selection.

- **D0:** the existing logistic model, [x, a, x ⊙ a]: 3K + 1 = 97 parameters.
- **D1, low-rank bilinear:** logit q̂ = c + u·x + v·a + s · xᵀ(I + PQᵀ)a, with P, Q ∈ R^{K×8}.
  - 2K + 2 + 16K = 578 parameters, fit by full-batch L-BFGS on the Bernoulli NLL with an L2 penalty of 1e-4 on
    (u, v, P, Q).
  - It can express the low-rank shift; the response factors only approximately.
- **D2, heterogeneity-aware:** logit q̂ = exp(v_αᵀx) (s · xᵀ(I + PQᵀ)a + γ) + v_βᵀx + v·a + c. It puts D1's
  interaction inside the calibration-aware head's form, with v_β as the user main effect and v·a as the item main
  effect: 611 parameters, the same fit.
- **The fit (D1 and D2).** Q starts as a seeded orthonormal matrix and P at 0, so the fit starts at the bilinear
  source model. c and s start at the one-dimensional logistic fit on xᵀa. The penalty is on ‖P‖² + ‖Q‖² and the
  linear terms. L-BFGS runs up to 400 iterations.
- **D3, target-weighted (gated):** D1 refit with row weights min(π̂(a|x)/μ(a|x), 10), π̂ the first-stage policy of the
  D1 arm.
  - The fold structure is kept: fold k's model never sees fold k's outcomes.
  - π̂ is fit on all training rows, a weak dependence that is stated.
  - Run only if D1 or D2 changes OPC's selected value or oracle bias in the pilot.
- **D4, variance-targeted (gated, MRDR-type):** q̂ fit to minimize Σ_i w_i (w_i − 1)(r_i − q̂_i)², the leading term of
  DR's variance. Run only with D3's condition, and if implementation and stability checks pass.
- **The D arms** train harmonic λ = 0.1 OPC with the D model in place of D0 (the cross-fitted lookup in training, the
  full-data model in selection). They isolate the reward model's effect on the current method; B5 (oracle q) is its
  ceiling.
- **Reward-model accuracy**, per model. RMSE against q_B under:
  - the logger;
  - the source;
  - the selected policy;
  - θ_value*'s policy (oracle diagnostic).

  Also: calibration (reliability in 10 bins of q̂ under the logger), RMSE on low-support pairs (μ < 1e-3, π-weighted),
  the resulting estimator bias and variance at the selected policy (§8), and the final policy value. Prediction
  accuracy and decision quality are reported separately.

### 5.5 Implementation (`models/adaptive_losses.py`, `models/reward_models.py`, `training/ae_arms.py`)

- **One criterion for families A–C:** `AdaptiveOPCLoss`.
  - With fixed weights and no term it equals the existing `DRPolicyLoss`, with its anchor, in value and gradient.
  - The training loop gains two generic hooks, inert for every existing criterion:
    - the minibatch's users go to a criterion that asks for them (the enumerated S2 and KL, delta-CRM);
    - `epoch_start` (the adaptive λ) and `epoch_end` (the multiplier) are called when a criterion defines them.
- **Family D:** `BilinearRewardLookup` serves q̂ with the trainer's lookup interface.
  - The D arms train with the cross-fitted D model and select natively with its full-data model.
  - The common selector keeps D0 for every arm.
- **One registry of the new arms:** `training/ae_arms.py`, shared by the runner, the run state and the analysis.
  - `shared_lr_opc_h{λ}` (A1); `_a2`, `_a3`; `_s2b{β}`, `_s2c{C}`, `_klb{β}` (B); `_crm{γ}` (C1); `_d1`, `_d2`.
  - Gated: `_hcrm{γ}` (C2), `_dcrm{γ}` (C3).
- **Per trial, recorded:** the adaptive λ (last and mean over epochs), the training S2 and the multiplier; per arm,
  the reward model's parameters and fit time.
- **Tests:** `tests/test_adaptive_losses.py` (5) and `tests/test_reward_models.py` (4).
  - The equivalence with the existing loss; the support terms against enumeration; the CRM terms; gradients against
    finite differences.
  - The adaptive λ's and the multiplier's rules.
  - The recovery of a known click model; the lookup interface; cross-fitting without a fold's own rows.
- **A smoke run** (ml, seed 200, outside the study's seeds, 2 trials of 3–4 epochs per arm): all 27 new arms run
  end to end on identical rows, none diverged.

### 5.6 The small shift level's world scan (statistics only, before any method)

`world_scan/world_scan_small.csv`, 3 datasets × seeds 100, 101, 300, 301 × response none / moderate / strong:
- all 36 worlds calibrate: ρ = 0.95; top-1 agreement 0.41–0.55; best-item CTR 23–30%; none pathological;
- the available gain is 1.7–4.5 points, and 1.7–3.2 at strong response (R1).

The pilot seeds' other worlds (`world_scan_pilot_seeds.csv`, 54 worlds) also calibrate with none pathological.

## 6. Training, search and budgets

- **Search:** the structured study's paired random search (seed label `shared`).
  - lr 1e-4–6.3246e-3; epochs 5–30; lr decay 0.8–1; the batch size by N; λ_anchor ∈ {0, 0.001, 0.01, 0.1, 1}.
  - Every arm of a world trains the same configurations.
- **Trials:** 10 per arm in Stage B pilots; 20 in Stages C–F.
- **Method parameters:** one arm per value (§5); no further search.
- **Costs** are recorded per arm: trial seconds, model fits (trials and reward-model fits), parameter counts.

## 7. Selection and metrics

- **Native:**
  - likelihood arms: validation NLL;
  - OPC arms: the 95% DR lower bound of π_θ on the validation rows, clip-10 weights, the full-data q̂ of the arm's own
    reward model (D0, except in the D arms).
- **Common:** the 95% DR lower bound of the greedy policy, clip 10, the full-data D0 model, the same for every arm.
- **Diagnostics only:** the oracle best of the trials and the mean over trials.
- **Primary metric:** TrueGain (greedy), with RecoveryFraction.
- **Primary comparisons**, paired by world: method − B3, method − B1, method − B2.
- **Secondary:** stochastic value; S2, ESS and weight tails (softmax and greedy); the oracle evaluator diagnostics at
  every selected policy (§8); gradient diagnostics on a subset (§10).

## 8. Oracle estimator-error diagnostics (`training/oracle_eval.py`)

**For every selected policy, softmax and greedy, plus the source and θ_value*:**
- the exact true value;
- every evaluator's error at that fixed policy, on fresh independent logs.

**Fixed reward model.** The selection-time full-data reward model of the world, fit on its training rows, so
independent of the fresh logs. Other models are fixed the same way.

**Exact moments.** With q̂ fixed, the rows are i.i.d. and every linear estimator f = DM_π(x) + g(w)(r − q̂) has
closed-form moments:

    E[f | x]  = DM_π(x) + Σ_a μ g(w)(q − q̂),
    E[f² | x] = Σ_a μ [ (DM_π(x) + g(w)(q − q̂))² + g(w)² q (1 − q) ],

summed exactly over the prior and the catalog. The same holds for the paired improvement f_π − f_π₀ on the same rows.
Exact bias, variance, MSE, and the covariance between V̂(π) and V̂(π₀).

**Estimators:**

| | estimator | moments |
|---|---|---|
| E1 | raw DR | exact |
| E2 | harmonic DR at λ ∈ {0.005, 0.01, 0.03, 0.1, 0.2} | exact |
| E3 | A2 (λ from the log's enumerated Ŝ2) | replicates |
| E3 | A3 (λ chosen on the log) | replicates |
| E5 | clip-10 DR (the selectors' weights) | exact |
| E5 | self-normalized DR | replicates |
| — | DM | exact |
| — | oracle-q DR (reference) | exact |
| E4 | each new training estimator, from Stage B | as applicable |

**Replicates.** R fresh logs of n = 25,000 rows: 50 in a pilot, 200 for the principal comparisons. The same logs serve
every estimator and policy of a world (paired). Per policy and estimator:
- bias (± its standard error); variance with a 95% χ² interval; MSE; RMSE;
- median and 90th-percentile absolute error;
- coverage of the plug-in 95% interval;
- the same for the paired improvement over the source, and for the improvement from two independent logs (exact);
- exact S2 and ESS fraction; Kish ESS, maximum weight and the 99th / 99.9th percentiles on the logs;
- the target mass on μ < 10⁻⁴ and 10⁻³, and on w > 10 and > 100.

**The checks:**
- Linear estimators: the replicates against the exact moments, bias in standard errors and variance ratio.
- Raw-weight estimators are heavy-tailed: a sample variance of such replicates usually falls short of its
  expectation. For them the exact variance is the reference, and a replicate variance is read with that caveat.
- Systematic bias is declared only against the exact moments, or for nonlinear estimators with a |bias| beyond 3
  standard errors of the replicate mean.

**Three error sources, kept apart:**
1. **The evaluator's conditional error** at a fixed policy and reward model: the above.
2. **Variation across independently trained policies:** across worlds and seeds.
3. **The adaptivity of selection:** the selected trial's own validation estimate against its true value, compared
   with the same for all trials of the arm (a winner's-curse measure), from the recorded trials.

## 9. Population objectives

As in the structured study §6, per world and adapter class:
- θ_value*, θ_lik*, θ_calib*, θ_harm*(λ = 0.1);
- the fixed-λ harmonic optima θ_harm*(λ) for λ ∈ {0.005, 0.01, 0.03, 0.2} with q̂_∞, and with the D1 and D2 reward
  models' population fits where feasible;
- the S2-constrained value optima for C ∈ {5, 10, 20, 50}: the loss imposed by the support constraint, V_g(θ_value*)
  − V_g(θ_C*), from the Lagrangian frontier of §5.2a.

Each is distinguished: the unconstrained optimum, the class optimum, each modified objective's optimum, the
constraint's loss, finite-sample and optimization error, and selection error.

## 10. Oracle gradient diagnostics

- **Subset:** R1 and R2, ml and kuairand, pilot seed 300 (4 worlds).
- **States:**
  - the source;
  - an intermediate policy: the selected trial replayed deterministically with a checkpoint at half its epochs;
  - the selected final stochastic policy.
- **Estimators:** raw DR; harmonic 0.1; A2's and A3's λ; the CRM objectives' gradients; the D1 and D2 reward models.
  Computed on 100 fresh datasets of N rows at fixed θ (`opc_gradients.estimator_gradients`).
- **Metrics** (regime study §4): bias norm, relative bias, variance, MSE, signal-to-noise ratio, cosine with the true
  gradient.

### 10a Amendment: how the states and estimators are computed (2026-10-13, before any gradient diagnostic)

**The selected policy's θ** is recovered from its exported vectors (`training/ae_gradients.py`, tested):
- the user vectors are s·x, so s is their least-squares ratio;
- the item vectors are a + a M with M = VUᵀ of rank 4, so M is an exact least-squares solve;
- (U, V) is taken as its balanced SVD factorization.

The policy is reproduced exactly. Gradients are reported in this parameterization, the same for every estimator at a
state.

**The intermediate state.** The point on the straight path θ(t) = (√t U, √t V, t log s), so M(t) = t M, whose true
value is halfway between the source's and the selected policy's (bisection on t). It replaces §10's replay of the
selected trial with a checkpoint, which the saved artifacts do not allow without re-running the trial. It is
deterministic and policy-meaningful, but it is not a point the optimizer visited.

**Fixed reward models.** The world's selection-time full-data models: D0, and D1 / D2 with the trainer's seeds. They
are fit on its training rows, so they are independent of the fresh datasets.
- For an estimator linear in the rows (a fixed λ), the expected gradient is the gradient of its exact expectation, so
  its bias is exact. Raw DR's expected gradient equals the true gradient for any fixed q̂ (tested).
- A2, A3 and the CRM objective (γ = 1 on raw DR) take the mean over the replicates.
- The variance comes from 100 fresh datasets of 25,000 rows.

**Estimators:** raw DR; h₀.₁; h₀.₂ (family A's representative); A2; A3; CRM (γ = 1); h₀.₁ with D1; h₀.₁ with D2;
h₀.₁ with the oracle q (reference).

**Subset:** R1 and R2 × ml and kuairand × seed 300, with harmonic OPC's (B3) selected pilot policy.

## 11. Stages and gates

**Stage A: audit and diagnostic baseline** (now).
- Reproduction: done (§1), bit-identical.
- The oracle evaluator diagnostics of §8 on R2, R3, C2 and S1 × 3 datasets × seeds 100, 101 (24 worlds), with the
  existing selected policies of the five arms, the source and θ_value*, softmax and greedy (14 policies per world).
  R = 50, then R = 200.
- The selection-adaptivity measure (§8.3) from the recorded trials.
- **Measured cost:** setup 7 s (ml) to 22 s (anime); exact sums 1.5–91 s; 0.15–0.65 s per replicate. About 1
  GPU-hour at R = 200, and 144 reward-model fits.
- **Gate:** the replicates agree with the exact moments for bounded-weight estimators (|bias z| < 4; variance within
  its χ² interval). Then an intermediate report.

**Stage B: individual method pilots.**
- Seeds 300, 301 × 3 datasets × {R2, R3, R1} = 18 worlds; N = 25,000; share 0.8; 10 paired trials per arm.
- **Arms:** B1–B5 (5); A1 (4); A2; A3; B-soft (4); B-hard (4); B-KL (2); C1 (3); D1; D2. 26 arms.
- **Budget:** 4,680 training fits plus 216 reward-model fits. Measured trial costs 2–6 s, so about 6 GPU-hours (about
  2.5 hours with 3 workers). Under both caps.
- **Reports:** per family as it completes.
- **C2 and C3** follow only by their gates (§5.3). **D3 and D4** follow only by theirs (§5.4).
- **Decision rule, per family, for Stage C.**
  - Among the variants that are stable (≤ 5% diverged trials, constraints within tolerance), take the one with the
    highest mean native TrueGain over the 18 pilot worlds.
  - If the best two differ by less than the standard error of their paired difference, take the earlier in a fixed
    order:
    - A: A2, A3, then A1 by the λ closest to A2's median λ;
    - B: B-hard by the largest C, then B-soft by the smallest β;
    - C: the smallest γ;
    - D: D1, then D2.
  - A family whose representative does not beat B3 still goes to Stage C as a documented null, unless it is unstable.

**Stage C: the controlled main comparison.**
- Seeds 100, 101 × 3 datasets × {R1, R2, R3, C1, C2, S1} = 36 worlds.
- Arms: B1–B5, reused where §4 allows (R1 new), and the 4 family representatives, frozen; 20 trials.
- About 2,900 new training fits plus R1's 600 baseline fits.
- The oracle evaluation at R = 200 for every selected policy; the gradient diagnostics (§10).
- Before Stage D: a written account of which mechanism each method changed and whether the true value improved.

**Stage D: justified combinations.**
- Only methods with complementary Stage C results, from the directive's priority list:
  - adaptive shrinkage + S2 support;
  - adaptive shrinkage + reward model;
  - CRM + reward model;
  - delta-CRM + support.
- 20 trials each. The budget is presented before launch.

**Stage E: sample size and support.**
- The strongest 2–3 methods with B1–B3.
- At least: R1 or R2; R3; one negative control (C1 or C2).
- N ∈ {5k, 25k, 100k} × logger greedy share ∈ {0.6, 0.8, 0.9}.
- A fixed-policy evaluation of the same selected policies across the three loggers.
- The budget, likely above the caps, is presented before launch.

**Stage F: held-out confirmation.**
- Seeds 900–903, all three datasets, R1–R3 with C1 and C2. Never restricted to worlds where earlier OPC results were
  positive.
- Frozen beforehand: implementations, parameters or adaptation rules, selection rules, regimes, comparators, metrics,
  analysis.
- No method change after its results.

## 12. Success criteria (from the directive) and analysis

1. **TrueGain > 0** where AvailableGain > 0: the mean over worlds with its CI above 0, per regime.
2. **Beats B3 and B1** in the primary regimes: the paired mean with its CI above 0, pooled over R1–R3 and per regime.
3. **More competitive with B2:** (B2 − method) below (B2 − B3), the paired CI of their difference excluding 0.
4. **Credible explanation:** the oracle diagnostics show lower estimator bias, variance or MSE at the selected policy,
   better support, or better selection, consistent with the gain.
5. **No simulation truth** in training or tuning: by construction, audited in the code (§14).

**Reporting.**
- Not every criterion must hold in every world; the report states which hold where.
- The unit is the world. Comparisons are paired by world; intervals are t intervals over worlds.
- Results are given per dataset, per regime, and dataset × regime. Trials of one world are never treated as
  replications.
- Analyses beyond the pre-specified comparisons are labelled exploratory. Negative and null results are kept.

## 13. Stop rules (from the directive)

Stop and diagnose when:
- the legacy reproduction fails;
- a method changes the simulator;
- arms receive different rows;
- a numerical gradient check fails;
- a supposedly unbiased estimator shows unexplained systematic population bias;
- a variance or ESS calculation disagrees with enumeration;
- a shrinkage rule reads the oracle truth in a practical run;
- a support constraint is violated beyond 5%;
- more than 5% of a cell's trials diverge;
- intervals become unreliable from extreme tails;
- confirmation data would be accessed early.

## 14. Tests before results are trusted

1. The fixed-policy evaluator: exact moments equal brute-force enumeration; the replicates agree with them; the
   source has unit weights (`tests/test_oracle_eval.py`, passing).
2. The small shift level: its calibration hits ρ = 0.95; the existing levels and labels are unchanged.
3. Each new objective at a fixed θ equals its definition computed directly. Its gradient matches finite differences
   (A2/A3 with λ held; B-soft; B-hard's Lagrangian; C1/C3's penalty).
4. S2 and KL per minibatch equal the enumerated values; a hard constraint's final S2 is checked exactly.
5. D1 and D2 recover a known bilinear logit on synthetic rows; their cross-fitting never fits fold k on fold k's rows.
6. No new arm reads the simulator's q, the truth adapter or θ_value* (a code audit and a test of the arms' inputs).
7. Old commands still produce old worlds and trials (the reproduction of §1, repeated after the new code).
8. The new arms share the paired search and the rows (data identity per world, as the structured study).

## 15. Compute management and provenance

- Before each expensive stage: measured trial times, GPU-hours, model fits, GPU memory and disk, conflicts, an ETA.
  A stage above 12 GPU-hours or 8,000 fits is presented before launch.
- Long runs from pinned worktrees, with run fingerprints and registry rows (`run_ae_*`).
- Artifacts in `artifacts/full_study/opc_adaptive_estimator/`, with a README.

## 16. Results

### 16.1 Stage A, the pilot (R = 50; 2026-10-13)

`run_ae_stage_a` (9022474), 24 worlds (R2, R3, C2, S1 × 3 datasets × seeds 100 and 101), 14 fixed policies each.
Tables: `artifacts/full_study/opc_adaptive_estimator/stage_a_R50/`. Linear estimators: exact moments; the others:
the replicates. Points; means over the 6 worlds of a regime. The principal pass (R = 200) follows in §16.2.

**Checks.**
- None of the 1,680 bounded-weight comparisons of replicates with exact moments has a bias beyond 4 standard errors.
- 99 (5.9%) fall outside their 95% variance interval: the nominal rate, clustered by world because every policy
  shares the world's logs.
- Raw-weight estimators are heavy-tailed: in 10% of the comparisons the replicates' variance is below 1% of the exact
  one (§8).

**At harmonic OPC's selected policy:**

| regime | raw DR: bias / SD | h₀.₁: bias / SD / RMSE | RMSE-best fixed λ (RMSE) | clip-10 DR: bias / RMSE | DM bias |
|---|---|---|---|---|---|
| R2 | 0 / 0.94 | −0.46 / 0.37 / 0.63 | 0.03 (0.61) | −0.25 / 0.61 | −0.95 |
| R3 | 0 / 1.92 | −0.51 / 0.38 / 0.73 | 0.03 (0.70) | −0.31 / 0.71 | −0.98 |
| C2 | 0 / 0.59 | 0.00 / 0.37 / 0.38 | 0.2 (0.32) | +0.01 / 0.51 | 0.00 |
| S1 | 0 / 122 | −0.57 / 0.33 / 1.11 | 0.01–0.03 (0.99) | −0.42 / 1.02 | −0.91 |

**The bias grows as a policy moves from the logger.**
- At the calibration-aware likelihood's greedy policy, h₀.₁ underestimates by 1.86 (R2), 2.05 (R3) and 4.21 (S1);
  clip-10 DR by 1.53, 1.68 and 3.95; DM by 2.66, 2.95 and 4.91. Raw DR's SD there is 43, 1,121 and 218,000.
- At θ_value*, harmonic DR's |bias| is about 2 points already at λ = 0.005 under moderate shift, and above 5 under
  strong shift.
- The estimators OPC trains with (h₀.₁) and selects with (clip 10) thus undervalue the policies that move furthest,
  the finite-sample counterpart of the population mismatch M_harm.

**Support (medians over worlds).**
- The value-optimal greedy policy has S2 ≈ 2,400 (R2), 2,900 (R3) and 2·10⁷ (S1), with 22%, 22% and 55% of its mass
  on weights above 100. Without shift (C2) it is the source's greedy policy, S2 ≈ 8.
- Harmonic OPC's selected policies: S2 ≈ 13 (R2), 22 (R3) and 140 (S1).
- Under shift, the gain thus lies largely where the logger (greedy share 0.8) gives little support. An S2 cap at
  C ≤ 50 excludes the optimum: Family B's population analysis (§9) measures the price.

**Adaptive λ at fixed policies.**
- A2's λ is ≈ 0.002–0.003 at the OPC policies (≈ 0.0007 in S1). Bias −0.11 to −0.32; SD close to raw DR's (0.68 R2,
  0.84 R3, 2.2 S1).
- A3 (λ chosen on each log) does not beat the best fixed λ (RMSE 0.62 R2, 0.82 R3, 1.43 S1).

**Paired improvement.** The paired estimate of V(π) − V(source) has, against two independent logs, a variance ratio:
- 0.96–1.00 for raw DR under shift (0.56 without);
- 0.45–0.57 for h₀.₁;
- 0.01–0.11 for DM.

Common noise cancels little for raw DR under shift: the evidence C3's gate reads (§5.3).

**Selection adaptivity.** The selected OPC trials' validation estimates (clip-10 DR) err by the evaluator's exact
conditional bias at those policies, within −0.006 ± 0.30 points over 72 world-arms: no detectable winner's curse.

**Reward model (D0) RMSE:** 0.065–0.079 under the logger, 0.073–0.094 under the selected OPC policies, 0.108–0.172
under θ_value*.

### 16.2 Stage A, the principal pass (R = 200)

Tables: `stage_a_R200/`. The linear estimators' moments are exact, so their numbers are those of §16.1.

**Checks.** No bounded-weight bias beyond 4 standard errors in 1,680 comparisons; 94 variance misses (5.6%). The
gate (§11) passes.

**The nonlinear estimators at R = 200** (bias / RMSE, points, means over 6 worlds):

| policy | estimator | R2 | R3 | C2 | S1 |
|---|---|---|---|---|---|
| harmonic OPC's selected softmax | SNDR | −0.03 / 0.87 | −0.02 / 1.07 | −0.00 / 0.59 | −0.10 / 2.55 |
| | A2 | −0.06 / 0.73 | −0.06 / 0.90 | −0.01 / 0.57 | −0.19 / 2.30 |
| | A3 | −0.26 / 0.67 | −0.31 / 0.80 | −0.01 / 0.43 | −0.49 / 1.38 |
| calibration-aware likelihood's greedy | SNDR | −0.11 / 2.44 | −0.12 / 2.62 | −0.01 / 0.59 | −0.62 / 6.06 |
| | A2 | −0.32 / 2.11 | −0.25 / 2.38 | −0.02 / 0.57 | −0.60 / 9.43 |
| | A3 | −1.09 / 1.65 | −1.21 / 1.82 | −0.05 / 0.43 | −3.14 / 4.09 |

**What this shows.**
- A2's λ removes most of the fixed λ's bias: at the far-moving greedy policies, −0.3 against −1.9 for λ = 0.1. It
  costs most of the variance that λ = 0.1 saves.
- A3 lands near the fixed λ = 0.03.
- Which trade-off helps learning is Stage B's question: the training objective's bias at the policies a run visits,
  against its noise.

### 16.3 Stage B, family A (adaptive shrinkage): the pilot

`run_ae_pilot_A` (8e4dada), seeds 300 and 301 × 3 datasets × {R1, R2, R3}: 18 worlds × 11 arms (the baselines and
family A) × 10 paired trials, N = 25,000. None of the 1,980 trials diverged. Tables:
`artifacts/full_study/opc_adaptive_estimator/stage_b_A/`.

**TrueGain** (native selection, points; per regime the mean over 6 worlds; the available gain is 2.36 / 4.47 / 4.85):

| arm | R1 | R2 | R3 | all 18 | − B3 [95% CI] (higher / 18) | recovery |
|---|---|---|---|---|---|---|
| raw DR (B4, λ = 0) | 0.38 | 0.87 | 1.02 | 0.76 | −0.28 [−0.45, −0.12] (2) | 0.19 |
| λ = 0.005 | 0.48 | 0.84 | 1.06 | 0.79 | −0.24 [−0.43, −0.05] (4) | 0.20 |
| λ = 0.01 | 0.47 | 0.99 | 1.04 | 0.83 | −0.20 [−0.35, −0.06] (4) | 0.22 |
| λ = 0.03 | 0.41 | 1.12 | 1.17 | 0.90 | −0.14 [−0.28, −0.00] (3) | 0.22 |
| λ = 0.1 (B3) | 0.62 | 1.22 | 1.27 | 1.04 | | 0.28 |
| λ = 0.2 | 0.55 | 1.31 | 1.52 | 1.13 | +0.09 [−0.01, +0.20] (12) | 0.29 |
| A2 | 0.46 | 1.00 | 0.95 | 0.81 | −0.23 [−0.42, −0.05] (3) | 0.21 |
| A3 | 0.50 | 0.92 | 1.40 | 0.94 | −0.10 [−0.33, +0.13] (7) | 0.24 |
| oracle q (B5) | 0.78 | 1.62 | 1.72 | 1.38 | +0.34 [+0.21, +0.47] (17) | 0.36 |
| ordinary likelihood (B1) | −0.82 | 0.73 | 2.39 | 0.76 | −0.27 [−1.01, +0.46] (8) | 0.13 |
| calibration-aware (B2) | 1.21 | 2.68 | 3.25 | 2.38 | +1.34 [+0.95, +1.74] (18) | 0.60 |

**Per dataset**, λ = 0.2 − B3 is +0.08 on each of ml, kuairand and anime. **Against B1**, B3 gains +1.45 in R1 and
+0.49 in R2 and loses 1.12 in R3. Every OPC variant is below B2 in all 18 worlds. **The adaptive λ:** A2's settles
at 0.002–0.007, so A2 trains as raw DR. A3's moves over the epochs (median 0.13), and in some trials it drops to 0
late, where S2 reaches the thousands.

**The decision (§11):** family A's representative is λ = 0.2. It has the highest mean, and the runner-up (A3) is more
than one paired standard error behind.

**Reading** (a pilot of 10 trials).
- At N = 25,000 less shrinkage trains worse: the curve rises monotonically from λ = 0 to 0.2.
- Shrinkage's bias cost is real: harmonic OPC's population target moves from about 2 points from the value optimum
  (λ = 0.2) to about 0.4 (λ = 0.005) on the first pilot worlds (§9). At this N its variance cost is larger.
- Reducing the reward model's error (the oracle q) is the one change here that improves on B3 in nearly every world.

### 16.4 Stage B, family D (reward models): the pilot

`run_ae_pilot_D` (ec808d4): the 18 pilot worlds × the two D arms × 10 paired trials. The baselines are those of
§16.3: the same worlds, rows and configurations. None of the 360 trials diverged; each D model took about five
minutes to fit per world (6 L-BFGS fits). Tables: `stage_b_D/`.

| arm | R1 | R2 | R3 | all 18 | − B3 [95% CI] (higher / 18) | − B1 | − B2 |
|---|---|---|---|---|---|---|---|
| B3 (D0) | 0.62 | 1.22 | 1.27 | 1.04 | | +0.27 | −1.34 |
| D1 | 0.58 | 1.62 | 1.64 | 1.28 | +0.24 [+0.07, +0.42] (16) | +0.52 (12) | −1.10 (0) |
| D2 | 0.50 | 1.64 | 1.71 | 1.28 | +0.25 [+0.05, +0.44] (13) | +0.52 (11) | −1.10 (0) |
| oracle q (B5) | 0.78 | 1.62 | 1.72 | 1.38 | +0.34 (17) | | |

- **Per dataset**, − B3: D1 +0.42 (ml), +0.27 (kuairand), +0.05 (anime); D2 +0.43, +0.28, +0.02; the oracle q +0.43,
  +0.43, +0.16.
- **Other selections:** best of 10 +0.25 (18 / 18) for D1, +0.28 (17) for D2; mean of the trials +0.12 (17) and
  +0.15 (16).
- **The decision (§11):** D1 and D2 are level (difference 0.001, under one paired SE of 0.046), so the fixed order
  takes D1.

**Reading.**
- The richer reward models recover about 70% of the oracle q's gain from the logs alone, where the logistic model is
  misspecified (moderate shift; ml and kuairand).
- They remove most of harmonic OPC's population mismatch (§9: M ≈ 0–0.16 against 0.5–1.7 points) without adding the
  variance that smaller λ adds.
- In R1, the small shift, they do not help.

### 16.5 Stage B, family B (support control): the pilot

`run_ae_pilot_B` (ec808d4, after §5.2a's fix): the 18 pilot worlds × 10 arms × 10 paired trials, on raw DR. None of
the 1,800 trials diverged. Tables: `stage_b_B/`.

| arm | R1 | R2 | R3 | all 18 | − B3 [95% CI] (higher / 18) |
|---|---|---|---|---|---|
| raw DR (B4) | 0.38 | 0.87 | 1.02 | 0.76 | −0.28 [−0.45, −0.12] (2) |
| S2, β = 0.0003 | 0.37 | 0.96 | 1.02 | 0.78 | −0.25 [−0.41, −0.10] (3) |
| S2, β = 0.001 | 0.47 | 1.01 | 1.00 | 0.83 | −0.21 [−0.36, −0.06] (4) |
| S2, β = 0.003 | 0.52 | 1.17 | 1.18 | 0.96 | −0.08 [−0.22, +0.06] (6) |
| S2, β = 0.01 | 0.59 | 1.44 | 1.42 | 1.15 | +0.11 [−0.03, +0.25] (12) |
| S2 ≤ 5 (binding: S2 ≈ 5.0) | 0.51 | 1.19 | 1.16 | 0.95 | −0.09 [−0.21, +0.04] (4) |
| S2 ≤ 10 / 20 / 50 (not binding: S2 ≈ 10–11.5) | | | | 0.74 / 0.78 / 0.75 | −0.30 / −0.26 / −0.28 |
| KL, β = 0.001 / 0.003 | | | | 0.82 / 0.84 | −0.22 / −0.20 |
| B3 (harmonic λ = 0.1) | 0.62 | 1.22 | 1.27 | 1.04 | |

- **Per dataset**, β = 0.01 − B3: ml +0.22, kuairand −0.05, anime +0.17.
- **Other selections:** best of 10 +0.11 [+0.03, +0.20] (14 / 18); mean of the trials +0.03 [−0.01, +0.07].
- **Against the likelihoods:** against B2, −1.23 (0 / 18); against B1, +0.39 (11 / 18).
- **The decision (§11):** β = 0.01, the highest mean; the next (β = 0.003) is more than one paired standard error
  behind. It is the grid's largest β. §5.2 has no edge rule for β, so a stronger penalty is untested here: a stated
  limitation, not an extension.

**Reading.**
- An S2 penalty raises raw DR by 0.39 points: it reduces the variance of the unbiased objective, whose gradient
  variance at the selected policies is about five times harmonic DR's (§16.7).
- At β = 0.01 it reaches fixed harmonic shrinkage's level.
- Hard caps act only where they bind.
- The price is conservatism, measured at the population level by the frontier (§9): 0.4–2.3 points for S2 ≤ 5, and
  0–0.55 for S2 ≤ 50.

### 16.6 Stage B, family C (CRM): the pilot, and the gates of C2 and C3

`run_ae_pilot_C` (ec808d4): the 18 pilot worlds × CRM on raw DR (γ = 0.5, 1, 2) × 10 paired trials. None of the 540
trials diverged. Tables: `stage_b_C/`.

| arm | R1 | R2 | R3 | all 18 | − raw DR | − B3 [95% CI] (higher / 18) |
|---|---|---|---|---|---|---|
| raw DR (B4) | 0.38 | 0.87 | 1.02 | 0.76 | | −0.28 |
| CRM γ = 0.5 | 0.46 | 0.86 | 1.08 | 0.80 | +0.04 | −0.23 [−0.40, −0.07] (3) |
| CRM γ = 1 | 0.46 | 1.00 | 0.96 | 0.81 | +0.05 | −0.23 [−0.41, −0.05] (4) |
| CRM γ = 2 | 0.37 | 1.08 | 1.03 | 0.83 | +0.07 | −0.21 [−0.36, −0.07] (3) |

- **Per dataset**, CRM − B3: ml −0.03 to +0.04, kuairand −0.41 to −0.57, anime −0.15 to −0.26.
- **The decision (§11):** the best two (γ = 2, γ = 1) differ by 0.019, under one paired standard error (0.049); the
  earlier in the fixed order is γ = 1. It goes to Stage C as a documented null (it does not beat B3).
- **C2's gate** (C1 stable: no diverged trial, every penalty finite) is open. Harmonic CRM is training as
  `run_ae_pilot_C2`; if it beats C1, it becomes family C's representative by the same rule.
- **C2, harmonic CRM** (`run_ae_pilot_C2`, ec808d4; the anime worlds finished 2026-10-14 after the interruption of
  §16.8): the 18 pilot worlds × γ = 0.5, 1, 2 on h₀.₁ × 10 paired trials, none diverged. Tables: `stage_b_C_C2/`.

  | arm | R1 | R2 | R3 | all 18 | − B3 [95% CI] (higher / 18) |
  |---|---|---|---|---|---|
  | B3 (h₀.₁) | 0.62 | 1.22 | 1.27 | 1.04 | |
  | CRM γ = 1 (C1) | 0.46 | 1.00 | 0.96 | 0.81 | −0.23 [−0.41, −0.05] (4) |
  | harmonic CRM γ = 0.5 | 0.55 | 1.30 | 1.24 | 1.03 | −0.01 [−0.08, +0.07] (8) |
  | harmonic CRM γ = 1 | 0.60 | 1.30 | 1.35 | 1.08 | +0.04 [−0.03, +0.12] (11) |
  | harmonic CRM γ = 2 | 0.57 | 1.31 | 1.39 | 1.09 | +0.05 [−0.02, +0.12] (10) |

  - Per dataset, harmonic CRM γ = 1 − B3: ml +0.11, kuairand −0.00, anime +0.01. It is above C1 (CRM γ = 1) by
    +0.27 (paired SE 0.07; 15 / 18 worlds): the penalty on h₀.₁ keeps B3's level, the penalty on raw DR does not.
  - **The decision (§11):** the best two (γ = 2, γ = 1) differ by 0.012, under one paired SE (0.025); the fixed order
    (the smallest γ) gives **γ = 1**. C2 replaces C1 as family C's representative; it does not beat B3 (CI includes 0)
    and goes to Stage C as a near-null.
  - The analysis' fixed order for family C listed only C1's arms, so its first output named γ = 2 (a gated arm
    outside the order fell to last place silently). Corrected before Stage C's C run: the order is the smallest γ, the
    ungated C1 first at equal γ, and an arm outside the order is an error (tests/test_analyze_adaptive.py).
- **C3's gate is closed.** Delta-CRM's penalty differs from C1's only through the cancellation of noise common to the
  learned policy and the source. Stage A measured it for raw DR: the paired variance is 0.96–1.00 of the independent
  one under shift (0.56 without; §16.1). Its penalty is therefore practically C1's, and C1 gains 0.05 over raw DR.

### 16.7 Stage B: the representatives, and Stage C

| family | representative | TrueGain (all 18) | − B3 [95% CI] (higher / 18) |
|---|---|---|---|
| A (shrinkage) | harmonic λ = 0.2 | 1.13 | +0.09 [−0.01, +0.20] (12) |
| B (support) | S2 penalty β = 0.01 | 1.15 | +0.11 [−0.03, +0.25] (12) |
| C (CRM) | harmonic CRM γ = 1 (C2; C1's γ = 1 gave 0.81) | 1.08 | +0.04 [−0.03, +0.12] (11) |
| D (reward model) | D1, the low-rank bilinear model | 1.28 | +0.24 [+0.07, +0.42] (16) |

The calibration-aware likelihood (B2) gains 2.38 (+1.34 over B3, 18 / 18).

**Stage C** (§11) runs A, B and D's representatives on the 36 development worlds (seeds 100 and 101 × 3 datasets ×
R1, R2, R3, C1, C2, S1) with 20 paired trials, and the five baselines on R1 (`run_ae_main_R1_baselines`). The other
regimes reuse `run_ss_main_25k`: the same worlds, rows, space and 20-trial search (§4). C's representative follows
once C2 is settled (harmonic CRM γ = 1, §16.6: `run_ae_main_C`, the same 36 worlds and search). The budget is about
3,000 training fits and 216 reward-model fits, about 7 GPU-hours, with C's 720 fits about 3,700 and 8 GPU-hours:
under the caps.

**The gradient diagnostics** (§10, §10a; `gradients/`, 4 worlds) at B3's selected policies:
- At the source and the intermediate state, every estimator's expected gradient is aligned with the true one (cosine
  1.00) with a signal-to-noise ratio of 15–33.
- At the selected policy the ratio falls to 0.14–0.35, and a single dataset's gradient has a cosine of 0.13–0.31 with
  the true gradient.
- There shrinkage tilts the expected gradient: cosine 0.61 (λ = 0.1), 0.48 (λ = 0.2) and 0.64 (A3), against 1.00 for
  raw DR. The D reward models restore it: 0.87 (D1), 0.92 (D2).
- Gradient MSE: raw DR 0.11, A2 0.10, CRM 0.11, λ = 0.1 0.024, λ = 0.2 0.014, D1 / D2 0.021.
- Late in training, then, the gradient is mostly noise. Variance reduction (shrinkage, the S2 penalty) wins over
  unbiasedness (raw DR, A2), and a better reward model reduces the bias without adding variance.

### 16.8 The interruption of 2026-10-13 and the resume

The workstation stopped abruptly at 23:45:01 on 2026-10-13 (the system journal ends without a shutdown) and restarted
at 07:09 on 2026-10-14. Every running job died; no committed result and no completed run folder was affected (all
3,040 result files load).
- **Interrupted:** Stage C's `run_ae_main_ABD` (ml, kuairand): 6 of 24 conditions complete (360 trials, ml seed 100);
  three kuairand seed-100 conditions held h₀.₂'s 20 trials each without a summary row. The pilot populations' anime
  seed-301 worlds (nothing written). Not yet started: C2's anime worlds, the anime oracle evaluations, Stage C's anime
  worlds.
- **The resume** (2026-10-14 08:28, the same commit ec808d4 and arguments; launchers in `ae_logs/launchers/`): the
  runner's skip-completed reran only the conditions without a summary row; an interrupted arm's rows are removed before
  it reruns, so no trial is counted twice. The 60 recomputed h₀.₂ trials equal the interrupted ones in all 37 numeric
  columns, and their selected policies are bit-identical (seeded, deterministic training).
- `run_ae_main_ABD`'s first invocation (2026-10-13 22:30) left no line in `run_invocations.jsonl` (it was killed); the
  resume's invocation is the run's first record.

### 16.9 Stage C: the controlled main comparison, training results

`run_ae_main_ABD` (A, B, D's representatives), `run_ae_main_C` (C's) and `run_ae_main_R1_baselines` (ec808d4; R2–S1's
baselines from `run_ss_main_25k`): 36 worlds (seeds 100, 101 × 3 datasets × R1, R2, R3, C1, C2, S1), 20 paired trials,
3,480 new training trials, none missing. Native selection; TrueGain in points. Tables: `stage_c/`
(`analyze_adaptive stage_c`).

| arm | TrueGain (36) | − B3 [95% CI] (higher / 36) | − B3 ml | kuairand | anime |
|---|---|---|---|---|---|
| B3 (h₀.₁) | 1.99 | | | | |
| A: h₀.₂ | 2.17 | +0.18 [+0.10, +0.26] (31) | +0.13 | +0.19 | +0.21 |
| B: S2 penalty β = 0.01 | 1.60 | −0.38 [−0.60, −0.17] (8) | −0.23 | −0.90 | −0.03 |
| C: harmonic CRM γ = 1 | 1.96 | −0.03 [−0.10, +0.05] (17) | −0.01 | −0.01 | −0.05 |
| D: D1 reward model | 2.49 | +0.50 [+0.31, +0.70] (31) | +0.34 | +0.70 | +0.47 |
| B4 raw DR | 1.48 | −0.51 [−0.69, −0.32] (5) | −0.62 | −0.54 | −0.36 |
| B5 oracle q | 2.31 | +0.32 [+0.07, +0.58] (24) | +0.35 | +0.07 | +0.54 |
| B1 likelihood | 1.92 | −0.07 [−0.51, +0.37] (16) | +0.07 | −0.18 | −0.10 |
| B2 calibration-aware likelihood | 3.35 | +1.36 [+0.98, +1.75] (36) | +1.58 | +1.02 | +1.49 |

− B3 by regime (R1 / R2 / R3 / C1 / C2 / S1):
- D1: −0.03 / +0.41 / +0.47 / +0.64 / +0.05 / +1.48 (6 / 6 higher in R2, R3, C1, S1).
- h₀.₂: +0.01 / +0.26 / +0.25 / +0.13 / +0.09 / +0.33.
- S2 penalty: −0.16 / −0.21 / −0.44 / −0.35 / −0.05 / −1.09.
- harmonic CRM: −0.08 / +0.01 / +0.01 / −0.14 / +0.07 / −0.02.

What the training results show, under these conditions (N = 25,000, share 0.8, the rank-4 adapter):
- **D1 and h₀.₂ replicate their pilot direction** on the 36 new worlds, with intervals excluding 0. D1's gain over
  B3 grows with the shift (largest in S1) and is absent in R1 and in the no-shift control C2. Under the best-trial and
  trial-mean rules D1 − B3 is +0.46 and +0.24, h₀.₂ − B3 +0.14 and +0.12 (36 / 36 higher by the trial mean).
- **The S2 penalty's pilot gain did not replicate.** In the pilot (R1–R3, seeds 300 / 301) β = 0.01 was +0.11
  [−0.03, +0.25] over B3; in Stage C it is below B3 in every regime, most on kuairand and in S1. β = 0.01 was the
  grid's edge (§16.5), and the pilot interval included 0.
- **Harmonic CRM is level with B3** (a null).
- **No OPC variant reaches the calibration-aware likelihood:** D1 − B2 −0.86 [−1.16, −0.56] (1 / 36), h₀.₂ − B2 −1.18.
  By the trial mean, D1 is level with B2 (+0.04 [−0.04, +0.13]); the gap is in native selection.
- D1 − B5 (the oracle-q harmonic baseline) is +0.18 (paired SE 0.12; 19 / 36): not distinguishable.
- Pending: the oracle evaluation (R = 200) of these selected policies (§8, §11) and their population analysis (§9);
  the mechanism account before Stage D (§11).

### 16.10 Stage C: the oracle evaluation (R = 200) and the population objectives

`run_ae_oracle_stage_c` (03106b0): the 9 arms' selected policies (softmax and greedy), the source and θ_value* on the
36 worlds, 200 fresh logs of 25,000 rows, the selection-time D0 as the fixed q̂. The replicates agree with the exact
moments (bounded-weight estimators: 3,753 of 3,960 checks pass, 5.2% misses, as in Stage A). `run_ae_population`
holds the 36 worlds' §9 optima. Tables: `stage_c_oracle/`, `stage_c_population/` (`stage_b_population/` for the pilot).

**Support of the selected policies** (median over the 36 worlds; greedy S2 / softmax S2 / softmax mass on w > 10):

| policy | greedy S2 | softmax S2 | mass w > 10 |
|---|---|---|---|
| source | 8.5 | 1 | 0 |
| B4 raw DR | 11.5 | 10.5 | 0.33 |
| B: S2 penalty β = 0.01 | 12.7 | 3.7 | 0.06 |
| C: harmonic CRM γ = 1 | 16.4 | 14.5 | 0.39 |
| B3 h₀.₁ | 18.2 | 15.6 | 0.39 |
| A: h₀.₂ | 18.7 | 16.7 | 0.42 |
| D: D1 | 20.7 | 18.8 | 0.42 |
| B2 calibration-aware likelihood | 149 | 5.5·10⁶ | 0.69 |
| θ_value* | 1,652 | 1,648 | 0.63 |

**The estimators at each arm's selected softmax policy** (bias / SD / RMSE, points, mean over worlds; D0 as q̂, so
the D1 and B5 rows are not their own estimators):

| arm | selection estimator (clip-10 DR) | training estimator |
|---|---|---|
| B3 | −0.23 / 0.51 / 0.69 | h₀.₁: −0.37 / 0.37 / 0.68 |
| A: h₀.₂ | −0.22 / 0.51 / 0.70 | h₀.₂: −0.42 / 0.29 / 0.73 |
| B: S2 penalty | −0.07 / 0.44 / 0.45 | raw DR: 0 / 0.48 / 0.48 |
| C: harmonic CRM | −0.24 / 0.51 / 0.69 | h₀.₁: −0.38 / 0.37 / 0.69 |
| D: D1 | −0.88 / 0.50 / 1.18 (D0) | h₀.₁ with D0: −1.07 / 0.36 / 1.24 |
| B4 raw DR | −0.20 / 0.51 / 0.63 | raw DR: 0 / 1.03 / 1.03 |

The adaptivity of selection (the selected trial's validation error beyond the selector's exact bias) is −0.03 to
+0.03 points for B1–B4, A and C (no winner's curse), −0.16 for the S2 penalty (conservative); not measurable for D1
and B5 (different q̂).

**Population mismatch M** (points, the value each objective's population optimum gives up; mean over worlds):

| objective | R1 | R2 | R3 | C1 | C2 | S1 | all |
|---|---|---|---|---|---|---|---|
| available gain | 2.65 | 4.99 | 5.59 | 5.99 | 0 | 10.61 | 4.97 |
| B3: harmonic 0.1 (q̂_∞) | 0.55 | 1.29 | 1.44 | 1.52 | 0.01 | 4.37 | 1.53 |
| A: harmonic 0.2 | 0.75 | 1.65 | 1.86 | 2.01 | 0.02 | 5.23 | 1.92 |
| D: harmonic 0.1 with D1 | 0.07 | 0.11 | 0.47 | 0.56 | 0.02 | 1.15 | 0.40 |
| B: raw value − 0.01 S2 | 0.87 | 1.99 | 2.25 | 2.40 | 0.02 | 5.67 | 2.20 |
| B1 likelihood | 1.84 | 2.01 | 0.93 | 0 | 1.47 | 2.37 | 1.44 |
| B2 calibration-aware likelihood | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 16.11 Before Stage D: which mechanism each method changed, and whether the true value improved

Under these conditions (N = 25,000, greedy share 0.8, the rank-4 adapter, 36 development worlds); not yet confirmed
(Stage F).
- **A, harmonic λ = 0.2 (shrinkage).** It changes the training estimator's bias-variance trade: at the selected
  policy its SD is 0.29 against h₀.₁'s 0.37 (the pilot gradient MSE 0.014 against 0.024), at a larger bias, and its
  population target is worse (M 1.92 against 1.53). The true value improved: +0.18 [+0.10, +0.26] (31 / 36), in
  every dataset (+0.13 to +0.21) and most in R2, R3 and S1 (+0.25 to +0.33), not in R1 (+0.01). The gain is a
  finite-sample variance effect, not a better target. The adaptive rules (A2, A3) did not improve on fixed λ in the pilot.
- **B, the S2 penalty (support).** It changes the selected policy's support: softmax S2 3.7 against 15.6, mass on
  w > 10 0.06 against 0.39, and the selection estimate becomes more accurate (bias −0.07 against −0.23). But the
  penalized objective's population optimum gives up 2.20 points (B3's 1.53), and the true value fell: −0.38 [−0.60,
  −0.17] (8 / 36), largest where that cost is (S1 −1.09). Reliability was bought with value; the pilot's +0.11 did
  not replicate.
- **C, harmonic CRM (variance penalty).** It changes nothing measurable: support, estimator errors and true value are
  B3's (−0.03 [−0.10, +0.05]). At N = 25,000 the penalty is too small against the objective to move the selected
  policy; on raw DR (C1) it added only +0.04 to +0.07 over raw DR in the pilot.
- **D, the D1 reward model (bias).** It changes the reward model inside the estimator and so the surrogate's bias: the
  population mismatch falls from 1.53 to 0.40 (S1: 4.37 to 1.15); in the pilot the evaluator's bias at θ_value* fell
  from −3.27 (D0) to −0.44 (D1) and the expected gradient's cosine with the true one rose from 0.61 to 0.87. Its
  policies move slightly further (greedy S2 20.7). The true value improved most: +0.50 [+0.31, +0.70] (31 / 36),
  growing with shift (S1 +1.48), none in R1 (−0.03) or without shift (C2). In R2–S1 it realizes 35–67% of the
  population gap it closes (R2 0.41 of 1.18, R3 0.47 of 0.97, C1 0.64 of 0.96, S1 1.48 of 3.22).
- **What none of them changes.** Every OPC objective keeps the policy close to the logger (greedy S2 11–21), while
  the value optimum lies far outside its support (S2 ≈ 1,650; S1 ≈ 2·10⁷). The calibration-aware likelihood, which
  does not weight by the logger, moves much further (greedy S2 149) and stays ahead of every OPC variant (D1 − B2
  −0.86 [−1.16, −0.56], 1 / 36).
- **For Stage D (§11: only complementary methods):** A and D change different things (variance; bias) and both
  improved the true value; D − A is +0.33 (paired SE 0.09; 25 / 36). Shrinkage + reward model is the combination the
  evidence supports. B (negative) and C (null) give no basis for a combination. Stage D is not started; its budget is
  presented before launch.

### 16.12 Stage C summary: weak logging support, population regret and estimator bias

**Two kinds of error, measured separately.**
- **Population-objective regret M** is a property of an objective: the value its own optimum gives up when data are
  unlimited (q̂ at its population fit; §9). It is zero for any unbiased objective (raw DR, the oracle q) and for the
  calibration-aware likelihood (it nests the truth).
- **Fixed-policy estimator bias** is a property of an estimator at one policy with finite logs: E[V̂(π)] − V(π) at
  n = 25,000 with the fixed selection-time q̂ (§8; exact moments). A constant bias leaves an objective's optimum where
  it is; regret comes from how the bias changes across policies, so the two need not move together. h₀.₂ is more
  biased than h₀.₁ at the selected policies (−0.42 against −0.37) and has more regret (1.92 against 1.53), yet a
  higher true gain.
- **The rest of the shortfall** (finite-sample estimation, optimization and selection together) is the residual in
  TrueGain = available gain − M − residual (points, means over the 36 worlds; the available gain is the population
  runs' 4.97):

| arm | M | residual | TrueGain | recovery |
|---|---|---|---|---|
| B4 raw DR | 0 | 3.49 | 1.48 | 30% |
| B5 oracle-q h₀.₁ | 0 | 2.66 | 2.31 | 46% |
| B3 h₀.₁ | 1.53 | 1.46 | 1.99 | 40% |
| C harmonic CRM (h₀.₁'s population objective) | 1.53 | 1.48 | 1.96 | 39% |
| A h₀.₂ | 1.92 | 0.89 | 2.17 | 44% |
| B S2 penalty β = 0.01 | 2.20 | 1.17 | 1.60 | 32% |
| D D1 | 0.40 | 2.09 | 2.49 | 50% |
| B1 likelihood | 1.44 | 1.62 | 1.92 | 39% |
| B2 calibration-aware likelihood | 0 | 1.62 | 3.35 | 67% |

  Unbiased weighting objectives have no regret and the largest residual; shrinkage trades regret for residual (A);
  the D1 reward model trades the other way (D): its target is better but lies further from the logger and is harder
  to reach with 25,000 logged rows.

**The practical limitation: weak logging support.** The logger (greedy share 0.8) puts little probability where the
value optimum acts. θ_value*'s greedy policy has S2 ≈ 1,650 (median; 2,300–3,200 in R2–C1, ≈ 2·10⁷ in S1), an exact
ESS fraction of 0.1% and 19% of its mass on importance weights above 100. Every OPC objective, raw or shrunk, settles
where its estimate is reliable: greedy S2 11–21, ESS 5–9%. The residual column is the cost: even with the true q (B5)
the weighting objective recovers 46% of the available gain. Shrinkage, support penalties and variance penalties
re-weight the trade near the logger; none creates support. A better reward model helps by carrying more of the
estimate outside it (D1, 50%), and the calibration-aware likelihood, which needs no weights, recovers 67%. Under
these conditions the weak support, not the choice of correction, bounds what OPC can learn; the logger's exploration
(Stage E's greedy share) is the direct lever.

## 17. Stage D proposal (draft, 2026-10-14; not launched, awaiting approval)

**Question.** Does stronger harmonic shrinkage (λ = 0.2) add to the D1 reward model? A lowered the residual at a
higher regret; D lowered the regret at a higher residual (§16.12). The combination is justified only if the effects
are complementary.

**Arms.** A 2 × 2 factorial in the reward model (D0, D1) and λ (0.1, 0.2), on the 36 development worlds, paired with
Stage C (same worlds, rows, search space, sampler and trial seeds):
- B3 = (D0, 0.1), A = (D0, 0.2), D = (D1, 0.1): Stage C's trials, reused;
- **AD = (D1, 0.2)**, `shared_lr_opc_h0.2_d1`: new, 720 trials;
- references, reused: B1 likelihood, B2 calibration-aware likelihood, B4 raw DR, B5 oracle-q h₀.₁;
- optional reference (not a practical method): oracle q at λ = 0.2, `shared_lr_opc_h0.2_oq`, 720 trials: shrinkage
  with a perfect reward model, the ceiling of the q̂ axis.
No other λ or reward model; no arm added after results.

**Estimands.** Primary: AD − D (does shrinkage add to D1?), native TrueGain, paired over the 36 worlds, mean [95% CI]
and worlds higher. Secondary: AD − A; the interaction I = (AD − D) − (A − B3); AD − B2; per dataset and regime.
Decision: AD − D's CI above 0 → AD is Stage E's candidate; CI including 0 → D1 alone (simpler) stays the candidate, a
null reported as such; CI below 0 → a negative result.

**Expected interaction (hypotheses, not claims).** Harmonic DR's bias at a fixed policy is −E_x Σ_a π (1 − h_λ(w)/w)
(q − q̂): shrinkage's bias is the reward model's error weighted by the shrinkage factor. D1 reduces |q − q̂|, so
shrinkage becomes cheaper (positive interaction). The variance of g(w)(r − q̂) is mostly reward noise q(1 − q), which
no q̂ removes, so shrinkage's variance benefit should persist. Against additivity: (i) as λ grows the estimate leans on
q̂ where weights are large, so D1's extrapolation error outside the logger's support enters undamped (S1, where D1's
M is still 1.15); (ii) shrinkage pulls policies toward the logger, against D1's better but more distant target; (iii)
D1's regret gain (1.53 → 0.40) leaves little population room, so further gains must come from the residual.

**Diagnostics (§8–§10).**
- Population (§9 extended): harmonic λ = 0.2 with the D1 population fit on the 36 worlds (M of AD's objective);
  the oracle-q objectives have M = 0.
- Fixed-policy estimator error: the oracle evaluator, R = 200, at the selected softmax and greedy policies of B3, A,
  D, AD (and the references), the source and θ_value*, twice: with D0 and with D1 as the fixed q̂ (`--reward-model
  lowrank`), so each arm's own training and selection estimators are measured; bias, SD, RMSE, coverage, the paired
  improvement's error; exact S2, ESS fraction, Kish ESS, maximum weight and mass on w > 10 / 100.
- Gradients (§10a): h_λ with D0 and D1 at λ 0.1 and 0.2, at the source, the half-value state and the selected policy:
  the 4 pilot worlds (B3's pilot policy) and 4 development worlds (R2, S1 × ml, kuairand, seed 100; AD's policy);
  bias norm, cosine with the true gradient, variance over 100 datasets, SNR, MSE.
- Policy improvement: TrueGain (exact), recovery, and the decomposition available − M − residual per arm.

**Objective, optimization or selection.** M isolates the objective (no data, no optimizer). The fixed-policy errors
compare estimators at the same policies. Every arm has the same search space, 20 trials and paired trial seeds; the
trial mean (a typical run) and the best trial (oracle choice among 20) separate optimization from selection; the
common selector (D0-based, identical for every arm) against the native one separates the selector; the adaptivity of
selection is measured with the matching q̂. A better objective shows as a lower M and a better trial mean; better
selection only as native minus trial mean.

**Budget and stopping.** Measured costs (Stage C): a D1 arm about 22 s per trial.
- Core: AD training about 5.5 GPU-hours (720 fits); population about 3; oracle (D0 and D1 q̂) about 3.5; gradients
  about 1. About 13 GPU-hours, about 6–8 hours wall-clock: above the 12-hour cap, presented here.
- Optional: the oracle-q λ = 0.2 reference about 3 GPU-hours (720 fits); AD on the 18 pilot worlds (10 trials) about
  1.2.
- Before launch: the arm code tested, and Stage C's B3 and D trials reproduced bit-identically at the new commit on
  one world.
- Stop: AD with more than 5% diverged trials (unstable, reported); GPU memory above 45 GB (pause); cost above 1.5× the
  estimate (pause and report). No tuning after results; every result kept. The confirmation seeds 900–903 stay
  untouched until Stage F.

