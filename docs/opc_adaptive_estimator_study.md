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
