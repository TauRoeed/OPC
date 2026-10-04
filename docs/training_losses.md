# Training Losses and Validation Scoring

This is the source of truth for the objectives used by the full Optuna study.
It intentionally uses plain-text equations so it renders consistently on GitHub,
in Cursor, and in ordinary text viewers.

Relevant code:

- `models/custom_losses.py`
- `training/trainer_trials.py`
- `training/run_full_study.py`

## 1. Notation

For each logged sample `i`:

- `x_i`: context/user
- `a_i`: action selected by the logging policy
- `r_i`: observed reward
- `pi_b_i`: logging-policy probability of the logged action
- `pi_i`: learned-policy probability of the logged action
- `q_i`: reward-model prediction for the logged action
- `q(x_i, a)`: reward-model prediction for any action
- `n`: batch or validation-split size

The learned policy is a softmax over the collaborative-filtering embeddings:

```text
pi_theta(a | x) = softmax(policy logits for context x)[a]
```

The reward model is trained separately on the regression split and is frozen
during policy training (when ``--reward-model regression``). Alternative
analytic sources skip fitting; see Section 1.1.

### 1.1 Shared reward model (`--reward-model`)

OPC DM / DR / SNDR terms need `q(x, a) ≈ E[r | x, a]`. The full study builds
one shared q-source per condition (same for OPC and no-propensity). Flag:

```text
--reward-model {regression, logging_score, oracle}
```

Default `regression`: fit a logistic `RegressionModel` on the interaction features
`[our_x, our_a, our_x ⊙ our_a]` (`--reward-features`; `concat` = `[our_x, our_a]`, the older
model) of the noisy vectors. Predictions are frozen for policy training and validation.

Its data (`--reward-data`; the study runners default to `train` with 5 cross-fitting folds):

- `train` (default): one model per train size, fit on that size's own training rows and shared by
  every arm, so each arm uses only the n logged rows it is given (the reward model and the policy
  share one budget). The reg slice is still drawn, so the train and validation rows are the same as
  in `external` mode (a matched comparison); condition folders get `__qhat=train`.
- `external` (the runs before 2026-09-26): a separate regression slice of the logged sim
  (`--shared-regression-size`, 50k rows), fit once per condition and used at every train size. The
  slice is extra data that only the arms using `q` (OPC, DM-only, the tempered logger's selection)
  benefit from: at 5k, a q fit on 50k rows beats the logger's own ranking by 2–5 CTR points on its
  own, while one fit on 5k rows does not.
- `--crossfit-folds K` (with `train`; default 5, off with `external`): users are split into K folds, fold k's model is fit on the
  other folds' training rows, and the training losses (SNDR, DM) take each user's `q` from the model
  that never saw that user's rows (`CrossFitScoresLookup`). Validation scoring and selection keep the
  model fit on all training rows (validation rows are never training rows). The cross-fitted `q` is
  built once per train size in the form the full model uses (a dense matrix when the full one is
  materialized, else one linear form over all folds), so a training batch costs the same:
  0.08 vs 0.09 ms per 1,024 rows on ml, and a condition's run time does not change. Folders get
  `__cf=K`.

`logging_score`: no fit. Use the same CTR link as the simulator on the
**noisy** logging embeddings:

```text
logits(x, a) = (our_x[x] · our_a[a]) / T_env
q(x, a)      = 1 / (1/ctr + exp(-logits(x, a)))
```

`oracle`: same CTR link on the **clean** environment embeddings
(`env.emb_x`, `env.emb_a`). Simulation diagnosis only — not available in a
real logged-bandit setting. If SNDR/DR improves a lot under `oracle` but not
under `regression`, the learned reward model is a likely bottleneck.

IPW and the no-propensity naive loss do not use `q` in the policy gradient
(`needs_qhat=False`); they still share the same bundle when DR validation /
selection needs it.

Relevant code: `AnalyticRewardModel` and `fit_shared_regression_bundle` in
`training/trainer_trials.py`.

## 2. Full-study objectives at a glance

Supported policy-loss names (`VALID_POLICY_LOSSES` / `--policy-losses`):

```text
dr (working development default), sndr, kl_crm, kl, ipw, crm, naive, dm
```

If more than one name is passed, Optuna treats `policy_loss` as a categorical.

### OPC arm

OPC uses logged propensities (`propensity_mode="logged"`) and IW / DR-style
losses: the reward model's value of the policy plus the weighted correction w·(r − q̂), no KL and no
CRM.

**Working development default (since f5cade9 (2026-09-27); revalidated on corrected logs 2026-10-04).** OPC
trains `dr` with the direct gradient (`--opc-gradient direct`), so the named estimate is the objective being
optimized. Its weights are `harmonic:0.1` (Metelli et al. 2021), and selection keeps `clip:10` with the 95% lower
bound. This is the working method for development runs, not the final paper choice.

The original choice (section 9, the decision record) rested on runs with the buggy logging simulator
(69fffab..c11b2b3). The revalidation on the fixed simulator
([simulator_fix_opc_revalidation_20261004.md](simulator_fix_opc_revalidation_20261004.md), Phase 2) re-tuned it:
- **weights:** a screen of raw, clip M ∈ {3, 10, 30, 100}, Su shrink λ ∈ {10, …, 10⁴} and harmonic λ ∈ {0.003, …,
  0.5}. Every regularized weighting beats raw DR, and harmonic 0.1–0.3 lead; `harmonic:0.1` is the most robust
  across sizes. `shrink:100` (Su et al. 2020) is within noise of it and is a robustness alternative; raw DR
  (`none`) stays the unregularized reference;
- **regime dependence:** the cap of 10 shared by harmonic:0.1 and shrink:100 makes OPC fragile to a misspecified
  reward model at large n: with concat features they lose about 4.5 points at 100k, raw DR nothing. Raw DR costs
  0.77 points with a well-specified q̂ (revalidation §3C). It is the robustness alternative wherever q̂ may be
  misspecified;
- **objective:** exact SNDR (`--sn-scope exact`) adds nothing over DR with harmonic weights;
- **gradient:** the log trick is never better than the direct gradient;
- **search space:** the study now searches lr 1e-4–2e-3 and 5–30 epochs (`--lr-range 1e-4 2e-3 --epochs-range 5
  30`; the search-space paragraph of section 4); the older range's optimum sat at its edge;
- **sharpness and regularization:** the learnable logit scale stays on; weight decay and post-hoc tempering stay off.

The objective variants remain available for reproducibility and diagnostics:
- **Legacy SNDR** (`sndr --sn-scope batch`) divides the correction by the minibatch mean weight, so
  its objective changes with the batch size.
- **`sndr --sn-scope global`** divides by a full-data mean weight held fixed for each epoch. It is not
  exact SNDR (section 3.4).
- **`sndr --sn-scope exact --opc-gradient direct`** (2026-10-04) follows the gradient of the full-data
  SNDR ratio. Its two means are refreshed every epoch (section 3.4). It is the legitimate SNDR
  alternative compared in the revalidation (`docs/simulator_fix_opc_revalidation_20261004.md`).
- **The previous defaults** are reproduced by `--policy-losses sndr --sn-scope batch --opc-gradient
  log-trick --train-weights shrink:100`. That configuration reproduces the OPC arm of the runs before
  f5cade9, and the H1 runner keeps it.

Legacy / ablation: `--policy-losses kl_crm` restores the unified loss

```text
OPC loss (kl_crm)
    = negative SNDR log-trick surrogate
    + kl_gamma  * batch-MC KL penalty
    + crm_lambda * CRM variance penalty
```

In `run_full_study.py`, OPC sets `use_log_trick_fixed=True` (not tuned by
Optuna). Other ablations: `--policy-losses ipw`, hurt-logging knobs,
`--optuna-selection r_hat`, etc.

**Importance-weight transforms.** Wherever inverse propensities are used, the weight
w = π_e/π_b goes through one transform (`utils/importance_weights.py`): `none`, `clip:M`
(`min(w, M)`), `shrink:λ` (Su et al. 2020, `λw / (w² + λ)`: at most √λ/2, falling back toward
0 past w = √λ) or `harmonic:λ`.

The harmonic transform comes from Metelli, Russo and Restelli (NeurIPS 2021), Definition 4.1: the
power-mean correction ((1 − λ) w^s + λ)^(1/s) with s = −1, which is `w / (1 − λ + λw)` for λ in
[0, 1]. It is the weighted harmonic mean of w and 1, with weights 1 − λ and λ:
- λ = 0 gives raw IS, and λ = 1 gives the constant 1.
- It is increasing and differentiable in w, and never exceeds 1/λ.
- Its derivative (1 − λ)/(1 − λ + λw)² is positive and bounded.

Their DR-λ estimator puts the corrected weight in the DR correction term, and their off-policy
learning ascends the estimate by its direct gradient. OPC does the same with `--policy-losses dr
--train-weights harmonic:λ --opc-gradient direct`; the study runner refuses harmonic training
weights under the log trick. The paper chooses λ from the sample size, a confidence level and the
2-Rényi divergence (λ*₂ = √(2 log(1/δ) / (3 I₂ n)), or a data-driven root, their Eq. (3)). Here λ is
fixed per run.

Two settings, not searched by Optuna:

- `--train-weights` (full-study default `STUDY_TRAIN_WEIGHTS` = `harmonic:0.1`; the trainer API and H1
  fall back to `DEFAULT_TRAIN_WEIGHTS` = `shrink:100`): the `sndr`, `dr`, `ipw` and `kl` training losses.
  `crm` / `kl_crm` keep their own clip `crm_M`, which Optuna searches.
- `--select-weights` (default `DEFAULT_SELECT_WEIGHTS`): the DR selection score and the post-hoc
  DR / SNIPW / SNDR estimates.

Both are recorded in `run_meta.json` and the summaries as the label (`train_weights`,
`select_weights`) and as `{train,select}_weight_mode` and `{train,select}_weight_param`.

`--train-weights none --select-weights clip:1` reproduces the older runs (unclipped training,
selection clipped at 1). Trials also log the ESS of the raw weights (`ess_raw`) next to the
transformed one, and `--log-select-weights SPEC ...` logs each trial's selection score under
other transforms, for tuning.

Defaults (interim, tuned on the true values with the spread logger on 2026-09-26): training
`shrink:100`, selection `clip:10`. For selection, every tight transform (clip:1 to clip:10,
shrink:10 to shrink:1000) picks trials worth at least 99.7% of the best trial's gain over the
logger, `clip:100` gets 93.5% and raw weights 77%. The training transform moves the true value by
at most about 0.2 points. These will be re-tuned on the sharpened logger, whose weights are heavier.

A fourth spec, `dm`, sets every weight to 0 so the DR score becomes the direct method: the DM-only
baseline's trial selection (`select_estimator="dm"`). It is not a training transform and not a
post-hoc setting.

### No-propensity arm

The no-propensity baseline always uses pure naive reward, regardless of the
OPC `--policy-losses` list (`_no_prop_policy_loss_types` returns `("naive",)`):

- `policy_loss = "naive"`
- `propensity_mode = "uniform"`
- no propensity or importance weight
- no reward-model / DM / DR / SNDR term
- no KL penalty
- no CRM penalty
- `use_log_trick_fixed = False`

Its loss is pathwise gradient descent on observed reward times the learned
probability of the logged action:

```text
naive loss = -(1 / n) * sum_i [r_i * pi_i]
```

Minimizing this loss increases the learned probability of logged actions that
received larger rewards. Zero-reward observations contribute zero directly to
the loss.

### Opt-in baselines (`--methods ... dm tempered_logger`)

Two more arms share the splits, the reward model, the selection weights and the search budget:

- `dm` (direct method): the policy is trained on the reward model alone and its trials are
  selected by the reward model alone. Loss `dm`, `-(1/n) sum_i sum_a pi(a | x_i) q_hat(x_i, a)`
  (the SNDR surrogate's DM term); selection `select_estimator="dm"`, i.e. the DR score with every
  weight 0. It uses no propensities and no logged rewards beyond those that fit `q_hat`.
- `tempered_logger`: no training. Each trial is the logger with its logits multiplied by a scale
  s searched in `TEMPER_SCALE_RANGE` (0.5 to 64, log-uniform), chosen by the same DR selection score
  as OPC. It measures how much a policy gains from sharpening (or flattening) the logger alone.

`--post-temper` chooses every trained policy's sharpness after training instead: its logits are
scaled by the factor in `POST_TEMPER_GRID` (0.25 to 16, 1 included) with the arm's best selection
score on validation, and the trial is that tempered policy from then on (`post_scale`). The ranking
never changes, only the sharpness; it avoids learning the scale through the noisy SNDR gradient.

`--learn-logit-scale` gives every trained policy (OPC, no-prop, DM) a learnable logit scale,
softmax(s · u·a / T) with s starting at 1 (`CFModel(learn_logit_scale=True)`). log s moves
`LOGIT_SCALE_SPEED` (30) times faster than the vector corrections under the same Adam steps, so
the searched lr × steps can reach a several-fold sharpening.

## 3. OPC training loss in detail (`kl_crm` ablation)

### 3.1 Importance weights

For OPC, the importance ratio for row `i` is:

```text
w_i = pi_i / pi_b_i
```

The SNDR term normalizes the correction by the batch mean weight:

```text
mean_w = (1 / n) * sum_i w_i
```

### 3.2 Direct-method value

The direct-method value for row `i` averages reward-model predictions under the
learned policy:

```text
DM_i = sum over actions a [q(x_i, a) * pi_theta(a | x_i)]
```

### 3.3 SNDR log-trick surrogate

The legacy OPC path (`sndr` with `--opc-gradient log-trick`, the default before f5cade9) and
`kl_crm` both use a policy-gradient surrogate (`use_log_trick=True`).
Probabilities used as coefficients are detached (treated as constants), while
gradients flow through log policy probabilities.

`stopgrad(z)` below means that `z` contributes a value but no gradient.

```text
w_i_detached = stopgrad(pi_i / pi_b_i)

correction_i
    = [w_i_detached / mean(w_detached)]
      * (r_i - q_i)
      * log(pi_i)

DM_log_i
    = sum over actions a [
          q(x_i, a)
          * stopgrad(pi_theta(a | x_i))
          * log(pi_theta(a | x_i))
      ]

SNDR_log_i = correction_i + DM_log_i

negative SNDR log-trick surrogate
    = -(1 / n) * sum_i SNDR_log_i
```

The correction uses observed reward to correct reward-model error. The direct
term supplies a policy-wide signal over all actions.

All probabilities passed to `log` are clamped to a small positive epsilon to
avoid `log(0)`.

When `use_log_trick=False` (direct path), the same SNDR value is used without
detaching or multiplying by `log pi`:

```text
SNDR_direct_i = DM_i + [w_i / mean(w)] * (r_i - q_i)
negative SNDR direct = -(1 / n) * sum_i SNDR_direct_i
```

This is the legacy minibatch form: `mean(w)` is over the minibatch. Section 3.4 derives what this and
the other two variants optimize.

### 3.4 What each objective variant optimizes

Notation for row i: weight w_i(θ) = π_θ(a_i|x_i) / π_b(a_i|x_i); transformed weight g(w_i), with
g(w) = w (`none`), min(w, M) (`clip:M`) or λw / (w² + λ) (`shrink:λ`); residual e_i = r_i − q̂(x_i, a_i);
DM value DM_i(θ) = Σ_a π_θ(a|x_i) q̂(x_i, a). n is the number of training rows and b the batch size.

Under the log trick the weights are detached and multiply ∇log π_θ(a_i|x_i) = ∇w_i / w_i. The ascent
direction on a minibatch B is therefore

```text
d_B(θ) = (1/b) Σ_{i∈B} [ ∇DM_i(θ) + (e_i / N) · (g(w_i) / w_i) · ∇w_i(θ) ]
       = (1/b) Σ_{i∈B} ∇[ DM_i(θ) + H(w_i(θ)) · e_i / N ],        H(w) = ∫_0^w g(t)/t dt,
```

where N is a number held fixed during backpropagation. The three variants differ only in N:

| variant | N | gradient through N | computed |
|---|---|---|---|
| `dr` | 1 | — | — |
| `sndr --sn-scope global` | c_e = mean over all training rows of g(w_j(θ_e)) | none (a number) | once per epoch, at its start (θ_e); stale after the first step |
| `sndr` (legacy, `--sn-scope batch`) | the batch's mean of g(w_j(θ)) | none under the log trick (the weights are detached) | every step, from the batch itself; legacy also uses 1/\|B\| in place of 1/b, so each batch's ratio counts once |

H is the weight whose exact gradient the log trick follows: H(w) = w for `none`; for `clip:M`,
H(w) = w up to M and M(1 + ln(w/M)) above it; for `shrink:λ`, H(w) = √λ · arctan(w/√λ), which rises
to √λ · π/2 (≈ 15.7 at λ = 100). So with a weight transform, the gradient is that of DM + H(w)e/N,
not of the transformed estimate DM + g(w)e/N. This convention is the same in all three variants.

**Log trick or direct gradient (`--opc-gradient`).** The objective the log trick optimizes and the
transformed DR estimate it is named after differ whenever the transform acts. The direct (pathwise)
gradient differentiates the transformed estimate itself: `--opc-gradient direct` trains OPC with
`use_log_trick=False`, whose ascent direction is exactly ∇ mean[DM_i + g(w_i) e_i / N]. The DM term is
exact either way. Per row, both move along ∇w_i = ∇π(a_i|x_i) / π_b(a_i|x_i), times e_i / N and a
coefficient: g(w)/w under the log trick and g′(w) directly.

| transform | w | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|---|
| `clip:10` | g(w) | 1 | 3 | 10 | 10 | 10 | 10 |
| | H(w) (log trick) | 1 | 3 | 10 | 21 | 33 | 44 |
| | log-trick coefficient g/w | 1 | 1 | 1 | 0.33 | 0.1 | 0.033 |
| | direct coefficient g′ | 1 | 1 | 0 above 10 | 0 | 0 | 0 |
| `shrink:100` | g(w) | 0.99 | 2.75 | 5 | 3 | 0.99 | 0.33 |
| | H(w) (log trick) | 1.0 | 2.9 | 7.9 | 12.5 | 14.7 | 15.4 |
| | log-trick coefficient g/w | 0.99 | 0.92 | 0.5 | 0.1 | 0.0099 | 0.0011 |
| | direct coefficient g′ | 0.97 | 0.77 | 0 | −0.08 | −0.0097 | −0.0011 |
| `shrink:10000` | g(w) | 1 | 3 | 9.9 | 27.5 | 50 | 30 |
| | H(w) (log trick) | 1 | 3 | 10 | 29 | 79 | 125 |
| | log-trick coefficient g/w | 1 | 1 | 0.99 | 0.92 | 0.5 | 0.1 |
| | direct coefficient g′ | 1 | 1 | 0.97 | 0.77 | 0 | −0.08 |

With raw weights (`none`) both coefficients are 1 and the two gradients are identical.

The log-trick coefficient is positive at every w. It follows a monotone objective: a row's weight
always counts in the direction of its residual, and the count saturates.

The direct gradient of the transformed estimate treats heavy rows differently:
- **`clip:M`:** rows above M get no gradient from the correction. Raising or lowering their
  probability is free.
- **`shrink:λ`:** past w = √λ the coefficient turns negative. The estimate is non-monotone in w (the
  shrunk weight falls back toward 0), so maximizing it lowers the probability of clicked heavy rows
  (e > 0) and raises that of unclicked ones (e < 0).

`tests/test_opc_gradient.py` checks both gradients against literal full-data autograd for `none`,
`clip:10`, `shrink:100` and `shrink:10000`, the closed forms of H against numerical integration, and
the sign reversal on a single clicked row with w = 20 under `shrink:100`. Section 9 reports the
paired comparisons (development runs).

- **`dr`** follows J_DR(θ) = (1/n) Σ_i [DM_i + H(w_i) e_i]. It is per-example additive, and with raw
  weights it is exactly the DR estimate. Summed over an epoch at fixed θ, its minibatch directions
  equal (n/b) · ∇J_DR for any batch size, the short final batch included.
- **`sndr --sn-scope global`** follows, during epoch e, J_e(θ) = (1/n) Σ_i [DM_i + H(w_i) e_i / c_e]:
  DR's direction with the correction divided by c_e. With raw weights at θ_e, J_e has the value of
  the full-data SNDR estimate, but not its gradient. The literal full-data SNDR ratio
  V(θ) = (1/n) Σ_i DM_i + Σ_i g(w_i) e_i / Σ_i g(w_i) has

  ```text
  ∇V = (1/n) Σ_i ∇DM_i + (1/(n c)) Σ_i g'(w_i) ∇w_i · (e_i − R),
       c = (1/n) Σ_i g(w_i),   R = Σ_i g(w_i) e_i / Σ_i g(w_i).
  ```

  At θ_e, ∇J_e differs from ∇V in three ways:
  - It lacks the term −(R/c) ∇c: the gradient through the denominator, which is self-normalization's
    baseline R.
  - It uses g(w)/w where ∇V has g'(w). The two coincide for raw weights.
  - After the first step, c_e is no longer c(θ).

  So `global` is a stop-gradient, epoch-stale SNDR surrogate. It is not exact SNDR.
- **`sndr --sn-scope exact`** (2026-10-04; direct gradient only) follows ∇V itself, at constants
  refreshed at the start of every epoch. With S = c and N = c·R computed over all training rows at θ_e
  (`training_utils.full_data_sn_constants`, with the training q̂), each row contributes
  DM_i + g(w_i)(e_i − N/S)/S. This is DR with the self-normalized mean residual R = N/S as a baseline,
  and the correction divided by S. Summed over the rows, its gradient at θ_e is exactly ∇V, for any
  batch size, the short final batch included. Within the epoch, S and N are stale, as in `global`.
  It refuses the log trick. `tests/test_sndr_exact.py` checks the direction against the literal ratio's
  autograd for `none`, `clip:2` and `shrink:4`.
- **Legacy `sndr`**: each batch follows its own stop-gradient ratio,
  (1/|B|) Σ_{i∈B} ∇[DM_i + H(w_i) e_i / w̄_B], where w̄_B is the batch's mean weight. That mean depends
  on which rows share the batch, so the epoch's direction, and the objective, change with the batch
  size. With the log trick off, the batch denominator carries its gradient, and one batch of all rows
  gives exactly ∇V; OPC runs with the log trick on.

Under the log trick the reported loss value is a surrogate: minus the sum of the detached
coefficients times log π. Only its gradient carries the meaning above; its value does not estimate
the policy's value. `tests/test_objective_gradients.py` checks each statement against literal
full-data autograd at fixed parameters, including the identity ∇V − ∇J_e = −(R/c) ∇c. It also checks
that the trainer sets the global normalizer once per epoch, and that the normalizer is stale after
every step that follows.

**Minibatch weighting (the short final batch).** The training DataLoader keeps the final short
batch (no `drop_last`), and the study's train sizes are not multiples of its batch sizes: 25,000 rows
in batches of 2048 leave a last batch of 424. For per-example additive losses (`dr`, `sndr`/`kl` with
`--sn-scope global`, `dm`, `naive`, `ipw`), `training_utils.minibatch_loss` scales a short batch's
mean by rows / b, so every row weighs 1/b in every epoch. Before this fix (commits up to b584edc),
the last batch's mean counted as much as a full batch's and upweighted its rows by b / rows (1.1× to
4.8× in the study's grid). This changes the dm-only and no-propensity arms slightly (their last batch
only), and `dr` and `global` compared with b584edc. Legacy SNDR and the CRM variance losses are
per-minibatch statistics: they keep one equal-weight mean per batch, as before. This accounting is of
gradients. Adam's per-coordinate step normalization and the per-step gradient-norm clip (max norm 1)
act on each step's direction, identically for every variant.

### 3.5 Batch Monte Carlo KL penalty

The KL term discourages the learned policy from moving too far from the logging
policy. It is estimated only at logged actions:

```text
KL_MC = (1 / n) * sum_i [log(pi_b_i) - log(pi_i)]
```

Because logged actions were sampled from the logging policy, this is a
single-action Monte Carlo estimate of `KL(pi_b || pi_theta)`.

Its contribution to the total loss is:

```text
kl_gamma * KL_MC
```

### 3.6 CRM variance penalty

First clip each importance ratio at Optuna parameter `crm_M`:

```text
clipped_w_i = min(w_i, crm_M)
u_i         = -r_i * clipped_w_i
```

Then penalize the standard error of these per-row risks:

```text
CRM_variance
    = crm_lambda * sqrt(sample_variance(u) / n + epsilon)
```

This discourages policies whose estimated risk has high sample variance. In the
unified `KLCRMPolicyLoss`, gradients flow through `clipped_w_i`, so
`crm_lambda` changes the policy update.

### 3.7 Complete `kl_crm` loss (legacy / ablation)

```text
OPC loss (kl_crm)
    = -(1 / n) * sum_i SNDR_log_i
      + kl_gamma * KL_MC
      + crm_lambda * sqrt(sample_variance(u) / n + epsilon)
```

OPC Optuna parameters for `kl_crm`:

- `lr`
- `num_epochs`
- `batch_size`
- `lr_decay`
- `kl_gamma`
- `crm_M`
- `crm_lambda`

(`use_log_trick` is fixed True in the full study; `policy_loss` is fixed when
only one `--policy-losses` value is passed. Default full-study loss is `sndr`,
which only searches lr / epochs / batch / lr_decay.)

## 4. No-propensity training loss in detail

For each row:

```text
naive_value_i = r_i * pi_i
```

The optimizer minimizes:

```text
no-propensity loss = -mean(naive_value)
```

This is implemented by `NaiveRewardPolicyLoss` with:

```text
policy_loss       = "naive"
propensity_mode   = "uniform"
use_log_trick     = false
```

The `scores` reward-model tensor and `pscore` propensity tensor are ignored by
this loss. Gradients flow directly through `pi_i`.

(The class also supports a log-trick variant `-mean(r * log pi)`, but the full
study keeps it off.)

No-propensity Optuna parameters:

- `lr`
- `num_epochs`
- `batch_size`
- `lr_decay`

**Search space (`trainer_trials.DEFAULT_SEARCH_SPACE`; OPC, no-propensity and DM-only share it).**
- **Defaults** (every run before 2026-10-04):
  - `lr` log-uniform on [1e-4, 1e-3];
  - `num_epochs` uniform on 5–25;
  - `lr_decay` uniform on [0.8, 1];
  - `batch_size` from `batch_schedule` by train size (512 / 1024 / 2048 up to 25k rows, 2048 / 4096 / 8192 up to
    100k).
- **Overrides** (both runners): `--lr-range`, `--epochs-range` and `--lr-decay-range` replace a range.
- **Revalidated study configuration (2026-10-04, corrected simulator):** `--lr-range 1e-4 2e-3 --epochs-range 5 30`,
  the lr decay and batch schedule unchanged, no weight decay. Chosen to minimize the largest loss of any trained arm
  against its own best candidate range (revalidation §2.3). The code defaults above are kept so that older commands
  reproduce.
- **Weight decay.** `--weight-decay-range LOW HIGH` adds AdamW weight decay, log-uniform. Every trained parameter
  starts at 0, so the decay pulls toward the logger. It needs `--sampler random`, and each trial draws its decay from
  its own seeded stream. The other parameters, and the pairing of trials across runs and arms, are therefore
  unchanged.
- **Recording.** Each run records its search space in `run_meta.json` and the manifest, and each trial its
  `param_weight_decay`, wall time (`trial_time_s`) and whether training stopped at a non-finite gradient
  (`diverged`). A diverged trial keeps its last finite parameters.

## 5. Validation scoring and Optuna selection

Each trial builds a per-row validation vector `row_value`, then forms:

```text
R_hat = mean(row_value)
SE    = sample_standard_deviation(row_value) / sqrt(n)
t     = 97.5th percentile of Student-t with n - 1 degrees of freedom
ci_low = R_hat - t * SE
```

`--optuna-selection` chooses what Optuna maximizes
(`VALID_OPTUNA_SELECTION`):

```text
ci_low         (default)  maximize R_hat - t * SE
r_hat                     maximize R_hat only
actual_reward             maximize true simulator reward (oracle; debug)
```

Larger is better. Recent hurt-logging ablations often use `r_hat` together with
`sndr` or `ipw`.

**Sampler and paired comparisons (`--sampler`).** The default TPE sampler draws each size's first 10
trials at random (seeded, so the same in every run) and then adapts to the trial values; every size
after the first also starts from the previous size's best trial (a warm start). Runs that differ only
in the objective therefore share just part of their configurations, and the rest follow each
objective's own search. `--sampler random` is seeded random search with no warm start: trial k has the
same configuration in every run with the same seeds and search space, and its seed (initialization,
batch order) depends only on (seed, arm, train size, trial number). Runs that differ only in
`--policy-losses`, `--sn-scope` or `--train-weights` are then a paired, replayed comparison: the same
configurations trained under each objective, compared trial by trial and after each objective's own
selection. The default TPE runs are the independent re-tuning. `--stage` records whether a run is a
development run (designing the method; the default) or a confirmatory one (the frozen method on fresh
seeds and conditions).

### 5.1 OPC validation row value

OPC keeps the doubly robust validation estimator (not self-normalized). For
Optuna / `ci_low` scoring, IW goes through `--select-weights` (shown for `clip:M`):

```text
DM_i = sum over actions a [q(x_i, a) * pi_theta(a | x_i)]
w_i  = min(pi_i / pi_b_i, M)

OPC row_value_i = DM_i + w_i * (r_i - q_i)
```

This evaluation formula uses the actual probabilities, not the detached
log-trick surrogate used to create training gradients. No-propensity never
applies this transform (pure naive `r_i * pi_i`).

### 5.2 No-propensity validation row value

No-propensity validation matches its pure naive training target:

```text
no-prop row_value_i = r_i * pi_i
```

It uses no propensities, reward-model predictions, DM, DR, or SNDR.

True simulator reward (`actual_reward`) is always logged. It is used for Optuna
selection only when `--optuna-selection actual_reward`.

## 6. Other supported policy losses

CLI `--policy-losses` accepts any of:

### 6.1 `dr` (working development default) and `sndr`

`dr`: DM(q̂) plus the weighted correction, without self-normalization, no KL, no CRM. `sndr`: the same,
self-normalized per minibatch (`--sn-scope batch`, legacy; the default before f5cade9) or by the
full-data mean weight held fixed for each epoch (`--sn-scope global`). Section 3.4 derives what each
variant optimizes.

### 6.2 `kl`

```text
Loss = negative SNDR surrogate + kl_gamma * KL_MC
```

### 6.3 `ipw`

Inverse-propensity reward objective at the logged action:

```text
w_i = pi_i / pi_b_i

log-trick:   Loss = -(1 / n) * sum_i [stopgrad(w_i) * r_i * log(pi_i)]
direct:      Loss = -(1 / n) * sum_i [w_i * r_i]
```

### 6.4 `crm`

Standalone Counterfactual Risk Minimization (`CRMPolicyLoss`):

```text
clipped_w_i = min(w_i, crm_M)

log-trick IPS:  -(1 / n) * sum_i [stopgrad(clipped_w_i) * r_i * log(pi_i)]
direct IPS:     -(1 / n) * sum_i [stopgrad(clipped_w_i) * r_i]

+ crm_lambda * sqrt(sample_variance(u) / n + epsilon)
```

with `u_i = -r_i * clipped_w_i`. Under the log trick, the variance term also
detaches `clipped_w_i` (unlike unified `kl_crm`, where CRM variance keeps
gradients through the clipped weights).

### 6.5 `naive`

Same as the no-propensity loss (section 4). Available as an OPC ablation loss
name, but the no-propensity arm always uses it.

Full-study defaults:

```text
OPC:           dr, propensity_mode=logged, direct gradient (--opc-gradient direct),
               --train-weights harmonic:0.1 / --select-weights clip:10 (working development defaults)
no-propensity: naive, propensity_mode=uniform, use_log_trick fixed False
```

## 7. Logging-damage and study knobs

These do not change the loss formulas, but they change the logged data that the
losses see (`run_full_study.py` / `utils/policies.py`):

### 7.1 Representation bias (`--bias-configs`)

Biased user and item vectors for the logger, the reward model and the policy: a global
warp, a group offset and a per-vector offset, each at `none` / `low` / `medium` / `high`.
All three at `low` / `medium` / `high` keep 90 / 75 / 50 % of the signal, calibrated per
dataset. See [representation_bias.md](representation_bias.md).

### 7.2 Logging–uniform mix (`--logging-uniform-mix α`)

Softmax logging policy mixed with uniform over actions:

```text
pi_b(a | x) = (1 - α) * pi_softmax(a | x) + α / |A|
```

Sampling draws from softmax with probability `1-α`, else uniform; stored
pscores are always the exact mixture above. `α = 0` is off. Typical hurt values
are `0.2`–`0.5`.

### 7.3 Logging spread (`--logging-spread`) and logger sharpness (`--logger-greedy-share`)

The spread sets the temperature `T` at which the clean logger's effective number of items is
`spread × |A|` (default 0.5); that spread logger calibrates the click model. The actual logger is
then sharpened per condition: its temperature is lowered until it earns `--logger-greedy-share`
(default 0.8) of its own greedy CTR. The learned policies start at the logger's temperature.
`--logger-greedy-share off` keeps `T` (the logger before 2026-09-26). A sharper logger gives
heavier importance weights; see [representation_bias.md](representation_bias.md).

### 7.4 Log trick (`--opc-gradient`, `--no-log-trick`)

Full study:

```text
OPC:           use_log_trick fixed by --opc-gradient (direct, the default since f5cade9 (2026-09-27), or log-trick)
no-propensity: use_log_trick fixed False
dm:            use_log_trick fixed False
```

Section 3.4 derives what each form optimizes. `--no-log-trick` only matters for trainers that search
`use_log_trick` (`search_use_log_trick=False`). Every arm of the full study has it fixed, so the flag
does not change the study's training (before `--opc-gradient`, passing it left OPC on the log trick).

## 8. Numerical checks

`tests/test_custom_losses.py` checks that:

- the unified OPC loss produces finite, nonzero gradients without NaNs
- the naive loss equals `-mean(r_i * pi_i)`
- the naive loss produces finite, nonzero pathwise gradients

## 9. Development evidence: objective, gradient form and importance weights (2026-09-27)

> **Historical: buggy logging simulator.** Every run in this section used the logging simulator of 69fffab..c11b2b3,
> in which each logged action reused its user's random draw (users received nearly fixed actions; the stored
> propensities were the logger's softmax probabilities). The empirical comparisons are superseded by the re-tuning
> on the fixed simulator ([simulator_fix_opc_revalidation_20261004.md](simulator_fix_opc_revalidation_20261004.md),
> Phase 2). The analytic statements of section 3.4 are unaffected. The section is kept as the record.

All runs below are **development runs**, used to design the method. They are not confirmatory: once
the objective and weighting are frozen, the paper protocol is evaluated on fresh seeds and conditions.
Runs are listed in `artifacts/full_study/run_registry.csv`, and the decision record is
`docs/decision_record_opc_objective_weighting.md`.

**Protocol.** OPC arm only:
- **Grid:** ml, kuairand and anime × medium and high bias × seeds 100 and 101, at train sizes 5k, 25k
  and 100k, with 20 trials per size.
- **Final setting:** logger share 0.8; q̂ on each size's own rows with 5-fold cross-fitting; 20,000
  validation rows; learnable logit scale; selection by the DR lower bound with clip:10.

Paired runs use `--sampler random`, so the same 20 configurations and trial seeds appear in every run
with no warm start. Differences are points of true CTR, averaged within each condition, with a 95% CI
over the 6 conditions per bias × size cell. "Per trial" compares identical configurations; "selected"
compares each run's own selected policy. All fixed-code results below are on code at or after b5efc7d,
which has the short-batch fix.

**Objective, all with the log trick and shrink:100 (paired, 120 trials per cell).**

| comparison | per trial, range over the 6 cells | trials where the first is better, per cell | selected |
|---|---|---|---|
| legacy SNDR − `dr` | −0.03 to −0.32 (all CIs exclude 0) | 1–13 of 120 | −0.07 to −0.41 |
| global SNDR − `dr` | −0.02 to −0.26 (all CIs exclude 0) | 0–5 of 120 | −0.07 to −0.41 |
| global − legacy SNDR | +0.00 to +0.05 | 89–113 of 120 | −0.05 to +0.10 (all CIs include 0) |

**Gradient form, `dr` direct − log trick (paired).**

| weights | per trial | selected |
|---|---|---|
| `none` (identical gradients: control) | ±0.00, CIs within ±0.01; per-trial absolute difference: median 0.000, 90th percentile ≤ 0.001 | within ±0.04 |
| `clip:10` | −0.06 to +0.03, mixed | −0.14 to +0.18, mixed |
| `shrink:100` | −0.02 to +0.14; at 25k the CIs exclude or touch 0 (high +0.14 [+0.01, +0.27], medium +0.06 [+0.00, +0.11]) | −0.04 to +0.27 |

Under both `clip:10` and `shrink:100`, the direct form trains policies closer to the logger. From 25k
up they have:
- fewer rows with a weight above 10;
- a larger raw-weight ESS;
- a smaller learned logit scale.

**Weights under the log trick (`dr`, paired, against `shrink:100`).** No difference at 5k. From 25k
up:
- `clip:10` is 0.10–0.20 points lower per trial.
- `none` is 0.15–0.36 points lower per trial.
- Under the direct gradient, `clip:10` is 0.06–0.33 points lower per trial.

**Earlier runs on pre-fix code (b584edc: last batch upweighted; TPE re-tuning).** Each variant was
re-tuned by its own TPE search, against legacy SNDR (Test 2's cross-fitted 0.8 run):
- `dr` + shrink:100: +0.07 to +0.45 (four of six CIs exclude 0).
- `dr` + none / clip:10 / shrink:10000, and global SNDR: within ±0.27, mostly including 0.

TPE's first 10 trials per size are seeded random draws, so they are identical across these runs. On
those identical trials:
- `dr` − legacy: +0.05 to +0.36 per trial.
- global − legacy: +0.00 to +0.05.
- global − `dr`: −0.04 to −0.30.

These agree in direction with the fixed-code results.

**Final bounded comparison: weighting only (`dr`, direct gradient, paired, 120 trials per cell).**
Raw DR and `shrink:100` are the 3bed8de runs, which reproduce bit for bit on d40aaef (one condition
each, all 60 trials). The harmonic runs are on d40aaef. λ was fixed at the prespecified values 0.05, 0.1
and 0.2 (caps 20, 10 and 5); no other value was run.

Selected true CTR (%), mean over the 6 conditions:

| cell | raw (`none`) | `shrink:100` | `harmonic:0.05` | `harmonic:0.1` | `harmonic:0.2` |
|---|---|---|---|---|---|
| high 5k | 14.92 | 15.14 | 15.10 | 15.38 | 15.71 |
| high 25k | 16.60 | 17.25 | 17.22 | 17.33 | 17.58 |
| high 100k | 17.50 | 18.06 | 18.02 | 18.16 | 18.26 |
| medium 5k | 20.90 | 20.75 | 20.84 | 20.88 | 21.01 |
| medium 25k | 22.08 | 22.33 | 22.25 | 22.45 | 22.49 |
| medium 100k | 22.55 | 22.78 | 22.87 | 22.88 | 22.90 |

Paired differences in points (range over the 6 cells):

| comparison | per trial | trials better, per cell | selected |
|---|---|---|---|
| `shrink:100` − raw | −0.03 to +0.51 (CIs exclude 0 from 25k up) | 62–112 of 120 | −0.15 to +0.65 |
| `harmonic:0.05` − `shrink:100` | −0.01 to +0.07 (CIs include 0 except medium 5k) | 62–118 of 120 | −0.08 to +0.09 |
| `harmonic:0.1` − `shrink:100` | +0.09 to +0.23 (all CIs exclude 0) | 108–119 of 120 | +0.09 to +0.24 (2 of 6 CIs exclude 0) |
| `harmonic:0.2` − `shrink:100` | +0.13 to +0.49 (all CIs exclude 0) | 117–119 of 120 | +0.12 to +0.57 (4 of 6 CIs exclude 0) |
| `harmonic:0.2` − raw | +0.16 to +1.00 (all CIs exclude 0) | 115–120 of 120 | +0.11 to +0.98 |

Diagnostics of the selected policies:
- **Raw-weight ESS on validation (rows of 20,000):** raw 297–955; `shrink:100` 273–2,362; harmonic
  234–1,504.
- **Share of rows with w > 10:** 1.5–2.9% for every method.
- **Learned logit scale:** raw 3.3–18.8; `shrink:100` 2.5–6.8; harmonic 2.7–16.9. Over all trials: raw
  1.6–6.6, `shrink:100` 1.5–3.9, `harmonic:0.2` 1.5–4.5.
- **Selection-estimate error:** the validation DR point estimate minus the truth is +0.07 to +1.55
  points for every method, with every CI including 0. The lower bound sits below the truth in all but
  one cell.
- **Selection quality:** the Spearman correlation between the selection score and the truth is ≥ 0.73,
  and the selection regret is ≤ 0.26 points.

On these development conditions:
- Both smooth corrections improve on raw DR from 25k up.
- `harmonic:0.05` matches `shrink:100`. At λ = 0.1 and 0.2 the harmonic correction is ahead by 0.1–0.5
  points per trial.
- The effect of λ is monotone, and its best value is at the edge of the prespecified set.
- The choice among the smooth corrections moves the selected policy's true CTR by up to about
  0.6 points. That is comparable to the OPC − DM differences at medium bias in the budget-fair runs
  (+0.5 to +1.1).

**What exists, and what is retained.** Every option above is implemented, tested and recorded in
the run metadata. Since f5cade9 (2026-09-27) the working development defaults are `dr`, the direct gradient,
`harmonic:0.1` training weights and `clip:10` selection weights. They are not the final paper choice.
The standard comparison is `shrink:100` with the direct gradient; the reference is raw DR.
- **Out of the main future grid, unless there is a specific scientific reason:** the log-trick form
  of transformed weights (the arctan-saturated objective for `shrink:λ`), `clip:10` and
  `shrink:10000` as training weights, and legacy and global SNDR. They stay available for
  reproducibility and appendix diagnostics.
- **Raw DR** (`dr`, `none`) remains the unregularized baseline.
- **The final paper method's weighting and λ** stay open for the scientific reassessment, among the
  direct-gradient candidates compared above.
