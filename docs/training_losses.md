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

Default `regression`: fit a logistic `RegressionModel` on
`concat(our_x, our_a)` from the regression slice of the logged sim (noisy
features). Predictions are frozen for policy training and validation.

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
kl_crm (default), kl, ipw, sndr, crm, naive
```

If more than one name is passed, Optuna treats `policy_loss` as a categorical.

### OPC arm

OPC uses logged propensities (`propensity_mode="logged"`) and IW / DR-style
losses. The full-study default is the unified `kl_crm` loss with the log trick
fixed on:

```text
OPC loss
    = negative SNDR log-trick surrogate
    + kl_gamma  * batch-MC KL penalty
    + crm_lambda * CRM variance penalty
```

In `run_full_study.py`, OPC sets `use_log_trick_fixed=True` (not tuned by
Optuna). Ablations can pass other losses, e.g. `--policy-losses sndr` or
`--policy-losses ipw`, often together with hurt-logging knobs and
`--optuna-selection r_hat`.

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

## 3. OPC training loss in detail (`kl_crm`)

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

The default OPC path uses a policy-gradient surrogate (`use_log_trick=True`).
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

### 3.4 Batch Monte Carlo KL penalty

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

### 3.5 CRM variance penalty

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

### 3.6 Complete OPC default loss

```text
OPC loss (kl_crm)
    = -(1 / n) * sum_i SNDR_log_i
      + kl_gamma * KL_MC
      + crm_lambda * sqrt(sample_variance(u) / n + epsilon)
```

OPC Optuna parameters for default `kl_crm`:

- `lr`
- `num_epochs`
- `batch_size`
- `lr_decay`
- `kl_gamma`
- `crm_M`
- `crm_lambda`

(`use_log_trick` is fixed True in the full study; `policy_loss` is fixed when
only one `--policy-losses` value is passed.)

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

### 5.1 OPC validation row value

OPC keeps the doubly robust validation estimator (not self-normalized):

```text
DM_i = sum over actions a [q(x_i, a) * pi_theta(a | x_i)]
w_i  = pi_i / pi_b_i

OPC row_value_i = DM_i + w_i * (r_i - q_i)
```

This evaluation formula uses the actual probabilities, not the detached
log-trick surrogate used to create training gradients.

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

### 6.1 `sndr`

Negative SNDR surrogate only (section 3.3), no KL, no CRM.

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

Full-study defaults remain:

```text
OPC:           kl_crm, propensity_mode=logged, use_log_trick fixed True
no-propensity: naive,  propensity_mode=uniform, use_log_trick fixed False
```

## 7. Logging-damage and study knobs

These do not change the loss formulas, but they change the logged data that the
losses see (`run_full_study.py` / `utils/policies.py`):

### 7.1 Noise levels (`--noise-levels`)

Combined-axis eps mix `(eps1, eps2, eps_meta)`:

```text
low:     (0.05, 0.05, 0.00)
medium:  (0.10, 0.15, 0.05)
high:    (0.20, 0.25, 0.10)
extreme: (0.35, 0.40, 0.20)
brutal:  (0.50, 0.50, 0.30)
```

Single-axis sweeps use the matching component only.

### 7.2 Logging–uniform mix (`--logging-uniform-mix α`)

Softmax logging policy mixed with uniform over actions:

```text
pi_b(a | x) = (1 - α) * pi_softmax(a | x) + α / |A|
```

Sampling draws from softmax with probability `1-α`, else uniform; stored
pscores are always the exact mixture above. `α = 0` is off. Typical hurt values
are `0.2`–`0.5`.

### 7.3 Policy temperature (`--policy-temperature`)

Softmax temperature for logging / evaluation dot-product policies (default
`1.0`). Larger temperature flattens `pi_b`.

### 7.4 Log trick (`--no-log-trick`)

Full study:

```text
OPC:           use_log_trick fixed True
no-propensity: use_log_trick fixed False
```

`--no-log-trick` disables the log-trick surrogate for searchable settings
(`search_use_log_trick=False`). With a fixed flag set by the study runner, Optuna
does not tune `use_log_trick`.

## 8. Numerical checks

`tests/test_custom_losses.py` checks that:

- the unified OPC loss produces finite, nonzero gradients without NaNs
- the naive loss equals `-mean(r_i * pi_i)`
- the naive loss produces finite, nonzero pathwise gradients
