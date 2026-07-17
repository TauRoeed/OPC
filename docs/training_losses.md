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
during policy training.

## 2. Full-study objectives at a glance

### OPC arm

OPC uses the unified `kl_crm` loss with logged propensities and the log trick.
Every OPC Optuna trial uses all three terms:

```text
OPC loss
    = negative SNDR log-trick surrogate
    + kl_gamma  * batch-MC KL penalty
    + crm_lambda * CRM variance penalty
```

There is no Optuna choice between separate KL and CRM losses. Optuna tunes both
regularization strengths in the same objective.

### No-propensity arm

The no-propensity baseline uses `naive`. It is deliberately simple:

- no propensity or importance weight
- no reward model
- no direct-method term
- no DR or SNDR term
- no KL penalty
- no CRM penalty
- no log trick

Its loss is ordinary pathwise gradient descent on observed reward multiplied by
the learned probability of the logged action:

```text
naive loss = -(1 / n) * sum_i [r_i * pi_i]
```

Minimizing this loss increases the learned probability of logged actions that
received larger rewards. Zero-reward observations contribute zero directly to
the loss.

## 3. OPC training loss in detail

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

The OPC implementation uses a policy-gradient surrogate. Probabilities used as
coefficients are detached (treated as constants), while gradients flow through
log policy probabilities.

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

### 3.6 Complete OPC loss

```text
OPC loss
    = -(1 / n) * sum_i SNDR_log_i
      + kl_gamma * KL_MC
      + crm_lambda * sqrt(sample_variance(u) / n + epsilon)
```

OPC Optuna parameters:

- `lr`
- `num_epochs`
- `batch_size`
- `lr_decay`
- `kl_gamma`
- `crm_M`
- `crm_lambda`

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

No-propensity Optuna parameters:

- `lr`
- `num_epochs`
- `batch_size`
- `lr_decay`

## 5. Validation scoring

Optuna maximizes a conservative lower confidence bound:

```text
R_hat = mean(row_value)
SE    = sample_standard_deviation(row_value) / sqrt(n)
t     = 97.5th percentile of Student-t with n - 1 degrees of freedom

validation score = R_hat - t * SE
```

Larger is better.

### 5.1 OPC validation row value

OPC keeps the doubly robust validation estimator:

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

The true simulator/on-policy reward is logged for later analysis, but Optuna
does not use it to select a trial.

## 6. Legacy optional losses

The CLI still accepts legacy losses for experiments:

- `sndr`: negative SNDR surrogate
- `kl`: negative SNDR surrogate plus KL penalty
- `ipw`: inverse-propensity reward objective
- `crm`: standalone clipped IPS risk plus CRM variance
- `naive`: pure observed reward times learned action probability

The full study defaults remain:

```text
OPC:           kl_crm, logged propensities, log trick enabled
no-propensity: naive, no propensities, log trick disabled
```

## 7. Numerical checks

`tests/test_custom_losses.py` checks that:

- the unified OPC loss produces finite, nonzero gradients without NaNs
- the naive loss equals `-mean(r_i * pi_i)`
- the naive loss produces finite, nonzero pathwise gradients
