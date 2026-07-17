# Training Losses and Validation Scoring

Living reference for what the Optuna learning trial actually optimizes.
Update this file when losses or selection scores change.

Code: `models/custom_losses.py`, `training/trainer_trials.py`.

---

## Setup

Logged bandit data: contexts $x_i$, actions $a_i \sim \pi_b(\cdot \mid x_i)$,
rewards $r_i$, propensities $\pi_b(a_i \mid x_i)$.

Learned policy $\pi_\theta(a \mid x)$ (softmax over CF embeddings).
Reward model $\hat{q}(x,a)$ is fit once on a held-out regression slice and frozen.

**Propensity modes**

| Mode | Training IW | Name in code |
|------|-------------|--------------|
| Off-policy (OPC) | $w_i = \pi_\theta(a_i \mid x_i) / \pi_b(a_i \mid x_i)$ | `propensity_mode="logged"` |
| No propensity | $w_i = 1$ | `propensity_mode="uniform"` |

---

## Default training loss: `kl_crm` (unified)

There is **no Optuna categorical split** between KL and CRM.
Every trial uses the same loss with both regularizer strengths tuned:

$$
\mathcal{L}_{\mathrm{kl\_crm}}
=
-\widehat{\mathrm{SNDR}}_{\mathrm{log}}
+ \gamma\,\widehat{\mathrm{KL}}
+ \lambda\,\widehat{\mathrm{CRM}}_{\mathrm{var}}
$$

Hyperparameters from Optuna: $\gamma$ (`kl_gamma`), $\lambda$ (`crm_lambda`),
clip $M$ (`crm_M`), plus `lr`, `num_epochs`, `batch_size`, `lr_decay`.

Log-trick is **on** for OPC in the full study.

### No-propensity baseline (naive GD)

No-propensity does **not** use `kl_crm`. It trains with:

- loss `sndr`
- `propensity_mode="uniform"` ($w_i = 1$)
- **no log-trick** (pathwise gradients through $\pi_\theta$)
- **no KL**, **no CRM**

So the training objective is ordinary gradient descent on the naive value surrogate:

$$
\mathcal{L}_{\mathrm{no\_prop}}
=
-\frac{1}{n}\sum_i
\Bigl(
\mathrm{DM}_i + (r_i - \hat{q}_i)
\Bigr),
\qquad
\mathrm{DM}_i = \sum_a \hat{q}(x_i,a)\,\pi_\theta(a \mid x_i).
$$

(The residual $r_i-\hat{q}_i$ is constant w.r.t. $\theta$ when $w_i=1$, so gradients come from the DM term.)

Optuna only tunes `lr`, `num_epochs`, `batch_size`, `lr_decay` for this arm.

### SNDR log-trick surrogate (OPC)

With $\log_\varepsilon \pi = \log(\max(\pi,\varepsilon))$:

**Log trick** (OPC default): detach policy in IW / DM coefficients; multiply by $\log \pi$.

$$
\begin{aligned}
\widehat{\mathrm{SNDR}}_{\mathrm{log},i}
&=
\underbrace{\frac{w_i^{\perp}}{\overline{w^{\perp}}}(r_i-\hat{q}_i)\,\log \pi_\theta(a_i \mid x_i)}_{\text{correction}}
\\
&\quad+
\underbrace{\sum_a \hat{q}(x_i,a)\,\pi_\theta^{\perp}(a \mid x_i)\,\log \pi_\theta(a \mid x_i)}_{\text{DM}}
\end{aligned}
$$

where ${}^{\perp}$ = stop-gradient, $w_i^{\perp}$ uses detached $\pi_\theta$ (or $w_i=1$ if no-prop),
and $\overline{w^{\perp}}$ is the batch mean of $w^{\perp}$.

Training minimizes the negative mean:

$$
-\widehat{\mathrm{SNDR}}_{\mathrm{log}} = -\frac{1}{n}\sum_i \widehat{\mathrm{SNDR}}_{\mathrm{log},i}.
$$

### Batch MC KL (logged actions)

$$
\widehat{\mathrm{KL}}
=
\frac{1}{n}\sum_i
\bigl(\log \pi_b(a_i \mid x_i) - \log \pi_\theta(a_i \mid x_i)\bigr)
\approx
\mathbb{E}_{a\sim\pi_b}\bigl[\mathrm{KL}(\pi_b \Vert \pi_\theta)\bigr]
\quad\text{(single-action MC)}.
$$

### CRM variance penalty

$$
u_i = -r_i\cdot \mathrm{clip}(w_i, M),
\qquad
\widehat{\mathrm{CRM}}_{\mathrm{var}}
=
\lambda\sqrt{\frac{\widehat{\mathrm{Var}}(u)}{n}+\varepsilon}.
$$

In `KLCRMPolicyLoss`, gradients flow through clipped $w_i$ in $u$, so $\lambda$
affects parameter updates (unlike standalone `CRMPolicyLoss`, which detaches $\mathrm{Var}(u)$
under the log trick).

---

## Legacy losses (still available via `--policy-losses`)

| Name | Formula |
|------|---------|
| `sndr` | $-\widehat{\mathrm{SNDR}}$ |
| `kl` | $-\widehat{\mathrm{SNDR}} + \gamma\,\widehat{\mathrm{KL}}$ |
| `ipw` | $-\frac{1}{n}\sum_i w_i^{\mathrm{pg}}\, r_i\, g_i$ with $g_i=\log\pi$ (log) or $1$ (direct) |
| `crm` | clipped IPS risk + $\lambda\sqrt{\mathrm{Var}(u)/n}$ |

Standalone CRM is clipped IPS risk

$$
-\overline{r\cdot \mathrm{clip}(w,M)^{\perp}\cdot g}
+ \lambda\sqrt{\mathrm{Var}(u)/n}.
$$

---

## Optuna validation score (what is maximized)

Each trial is scored on the **validation** logged split by a conservative lower confidence bound
of the same value estimator the arm is training toward.

### Off-policy (`logged`)

Doubly robust per row:

$$
\widehat{R}_i
=
\mathrm{DM}_i + \frac{\pi_\theta(a_i \mid x_i)}{\pi_b(a_i \mid x_i)}\,(r_i-\hat{q}_i),
\qquad
\mathrm{DM}_i = \sum_a \hat{q}(x_i,a)\,\pi_\theta(a \mid x_i).
$$

$$
\widehat{R} = \overline{\widehat{R}_i},
\quad
\mathrm{se} = \frac{s(\widehat{R}_i)}{\sqrt{n}},
\quad
\mathrm{score} = \widehat{R} - t_{0.975,\,n-1}\,\mathrm{se}.
$$

### No propensity (`uniform`)

Training and selection both avoid propensities. Validation uses the same naive LCB:

$$
\widehat{V}_i = \mathrm{DM}_i + (r_i - \hat{q}_i),
\qquad
\mathrm{score} = \overline{\widehat{V}_i} - t_{0.975,\,n-1}\,\mathrm{se}(\widehat{V}).
$$

True on-policy reward is logged for analysis but **not** used for trial selection.

---

## Gradient sanity

`tests/test_custom_losses.py` checks that `KLCRMPolicyLoss` with log trick yields
finite loss and finite, non-NaN, nonzero gradients for both `logged` and `uniform` modes.
