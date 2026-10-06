# What each arm optimizes: BLOB-NQ, CausE-capacity-matched at ρ = 0 and OPC (audit, 2026-10-06)

*A read-only audit of the executable code behind the controlled comparison's reported results: what each arm's training
loss and selection rule are, what is fixed and what is learned, and what anchors the learned correction to the source
representation. Every statement was checked against the code at the commit of the run that produced the reported rows,
and against the runs' manifests and saved trial records. Where the reports' wording and the code disagree, the code
wins; §6 lists the disagreements.*

## 1. The implementations behind the reported results

| arm | run, code commit (clean worktree) | path from the runner to the loss |
|---|---|---|
| **OPC** | `run_reval_stage2_opc_{mlkr,anime}` @ `d791849` (the replays at `d0a8a77` are bit-identical) | `training/run_full_study.py` `_run_condition` → `training/trainer_trials.py` `regression_trainer_trial` → `models/models.py` `CFModel` + `GlobalLinearCorrection` → `models/custom_losses.py` `DRPolicyLoss` → `dr_sndr_surrogate` (direct branch) → `training/training_utils.py` `train` / `run_train_loop` |
| **CausE-cap-C, ρ = 0** (`causecap_c_r000`) | `run_cause_fair_cap_25k` @ `78b5a42` | `_run_condition` → `training/cause_trials.py` `cause_trainer_trial` (family `cap`) → `models/cause.py` `CausELinBatchModel.loss` → `fit_cause_batch` |
| **BLOB-NQ** (`blob_nq`) | `run_blob_main_25k_nq` @ `ca842cc` | `_run_condition` → `training/blob_trials.py` `blob_trainer_trial` → `models/blob.py` `BlobBanditBatch.neg_elbo` → `fit_blob_batch` |

Configuration, from each run's `run_manifest.json`:
- **OPC.**
  - Loss and selection: `policy_loss_types [dr]`, `opc_gradient direct`, `train_weights harmonic:0.1`,
    `select_weights clip:10`, `optuna_selection ci_low`.
  - Policy: `learn_logit_scale true`, `policy_transform linear`.
  - Reward model: `reward_data train`, `crossfit_folds 5`, `reward_features interaction`.
  - Search: lr 1e-4–2e-3 (log-uniform), 5–30 epochs, lr decay 0.8–1, no weight decay; 20 random trials per size.
- **CausE-cap.**
  - `family cap`, optimizer `momentum_decay`, batch 512.
  - Search: lr 3e-4–3e-2; epochs {30, 100, 300}; `l2` ∈ {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3};
    `cf` ∈ {0, 0.01, 0.1, 1, 10, 100}; tie direction {one-way, symmetric}; intercept starting at the base rate.
    20 trials.
- **BLOB-NQ.**
  - Family `nq`, batch 1024, `kappa_s` 0.1.
  - Search: lr 3e-3–1e-1; epochs {10, 30, 100, 300, 1000}; `wa_m` ∈ {−1, 1, 3}; `wb_m` ∈ {−6, −3, 0}. 20 trials.

## 2. Notation

- `x_u` and `a_j` are the logger's biased user and item vectors (`our_x`, `our_a`, `utils/representation_bias.py`
  `build_world`); K = 32; P = 3,533 / 7,579 / 10,803 items (ml / kuairand / anime).
- The logger is π0(j|u) = softmax_j(x_uᵀ a_j / T), T its sharpened temperature (`policy_temperature`).
- The n = 25,000 training rows are (u_i, j_i, r_i, p_i = π0(j_i|u_i)); the 20,000 validation rows are shared by
  every arm. ℓ(r, z) is the logistic negative log-likelihood (NLL).

## 3. The objectives

### OPC

```text
policy:  u'_u = (I + D_u) x_u + b_u,   a'_j = (I + D_a) a_j + b_a
         π_θ(j|u) = softmax_j( s · u'_uᵀ a'_j / T ),   s = exp(30 θ_s)
learned: θ = (D_u, b_u, D_a, b_a, θ_s)           fixed: x, a (frozen), T, p_i, q̂
init:    θ = 0  ⇒  π_θ = π0 exactly

max_θ  (1/n) Σ_i [ Σ_j π_θ(j|u_i) q̂⁻ᵏ(u_i, j) + g(π_θ(j_i|u_i) / p_i) · (r_i − q̂⁻ᵏ(u_i, j_i)) ]
       g(w) = w / (0.9 + 0.1 w)        (harmonic weights, λ = 0.1: increasing, at most 10)
```

- **Plain DR, no self-normalization.** `DRPolicyLoss` fixes the normalizer to "none"; the manifest's `sn_scope batch`
  has no effect on it. The gradient is pathwise through π in both terms, including inside g (all 600 reported
  trials ran with the direct gradient).
- **The propensity** enters only through g(π/p_i) at the logged action.
- **The reward model q̂** is a logistic regression (sklearn defaults, L2 with C = 1) on [x, a, x⊙a], fit on the
  same 25k rows. The training loss uses the model fit without the user's fold (5 user folds); selection uses the model
  fit on all 25k rows. Its interaction features are diagonal (x⊙a), so q̂ cannot represent a general warp, which the
  policy class can.
- **No explicit penalty, KL, entropy, variance or trust-region term.** Implicit regularization only:
  - Adam with no weight decay, the learning rate multiplied by the decay factor each epoch;
  - the gradient norm clipped at 1.0 every step (`training/training_utils.py`);
  - a fixed number of epochs (no early stopping inside a trial);
  - harmonic weights bounded at 10;
  - selection by the 95% DR lower bound of π_θ on validation (clip:10 weights, full-data q̂), which penalizes
    high-variance policies far from π0.
- **What is optimized vs what is scored.** OPC optimizes the value of its softmax policy (with a learned sharpness:
  median s = 20 in the selected trials); the primary metric scores its argmax.

### CausE-capacity-matched at ρ = 0

```text
model:   u'_u = (I + D_u) x_u + b_u,  θ^c_j = (I + D_c) a_j + b_c,  θ^t_j = (I + D_t) a_j + b_t
         z^c(u, j) = α · u'_uᵀ θ^c_j + b               (ρ = 0: every training row is a control row)
learned: D_u, b_u, D_c, b_c, D_t, b_t, α, b (3,170 parameters); x and a are frozen buffers
init:    every D and b_* at 0; α = 1e-8; b = logit(click rate of its 25k rows)

min  (1/n) Σ_i ℓ(r_i, z^c(u_i, j_i))
   + λ₂ · ½ (‖D_u‖² + ‖b_u‖² + ‖D_c‖² + ‖b_c‖² + ‖D_t‖² + ‖b_t‖²)
   + λ_cf · (1/n) Σ_i ‖ θ^c_{j_i} − sg(θ^t_{j_i}) ‖₁            (one-way; symmetric: no stop-gradient)
```

- **Same family as OPC.** The maps are OPC's `GlobalLinearCorrection` on the frozen `our_x`, `our_a`; the evaluated C
  policy uses the user and control maps, with α in place of s/T (b is constant over items).
- **Data at ρ = 0.** All N rows of the shared training split; no uniform rows (the uniform pool is simulated but
  unused). No propensities in training or in the NLL selection.
- **What the penalties anchor.**
  - λ₂ pulls every D and b toward 0, which is the source. α and b are not penalized.
  - The one-way tie pulls control toward treatment. At ρ = 0 the treatment map gets no gradient at all, so it stays
    exactly at the source (‖(D_t, b_t)‖ = 0.0 in all 345 one-way trials), and the tie is exactly
    λ_cf · mean_i ‖D_c a_{j_i} + b_c‖₁: an L1 anchor of the item-side correction to the source, weighted by how often
    each item was logged.
  - The symmetric tie couples the control map to a free copy that follows it (median gap 6e-4), so it anchors
    nothing beyond λ₂.
- **The selected models at ρ = 0** (30 worlds):

  | selected configuration | worlds | item map ‖(D_c, b_c)‖ |
  |---|---|---|
  | one-way tie, λ_cf > 0 | 18 | 1.6e-5 to 8.8e-4: held at the source |
  | symmetric tie, λ_cf > 0 | 7 | 0.32–0.75, moving with the treatment map |
  | λ_cf = 0 (2 one-way, 3 symmetric) | 5 | 0.83–1.33, moving freely |

  λ₂ > 0 in 28 of 30 selected models, always ≤ 1e-5: n·λ₂ ≤ 0.25 in sum-of-NLL units, a prior standard deviation of
  at least 2 on each entry of D and b.
- **No extra capacity.** ⟨(I + D_u)x + b_u, a⟩ = xᵀWa + vᵀa already spans OPC's whole ranking class, so pinning the
  item map changes only the parameterization, and the treatment map never enters the C policy. CausE-cap and OPC have
  the same K² + K = 1,056 ranking degrees of freedom.
- **Optimizer.** Momentum 0.9 with the learning rate decayed linearly to 0; no clipping, no weight decay.
- **Selection.** The lowest validation NLL of the C click model over the 20 trials.

### BLOB-NQ

```text
fixed:   ω̂_u = x_u / RMS(x)                 (one global scalar)
         Ψ̃ = a with each column divided by its norm over the P items (the released in-place aliasing)
         L = chol(Ψ̃ᵀ Ψ̃ / P)
model:   z(u, j) = s+(w_a) Ψ̃_j ω̂_u + s+(w_b) Ψ̃_j ζ Lᵀ ω̂_u + w_c + κ_j
               = Ψ̃_j M ω̂_u + κ_j + w_c,      M = s+(w_a) I + s+(w_b) ζ Lᵀ       (s+ = softplus)
prior:   ζ_jk ~ N(0, 1);  κ_j ~ N(0, 0.1²);  w_a ~ N(m_a, 1);  w_b ~ N(m_b, 1);  w_c ~ N(−4.5, 10²)
q (NQ):  a fully factorized Gaussian (a mean and a std for each entry of ζ and κ and for each scalar)

min_q  (1/n) Σ_i E_q[ ℓ(r_i, z(u_i, j_i)) ]  +  (1/n) KL(q ‖ prior)
```

- **What it learns.** One global K×K map M (prior mean: a scaled identity on the normalized source), one scalar
  intercept per item and a global intercept; 2K² + 2P + 6 variational parameters (9,120 / 17,212 / 23,660). No user or
  item embeddings are learned.
- **The expected NLL** is estimated with one local-reparameterization sample per row per step. TF1 Adam, initialized at
  the prior, no clipping, decay or weight decay.
- **Likelihood plus a quadratic penalty only in the zero-variance limit.** There, the mean-dependent part (× n) is

  ```text
  Σ_i ℓ(r_i, z̄_i) + ½‖ζ‖² + ½‖κ‖²/0.1² + ½(w_a − m_a)² + ½(w_b − m_b)² + ½(w_c + 4.5)²/10²
  with ½‖ζ‖² = (P / 2 s+(w_b)²) · tr(ΔM (Ψ̃ᵀΨ̃)⁻¹ ΔMᵀ),   ΔM = M − s+(w_a) I
  ```

  a generalized ridge on the map's deviation from the scaled source map with precision ∝ P / s+(w_b)², and a strong
  ridge on the item intercepts. As implemented the data term is the expected NLL (the learned variances add a penalty)
  and w_b is learned, so the effective penalty on ΔM is not a fixed quadratic.
- **NQ vs MNQ.** NQ has K² standard deviations for ζ, used in the noise term. MNQ factorizes them as σ1_j·σ2_k, and as
  released its noise term uses the prior standard deviations. Point prediction, priors and selection are the same.
- **BLOB-Pnorm** (`docs/blob_prior_calibration.md`) is identical except that L is multiplied by √(P/P₀).
- **Selection.** The lowest validation NLL of the posterior-mean click model (with w_c), over 20 trials.
- **Selected models.** s+(w_a) median 6.1; s+(w_b) median 3.1 (anime 0.05); the correction is a median 0.9% of the
  source term (anime ≈ 0); the item intercepts barely move (RMS ≈ 0.01, posterior std 0.0998 ≈ the prior's).

## 4. Initialization versus anchoring

| role of the source | BLOB-NQ | CausE-cap, ρ = 0 | OPC |
|---|---|---|---|
| fixed input | yes (rescaled ω̂, column-normalized Ψ̃) | yes | yes |
| parameterization around the source | yes: M = s+(w_a)I + ΔM | yes: three (I + D)z + b maps | yes: two maps |
| initialization | the prior mean (ΔM = 0); its ranking is ⟨x, Ψ̃_j⟩, not the logger's (84% top-item agreement on biased worlds) | identity maps with α ≈ 0: the logger's greedy ranking, flat logits | identity maps, s = 1: exactly π0 |
| explicit prior mean | yes: Gaussian prior centred at M = s+(w_a)I, κ = 0 | no | no |
| explicit regularization target | yes: the KL pulls ζ and κ to 0, with strength ∝ P | weak: λ₂ ridge toward the source; the one-way tie anchors the item map (18/30); neither restricts the ranking class | none |

For OPC the source is a warm start and a parameterization with no explicit source-anchoring term; the pulls toward the
logger are the implicit ones listed in §3.

## 5. The three arms side by side

| property | BLOB-NQ | CausE-cap, ρ = 0 | OPC |
|---|---|---|---|
| parameters (ranking degrees of freedom) | 2K² + 2P + 6 (K² + P, κ heavily shrunk) | 3,170 (the C policy uses 2,113; ranking 1,056) | 2,113 (ranking 1,056) |
| training data | 25k biased-logger rows | the same 25k | the same 25k, their propensities, and q̂ fit on them |
| randomized data | no | no | no |
| propensities | no (only the post-hoc tempering of the stochastic metric) | no (same tempering) | yes: training g(π/p), selection min(π/p, 10) |
| data-fit objective | expected NLL under q | NLL | harmonic-weighted DR value of π_θ |
| source anchoring | strong Gaussian prior centred on the scaled source map, precision ∝ P | weak ridge toward the source; L1 item-map anchor | none |
| other regularization | KL on κ and w_c; Monte Carlo noise; fixed epochs | momentum with linear decay; fixed epochs | Adam with decay; gradient clip 1; bounded weights; fixed epochs |
| policy evaluated | argmax of the posterior-mean click model | argmax of α u'ᵀθ^c | argmax of π_θ's logits (and π_θ itself) |
| selection | validation NLL | validation NLL | DR lower bound, clip 10 |

```text
BLOB:      E_q[logged NLL] + KL(q ‖ Gaussian prior centred on the column-normalized, scaled source map; precision ∝ P)
CausE-cap: logged NLL + weak ridge toward the source + L1 item-map anchor (the CausE tie, degenerate at ρ = 0)
OPC:       −V̂_DR,harmonic(π_θ) from a cross-fitted logistic q̂ and the logged propensities; no explicit penalty
```

**What the CausE-cap vs OPC contrast tests.** Within the same policy class and data: a (penalized) click likelihood
selected by NLL against a DR value objective selected by the DR lower bound. The selection rule, the optimizer and the
search spaces (each tuned by its own procedure) differ too. **The BLOB vs CausE-cap contrast** changes the prior (strong
and P-dependent against weak), the item-intercept structure, the starting ranking and the optimizer, with the same
selection rule. **Naming.** At ρ = 0 CausE's own mechanism (a treatment representation learned from randomized rows
that regularizes the control one) is absent; CausE-cap there is a capacity-matched penalized-likelihood baseline in
CausE's objective form.

## 6. Where the reports and the code disagree

- **"Plain likelihood (learner)"** for CausE-cap at ρ = 0 (`docs/blob_controlled_integration.md`, and the arm label in
  `training/analyze_blob.py` and the tables it generates): the executed loss had λ₂ > 0 in 28 of 30 selected models and
  a tie in 25 of 30. The penalties do not restrict the ranking class, so "likelihood learner in OPC's class" is fair;
  "plain" is not exact.
- **`docs/cause_fair_comparison_25k.md` §8.2:** "at ρ = 0 both item maps stay at the source (norm 0.001)" holds in 18 of
  30 worlds. In the other 12 the control map moved (norm 0.32–1.33).
- **The same report's §1.3** calls C "a likelihood fit" without its penalties. Its §1.2 remark that L2 "shrinks toward
  0, not toward the source" is about CausE-warm; for CausE-cap the same L2 acts on D and b, so it shrinks toward the
  source.
- **Inert manifest fields:** OPC's `sn_scope batch` (ignored by the DR loss) and the OPC-only fields in the CausE and
  BLOB runs' manifests (`train_weights`, `learn_logit_scale`).
