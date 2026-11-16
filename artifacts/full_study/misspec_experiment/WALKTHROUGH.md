# What we are actually doing — walkthrough with code

Self-contained tour of the residual-dial / hybrid confirmatory study.  
Each section states the math, then shows the **real calculation** in the repo.

Companion memos: `MISSPEC_FRAMING.md`, `SOFTAND_SOFTTHRESH_EXPLAINED.md`, `FAMILIES_EXPLAINED.md`, `PROTOCOL_DRAFT_hybrid.md`, `PROTOCOL_DRAFT_capacity.md`.

---

## 0. One-paragraph picture

We build a click world whose logit is **bilinear truth + γ · (unit-SD residual φ)**.  
A frozen logging policy logs `(user, action, click, propensity)`.  
Learners never see φ; they only see biased embeddings and logged clicks.  
We train **NLL**, **OPC/DR**, and **hybrid NLL→DR** in a locked score class \(\mathcal{F}\) (usually linear), then score them by **exact-greedy CTR** under the true \(q_\gamma\).  
The scientific contrast is how the method gap moves when γ rises (DiD / interaction), not “is there confounding.”

Grid driver: `run_full_hybrid_grid.sh`  
Analysis: `analyze_hybrid_by_cell.py` → PDF via `build_hybrid_results_pdf.py`

---

## 1. World build order

For each `(dataset, seed, bias=high, residual_kind, γ)`:

1. Calibrate clean embeddings → scale / offset / α (linear-logit SD).  
2. Apply **representation bias** → learner/logger vectors `our_x`, `our_a` (signal kept ~50% in `high`).  
3. Sharpen logger temperature **on the γ=0 env** → freeze `T_log`.  
4. If γ>0, wrap truth in `ResidualBanditEnv`, re-solve offset \(b'(\gamma)\) so reference CTR stays ~5%.  
5. Log data under frozen \(\pi_0\); evaluate policies under \(q_\gamma\).

```617:694:utils/representation_bias.py
def build_world(emb_x, emb_a, bias, *, seed: int, config: WorldConfig | None = None,
                ...
                residual_gamma: float = 0.0, residual_kind: str = "none",
                residual_spec=None) -> dict:
    ...
    env0 = SyntheticBanditEnv(emb_x=X, emb_a=A, scale=cal["scale"], offset=cal["offset"], ctr=float(config.target_ctr))
    ...
    sharp = sharpen_logger(our_x, our_a, cu, cal["logging_temperature"], env0, share)
    T_log = sharp["temperature"] if share > 0.0 else float(cal["logging_temperature"])

    r_gamma = float(residual_gamma)
    r_kind = str(residual_kind or "none").lower()
    ...
    if r_gamma == 0.0 or r_kind in ("none", "", "identity"):
        env = env0
        ...
    else:
        env = make_residual_env(
            env0, gamma=r_gamma, kind=r_kind, seed=seed, alpha=float(cal["alpha"]), spec=r_spec,
            taste_dim=int(d),
        )
        resolve_offset_prime(env, cu, cal["_ctr_items"], target_ctr=float(config.target_ctr))
        # freeze T_log: only recompute logger stats under env_g (share=0 keeps f=1)
        sharp_g = sharpen_logger(our_x, our_a, cu, T_log, env, share=0.0)
```

**Important:** γ=0 is *not* fully “well-specified” for the learner — truth is bilinear in **clean** embeddings, learners see **biased** ones. Prefer: “baseline = representation mismatch only.”

---

## 2. Truth logit (the residual dial)

\[
\ell_\gamma(u,a)
=
\underbrace{\alpha\,\langle x_u,a_a\rangle}_{\text{scale·dot in code}}
+ b'(\gamma)
+ \gamma\,\alpha\,\phi(u,a),
\qquad
q_\gamma=\sigma(\ell_\gamma),\quad \mathrm{sd}(\phi)=1.
\]

Hot path:

```596:615:utils/residual_env.py
    def logit_block(self, users: np.ndarray, a0: int, a1: int) -> np.ndarray:
        logit = self._linear_logit_block(users, a0, a1)
        if self.residual_active:
            logit = logit + float(self.gamma) * float(self.alpha) * self._phi_block(users, a0, a1)
        return logit

    def reward_prob(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        ...
        dots = np.einsum("ij,ij->i", self.emb_x[users], self.emb_a[actions], dtype=np.float64)
        logit = float(self.scale) * dots + float(self.offset)
        logit = logit + float(self.gamma) * float(self.alpha) * self._phi_pairs(users, actions)
        return expit(logit)
```

Offset re-solved so mean \(q\) on calibration pairs stays at the target CTR:

```1094:1114:utils/residual_env.py
def resolve_offset_prime(
    env: ResidualBanditEnv,
    users: np.ndarray,
    actions: np.ndarray,
    *,
    target_ctr: float = 0.05,
) -> float:
    ...
    dots = np.einsum("ij,ij->i", env.emb_x[users], env.emb_a[actions], dtype=np.float64)
    phi = env._phi_pairs(users, actions) if env.residual_active else np.zeros_like(dots)
    zs = float(env.scale) * dots + float(env.gamma) * float(env.alpha) * phi
    b = _solve_b(zs, 1.0, float(target_ctr))
    env.offset_prime = float(b)
```

---

## 3. Shared residual scaffolding

All locked SoftAND* / SoftThresh / (latent) families share:

1. **Facet projections** \(t_u, t_a\) (z-scored; RNG keyed by `proj_tag`).  
2. A family-specific **raw** φ.  
3. **⊥** = remove best affine-bilinear fit under uniform catalog pairs on **clean** \(X,A\).  
4. **Unit-SD** (`norm=v2`) so catalog pair-SD of φ is 1 → designed `sd_ratio ≈ γ`.

```346:360:utils/residual_env.py
def _shared_proj_coords(
    X: np.ndarray, A: np.ndarray, *, seed: int, m: int, proj_tag: str
) -> tuple[np.ndarray, np.ndarray]:
    """z-scored projected coords; RNG key independent of kind/τ/c/γ."""
    d = X.shape[1]
    rng = np.random.default_rng(derive_seed(seed, "proj", f"m={m}", f"tag={proj_tag}"))
    G = rng.normal(size=(d, m))
    H = rng.normal(size=(d, m))
    t_u = X @ G
    t_a = A @ H
    t_u = t_u - t_u.mean(axis=0, keepdims=True)
    t_a = t_a - t_a.mean(axis=0, keepdims=True)
    sd_u = np.maximum(t_u.std(axis=0, keepdims=True), 1e-12)
    sd_a = np.maximum(t_a.std(axis=0, keepdims=True), 1e-12)
    return t_u / sd_u, t_a / sd_a
```

⊥ for factored residuals (SoftAND*, etc.):

```314:343:utils/residual_env.py
def _perp_factored(
    fu: np.ndarray, fa: np.ndarray, U: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove P_S(fu@fa.T) via augmented factors [fu, U] @ [fa, -Y M.T].T."""
    Gu = U.T @ U
    Gy = Y.T @ Y
    C = (U.T @ fu) @ (fa.T @ Y)
    M = _solve_bilinear_M(C, Gu, Gy)
    ...
    fu_aug = np.concatenate([fu, U], axis=1)
    fa_aug = np.concatenate([fa, -(Y @ M.T)], axis=1)
```

Unit-SD (`v2` divides one side only):

```279:285:utils/residual_env.py
def _unit_sd_factors(fu: np.ndarray, fa: np.ndarray, *, norm: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale factors so pair-field sd(phi)=1. norm=v2 divides one side only; v1 divides both (legacy)."""
    sd = _factor_sd(fu, fa)
    sd = max(sd, 1e-300)
    if norm == "v2":
        return fu / sd, fa, 1.0
    return fu / sd, fa / sd, 1.0
```

---

## 4. Residual families (geometry of φ only)

Learners do **not** change across families — only φ’s shape does.

### 4.1 R* (tanh) — second index + even warp

\[
\psi(t)=\tanh\!\big(s(t^2-1)\big),\qquad
\phi_{\mathrm{raw}}=m^{-1/2}\sum_k \psi(t_{u,k})\,\psi(t_{a,k}).
\]

Locked: `s=1,m=4,psi=tanh` (v1 path; louder at same nominal γ than SoftAND/ST).

```245:249:utils/residual_env.py
def _psi(t: np.ndarray, *, kind: str, s: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if kind == "square":
        return t * t - 1.0
    return np.tanh(float(s) * (t * t - 1.0))
```

```720:730:utils/residual_env.py
    if kind == "pl":
        fu = scale_m * t_u
        fa = t_a
    elif kind == "rstar":
        fu = scale_m * _psi(t_u, kind=spec["psi"], s=float(spec["s"]))
        fa = _psi(t_a, kind=spec["psi"], s=float(spec["s"]))
    ...
    fu, fa = _project_orthogonal_to_linear(fu, fa, X, A)
    fu, fa, _ = _double_center_factors(fu, fa)
    fu, fa, _ = _unit_sd_factors(fu, fa, norm="v1")
```

### 4.2 SoftAND* — same-facet soft AND

\[
g_{u,k}=\mathrm{sp}\!\big((t_{u,k}-c)/s\big),\quad
h_{a,k}=\mathrm{sp}\!\big((t_{a,k}-c)/s\big),\quad
\phi_{\mathrm{raw}}=m^{-1/2}\sum_k g_{u,k}h_{a,k},
\]
then DC → ⊥ → unit-SD.

Locked: `m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2`

```786:799:utils/residual_env.py
    if kind == "softand":
        c = float(spec["c"])
        gate_s = float(spec["gate_s"])
        fu = scale_m * _softplus((t_u - c) / gate_s)
        fa = _softplus((t_a - c) / gate_s)
        fu, fa, _ = _double_center_factors(fu, fa)
        fu, fa, M, diag = _perp_factored(fu, fa, U, Y)
        ...
        fu, fa, _ = _unit_sd_factors(fu, fa, norm="v2")
```

### 4.3 SoftThresh (τ=1.5) — soft threshold on \(z_2\)

\[
z_2=m^{-1/2}\sum_k t_{u,k}t_{a,k},\quad
\tilde z_2=(z_2-\mu)/\sigma,\quad
\phi_{\mathrm{raw}}=\mathrm{sp}\!\big((\tilde z_2-\tau)/s_{\mathrm{th}}\big),
\]
then ⊥ → unit-SD. **Not** finite-rank separable after the softplus-of-sum.

Locked: `m=4,tau=1.5,s_th=0.25,proj_tag=default,norm=v2`

Hot-path evaluation (raw − bilinear correction) / `phi_sd`:

```516:522:utils/residual_env.py
    def _phi_softthresh_block(self, users: np.ndarray, a0: int, a1: int) -> np.ndarray:
        scale_m = 1.0 / np.sqrt(int(self.spec["m"]))
        z2 = scale_m * (self.t_u[users] @ self.t_a[a0:a1].T)
        z_t = (z2 - float(self.z2_mu)) / float(self.z2_sd)
        raw = _softplus((z_t - float(self.spec["tau"])) / float(self.spec["s_th"]))
        corr = self.soft_U[users] @ self.soft_M @ self.soft_Y[a0:a1].T
        return (raw - corr) / float(self.phi_sd)
```

### 4.4 Dial calibration (realized loudness)

Designed: `sd(γ·α·φ)/sd(linear) ≈ γ` after unit-SD.  
Realized shares differ by family — **do not compare families at the same nominal γ**.

```7:14:artifacts/full_study/misspec_experiment/measure_reward_ratios.py
  sd_ratio      = sd(γ·α·φ) / sd(linear)          ≈ γ by construction
  var_share_φ   = sd(res)^2 / (sd(lin)^2 + sd(res)^2)
  mean_q_full   = E[σ(lin + b' + res)]
  mean_q_gt     = E[σ(lin + b')]
  ...
  ctr_lift_φ    = mean_q_full - mean_q_gt
```

```81:84:artifacts/full_study/misspec_experiment/measure_reward_ratios.py
    sd_lin = float(np.std(lin))
    sd_res = float(np.std(res))
    sd_ratio = sd_res / max(sd_lin, 1e-12)
    var_share = (sd_res ** 2) / max(sd_lin ** 2 + sd_res ** 2, 1e-12)
```

At γ=0.5 on ML: SoftAND/ST ≈ 20% var share; R* ≈ 59%.

---

## 5. Logging and propensities

Logger \(\pi_0\) is softmax of **biased** scores at frozen \(T_{\log}\).  
Rows: \(u\sim\mathrm{prior}\), \(a\sim\pi_0(\cdot|u)\), \(p=\pi_0(a|u)\), \(r\sim\mathrm{Bern}(q_\gamma(u,a))\).

Propensity is **simulator-true** (known). This study is **not** about confounding / propensity misspec.

---

## 6. What learners optimize

Same policy class \(\mathcal{F}\) across methods in a cell (default: `--policy-transform linear`).  
Reward model \(\hat q\) is a separate nuisance for DR (regression on interaction features; cross-fit).

### 6.1 NLL — logged logistic likelihood

\[
L_{\mathrm{NLL}}
=
\mathrm{BCEWithLogits}\big(z_\theta(u,a_{\mathrm{logged}}),\, r\big).
\]

No IPS, no \(\hat q\). Selection: val NLL.

```475:496:models/custom_losses.py
class LoggedNLLPolicyLoss(_BanditPolicyLossBase):
    """Logged logistic NLL on click logits (matched-capacity likelihood baseline).
    ...
    """
    ...
    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions,
                click_logits=None):
        ...
        z = click_logits[idx, original_policy_actions.long()].squeeze()
        y = original_policy_rewards.reshape_as(z).to(z.dtype)
        return torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

### 6.2 OPC / raw DR — value surrogate

Doubly robust policy objective (direct method + IPS correction), **no** self-normalization on the train loss:

\[
\widehat V_{\mathrm{DR}}
=
\sum_a \hat q(u,a)\,\pi_\theta(a|u)
+
w\,\big(r-\hat q(u,a_{\mathrm{logged}})\big),
\quad
w=\pi_\theta(a|u)/\pi_0(a|u)
\]
(with raw weights under the study lock; train minimizes \(-\widehat V\)).

```178:206:models/custom_losses.py
def dr_correction(iw, rewards, q_at_action, normalizer="batch"):
    """Per-row DR correction w_i (r_i - q_i) / c.
    ...
    """
    ...
    if normalizer is None:
        return iw * (rewards - q_at_action)
    return iw * (rewards - q_at_action) / float(normalizer)
```

```214:259:models/custom_losses.py
def dr_sndr_surrogate(
    scores, policy_prob, actions, rewards, pscore, *,
    use_iw: bool, use_log_trick: bool, ...
    normalizer="batch",
):
    ...
    else:
        iw = transform_importance_weights(pi_a, pscore, **wkw)
        corr_term = dr_correction(iw, rewards, q_factual, normalizer)
        dm_term = dm_reward(scores, policy_prob)

    return corr_term + dm_term
```

```443:452:models/custom_losses.py
class DRPolicyLoss(SNDRPolicyLoss):
    """Doubly robust policy objective without self-normalization: DM(q_hat) + w (r - q_hat), ...
    """
    def __init__(self, log_eps=1e-10, use_log_trick=True, propensity_mode="logged", weights="none"):
        super().__init__(..., weights=weights, normalization="none")
```

Grid lock (excerpt): `--policy-losses dr --opc-gradient direct --train-weights none --select-weights clip:10`, select by DR lower bound.

### 6.3 Hybrid sequential — NLL then DR

Not a mixed loss. Train NLL for fraction \(f\) of epochs, **freeze click bias**, finish with raw DR; select by DR-LB.

\[
\text{epochs}_{\mathrm{NLL}}=\mathrm{round}(f\cdot E),\qquad
\text{epochs}_{\mathrm{DR}}=E-\text{epochs}_{\mathrm{NLL}}.
\]

Arms: `hybrid_seq_f0.25`, `f0.5`, `f0.75`, and `hybrid_seq_adapt` (val-NLL plateau switch with \(f\in[0.25,0.75]\)).

```121:181:training/training_utils.py
def split_hybrid_epochs(num_epochs: int, nll_frac: float) -> tuple[int, int]:
    ...
    epochs_a = int(round(f * e))
    ...
    return epochs_a, e - epochs_a


def train_hybrid_seq(
    model, train_loader, scores_all, criterion_nll, criterion_dr,
    num_epochs=1, nll_frac: float = 0.5, ...
):
    """NLL for ``round(nll_frac * epochs)``, freeze click bias, then raw/DR for the rest (one Adam)."""
    epochs_a, epochs_b = split_hybrid_epochs(num_epochs, nll_frac)
    ...
    if epochs_a > 0:
        optimizer = train(..., criterion_nll, num_epochs=epochs_a, ...)
        freeze_click_bias(model)
    if epochs_b > 0:
        optimizer = train(..., criterion_dr, num_epochs=epochs_b, optimizer=optimizer, ...)
```

Wiring in the study runner:

```751:760:training/run_full_study.py
        elif h_frac is not None:
            arm = dict(
                policy_loss_types=("dr",),
                learn_logit_scale=True,
                learn_click_bias=True,
                post_temper=False,
                optuna_selection="ci_low",
                select_estimator="dr",
                hybrid_nll_frac=float(h_frac),
            )
```

Also present: `nll_dr` = NLL **train** + DR-LB **select** (selection hybrid only — kill arm vs sequential).

---

## 7. Evaluation — exact-greedy CTR

Primary metric: true value of the learned ranking’s top item,

\[
V_{\mathrm{gr}}(\pi)
=
\sum_u \mathrm{prior}(u)\, q_\gamma\big(u,\, \arg\max_a s_\theta(u,a)\big).
\]

Stored as `policy_rewards_greedy` in `summary_metrics.csv`.

```434:439:utils/simulation_utils.py
def calc_greedy_reward(dataset: dict, user_emb: np.ndarray, item_emb: np.ndarray, *, chunk_size: int = 2048) -> float:
    """True value of the policy's greedy (exploitation) part: sum_u prior[u] * q(u, argmax_a x_u . a_a).
    ...
    """
```

Residual worlds route \(q\) through `env.q_block_torch` / `reward_prob*` so evaluation sees the residual dial.

---

## 8. Contrasts and “DiD”

Per seed (CRN across γ), in CTR percentage points:

\[
\Delta_{m,m'}(\gamma)
=
100\big(V_m(\gamma)-V_{m'}(\gamma)\big),
\qquad
\mathrm{DiD}
=
\Delta(\gamma)-\Delta(0).
\]

Primary: \(\mathrm{H}=\texttt{hybrid\_seq\_f0.5}\) vs NLL; also OPC−NLL.

This is a **method × γ interaction contrast** identified by design (manipulated dial + paired seeds). Not classical DiD / parallel trends.

```235:253:artifacts/full_study/misspec_experiment/analyze_hybrid_by_cell.py
            d_g0 = 100.0 * (piv0.loc[seeds, a] - piv0.loc[seeds, b]).to_numpy()
            d_g1 = 100.0 * (piv1.loc[seeds, a] - piv1.loc[seeds, b]).to_numpy()
            did = d_g1 - d_g0
            for gamma, arr in ((g0, d_g0), (g1, d_g1)):
                mean, lo, hi, n = _ci(arr)
                contrast_rows.append(...)
            mean, lo, hi, n = _ci(did)
            did_rows.append(
                dict(
                    dataset=ds, misspec=misspec, g0=g0, g1=g1, contrast=label,
                    n=n, DiD_pt=mean, ci_lo=lo, ci_hi=hi,
                    signs=int((did > 0).sum()), note=note,
                )
            )
```

CIs: Student-\(t\) on paired seed vectors (`_ci`).

---

## 9. Confirmatory grid (what is running)

From `run_full_hybrid_grid.sh`:

| Axis | Levels |
|------|--------|
| methods | `opc`, `nll`, `hybrid_seq_f{0.25,0.5,0.75}`, `hybrid_seq_adapt` |
| families | R*, SoftAND*, SoftThresh τ=1.5 |
| γ | 0, 0.25, 0.5, 0.75, 1, 2, 4, 10, 100 (+ anime 0.6) |
| datasets | ml, anime, myket, kuairand/kuairec full + mlsize |
| seeds | 300–339 (n=40) |
| capacity | `linear` (main grid) |
| n / val | 100k train, 20k val; Optuna 20 trials |

Tags look like `nlres_grid_{ds}_{kind}_g{γ}`.

---

## 10. Capacity trial (follow-on)

Question: if we enlarge \(\mathcal{F}_c\) to `linear+mlp` (hidden 64 / 256), does H−NLL / DiD **shrink** (φ was absorbable) or **persist** (more intrinsic / non-class)?

Same estimands, matched methods within cell; lean seeds 300–309; families R* + SoftThresh; γ ∈ {0, 0.5, 2}.  
Protocol: `PROTOCOL_DRAFT_capacity.md`; driver: `run_capacity_trial.sh`.

---

## 11. Framing — what kind of “misspec” this is

**Tight label:** controlled **class / mean-function misspecification** of the click/policy score relative to locked \(\mathcal{F}\), studied via a residual dial and interaction contrasts.

| Is | Is not |
|----|--------|
| \(\ell_\gamma \notin \mathcal{F}_{\mathrm{linear}}\) for γ>0 (by construction after ⊥) | Unobserved confounding |
| Different pseudo-true targets: NLL ≈ KL projection of \(q^\star\); DR ≈ value | Propensity misspec (π₀ known) |
| Stress / sensitivity analysis in γ | “The world is misspecified” |
| Score-class misspec (and separately: \(\hat q\) nuisance misspec) | Cross-family claims at equal nominal γ |

Review corrections live in `MISSPEC_FRAMING.md` §9.

---

## 12. File map

| Piece | Path |
|-------|------|
| Residual DGP | `utils/residual_env.py` |
| World + frozen logger | `utils/representation_bias.py::build_world` |
| Exact value / greedy | `utils/simulation_utils.py` |
| NLL / DR losses | `models/custom_losses.py` |
| Hybrid schedule | `training/training_utils.py` |
| Method wiring | `training/run_full_study.py` |
| Full grid | `artifacts/.../run_full_hybrid_grid.sh` |
| Levels / DiD | `artifacts/.../analyze_hybrid_by_cell.py` |
| Dial loudness | `artifacts/.../measure_reward_ratios.py` |
| Results PDF | `artifacts/.../build_hybrid_results_pdf.py` |
| Framing | `artifacts/.../MISSPEC_FRAMING.md` |

---

## 13. Reading order for a new reader

1. This file (§0–§2, §6–§8).  
2. `SOFTAND_SOFTTHRESH_EXPLAINED.md` + `FAMILIES_EXPLAINED.md`.  
3. `MISSPEC_FRAMING.md` (esp. §9).  
4. Skim `PROTOCOL_DRAFT_hybrid.md` for why sequential hybrid (not a mixed loss).  
5. Open `utils/residual_env.py` and `models/custom_losses.py` for the arithmetic.
