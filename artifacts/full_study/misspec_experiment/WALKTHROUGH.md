# What we are actually doing — walkthrough with code

Self-contained tour of the residual-dial / hybrid confirmatory study. Each section presents the **math**, restates it **in plain words**, and then points to the **real calculation** in the repo.

> **How to read the formulas in Cursor**
>
> 1. Built-in preview: open this file, open the command palette, and choose **“Markdown: Open Preview”** (`markdown.math.enabled` is already set).
> 2. Richer KaTeX preview: choose **“Markdown Preview Enhanced: Open Preview to the Side”** (extension `shd101wyy.markdown-preview-enhanced` is installed).
> 3. Formulas use `$...$` and `$$...$$`, which both previews render.

Companion memos: `MISSPEC_FRAMING.md`, `SOFTAND_SOFTTHRESH_EXPLAINED.md`, `FAMILIES_EXPLAINED.md`, `PROTOCOL_DRAFT_hybrid.md`, `PROTOCOL_DRAFT_capacity.md`, `CLAIMS_FAIR_PI.md`, `capacity_fair_pi/REPORT.md`.

---

## 0. One-paragraph picture

We build a click world whose click **logit** (pre-sigmoid score) is

> bilinear “taste” truth **plus** a dialed residual $\gamma \cdot \alpha \cdot \phi$.

A frozen logging policy writes rows `(user, action, click, propensity)`.
Learners never see $\phi$ as a feature — only biased embeddings and logged clicks.
We train **NLL**, **OPC/DR**, and **hybrid NLL→DR** inside a locked score class $\mathcal{F}$ (usually linear), then score policies by **exact-greedy CTR** under the true click probability $q_\gamma$.

The scientific contrast is **how the method gap moves when $\gamma$ rises** (an interaction / “DiD”-style contrast), not “is there confounding.”

- Grid driver: `run_full_hybrid_grid.sh`
- Analysis: `analyze_hybrid_by_cell.py` → PDF via `build_hybrid_results_pdf.py`

### Symbol glossary

- $u$ and $a$: a user and an action (item); $x_u$ and $a_a$ are their **clean** embeddings. Tildes, as in $\tilde x_u$ and $\tilde a_a$, mark the biased embeddings available to the logger and learners.
- $\langle x_u,a_a\rangle$: the clean bilinear taste score; $\alpha$ puts that score on the calibrated logit scale.
- $\gamma$: residual-strength dial. It multiplies $\phi(u,a)$, the nominally unit-SD residual field whose geometry changes across R*, SoftAND*, and SoftThresh. The locked legacy R* normalization is a realized-scale exception discussed in §3.3–§4.4.
- $b'(\gamma)$: intercept re-solved at each $\gamma$ to preserve the reference mean CTR.
- $\ell_\gamma$ and $q_\gamma=\sigma(\ell_\gamma)$: the true click logit and click probability.
- $\mathcal F$: the locked learner score class (linear in the main grid); $s_\theta$ and $\pi_\theta$ are a learned score and its induced softmax policy.
- $\pi_0$: frozen logging policy; $p=\pi_0(a\mid u)$ is a logged action's known propensity.
- $\hat q$: cross-fitted reward model used by DR as a nuisance estimate; it is separate from the true $q_\gamma$.
- $V_m(\gamma)$: exact greedy CTR for method $m$ in the environment at dial setting $\gamma$.

---

## 1. World build order

For each `(dataset, seed, bias=high, residual_kind, γ)`:

1. Calibrate clean embeddings → get `scale`, `offset`, and $\alpha$, the scale used to express residual strength in linear-logit SD units.
2. Apply **representation bias** → learner/logger vectors `our_x`, `our_a` (in the locked `high` cell, only ~50% of the clean signal is kept).
3. Sharpen the logger temperature **on the $\gamma=0$ environment**, then **freeze** that temperature $T_{\log}$.
4. If $\gamma>0$, wrap the truth env in `ResidualBanditEnv` and re-solve the intercept $b'(\gamma)$ so the reference CTR stays about 5%.
5. Log data under the frozen logger $\pi_0$; evaluate learned policies under true $q_\gamma$.

**In words:** we first fix how the logger behaves in a bilinear world. We then add nonlinear structure to the *true* click mechanism without changing the logger's scores or temperature. Thus the logging distribution—and therefore the propensity rule—stays fixed across $\gamma$ even though the click probabilities change.

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

**Important:** $\gamma=0$ is *not* fully “well-specified” for the learner. Truth is bilinear in **clean** embeddings; learners see **biased** ones. Prefer: “baseline = representation mismatch only.”

---

## 2. Truth logit (the residual dial)

### 2.1 Definition

$$
\ell_\gamma(u,a)
=
\alpha\,\langle x_u, a_a\rangle
+ b'(\gamma)
+ \gamma\,\alpha\,\phi(u,a)
$$

$$
q_\gamma(u,a)=\sigma\big(\ell_\gamma(u,a)\big)
=
\frac{1}{1+e^{-\ell_\gamma(u,a)}}
$$

The canonical dial targets $\mathrm{sd}(\phi)=1$ over catalog user–item pairs after normalization. The locked R* `v1` path is a legacy exception, so cross-family comparisons use realized scale diagnostics rather than nominal $\gamma$ alone.

**In words:**

- $\langle x_u,a_a\rangle$ is the clean bilinear taste score (dot product of user and item embeddings).
- $\alpha$ (called `scale` / calibration `alpha` in code) sets the units of that score.
- $b'(\gamma)$ is an intercept re-chosen so average click rate on fixed calibration pairs stays near the target (usually 5%).
- $\phi(u,a)$ is a unit-SD residual field — the “extra structure.”
- $\gamma$ is the **dial**: under unit-SD normalization, it is the intended number of linear-SDs of residual added. At $\gamma=0$ the residual term vanishes; at large $\gamma$ the residual dominates ranking. Realized scale is checked separately in §4.4.

Code writes the linear part as `scale * dots + offset`. In the notation above, that calibrated linear term is $\alpha\langle x_u,a_a\rangle+b'(\gamma)$. In a residual environment, `offset_prime` supplies $b'(\gamma)$, while the stored `alpha` scales the residual in linear-logit SD units.

### 2.2 Hot path

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

### 2.3 Re-solving the intercept

We choose $b'(\gamma)$ so that on the same calibration pairs used for the $\gamma=0$ world,

$$
\frac{1}{N}\sum_{i=1}^{N}
\sigma\big(
\alpha\langle x_{u_i},a_{a_i}\rangle
+ b'(\gamma)
+ \gamma\alpha\phi(u_i,a_i)
\big)
\approx 0.05.
$$

**In words:** turning $\gamma$ on would otherwise inflate or deflate average CTR. We slide the intercept until the mean predicted click rate on the reference pairs is again about 5%. Best-item CTR is *not* re-targeted (it drifts; we record it).

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

The locked SoftAND* and SoftThresh constructions use the following pipeline. R* follows the same conceptual sequence—shared projections, a family-specific raw field, removal of linear structure, and rescaling—but retains the legacy `v1` normalization noted below.

1. **Facet projections** $t_u, t_a$ (z-scored; RNG keyed by `proj_tag`).
2. A family-specific **raw** field $\phi_{\mathrm{raw}}$.
3. **Perp ($\perp$)**: remove the best affine-bilinear fit under uniform catalog pairs on **clean** $X,A$.
4. **Unit-SD**: normalize the catalog pair-SD of $\phi$ to 1, so the intended `sd_ratio` is approximately $\gamma$.

### 3.1 Facet projections

Draw random matrices $G,H\in\mathbb{R}^{d\times m}$ and set

$$
t_u = \mathrm{zscore}(X G),\qquad
t_a = \mathrm{zscore}(A H).
$$

**In words:** each user and item gets $m$ “facet” coordinates by projecting embeddings onto random directions and standardizing. Same draw for every family that shares a `proj_tag`.

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

### 3.2 Bilinear perpendicular (⊥)

Let $\phi_{\mathrm{raw}}(u,a)$ be the raw residual. Fit the best affine-bilinear field

$$
P\phi(u,a)
=
\begin{bmatrix}1\\ x_u\end{bmatrix}^{\!\top}
M
\begin{bmatrix}1\\ a_a\end{bmatrix}
$$

in least squares over all catalog pairs, then set

$$
\phi_\perp = \phi_{\mathrm{raw}} - P\phi.
$$

For **factored** residuals $\phi = f_u^\top f_a$, this is implemented by augmenting factors so the product field subtracts $UMY^\top$.

**In words:** strip whatever a clean bilinear (plus user/item intercepts) could already explain. What remains is “structure that the locked linear class cannot soak up under uniform pairs.” Caveat from review: ⊥ is w.r.t. **clean** embeddings and uniform pairs — not the learner’s biased features or the logging measure — so some linear visibility can remain on `our_x`/`our_a`.

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

### 3.3 Unit standard deviation

After ⊥ (and double-centering when used), rescale so

$$
\mathrm{sd}_{\text{pairs}}(\phi)=1.
$$

With `norm=v2` we divide **one** factor side only (avoids the legacy double-divide that made v1 residuals quieter).

**In words:** with `norm=v2`, the final $\phi$ has one unit of standard deviation over uniformly weighted catalog user–item pairs. This makes $\gamma$ an intended residual-to-linear SD ratio. The locked R* arm uses legacy `norm=v1`, which divides both factor sides and does not preserve that same realized scaling; this is why §4.4 treats measured `sd_ratio` and `var_share` as authoritative for cross-family comparisons.

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

## 4. Residual families (geometry of $\phi$ only)

Learners do **not** change across families — only the shape of $\phi$ does. That is how we check that hybrid wins are not an R*-only artifact.

### 4.1 R* (tanh) — second index + even warp

$$
\psi(t)=\tanh\big(s(t^2-1)\big)
$$

$$
\phi_{\mathrm{raw}}(u,a)
=
m^{-1/2}\sum_{k=1}^{m}
\psi(t_{u,k})\,\psi(t_{a,k}).
$$

Locked: `s=1,m=4,psi=tanh` (v1 path; **louder** at the same nominal $\gamma$ than SoftAND/SoftThresh).

**In words:** project to $m$ facets, apply an *even* nonlinearity (sign-flip of a facet leaves $\psi$ unchanged), then take a second “bilinear-like” product of warped facets. Intuition: a second interaction channel that is roughly uncorrelated with the linear score under symmetric facets.

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

With softplus $\mathrm{sp}(z)=\log(1+e^{z})$ and locked $c=1$, $s=\texttt{gate\_s}=0.25$:

$$
g_{u,k}=\mathrm{sp}\big((t_{u,k}-c)/s\big),
\qquad
h_{a,k}=\mathrm{sp}\big((t_{a,k}-c)/s\big)
$$

$$
\phi_{\mathrm{raw}}(u,a)
=
m^{-1/2}\sum_{k=1}^{m} g_{u,k}\,h_{a,k}.
$$

Then double-center → ⊥ → unit-SD.

Locked: `m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2`

**In words:** both user and item must “light up” the *same* facet above a soft threshold (AND-like). Average that concordance over facets. Unlike R*, the gate is one-sided (softplus), not an even $\tanh(t^2-1)$ warp. Still a product of per-side factors *before* ⊥.

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

### 4.3 SoftThresh ($\tau=1.5$) — soft threshold on $z_2$

$$
z_2(u,a)=m^{-1/2}\sum_{k=1}^{m} t_{u,k}\,t_{a,k}
$$

$$
\tilde z_2=\frac{z_2-\mu}{\sigma},
\qquad
\phi_{\mathrm{raw}}=\mathrm{sp}\big((\tilde z_2-\tau)/s_{\mathrm{th}}\big).
$$

Then ⊥ → unit-SD. Locked $\tau=1.5$, $s_{\mathrm{th}}=0.25$.

**In words:** form a second bilinear index $z_2$, standardize it, and give a sparse bonus only when it is large. SoftThresh cares that the *sum* of facet products is large, not per-facet co-activation. Softplus-of-a-sum is **not** finite-rank separable — different inductive bias from R*/SoftAND*.

Hot path = (raw − bilinear correction) / `phi_sd`:

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

Define the residual term $r=\gamma\alpha\phi$ and the linear term $\ell_{\mathrm{lin}}=\alpha\langle x,a\rangle$ (plus intercept when measuring CTR):

$$
\mathrm{sd\_ratio}
=
\frac{\mathrm{sd}(r)}{\mathrm{sd}(\ell_{\mathrm{lin}})}
\approx \gamma
\quad\text{(by unit-SD of $\phi$)}
$$

$$
\mathrm{var\_share}_\phi
=
\frac{\mathrm{sd}(r)^2}{\mathrm{sd}(\ell_{\mathrm{lin}})^2+\mathrm{sd}(r)^2}
$$

$$
\mathrm{CTR\_lift}_\phi
=
\mathbb{E}[\sigma(\ell_{\mathrm{lin}}+b'+r)]
-
\mathbb{E}[\sigma(\ell_{\mathrm{lin}}+b')].
$$

**In words:** `sd_ratio` checks the dial's realized scale; `var_share` reports the residual's share of the sum of the two marginal logit variances; and CTR lift reports the change in average click probability when $r$ is added while holding $b'$ fixed. The quantity is called a lift but need not be positive by definition. **Do not compare families at the same nominal $\gamma$**: realized `var_share` differs by family. For example, at $\gamma=0.5$ on ML, SoftAND/SoftThresh have ~20% var share while R* has ~59%.

```81:84:artifacts/full_study/misspec_experiment/measure_reward_ratios.py
    sd_lin = float(np.std(lin))
    sd_res = float(np.std(res))
    sd_ratio = sd_res / max(sd_lin, 1e-12)
    var_share = (sd_res ** 2) / max(sd_lin ** 2 + sd_res ** 2, 1e-12)
```

---

## 5. Logging and propensities

The logger $\pi_0$ is a softmax over **biased** scores at frozen temperature $T_{\log}$:

$$
\pi_0(a\mid u)
=
\frac{\exp\big(\langle \tilde x_u,\tilde a_a\rangle / T_{\log}\big)}
{\sum_{a'}\exp\big(\langle \tilde x_u,\tilde a_{a'}\rangle / T_{\log}\big)}.
$$

A logged row is

$$
u\sim\mathrm{prior},\quad
a\sim\pi_0(\cdot\mid u),\quad
p=\pi_0(a\mid u),\quad
r\sim\mathrm{Bernoulli}\big(q_\gamma(u,a)\big).
$$

**In words:** users are drawn from a fixed prior; items are drawn from the logger; we store the true propensity $p$; clicks are coin flips with success probability equal to the *true* residual-aware CTR. Propensity is simulator-true. This study is **not** about confounding or propensity misspecification.

**Sanity check:** biased embeddings create representation and score-class mismatch, not unobserved confounding. The simulator records the action probabilities generated by $\pi_0$, so the propensity denominator is known.

---

## 6. What learners optimize

Same policy score class $\mathcal{F}$ across methods in a cell (default: `--policy-transform linear`).
Separately, DR uses a reward model $\hat q$ (regression on interaction features; cross-fit) as a **nuisance**.

A policy scores actions with $s_\theta(u,a)$ and (when needed) a softmax policy

$$
\pi_\theta(a\mid u)
=
\frac{\exp(s_\theta(u,a)/T)}{\sum_{a'}\exp(s_\theta(u,a')/T)}.
$$

**In words:** $s_\theta$ is the learner's score, $\theta$ denotes its fitted parameters, and $T$ controls how sharply those scores are converted into action probabilities during policy optimization. Exact-greedy evaluation later uses only the score ordering.

### 6.1 NLL — logged logistic likelihood

$$
L_{\mathrm{NLL}}(\theta)
=
\mathrm{BCEWithLogits}\big(z_\theta(u,a_{\mathrm{logged}}),\, r\big)
=
-\Big[
r\log\sigma(z)
+
(1-r)\log(1-\sigma(z))
\Big]
$$

with $z=z_\theta(u,a_{\mathrm{logged}})$.

**In words:** treat the logged action’s click as a binary label and fit a logistic model on the model’s click logit for that action. No importance weights, no $\hat q$. Hyperparameters selected by validation NLL.

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

Under class misspecification, NLL’s population target is roughly a **KL / likelihood projection** of $q^\star$ onto $\mathcal{F}$ — not necessarily the best ranking for value.

### 6.2 OPC / raw DR — value surrogate

Doubly robust policy objective (direct method + IPS correction), **no** self-normalization on the train loss:

$$
w
=
\frac{\pi_\theta(a_{\mathrm{logged}}\mid u)}{\pi_0(a_{\mathrm{logged}}\mid u)}
$$

$$
\widehat V_{\mathrm{DR}}(u;\theta)
=
\underbrace{\sum_a \hat q(u,a)\,\pi_\theta(a\mid u)}_{\text{direct method (DM)}}
+
\underbrace{w\,\big(r-\hat q(u,a_{\mathrm{logged}})\big)}_{\text{IPS correction}}
$$

Train minimizes $-\widehat V$ (ascend estimated value). Study lock: raw train weights, direct gradients, select by DR lower bound with clipped weights.

**In words:**

- DM: under the reward model, what CTR would this policy earn on this user?
- Correction: if the logged action was under/over-weighted relative to $\pi_0$, add an importance-weighted residual $(r-\hat q)$.
- If either $\hat q$ is right *or* propensities are right, the estimator of $V(\pi)$ is consistent (double robustness). Here propensities are exact; $\hat q$ can still be wrong.

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

Grid lock (excerpt): `--policy-losses dr --opc-gradient direct --train-weights none --select-weights clip:10`.

### 6.3 Hybrid sequential — NLL then DR

**Not** a mixed loss $\lambda L_{\mathrm{NLL}}+(1-\lambda)L_{\mathrm{DR}}$. Instead a schedule:

$$
E_{\mathrm{NLL}}=\mathrm{round}(f\cdot E),
\qquad
E_{\mathrm{DR}}=E-E_{\mathrm{NLL}}.
$$

1. Train with NLL for $E_{\mathrm{NLL}}$ epochs (learn click bias).
2. **Freeze** click bias.
3. Continue with raw DR for $E_{\mathrm{DR}}$ epochs (same Adam).
4. Select the trial by DR lower bound.

Arms: `hybrid_seq_f0.25`, `f0.5`, `f0.75`, and `hybrid_seq_adapt` (val-NLL plateau switch with $f\in[0.25,0.75]$).

**In words:** first fit the click-likelihood geometry, then switch to a value objective without inventing a shared scale for two losses. The same optimizer continues across phases, but the click-bias parameter is frozen at the switch. Also present: `nll_dr` = NLL **train** + DR-LB **select** only (a selection-only comparison arm, not sequential DR training).

**Sanity check:** `hybrid_seq` is neither a weighted mixed loss nor merely NLL training with DR-based model selection. It performs NLL updates first and DR updates second.

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

---

## 7. Evaluation — exact-greedy CTR

Primary metric: true value of recommending each user the policy’s top item,

$$
V_{\mathrm{gr}}(\pi_\theta)
=
\sum_u \mathrm{prior}(u)\,
q_\gamma\big(u,\, \arg\max_a s_\theta(u,a)\big).
$$

**In words:** for each user, pick the action with the highest learned score (argmax; temperature does not matter), look up the *true* click probability $q_\gamma$ for that pair, and average over users with the user prior. Stored as `policy_rewards_greedy` in `summary_metrics.csv`. Residual worlds route $q$ through `env.q_block_torch` / `reward_prob*` so evaluation sees the dial.

```434:439:utils/simulation_utils.py
def calc_greedy_reward(dataset: dict, user_emb: np.ndarray, item_emb: np.ndarray, *, chunk_size: int = 2048) -> float:
    """True value of the policy's greedy (exploitation) part: sum_u prior[u] * q(u, argmax_a x_u . a_a).
    ...
    """
```

---

## 8. Contrasts and “DiD”

Per seed (common random numbers across $\gamma$), in CTR **percentage points**:

$$
\Delta_{m,m'}(\gamma)
=
100\big(V_m(\gamma)-V_{m'}(\gamma)\big)
$$

$$
\mathrm{DiD}_{m,m'}(\gamma)
=
\Delta_{m,m'}(\gamma)-\Delta_{m,m'}(0).
$$

Primary: $m=\texttt{hybrid\_seq\_f0.5}$, $m'=\texttt{nll}$ (also OPC vs NLL).

Expanding the primary contrast:

$$
\mathrm{DiD}_{\mathrm{H,NLL}}(\gamma)
=
\big(V_{\mathrm{H}}(\gamma)-V_{\mathrm{NLL}}(\gamma)\big)
-
\big(V_{\mathrm{H}}(0)-V_{\mathrm{NLL}}(0)\big).
$$

**In words:** first measure “hybrid minus NLL” at a given $\gamma$; then subtract the same gap at $\gamma=0$. A positive DiD means hybrid’s advantage over NLL *grew* when the residual was turned on (relative to the $\gamma=0$ baseline). This is a **method × γ interaction contrast** identified by design (manipulated dial + paired seeds). It is **not** classical difference-in-differences with a parallel-trends assumption.

The subtraction does not make $\gamma=0$ a perfectly specified control condition: that baseline still contains representation mismatch from the biased embeddings. It removes the methods' baseline gap, whatever its source, and isolates how that gap changes with the residual dial.

CIs: Student-$t$ intervals on the paired seed vector.

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

---

## 9. Confirmatory grid (what is running)

From `run_full_hybrid_grid.sh`:

| Axis | Levels |
|------|--------|
| methods | `opc`, `nll`, `hybrid_seq_f{0.25,0.5,0.75}`, `hybrid_seq_adapt` |
| families | R*, SoftAND*, SoftThresh $\tau=1.5$ |
| $\gamma$ | 0, 0.25, 0.5, 0.75, 1, 2, 4, 10, 100 (+ anime 0.6) |
| datasets | ml, anime, myket, kuairand/kuairec full + mlsize |
| seeds | 300–339 ($n=40$) |
| capacity | `linear` (main / identifying grid) |
| $n$ / val | 100k train, 20k val; Optuna 20 trials |

Tags look like `nlres_grid_{ds}_{kind}_g{γ}`.

**Fair reasonable $\pi$ (claims lock):** matched `--policy-transform linear+mlp --mlp-hidden 64` is the production-like head for method comparisons under residual capacity; linear remains the identifying small-$\mathcal F$ arm. See `CLAIMS_FAIR_PI.md` and `capacity_fair_pi/REPORT.md`.

---

## 10. Capacity trial (fair reasonable $\pi$)

Hold the residual dial fixed and enlarge the policy class to `linear+mlp` (primary hidden **64**; **256** as sensitivity). Ask whether

$$
\mathrm{DiD}_c
=
\Delta_{\mathrm{H,NLL}}(\gamma;c)-\Delta_{\mathrm{H,NLL}}(0;c)
$$

**shrinks** as capacity $c$ grows ($\phi$ was misspecification relative to a small class) or **persists** (the structure remains hard for the expanded class, or other mechanisms matter).

**In words:** for each capacity setting $c$, recompute how the hybrid–NLL gap changes from $\gamma=0$ to the chosen positive $\gamma$, then compare that interaction across widths. More capacity would support a simple capacity explanation only if this contrast systematically moved toward zero.

**Lean result (ml, $n=10$, R* + SoftThresh, $\gamma\in\{0,0.5,2\}$):**

- **R***: H−NLL DiD stays clearly positive under mlp64 (**+0.99** at 0→0.5; **+3.22** at 0→2) and mlp256 — not an artifact of refusing the MLP.
- **SoftThresh:** mid-$\gamma$ softens / CI crosses 0 under MLP (NLL absorbs $\phi$ — class-relative); loud SoftThresh only partially shrinks.
- At $\gamma=0$, hybrid’s cost vs NLL **worsens** with width (≈−0.63 pt @64 vs ≈−1.19 @256) → prefer width **64**.

**SoftAND* gap:** capacity wave omitted SoftAND*; run `run_capacity_softand_mlp64.sh` (γ∈{0, 0.5, 100}, mlp64, matched methods).

Protocol: `PROTOCOL_DRAFT_capacity.md`, `CLAIMS_FAIR_PI.md`; drivers: `run_capacity_trial.sh`, `run_capacity_softand_mlp64.sh`; report: `capacity_fair_pi/REPORT.md`.

---

## 11. Framing — what kind of “misspec” this is

**Tight label:** controlled **class / mean-function misspecification** of the click/policy score relative to a locked $\mathcal{F}$, studied via a residual dial and interaction contrasts.

For $\gamma>0$ (after ⊥), by construction

$$
\ell_\gamma \notin \mathcal{F}_{\mathrm{linear}}
$$

in the clean catalog sense — the true logit is outside the bilinear score class the linear learner uses.

**In words:** after the residual is made perpendicular to clean affine-bilinear structure, no member of the locked linear score class exactly represents the resulting true logit over the clean catalog. This is a class-relative statement, not a claim that every richer model must fail.

| Is | Is not |
|----|--------|
| Score-class misspec of $\mathcal{F}$ (and separately: $\hat q$ nuisance misspec) | Unobserved confounding |
| Different pseudo-true targets: NLL ≈ KL projection of $q^\star$; DR ≈ value | Propensity misspec ($\pi_0$ known) |
| Stress / sensitivity analysis in $\gamma$ | “The world is misspecified” |
| Class-relative claim until capacity says otherwise | Cross-family claims at equal nominal $\gamma$ |

Review corrections: `MISSPEC_FRAMING.md` §9.

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
| Fair-π capacity report | `artifacts/.../capacity_fair_pi/REPORT.md` |
| Fair-π claims lock | `artifacts/.../CLAIMS_FAIR_PI.md` |
| Capacity analyzer | `artifacts/.../analyze_capacity_fair_pi.py` |
| SoftAND mlp64 driver | `artifacts/.../run_capacity_softand_mlp64.sh` |
| Dial loudness | `artifacts/.../measure_reward_ratios.py` |
| Results PDF | `artifacts/.../build_hybrid_results_pdf.py` |
| Framing | `artifacts/.../MISSPEC_FRAMING.md` |

---

## 13. Reading order for a new reader

1. This file (§0–§2, §6–§8).
2. `SOFTAND_SOFTTHRESH_EXPLAINED.md` + `FAMILIES_EXPLAINED.md`.
3. `MISSPEC_FRAMING.md` (especially §9).
4. Skim `PROTOCOL_DRAFT_hybrid.md` for why sequential hybrid (not a mixed loss).
5. Open `utils/residual_env.py` and `models/custom_losses.py` for the arithmetic.
