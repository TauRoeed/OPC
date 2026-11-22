# Scaling laws under misspecification — protocol draft

**Status.** Draft for discussion (2026-11-21). Built by synthesizing four independent critiques:
math/functional forms, experimental protocol, theory–practice claims, and LLM-scaling analogy audit.

**One-line stance.** Do **not** port Kaplan/Chinchilla as “compute-optimal N vs C for recommenders.” Port the *floor + power-law excess* structure, then specialize to **objective-dependent floors**, **feature-gated capacity**, and **logged-data / ESS** as the scarce resource.

**Notation note.** All formulas below are plain ASCII (phone-safe). Greek letters are spelled out where helpful: gamma = misspec dial, phi = residual, pi0 = logger, Delta = gap.

---

## 0. Why LLM scaling analogies mostly break here

| LLM assumption | Offline OPL reality (this project) |
|---|---|
| Train loss ≈ eval target | Logged NLL ≠ greedy CTR when gamma>0 (Goodhart / proxy–gold) |
| One irreducible floor E | Stacked floors: feature observability, class capacity, **Delta_proj** (NLL vs value) |
| More params help given data | When R2(phi | biased features) ≈ 0, width buys ~0 headroom and can **hurt** DR via IPS variance |
| Compute is binding | Policy models are tiny; binding axes are **N**, ESS, HPO budget |
| Smooth power laws in gamma | gamma is a **stress dial**, not a resource; expect quadratic onset then plateau |

**Transferable recipes (keep):**
1. Fit **reducible** quantities (regret to a pinned oracle floor; KL minus H(q)), never raw CTR/CE.
2. Pin floors from oracles; do not free-fit E after looking at curves.
3. Three-regime curves (small-data → power-law → floor); report the knee.
4. Hold-out largest-N prediction before claiming a “law.”
5. Matched HPO / selection across every scale point.

---

## 1. Mathematical core

### 1.1 Axes

Scale point:

    theta = (N, |U|, |A|, d, C, F, gamma, s)

where
- N = logged training rows
- |U|, |A| = users, catalog size
- d = embedding dim
- C = policy capacity (params / width)
- F = training compute (secondary here)
- gamma = residual dial
- s = signal_kept (representation bias)

For DR/IPS use effective sample size:

    N_eff = (sum_i w_i)^2 / (sum_i w_i^2)

    w_i = pi(a_i | u_i) / pi0(a_i | u_i)

Distinguish experiments: N fixed vs N/|A| fixed vs N/|U| fixed.

### 1.2 Regret and additive law

    R_m(N, C, gamma, s) = V_ref(C, gamma) - V_m(N, C, gamma, s)

Pin the reference **before** fitting (never “% of oracle”):

    V_ref(C, gamma) = max( V_lin_val(gamma), V_repair_C(gamma) )

Candidate form (only identify when axes are varied independently):

    R_m = B_m(gamma, s)
        + a_m * N_eff_m ^ (-alpha_m)
        + b_m * C ^ (-beta_m)
        + c_m * F ^ (-delta_m)

B_m is the asymptotic floor. Expect B_NLL >> B_DR under misspec because of Delta_proj.

### 1.3 Exact regret decomposition

    V_ceil - V_m
      = Delta_feat(gamma; s, R2_phi)          # unrecoverable by N, C, method
      + Delta_class(gamma, C; R2_nl)          # shrinks only if R2_nl > 0
      + Delta_obj_m(gamma, C)                 # NLL: Delta_proj_eff; DR ≈ 0
      + eps_est_m(N, C, pi0, gamma)           # finite-sample + optimization

Representation floor (local loss intuition):

    B_repr_L  proportional to  (gamma * alpha)^2 * Var(phi) * [1 - R2(phi | biased_z)]

Value floors need not be quadratic (ranking margins); fit exponent q on:

    B_repr_V  proportional to  |gamma|^q * [1 - R2]

### 1.4 Two laws worth claiming

**Law 1 — Misspecification–variance crossover (primary).**  
At matched capacity C:

    D(gamma; N, C)
      = V_DR - V_NLL
      = Delta_proj_eff(gamma, C) - Delta_eps_est(N, C, pi0, gamma)

Define the dimensionless ratio:

    Psi = Delta_proj_eff / Delta_eps_est

- DR beats NLL when Psi > 1
- Hybrid (f ≈ 0.5) wins when Psi ≳ lambda(f) ≈ 0.2

**Headline scaling object:** crossover N_star(gamma) (falls as gamma / DeltaDelta_proj grows), not a universal exponent.

**Law 2 — Feature-gated capacity ceiling (secondary).**  
Capacity gains proportional to R2_nl(phi | biased_z).  
When R2_nl ≈ 0 (current high-bias cell), value saturates near V_lin_val and extra width can worsen finite-sample DR/hybrid.

### 1.5 Hybrid schedule as the “compute” dial

True Chinchilla N–C allocation is mostly decorative here. The practical dial is objective schedule f = fraction of epochs on NLL before DR fine-tune. Empirically f_star = 0.5 cuts IPS tax while keeping most of Delta_proj.

---

## 2. Experimental roadmap

### Precondition (Phase 0) — must pass before any N-curve claim

| ID | Audit | Kill / pass |
|---|---|---|
| A1 | **Nested data:** one pool at N_max per seed; train prefixes nested; val fixed 20k | Current code folds train_size into sim seed → independent draws. **Fix first.** |
| A2 | Equal data accounting across methods (reg_data / q-hat folds) | Unequal subsidy ⇒ stop |
| A3 | NLL asymptote vs V_lin_lik at huge N (no HPO) | If NLL never approaches lik-oracle, drop F2 pin for NLL |
| A4 | HPO 20 vs 60 trials at small & large N | If best-of-20 ≪ best-of-60, HPO is binding |
| A5 | Nested 100k ≡ legacy nlres_grid_ml_* (±0.5 pt) | Licenses overlay of old anchors |
| A6 | Finish capacity gamma=2 cells to n=40; extend oracles to scaling seeds | Anchors complete |

**Also:** finish / deprioritize remaining hybrid grid; free GPUs from vLLM before Phases 2–5.

### Phase 1 — cheap probe (seeds 100–104)

- ml, **linear**, methods {nll, opc, hybrid_f0.5}
- N in {25k, 50k, 100k, 200k, 400k, 800k}
- Worlds: R* gamma=0, R* gamma=2, **well-spec control** (bias=none, gamma=0)

**Kill switches:** flat curves (K1), HPO-bound (K2), selection-bound (K3), control fails (K4: NLL must ≥ OPC and converge).

### Phase 2 — identifying N × gamma (linear)

- Seeds **300–309**, nested; gamma in {0, 0.5, 2} × {R*, SoftThresh} (+ shared gamma=0)
- N up to 1.6M; fit beta_DiD, N_star; forms by **leave-largest-N-out**

### Phase 3 — N × C surface

- C in {linear, mlp16, mlp64, mlp256, mlp1024}
- Worlds {gamma0, R* gamma2, ST gamma2}; sparse N grid
- **H_C1:** capacity does not substitute for data (N×C on DiD null/shrink)
- q-hat class **fixed** while policy C varies

### Phase 4 — generality

- 4a multi-ds (matched catalog size)
- 4b catalog subsample **within** ml (index by measured DeltaDelta_proj, not gamma)
- 4c ESS collapse: DR curves vs (N * ESS) better than vs N?

### Phase 5 — confirmatory (seeds 400–419)

- Freeze form + params + script hash; predict N=3.2M
- ≥80% of cell means inside frozen 90% PIs, else claim **trends only**

**Budget sketch:** ~18–20k RU (~100–130 dedicated GPU-hours) after Phase 0.

---

## 3. Pre-registered claims (decision rules)

| ID | Claim | Support if… | Refute if… |
|---|---|---|---|
| S1 | Misspec advantage is bias, not small-sample | beta_DiD CI LB > 0 | CI UB < 0 |
| S2 | Inverse / saturating NLL under gamma>0 | F4 or negative slope + NLL↔V divergence | Monotone NLL gains forever |
| S3 | Crossover N_star(gamma) falls with DeltaDelta_proj | Slope CI LB < 0 | Flat / rising |
| S4 | Well-spec control: NLL ≥ OPC for all N | Required for S1–S3 | Pipeline artifact |
| S5 | Capacity does not substitute for data when R2_nl ≈ 0 | N×C null / DiD persists | DiD vanishes with width |
| S6 | ESS is DR’s data unit | Collapse residual down ≥30% | No collapse |
| S7 | Extrapolation to 3.2M | Coverage ≥80% | Trends-only paper |

---

## 4. Figure set (paper-ready)

1. **Regret stack vs gamma** — approx / Delta_proj / estimation  
2. **Reducible KL vs C** (log-log), floors dashed  
3. **Value regret vs C** — NLL / DR / hybrid asymptotes  
4. **Regret vs N** + inset N_star(gamma)  
5. **Iso-regret (N×C) heatmap** for D = V_DR − V_NLL (L-shaped above floor)  
6. **DiD(gamma)** with quadratic-onset fit (not a power law in gamma)  
7. **Collapse:** DiD vs Delta_proj across families  
8. **Mirage:** continuous vs discontinuous metrics vs gamma  
9. **OPE fidelity** vs exact V (sigma/sqrt(ESS) floor)  
10. **Proxy–gold:** held-out NLL vs oracle V, Spearman vs gamma  

---

## 5. Decision tree (practitioner lever)

```
if R2_nl(phi | biased) ≈ 0 and s low:
    do NOT scale policy C
    if gamma unknown/mid/high → Hybrid f=0.5; scale N_eff / features
    if gamma ≈ 0 → pure NLL; scale N or raise s
elif R2_nl > 0.05:
    scale C + Hybrid; keep N_eff proportional to C
```

---

## 6. What not to do

1. Fit power laws on **non-nested** N draws without saying so.  
2. Free-fit asymptotes after peeking.  
3. Claim exponents on <1 decade / <5 levels.  
4. Confound capacity with method (OPC-mlp vs NLL-linear).  
5. Treat FLOPs / “20 tokens per param” as binding.  
6. Call gamma-onset “emergence.”  
7. Index misspec by gamma across datasets (use DeltaDelta_proj).  
8. Use hybrid_seq_adapt as a primary scaling arm (f entangled with N).  
9. Confirm on dev seeds 100–104.  
10. Drop collapsed cells; use ITT.

---

## 7. Immediate next actions (ordered)

1. **Phase 0 A1:** implement nested-data train prefixes (block everything else).  
2. **A3** asymptote audit (NLL vs V_lin_lik) — decides F2 for NLL.  
3. **A6** finish capacity gamma=2 to n=40 when GPUs free.  
4. Phase 1 probe on reserved GPU; only then Phase 2.  
5. Keep recoverability CSVs as fixed V_ref / R2_phi tables; extend seeds as needed.

---

## 8. Open follow-on papers

1. Quantitative (N, C, kappa(pi0), gamma) iso-value frontier (true OPL data law).  
2. Closing the ~10 pt gap Hybrid → V_lin_val (nuisance C_qhat vs policy C_pi).  
3. Why oracle Delta_proj fails to predict SoftAND/ST DiD until gamma→inf (ranking-invariance geometry).  
4. Unfreeze embeddings: can Hybrid unlock V_info → V_ceil?  
5. Theory of NLL→DR as localized regularization around pi_KL.  
6. Phase diagram with propensity misspec eps_pi × gamma.

---

## Appendix — synthesis sources

| Pass | Focus |
|---|---|
| Math draft | Axes, R = B + a N^(-alpha) + …, decomposition, crossovers |
| Protocol critic | Nested-data precondition, phased RU budget, S1–S7, kill switches |
| Theory–practice | Law 1/2 wording, Psi ratio, lever matrix |
| Analogy audit | Transferable recipes, false analogies, Fig 1–10 |

*This draft does not authorize new GPU batches until Phase 0 A1–A3 pass and hybrid/capacity anchors are completed or deprioritized.*
