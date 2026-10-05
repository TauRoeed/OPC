# BLOB's prior and catalog size: derivation, calibration and the 25k check

*Development stage. Branch `blob-prior-calibration`, from `blob-controlled-integration` (1a01084). §1 (the derivation)
and §2 (the pre-registered calibration and decision rule) were committed before any calibration run. Results come
after them. The published/default BLOB rows of `docs/blob_controlled_integration.md` are unchanged and stay the
reference. The calibrated variant is a separate, explicitly named arm.*

**The question.** The controlled study found that default BLOB, transferred faithfully to our catalogs (P = 3,533–10,803
items), barely adapts from its source and trails CausE-cap, OPC and DM-only on biased worlds. Is that weak adaptation
intrinsic to BLOB here? Or is it largely a consequence of a prior and parameterization calibrated in the paper's
P ≈ 100 regime?

**In brief.**
- **Where P enters.** In the released code every entry of BLOB's K × K target correction has prior variance
  s+(w_b)² s_ζ² / P (§1). The 1/P inside L is applied after Ψ's columns are normalized, so the prior tightens with
  catalog size: 35–108 times tighter in our catalogs than at the paper's P = 100.
- **The calibration.** Computing L at a fixed reference size P₀ removes the dependence. Of P₀ = 1,000, 100 and 10,
  the pre-registered rule chose P₀ = 10 on the 18 tuning worlds, 0.12 points above P₀ = 100 (§3).
- **The 25k result.** On the 30 main worlds, BLOB-Pnorm-NQ (P₀ = 10) gains +2.82 on biased worlds (§4).
  - +1.40 [+0.81, +1.98] over default BLOB, in 24/24 worlds.
  - Level with OPC: +0.13 [−0.08, +0.34].
  - Just below CausE-cap: −0.28 [−0.57, +0.01], all of it under warp.
  - Above DM-only: +0.66.
- **The mechanism.** The correction is 8.5 times larger and the click model fits better, and the training gap closes
  (4.72 → 3.56 points). Without bias it stays as close to the logger as default BLOB (§5–§6).
- **Outcome: B** (§7), with C's condition partly met: tied strongest under combined bias. The published BLOB's rows
  stay as they were.

## 1. Where catalog size enters BLOB's prior (Phase 1)

Sources: the paper, §2 and the hierarchical model (arXiv 2008.12504); the released graph,
`criteo-research/blob` @ e15cb38, `models/models_organic_bandit.py`, lines 225–337 (MNQ) and 606–703 (NQ).

### 1.1 The model as released

The paper's prior on the P × K bandit embedding is a matrix normal:

```text
β | Ψ  ~  MN( s+(w_a) Ψ,  s+(w_b) Ψ Ψᵀ,  s+(w_b) (1/P) Ψᵀ Ψ )
```

It is sampled through a K × K variable:

```text
ζ ~ MN(0, I_K, I_K),   L = chol( (1/P) Ψᵀ Ψ ),   β = s+(w_a) Ψ + s+(w_b) Ψ ζ Lᵀ
```

The released graph does exactly this, with one addition: the released setting `norm=True` normalizes Ψ's columns.
- `Psi_cov = Psi_loc`, then `Psi_cov /= np.linalg.norm(Psi_cov, axis=0)` in place, so Ψ_loc is normalized too
  (lines 228–229).
- `cov_k = Psi_cov.T.dot(Psi_cov) / self.P` and `L = np.linalg.cholesky(cov_k)` (lines 232–233).
- The ζ prior std is the constant `s_zeta = 1.0` (line 243). It is not an argument.

Write Ψ̃ for the normalized Ψ. Every column of Ψ̃ has unit norm over the P items, so diag(Ψ̃ᵀΨ̃) = 1 exactly. A
logged row (user u, item a) has the logit (lines 314–319 for MNQ, 698–703 for NQ, and the point estimate on line 382):

```text
logit(u, a) = Ψ̃_a M ω̂_u + κ_a + w_c,     M = s+(w_a) I_K + s+(w_b) ζ Lᵀ        (Ψ̃_a a 1 × K row)
```

**How ζ enters.**
- The learned target correction is the K × K matrix ΔM = s+(w_b) ζ Lᵀ, added to the source map s+(w_a) I_K.
- Its point estimate is s+(μ_wb) μ_ζ Lᵀ, the β̂ of line 382.
- Users and items are corrected only through this one map (plus the intercepts κ).

### 1.2 The prior on the correction, exactly

With ζ_ij iid N(0, s_ζ²), each row of ΔM is s+(w_b) ζ_i,: Lᵀ, a Gaussian with covariance:

```text
Cov(ΔM_i,:) = s+(w_b)² s_ζ² L Lᵀ = s+(w_b)² s_ζ² Ψ̃ᵀ Ψ̃ / P          (the same for every row i)
```

Because diag(Ψ̃ᵀΨ̃) = 1, **every entry of the correction has prior variance s+(w_b)² s_ζ² / P, exactly.** The off-diagonal
structure, Ψ̃ᵀΨ̃ (the item factors' correlations), does not depend on P.

**Where P enters, and why ‖L‖ changes with P.**
1. **Through the column normalization.** It puts Ψ̃ᵀΨ̃ at unit scale whatever P is.
2. **Through the 1/P inside L.** In the paper's own formula (raw Ψ), (1/P)ΨᵀΨ is the item factors' mean second
   moment, which does not depend on P: there the 1/P is a normalization. Applied after the column normalization, the
   same 1/P normalizes a second time. So LLᵀ = Ψ̃ᵀΨ̃/P has diagonal 1/P, and ‖L‖_F = √(K/P).
   - The combination makes the correction's prior shrink as 1/P.
   - It is the released default, used in the paper's experiments at P = 100 (Table 3) and P = 1,000 (Table 4), with
     the same priors.
3. **Through the source term**, which is minor (§1.4).

**Which hyperparameters set the correction's prior scale.**
- The ζ prior std s_ζ, fixed at 1 in the release.
- The prior of w_b, N(μ_wb, 1), which scales ΔM by s+(w_b).
- The constant inside L (P), the one that depends on the catalog.

### 1.3 The invariant and the normalization

**The invariant.** The quantity to hold fixed when moving from the paper's P₀ to our P is the per-entry prior variance
of the correction to the user-side map, s+(w_b)² s_ζ² diag(LLᵀ), at given w_b and s_ζ.
- M is K × K whatever P is, and its entries are what the bandit data must determine.
- So the per-entry prior is what makes the prior "the same".
- It does not involve K. The paper used K = 20 and we use K = 32, and the per-entry prior does not change with K.

**The earlier report's numbers, corrected.** The previous report said 4.5–8× larger ζ and a 20–64× tighter prior.
That compared ‖L‖_F = √(K/P) across K = 20 and K = 32, which mixes the dimension change into the catalog effect. The
exact catalog factor, at fixed K and per entry, is √(P/P₀):

| dataset | P | √(P/100) | prior variance per entry, relative to P₀ = 100 |
|---|---|---|---|
| ml | 3,533 | 5.94 | 1/35 |
| kuairand | 7,579 | 8.71 | 1/76 |
| anime | 10,803 | 10.39 | 1/108 |

**The catalog-normalized parameterization.** Compute L at a reference catalog size P₀ instead of P:

```text
L_P₀ = chol( Ψ̃ᵀ Ψ̃ / P₀ ) = √(P/P₀) · L
```

Everything else stays as released: the normalization, ζ ~ N(0, I) with s_ζ = 1, the w priors, the noise term, Adam
and selection.
- With P₀ = 100, the paper's Table 3 catalog, every entry of the correction has the prior variance it had at
  P = 100: s+(w_b)² s_ζ² / 100.
- **The scale differs by dataset because P differs:** the factor is 5.94 (ml), 8.71 (kuairand) and 10.39 (anime). One
  constant for all datasets would leave the prior catalog-dependent across our own datasets.
- **Name: BLOB-Pnorm (P₀ = 100)**, the arm `blob_l100_nq`. The published/default BLOB is the special case P₀ = P.

**Why scale L and not s_ζ.** Multiplying s_ζ by √(P/P₀) gives the same prior on ΔM. It is not the same model to
optimize.
- The ELBO is the same function of the correction in both forms: ζ_σ = √(P/P₀) ζ_L. So the two have the same
  optimum, the same posterior over ΔM.
- But Adam takes steps of roughly constant size in parameter units, and one unit of ζ moves ΔM by s+(w_b)·L.
- Released L at our P is 6–10× smaller than at P = 100. So the released parameterization also learns the correction
  6–10× more slowly in function space per step. Under the s_ζ-only change it still does.
- L_P₀ restores both the prior and the step geometry of the P₀ regime. That is what "transporting the P = 100
  calibration" means here.
- The s_ζ-only version is run as a diagnostic (§2). It separates the prior from the optimization geometry.

### 1.4 What does not need normalizing

- **The source term.** s+(w_a) Ψ̃_a ω̂ also scales with P, as √(K/P) at given w_a, because the columns are normalized.
  - The learned s+(w_a) absorbs this. It is one scalar whose prior is weak against 25,000 rows: the controlled study
    learned s+(w_a) ≈ 5.7.
  - Normalizing it too would change the source's prior mean, which is not the question here.
- **κ, w_c and their priors** do not involve P per entry.
- **The KL/N weighting** depends on N, not P. N = 25,000 here; it is a data-size effect and is left as is.

## 2. Pre-registered calibration (Phases 2–4)

Fixed before any calibration run (this section's commit).

**Hypothesis.** BLOB under-adapts in our environment because the released prior and parameterization of the correction
become unintentionally stronger with catalog size.

**Setting, held fixed** (isolating the prior):
- The NQ family, the controlled study's primary family. MNQ is not run: its released noise uses the prior stds and
  would mix the noise with the prior.
- σ_κ = 0.1. The other dimensions searched as in the controlled main grid:
  - epochs {10, 30, 100, 300, 1,000};
  - μ_wa {−1, 1, 3};
  - μ_wb {−6, −3, 0};
  - lr log-uniform, with the upper end extended by one geometric step (Phase 3): [3e-3, 3e-1], against [3e-3, 1e-1]
    before.
- The 18 tuning worlds of the controlled study: seeds 200/201 × ml, kuairand, anime × warp, vector and combined high;
  N = 25,000.

**The prior grid** (one study per world; each configuration trained under every variant, paired):

| variant | arm label | L | s_ζ | per-entry prior variance of ΔM, at given w_b |
|---|---|---|---|---|
| A. released / default | `blob_nq` | chol(Ψ̃ᵀΨ̃/P) | 1 | s+(w_b)²/P |
| B. **BLOB-Pnorm, P₀ = 100** (the derived normalization) | `blob_l100_nq` | chol(Ψ̃ᵀΨ̃/100) | 1 | s+(w_b)²/100 |
| C. stronger anchor, P₀ = 1,000 (the paper's Table 4 catalog) | `blob_l1000_nq` | chol(Ψ̃ᵀΨ̃/1,000) | 1 | s+(w_b)²/1,000 |
| D. weaker anchor, P₀ = 10 | `blob_l10_nq` | chol(Ψ̃ᵀΨ̃/10) | 1 | s+(w_b)²/10 |
| E. prior only, P₀ = 100 (diagnostic, not a candidate) | `blob_s100_nq` | chol(Ψ̃ᵀΨ̃/P) | √(P/100) | s+(w_b)²/100 |

- **The anchors are geometric**, a factor √10 apart in the L scale (10× in prior variance). C is the paper's own
  larger catalog; D is a meaningfully weaker prior than the P = 100 regime.
- **E has B's prior with the released step geometry.** B − E isolates the optimization geometry, and E − A isolates
  the prior.
- **Pairing.** 30 configurations (lr, epochs, μ_wa, μ_wb) are drawn per world from a fresh seed tag (`calibration`).
  Every configuration is trained under all 5 variants, with the same batch order and the same noise draws: 150 trials
  per world. Variants differ only in L or s_ζ.

**Decision rule (Phase 4)**, the controlled study's principles:
- **Score of a variant.** A 20-trial study (the main grid's budget) is simulated from the variant's 30 trials, by 300
  resamples. It selects by validation NLL, and the score is that trial's true greedy gain over the logger's greedy
  value, averaged over the 18 worlds. Paired differences against A come with 95% CIs over worlds.
- **The calibrated configuration is chosen among B, C and D** (A is the reference, E a diagnostic). It is the
  best-scoring anchor. If that anchor beats B by less than 0.10 points, B is kept, the value derived a priori.
- **One anchor for every dataset and bias type.** No per-bias choice; nothing is tuned on the 30 main worlds.
- **The learning-rate edge (Phase 3).**
  - For A and the chosen anchor, the lr marginal is reported per half-decade. So is the score of A restricted to the
    previous range, lr ≤ 1e-1, which tells whether the extension changed the released BLOB.
  - If the best half-decade is again the top one, the range is not widened again; it is reported as a limitation.
- **Main grid (Phase 5).**
  - The chosen calibrated arm, NQ, σ_κ = 0.1, with the extended space above.
  - 20 trials per world (no seed tag), on the 30 main worlds of the controlled study, once.
  - Pick diagnostics and saved policies, as before.
  - Every comparator is reused (§3): default BLOB-NQ and BLOB-MNQ, CausE-cap at ρ = 0, OPC, DM-only and the tempered
    logger.

**Outcome classes (Phase 8)**, read from effect sizes, intervals and the mechanism diagnostics together:
- **A, calibration does little.** Calibrated BLOB stays clearly below CausE-cap and OPC on biased worlds.
- **B, the gap substantially closes.** Calibrated BLOB becomes competitive but does not dominate.
- **C, the ordering changes.** Calibrated BLOB is the strongest or tied strongest in important regimes. In that case,
  stop and analyze before any simulator change.

## 3. Calibration result and the frozen choice

**Run.** `run_blob_calib_s200`, code f9b258b on a pinned worktree, 2026-10-05 20:35–21:31, 3 workers. 18 tuning worlds
× 30 configurations × 5 variants = 2,700 trials; none diverged. Tables are in
`artifacts/full_study/blob_prior_calibration/calibration/`: `calibration_decision.csv`, `calibration_mechanism.csv`,
`calibration_lr_*.csv` and `calibration_edges.csv`. Rebuild them with
`python -m training.analyze_blob calib --runs artifacts/full_study/run_blob_calib_s200 --out …`.

**Scores** (a simulated 20-trial study selected by validation NLL; greedy gain over the logger's greedy value, CTR points,
mean over the 18 worlds; paired 95% CIs over worlds):

| variant | score | − released | − P₀ = 100 | best of 20 | selection regret |
|---|---|---|---|---|---|
| A. released | +1.79 | — | −1.74 [−2.70, −0.78] | +2.52 | 0.73 |
| C. P₀ = 1,000 | +2.85 | +1.06 [+0.42, +1.70] | −0.68 [−1.03, −0.32] | +3.16 | 0.32 |
| B. P₀ = 100 (derived) | +3.52 | +1.74 [+0.78, +2.70] | — | +3.80 | 0.28 |
| **D. P₀ = 10** | **+3.65** | +1.86 [+0.84, +2.88] | **+0.12 [+0.05, +0.20]** | +3.91 | 0.26 |
| E. prior only, s_ζ at P₀ = 100 | +3.48 | +1.70 [+0.73, +2.66] | −0.04 [−0.12, +0.04] | +3.65 | 0.17 |

**The rule's choice is BLOB-Pnorm with P₀ = 10, the arm `blob_l10_nq`.** It beats the derived P₀ = 100 by 0.12 points.
The pre-registered tie-break keeps P₀ = 100 only for a margin below 0.10, so it does not apply. The two are close (CI
[0.05, 0.20]), and the choice is a weaker prior than the derived one: L = chol(Ψ̃ᵀΨ̃/10), √(P/10) = 18.8, 27.5 and 32.9
times the released L.

**What the variants do** (the NLL-selected trial of each world, means over the 18 worlds):

| variant | correction ‖ΔM‖ / ‖s+(w_a) I‖ | users on the logger's top item | median pick rank | validation NLL | population NLL under π0 (exact) |
|---|---|---|---|---|---|
| released | 0.03 | 60% | 0.2 | 0.3962 | 0.3894 |
| P₀ = 1,000 | 0.11 | 52% | 0.8 | 0.3940 | 0.3874 |
| P₀ = 100 | 0.24 | 43% | 2.4 | 0.3928 | 0.3862 |
| P₀ = 10 | 0.28 | 42% | 3.5 | 0.3926 | 0.3860 |
| prior only, P₀ = 100 | 0.21 | 44% | 2.5 | 0.3928 | 0.3862 |

- **The catalog-size prior is what held BLOB back on the tuning worlds.** Weakening the prior toward the paper's P₀
  regime raises the correction from 3% to 24–28% of the source term and moves more users off the logger's choice
  (60% → 42–43%). The click model fits better: the exact population NLL falls by 0.003, and validation AUC rises
  from 0.678 to 0.692.
- **The effect is the prior, not the optimizer geometry.** The prior-only variant (E) scores like the
  L-normalized one at the same prior (−0.04 [−0.12, +0.04]). Released L with the P₀ = 100 prior on ζ adapts as much
  within this budget, though with a larger ζ (‖ζ‖ ≈ 77 against 10).
- **The learning-rate caveat is resolved.**
  - The one-step extension (lr up to 3e-1) changed the released prior's score by +0.05 (1.74 within the previous
    range, 1.79 extended) and P₀ = 10's by −0.004.
  - The chosen variant's best half-decade is 3e-2–1e-1, an interior value. The new top half-decade, 1e-1–3e-1, is its
    worst.
  - The previous range was not what limited BLOB.
- **Edges.** In the chosen variant's trials, μ_wb's best value is its top, 0, beating −3 by 0.37 points. 1,000 epochs
  is the top value too, but within 0.03 of 300. A larger μ_wb means a larger initial correction scale, the same
  direction as the weaker prior. As fixed in §2, the space is not widened again; it is a limitation.

**Frozen for the main grid (Phase 5):**
- the arm `blob_l10_nq`, BLOB-Pnorm-NQ with P₀ = 10, L = chol(Ψ̃ᵀΨ̃/10);
- σ_κ = 0.1;
- lr log-uniform on [3e-3, 3e-1], epochs {10, 30, 100, 300, 1,000}, μ_wa {−1, 1, 3}, μ_wb {−6, −3, 0};
- 20 trials per world, no seed tag, on the 30 main worlds, once;
- pick diagnostics and saved policies.

The derived P₀ = 100 is not run on the main worlds. On the tuning worlds it is 0.12 points below P₀ = 10, with
the same mechanism.

## 4. The 25k check on the 30 main worlds (Phase 5)

**Run.** `run_blob_pnorm_main_25k`, code d59ab80 on a pinned worktree, 21:33–22:42, 3 workers. 30 worlds × 20
trials; none diverged.
- The arm is `blob_l10_nq`; L scales √(P/10) are 18.8 (ml), 27.5 (kuairand) and 32.9 (anime).
- Every comparator is reused unchanged, including default BLOB-NQ and BLOB-MNQ from the controlled study.
- Data identity: the arm trained on the same 25,000 rows as every other arm and selected on the same 20,000 rows, in
  all 30 worlds.
- Tables and figures: `artifacts/full_study/blob_prior_calibration/compare/`. Pick diagnostics:
  `…/diagnostics/`.

**Greedy value** (CTR points over the logger's greedy value; mean over worlds, 95% CI for the 24 biased worlds pooled):

| arm | biased (24) | no bias | warp | group | vector | combined |
|---|---|---|---|---|---|---|
| **BLOB-Pnorm-NQ (P₀ = 10)** | **+2.82 [+1.84, +3.79]** | −0.21 | +2.99 | +1.18 | +1.06 | +6.03 |
| BLOB-NQ (default, published prior) | +1.42 [+0.85, +1.98] | −0.17 | +1.58 | +0.70 | +0.84 | +2.56 |
| BLOB-MNQ (default) | +0.94 [+0.53, +1.35] | −0.11 | +0.71 | +0.54 | +0.75 | +1.75 |
| CausE-cap-C, ρ = 0 | +3.09 [+2.10, +4.08] | −0.55 | +4.23 | +1.22 | +1.03 | +5.89 |
| OPC (harmonic:0.1) | +2.69 [+1.84, +3.53] | −0.61 | +3.03 | +1.32 | +0.96 | +5.42 |
| DM-only (own range) | +2.15 [+1.30, +3.01] | −0.55 | +2.09 | +0.81 | +0.64 | +5.07 |

**Paired differences, BLOB-Pnorm minus each arm** (mean [95% CI]; worlds where BLOB-Pnorm is higher):

| b | biased (24) | no bias | warp | group | vector | combined |
|---|---|---|---|---|---|---|
| default BLOB-NQ | **+1.40 [+0.81, +1.98] (24/24)** | −0.03 [−0.34, +0.28] (5/6) | +1.41 [+0.78, +2.05] | +0.48 [+0.20, +0.77] | +0.22 [+0.06, +0.38] | +3.47 [+2.70, +4.25] |
| CausE-cap-C | −0.28 [−0.57, +0.01] (8/24) | +0.34 [+0.05, +0.64] (5/6) | −1.24 [−1.84, −0.64] (0/6) | −0.04 [−0.36, +0.29] | +0.02 [−0.21, +0.26] | +0.14 [−0.32, +0.60] |
| OPC | +0.13 [−0.08, +0.34] (17/24) | +0.40 [+0.20, +0.60] (6/6) | −0.04 [−0.42, +0.33] | −0.14 [−0.50, +0.22] | +0.09 [−0.07, +0.26] | +0.61 [−0.07, +1.30] (6/6) |
| DM-only (own range) | +0.66 [+0.30, +1.03] (20/24) | +0.34 [+0.06, +0.62] | +0.89 [+0.52, +1.27] | +0.37 [+0.09, +0.65] | +0.41 [+0.10, +0.73] | +0.97 [−0.78, +2.72] |

- **Against default BLOB.** BLOB-Pnorm gains 1.40 points on biased worlds, in 24 of 24 worlds, and repairs 0.28 of
  the representation loss against 0.15. The gain is largest where the source is most wrong: combined (+3.47) and warp
  (+1.41). It is smallest under vector bias (+0.22), where no learner repairs much.
- **Against the other learners.**
  - BLOB-Pnorm ties OPC (+0.13 [−0.08, +0.34], 17/24).
  - It sits just below CausE-cap (−0.28 [−0.57, +0.01], 8/24); the whole deficit is warp (−1.24, 0/6).
  - It is above DM-only by 0.66 (20/24).
  - Under combined bias it has the highest mean of all arms: +0.14 over CausE-cap (CI includes 0; higher in 2 of 6
    worlds) and +0.61 over OPC (higher in 6 of 6).
  - Under group and vector bias every model-based arm is level.
- **Stochastic value** (BLOB and CausE tempered by the DR lower bound; OPC its learned scale): BLOB-Pnorm +6.84
  [6.15, 7.52] against CausE-cap +7.13 (−0.29 [−0.56, −0.02]), OPC +6.75 (+0.08 [−0.12, +0.28]) and default BLOB
  +5.45. The ordering is the greedy one.
- **Without bias the protection holds on average.** BLOB-Pnorm loses 0.21 points against the logger, against 0.17 for
  default BLOB (−0.03 [−0.34, +0.28]) and 0.55–0.61 for CausE-cap and OPC. One world changes the most: anime seed
  100, −0.74 against −0.15; anime seed 101 moves the other way, −0.11 against −0.42.

## 5. Does weakening the prior change how BLOB adapts? (Phase 6)

The selected policy of each world, biased worlds unless stated (`table_pick_diagnostics.csv`, `table_pick_pairs.csv`):

| policy | correction ‖ΔM‖ / ‖s+(w_a) I‖ | users on the logger's top item | median pick rank | true CTR at the moved picks | true CTR at the kept picks | population NLL under π0 | click error, rarely / often logged pairs |
|---|---|---|---|---|---|---|---|
| BLOB at its prior mean (its start) | 0 | 84% | 0 | 17.9% | 21.0% | — | — |
| default BLOB-NQ | 0.025 | 64% | 0.2 | 21.6% | 22.0% | 0.4020 | 0.0126 / 0.0383 |
| **BLOB-Pnorm-NQ** | **0.216** | **48%** | 2.0 | **22.4%** | 24.2% | 0.3991 | 0.0120 / 0.0352 |
| CausE-cap-C | — | 38% | 2.8 | 21.9% | 25.9% | 0.3976 | 0.0100 / 0.0330 |
| OPC | — | 42% | 1.9 | 21.9% | 24.5% | — | — |
| likelihood oracle, BLOB's class | — | 21% | 36.6 | 23.9% | 33.1% | 0.3919 | 0.0061 / 0.0213 |

1. **Does the normalized prior increase the learned correction? Yes, about 8.5-fold.** It rises from 2.5% to 22% of
   the source term. The learned scale s+(w_b) falls from 4.0 to 0.87; the larger L carries the correction instead.
2. **Does BLOB move more users away from the logger's choice? Yes.**
   - 52% of users get a different item than the logger's top one, against 36% for default BLOB. That is between OPC
     (58%) and CausE-cap (62%).
   - The moves pay: the true CTR at its moved picks (22.4%) is the highest of the learners.
   - Against default BLOB, +1.22 of the +1.40 comes from the 28% of users it moves further from the logger.
3. **Does its click model become more accurate? Yes, partly.**
   - The population NLL falls by 0.003, against the class floor of 0.392. Validation AUC rises from 0.688 to 0.697.
   - The prediction error falls in every logging-propensity bin. It stays above CausE-cap's.
   - It is more pessimistic at its picks (−1.86 points, against −1.30).
4. **Does the target value improve together with those changes? Yes, in the same worlds and in the same direction.**
   The greedy value rises 1.40 points, mostly where the correction grew most. The four bias types rank the same by
   the correction's size (combined 0.53, warp 0.20, group 0.11, vector 0.04, against 0.01–0.05 before) and by the
   value gained (+3.47, +1.41, +0.48, +0.22).
5. **Does no-bias performance deteriorate? Not on average.**
   - Without bias BLOB-Pnorm keeps 81% of users on the logger's top item, the same share as default BLOB (81%; OPC
     and CausE-cap keep 68–69%). Its correction there is 0.7% of the source term, and its population NLL (0.4928)
     is level with default BLOB's and the lowest of the learners.
   - The weaker prior lets BLOB adapt where the logged clicks call for it, and it does not adapt where they do not.
   - The mean no-bias change is −0.03 [−0.34, +0.28], with one anime world losing 0.6 points.

**Against OPC and CausE-cap, the remaining differences are small and specific.**
- OPC − BLOB-Pnorm is −0.13 [−0.34, +0.08]. The two pick the same item for 58% of users.
  - On the 17% where BLOB-Pnorm's pick is the less-logged one, it gains 0.21 points.
  - On the 24% where OPC's is, OPC gains 0.08.
- CausE-cap − BLOB-Pnorm is +0.28 [−0.01, +0.57]. Most of it comes from the 28% of users CausE-cap moves further,
  mainly in warp worlds.

## 6. Capacity, training and selection (Phase 7)

**The ceiling is unchanged.** BLOB-Pnorm is the same score class as default BLOB (xᵀMa + κ_a). Normalizing L
changes the prior, not the family, so its value oracle is BLOB's: +6.70 on biased worlds. No oracle was rerun.

| a − b (biased, 24) | Δgain | Δceiling | Δtraining gap | Δselection regret |
|---|---|---|---|---|
| BLOB-Pnorm − default BLOB-NQ | +1.40 [+0.81, +1.98] | 0 | −1.16 [−1.57, −0.75] | −0.24 [−0.65, +0.17] |
| BLOB-Pnorm − OPC | +0.13 [−0.08, +0.34] | +0.07 [+0.05, +0.10] | −0.22 [−0.43, −0.02] | +0.17 [+0.05, +0.28] |
| BLOB-Pnorm − CausE-cap-C | −0.28 [−0.57, +0.01] | +0.07 [+0.05, +0.10] | +0.15 [−0.12, +0.42] | +0.20 [+0.07, +0.32] |

(Δgain = Δceiling − Δtraining − Δselection.)

- **Normalization closes BLOB's training gap.** The gap from the class ceiling to the best of 20 trials falls from
  4.72 to 3.56 points. That is now smaller than OPC's (3.78) and close to CausE-cap's (3.41).
  - The best trial reaches +3.14, against +1.98 before, +2.84 for OPC and +3.22 for CausE-cap.
  - Selection also improves, from 0.56 to 0.32 points of regret.
- **What separates it from the others now.**
  - Against OPC: it trains better (0.22) and selects worse (0.17).
  - Against CausE-cap: the deficit splits between training (0.15, mostly warp: 1.03 there) and selection (0.20).
- **Selection is BLOB's remaining weakness.** NLL selection loses 0.32 points against the best of its 20 trials,
  against 0.13 (CausE-cap) and 0.16 (OPC).

## 7. Decision (Phase 8)

**Outcome: B, with C's condition partly met. Calibration substantially closes the gap; BLOB-Pnorm becomes competitive
without clearly dominating.**
- **Against default BLOB:** +1.40 (24/24), through the training gap the controlled study identified.
- **Against OPC:** level, +0.13 [−0.08, +0.34].
- **Against CausE-cap:** slightly below, −0.28 [−0.57, +0.01], all of it under warp.
- **Against DM-only:** above, by 0.66.
- **C's condition is partly met.** BLOB-Pnorm is tied strongest under combined bias: it has the highest mean, +0.14
  [−0.32, +0.60] over CausE-cap. It is also tied strongest under group and vector bias, but there every model-based
  arm is level.
  - It is not the strongest anywhere with an interval that excludes the others.
  - It is clearly below CausE-cap under warp (−1.24, 0/6), so on the pooled biased worlds the ordering is CausE-cap ≥
    BLOB-Pnorm ≈ OPC > DM-only.
  - The baseline landscape changes: BLOB is no longer dominated.
  - C's action, stop and analyze before the simulator changes, is what this phase does in any case.

**Answers to the question of the phase.**
- **Is BLOB's weak adaptation intrinsic, or a consequence of a prior calibrated for P ≈ 100? Largely the latter.**
  - Under the released parameterization, every entry of the correction's prior has variance ∝ 1/P (§1). In our
    catalogs that is 35–108 times tighter than at P = 100.
  - Making the prior independent of P raises BLOB's correction about 8.5-fold and improves its click model's fit.
  - It raises BLOB's value by 1.40 points on the main worlds (P₀ = 10), and by 1.74 (P₀ = 100) and 1.86 (P₀ = 10)
    on the tuning worlds.
  - The gap to OPC goes from −1.27 to +0.13 and the gap to CausE-cap from −1.67 to −0.28.
  - The prior-only variant shows the prior is the cause, not the optimizer geometry.
- **Both answers stand.**
  - The published BLOB, transferred faithfully to our catalogs, adapts very little and trails CausE-cap, OPC and
    DM-only on biased worlds (the controlled study, unchanged).
  - With its prior normalized for catalog size, BLOB is a competitive model-based baseline: level with OPC and just
    below CausE-cap.
- **Note on the chosen anchor.** It is P₀ = 10: each entry of the correction has ten times the prior variance it
  has in the paper's P = 100 regime.
  - On the tuning worlds P₀ = 100 scored 0.12 points lower, with the same mechanism (§3).
  - The rule made the choice, and the main worlds were not used to choose.

**Recommendation for the next stage.**
- **Keep BLOB-Pnorm as a core model-based baseline** in the richer representation-mismatch study. Label it as such:
  "BLOB-Pnorm-NQ, L = chol(Ψ̃ᵀΨ̃/10)". Keep the published BLOB as the faithful reference in a smaller number of cells.
- **Report its selection separately.** NLL selection is its remaining weakness; a DR-lower-bound selection was not
  tried, and would be a different protocol.
- **Before the next simulator change, look at warp.** Warp is the one regime where CausE-cap still leads (−1.24). There
  the difference is mostly training (1.03 points) within classes of the same ceiling.
- **Carry the P-dependence of the released prior into any RecoGym work**, where P can be varied.

**Limitations.**
- **Scope.** Development seeds, 25k, one calibrated anchor on the main worlds.
- **The edges.** μ_wb's best tuning value sits at its upper edge, and 1,000 epochs is at the top of its range by a
  hair (§3); neither was widened.
- **MNQ.** Not calibrated.
- **Interpretation of the anchors.** The calibrated anchors' prior is a function of P₀ chosen on tuning worlds. A
  different reference catalog changes the strength by √(P₀/P₀′).

## 8. Tests and reproducibility

- **Tests added on this branch.**
  - Per-trial L scales in the batched layer equal separate single runs (MNQ and NQ).
  - Scaling L by c is the same correction as scaling ζ by c.
  - Paired variants share their noise stream.
  - Adding prior variants leaves the released arm's trials unchanged.
  - The calibration rule's choice and tie-break.
- **The released graph's step-by-step replay is unchanged** (`tests/test_blob_tf_reference.py`), since the default
  L scale is 1.
- **The controlled study's comparison tables regenerate byte for byte** with the extended analysis.
- Suites: see §8.1.
