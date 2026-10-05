# BLOB's prior and catalog size: derivation, calibration and the 25k check

*Development stage. Branch `blob-prior-calibration`, from `blob-controlled-integration` (1a01084). §1 (the derivation)
and §2 (the pre-registered calibration and decision rule) were committed before any calibration run. Results come
after them. The published/default BLOB rows of `docs/blob_controlled_integration.md` are unchanged and stay the
reference. The calibrated variant is a separate, explicitly named arm.*

**The question.** The controlled study found that default BLOB, transferred faithfully to our catalogs (P = 3,533–10,803
items), barely adapts from its source and trails CausE-cap, OPC and DM-only on biased worlds. Is that weak adaptation
intrinsic to BLOB here? Or is it largely a consequence of a prior and parameterization calibrated in the paper's
P ≈ 100 regime?

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
