# When Click-Model Misspecification Favors Doubly Robust Offline Learning

**Working draft** · Offline policy learning / recommender bandits · Target: AISTATS / ICML / RecSys  
**Protocol:** misspec-DR-v1 + latent-alt-v1.0 + hybrid-seq · Artifacts under `artifacts/full_study/misspec_experiment/`

> **One-line claim.** Under a controlled dial that pushes the true click model off the learner’s bilinear score class, matched-capacity doubly robust (DR) policy optimization widens its advantage over matched-capacity logged NLL—measured as a difference-in-differences (DiD)—and a simple NLL→DR training schedule interpolates the well-specified and misspecified regimes.

---

## Abstract

Offline policy learning for recommender bandits often pits structural likelihood fitting (logged NLL) against doubly robust (DR) value optimization. Empirical comparisons are hard to interpret when capacity, hyperparameters, and the degree of model misspecification are entangled. We introduce a **controlled misspecification dial** \(\gamma\) on a synthetic click world built from MovieLens BPR factors: the true logit is an affine-bilinear score plus a unit-SD residual \(\phi\), with logging policy and HPO held fixed by common random numbers. The primary estimand is

\[
D(\gamma)=V(\pi^{\mathrm{DR}}_\gamma)-V(\pi^{\mathrm{NLL}}_\gamma),\qquad
\mathrm{DiD}=D(\gamma^\star)-D(0).
\]

On the pre-registered even residual family \(R^\star\), confirmatory DiD at \(\gamma^\star=0.5\) is **\(+1.53\) CTR points** (95% CI \([+0.97,+2.09]\), \(10/10\) seeds). Latent soft-gate residuals (SoftAND\*/SoftThresh) open large *oracle* projection gaps at moderate \(\gamma\) but yield **null** linear DiD until \(\gamma\to\infty\) (\(\gamma=100\)), where DR wins by \(+3.8\) to \(+7.2\) pt. A sequential hybrid that trains NLL for fraction \(f=0.5\) of epochs then fine-tunes with raw DR retains near-NLL value at \(\gamma=0\) and beats both pure NLL and pure DR under mid and extreme misspec (e.g. SoftAND \(\gamma=100\): hybrid \(21.8\) vs NLL \(18.0\) vs DR \(20.6\)). We argue for reporting DiD under locked misspec protocols rather than raw horse races, and for hybrid schedules when the degree of class misspecification is unknown.

---

## 1. Introduction

**Problem.** Industrial recommender logging still optimizes logged logistic likelihood. Offline policy learning (OPL) theory recommends IPS/DR when propensities are known, but practitioners lack a clean answer to: *when* does DR beat a well-tuned NLL baseline of the *same* capacity?

**Gap.** Prior sims often change bias, model class, and objective together. Wins for DR can be capacity or HPO artifacts. Wins for NLL can be well-specification in disguise.

**Approach.** Fix representation bias and logger temperature. Dial only a residual \(\phi\) in the *true* click model. Compare linear DR vs linear NLL under shared Optuna draws (CRN). Primary contrast is DiD of the value gap, not a single-γ horse race.

**Contributions.**

1. A locked misspecification protocol (oracle gates, foils, seed roles, kill switches) for recommender bandit sims.
2. Positive confirmatory DiD for the even residual \(R^\star\) at moderate \(\gamma\); null DiD for SoftAND\*/SoftThresh at the same \(\gamma\), with recovery at \(\gamma\approx\infty\).
3. Evidence that **NLL→DR sequential fine-tuning** interpolates regimes: near-NLL when well-specified, near-or-above DR when \(\phi\) dominates.
4. Negative/foil controls (NC1, PL) and selection hybrids (`nll_dr`) that separate train objective from selection.

**What we do *not* claim.** DR does not “fix confounding” here (π₀ is simulator-true). SoftAND\* does not escape arbitrary bilinear/Blob capacity. Hybrid is neither MLE nor doubly robust—only an empirical interpolant.

---

## 2. Related work

- **DR / OPL:** Dudík et al.; SNIPS; continuous-action and contextual bandit off-policy estimators. We use raw DR with cross-fit \(\hat q\) as the Phase-1 identifying loss.
- **Logged bandits & recommenders:** CausE, BLOB / score-class methods, BPR embeddings as simulators. We treat CausE/BLOB as Phase-2 references, not the identifying NLL cell.
- **Misspecification studies:** Often unstructured noise or reward-model error. We target **class misspecification** of the click logit relative to affine-bilinear learners.
- **Multi-objective OPL:** KL-to-logger and CRM variance penalties exist in our codebase; they regularize toward π₀, not toward likelihood geometry. Our hybrid is NLL→DR, distinct from KL.

---

## 3. Setup and estimand

### 3.1 World

Clean user/item factors from BPR on MovieLens-1M. Optional combined-high representation bias produces logging features `our_*`. Truth uses clean factors. Logger softmax temperature is **frozen at \(\gamma=0\)** within each seed; reference CTR is held at 5% by resolving an offset \(b'(\gamma)\).

### 3.2 Misspecification dial

\[
\ell^*_\gamma(u,a)=\mathrm{scale}\langle x_u,y_a\rangle+b'(\gamma)+\gamma\,\alpha\,\phi(u,a),
\qquad \mathrm{sd}(\phi)=1.
\]

Thus \(\gamma\) is the residual-to-linear SD ratio. \(\gamma=0\) is bilinear truth; \(\gamma=100\) is effectively \(\phi\)-dominated ranking (empirically \(\mathrm{sd}(\mathrm{residual})/\mathrm{sd}(\mathrm{linear})\approx 90\text{–}100\)).

### 3.3 Residual families

| Family | Construction | Role |
|---|---|---|
| \(R^\star\) | Second random projection index; even map \(\psi(t)=\tanh(s(t^2-1))\); double-center; unit-SD | **Primary** positive result |
| SoftAND\* | Softplus gates on projected coords (\(c=1,s=0.25\)); product; ⊥ to clean affine-bilinear span; unit-SD | Latent alternative |
| SoftThresh | Softplus threshold on standardized \(z_2\); catalog ⊥ | Latent alternative |
| PL | Pure bilinear \(z_2\) (no ⊥) | Foil: second linear channel |
| NC1 | Feature-free noise, energy-matched (`nc1_amp=0.5`) | Negative control |

### 3.4 Learners (Phase 1)

Matched capacity: `GlobalLinearCorrection` on both sides + learnable logit scale.  
Arms: **OPC/DR** (raw DR, direct gradient, DR-LB select), **NLL/NLL**, kill arms **NLL/DR** and **DM/DR**.  
\(n=10^5\) train, val \(2\cdot10^4\), 20 Optuna trials, shared random-search configs across \(\gamma\).

### 3.5 Estimand and go rule

Exact greedy CTR \(V_{\mathrm{gr}}\). SEOI \(=0.5\) pt. Go: DiD point \(\ge 0.5\), CI LB \(>0\), same sign \(\ge 7/10\) seeds; NC1 \(|\mathrm{DiD}|\le 0.3\) (R\* protocol).

---

## 4. Protocol discipline

Phase-0a oracle gates (flip, \(\Delta\Delta_{\mathrm{proj}}\), headroom, logger drift, NC1, foil separation) lock \((\text{family},\gamma^\star,\text{bias})\) *before* confirmatory training. SoftAND\* won lexicographically over SoftThresh on combined-high at \(\gamma^\star=0.5\) despite SoftThresh’s larger \(\Delta\Delta_{\mathrm{proj}}\)—by design, not max-gap cherry-picking. Confirmatory seeds are disjoint from calibration (R\*: 300–309; SoftAND: 310–319).

---

## 5. Main results

### 5.1 \(R^\star\): moderate misspec favors DR (primary)

**Table 1.** Mean greedy CTR (%) and \(D=V(\mathrm{DR})-V(\mathrm{NLL})\) on ml, combined-high, seeds 300–309.

| \(\gamma\) | NLL | DR (OPC) | \(D\) (pt) |
|---:|---:|---:|---:|
| 0 | 18.06 | 16.99 | −1.07 |
| 0.25 | 17.96 | 17.26 | −0.70 |
| 0.5 | 17.96 | **18.42** | **+0.46** |

**DiD** \(=D(0.5)-D(0)=\) **\(+1.53\)** pt · 95% CI \([+0.97,+2.09]\) · **10/10** seeds positive · **GO**.

NC1 control DiD \(=+0.20\) (within \(\pm 0.3\)). Kill switches: win does not collapse to DM; does not vanish under NLL→DR selection (`nll_dr`).

BLOB-class lean DiD (5 seeds): \(+1.31\) pt, CI crosses the 7/10 rule (underpowered) but same sign pattern.

### 5.2 SoftAND\* / SoftThresh: oracle ≠ learner at moderate \(\gamma\)

**Phase-0a (oracle).** SoftAND\* @ \(\gamma=0.5\): mean \(\Delta\Delta_{\mathrm{proj}}=4.35\) pt, passes G0–G6. SoftThresh \(\tau=1.5\): \(6.95\) pt (larger gap, lower lex priority).

**Phase-1 DiD @ \(\gamma\in\{0.25,0.5,0.75\}\) (seeds 310–319).** All SoftAND/SoftThresh DiDs are **null** (SoftAND @0.5: \(+0.09\), CI \([-0.11,+0.29]\)). Absolute CTRs rise for *both* NLL and DR under SoftThresh; the gap does not flip.

**Interpretation.** Projection-gap oracles overstate transferable misspec when a large linear channel remains: NLL’s bilinear fit still tracks value.

### 5.3 \(\gamma=100\): \(\phi\)-dominated regime

**Table 2.** DiD at \(\gamma=100\) vs \(\gamma=0\).

| Residual | DiD (pt) | 95% CI | Signs |
|---|---:|---|---|
| SoftAND\* | +3.76 | [+2.07, +5.46] | 10/10 |
| SoftThresh \(\tau=1\) | +4.64 | [+2.84, +6.45] | 10/10 |
| SoftThresh \(\tau=1.5\) | +6.09 | [+4.45, +7.73] | 10/10 |
| SoftThresh \(\tau=2\) | +7.17 | [+5.24, +9.10] | 10/10 |
| \(R^\star\) | +6.10 | [+4.58, +7.62] | 10/10 |

At \(\gamma=100\), DR overtakes NLL for every family. SoftThresh’s larger oracle gaps finally appear as larger DiDs once the linear channel is drowned out.

### 5.4 Hybrid: NLL → DR (\(f^\star=0.5\))

Train logged NLL for fraction \(f\) of epochs (click bias on), freeze bias, continue with raw DR; select by DR-LB. Pilot (seeds 100–102) locks \(f^\star=0.5\).

**Table 3.** Confirmatory mean greedy CTR (%). H = `hybrid_seq_f0.5`.

*SoftAND\** (310–319)

| \(\gamma\) | H | NLL | DR | nll_dr | \(V(H)-V(\mathrm{NLL})\) | Gates |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 18.23 | 18.49 | 17.34 | 16.99 | −0.26 | G_well ✓ |
| 0.5 | 18.97 | 18.73 | 17.67 | 17.13 | +0.24 | misspec ✗ (CI) |
| 100 | **21.79** | 18.01 | 20.63 | 18.01 | **+3.78** | misspec ✓ · >DR |

*R\** (300–309)

| \(\gamma\) | H | NLL | DR | nll_dr | \(V(H)-V(\mathrm{NLL})\) | Gates |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 17.91 | 18.06 | 16.99 | 14.10 | −0.15 | G_well ✓ |
| 0.5 | **19.50** | 17.96 | 18.42 | 14.57 | **+1.54** | misspec ✓ · >DR |

Hybrid beats selection-only `nll_dr` everywhere meaningful (G_kill).

**Table 4 (sensitivity).** All three \(f\) on confirmatory seeds — mean greedy CTR (%).

*SoftAND\**

| \(\gamma\) | \(f{=}0.25\) | \(f{=}0.5\) | \(f{=}0.75\) | NLL | DR |
|---:|---:|---:|---:|---:|---:|
| 0 | 18.13 (−0.36) | **18.23** (−0.26) | 18.22 (−0.27) | 18.49 | 17.34 |
| 0.5 | 18.82 (+0.10) | **18.97** (+0.24) | 18.95 (+0.22) | 18.73 | 17.67 |
| 100 | 21.74 (+3.74) | **21.79** (+3.78) | 21.58 (+3.57) | 18.01 | 20.63 |

*R\**

| \(\gamma\) | \(f{=}0.25\) | \(f{=}0.5\) | \(f{=}0.75\) | NLL | DR |
|---:|---:|---:|---:|---:|---:|
| 0 | 17.38 (−0.68)⚠ | **17.91** (−0.15) | 17.95 (−0.11) | 18.06 | 16.99 |
| 0.5 | 19.10 (+1.14) | 19.50 (+1.54)✓ | **19.56** (+1.60)✓ | 17.96 | 18.42 |

(Parentheses = vs NLL in CTR pt. ⚠ fails G_well; ✓ passes G_misspec.)

**Reading.** \(f{=}0.5\) and \(f{=}0.75\) are nearly tied; \(f{=}0.25\) is too DR-heavy (fails G_well on R\*@0). Lex lock on SoftAND@100 pilot correctly preferred \(f{=}0.5\); confirmatory would also accept \(f{=}0.75\).

### 5.5 Phase-2 references: CausE / BLOB (standalone; already run)

Identifying contrast remains Phase-1 linear. Phase-2 asks whether the R\* DiD pattern survives against prior-work recipes under matched mlp capacity. Full tables: `paper/PHASE2_CAUSE_BLOB_ADDENDUM.md`.

**Table 5.** OPC-mlp64 vs CausE-cap-C-mlp64 (seeds 300–309). \(D=V(\mathrm{OPC})-V(\mathrm{CausE})\).

| \(\gamma\) | \(D\) (pt) |
|---:|---:|
| 0 | −1.46 |
| 0.25 | −0.87 |
| 0.5 | **+0.87** |

**DiD = +2.32** pt · 95% CI \([+1.11,+3.54]\) · **9/10** signs.

Absolute at \(\gamma^\star\): OPC-mlp **19.55%**, CausE-C-mlp 18.68%, CausE-C-linear 19.04%, **BLOB-l10-nq 20.09%**. Refs DiDs (OPC-mlp − ref): vs CausE-linear **+1.86** (10/10); vs BLOB **+1.20** (8/10) — OPC *closes gap* but still **trails BLOB in level**. Do not claim “OPC beats BLOB.”

**Two BLOBs.** `BLOB-l10-nq` is the published ELBO prior-work arm (`methods=blob`). The lean **blobcls** DiD (§5.1, +1.31 / 4/5 seeds) uses `policy_transform=blob` (ScoreClass \(x^\top Ma+\kappa\)) under matched OPC vs NLL — same phenomenon, different method. CausE/BLOB *train* hybrids are not in `hybrid_seq` today; see `PROTOCOL_DRAFT_phase2b.md`.

---

## 6. Discussion

**Mechanism.** DiD tracks whether misspecification creates a *value–likelihood* projection gap that linear NLL cannot close. \(R^\star\)’s even structure does so at moderate \(\gamma\). SoftAND\*/SoftThresh need near-infinite \(\gamma\) before the same happens—oracle \(\Delta\Delta_{\mathrm{proj}}\) alone is not sufficient for Phase-1 transfer.

**Hybrid.** When misspecification strength is unknown, scheduling likelihood then value avoids committing to either pure objective. It is a practical compromise, not a new estimator with DR guarantees.

**Phase-2.** Matched mlp OPC vs CausE-cap reverses with \(\gamma\) (DiD +2.32). Published BLOB remains a strong absolute baseline; relative DiD ≠ level win. Score-class (blobcls) DiD should not be read as a test of the BLOB ELBO.

**Limits.** Simulator-true π₀; one dataset (ml) in confirmatory tables; linear identifying contrast only; SoftAND\* still low-rank after residualization; \(\gamma=100\) is a stress test, not a claim about production γ.

**Practice.** (i) Prefer DiD / locked protocols when claiming “DR helps under misspec.” (ii) Do not trust oracle projection gaps without learner DiD. (iii) Consider NLL→DR schedules when both well-spec and misspec regimes matter. (iv) Name CausE/BLOB refs carefully; separate score-class from ELBO.

---

## 7. Conclusion

Controlled class misspecification can reverse the NLL–DR ranking under matched capacity. The even residual \(R^\star\) yields a clean confirmatory DiD at moderate \(\gamma\); soft latent residuals mostly do not until \(\phi\) dominates. Sequential NLL→DR hybridization captures both ends of the dial. We release protocols, seeds, and analysis scripts for replication.

---

## Reproducibility

| Item | Location |
|---|---|
| Locked protocols | `PROTOCOL_LOCKED.md`, `PROTOCOL_APPROVAL_latent_alt.md`, `PROTOCOL_DRAFT_hybrid.md` |
| Phase-1 R\* | `phase1_did/`, seeds 300–309 |
| SoftAND / SoftThresh | `phase1_softand_did/`, `phase1_st*_did/`, `*_g100_did/` |
| Hybrid | `hybrid_seq_h1/`, `run_hybrid_seq.sh` |
| Phase-2 CausE/BLOB | `phase2/`, `blobcls_did/`, `PHASE2_CAUSE_BLOB_ADDENDUM.md` |
| Phase-2b draft | `PROTOCOL_DRAFT_phase2b.md` |
| Numbers cache | `paper_results_cache.json` |

---

## Appendix stubs (to expand)

- A. Residual definitions (R\*, SoftAND\*, SoftThresh, ⊥ projector)  
- B. Full Phase-0a gate tables  
- C. Per-seed DiD forests  
- D. Kill-switch and ESS diagnostics  
- E. Hybrid \(f\in\{0.25,0.5,0.75\}\) full confirmatory — see Table 4; reports in `hybrid_seq_h1/*_allf_report.txt`  
- F. Phase-2 mlp / CausE / BLOB — see §5.5 and `PHASE2_CAUSE_BLOB_ADDENDUM.md`

---

## Title candidates (pick one)

1. *When Misspecification Favors Doubly Robust Offline Learning: A Difference-in-Differences Study*  
2. *Logged NLL vs DR under Controlled Click-Model Misspecification*  
3. *From Likelihood to Value: Hybrid Offline Learning under Class Misspecification*
