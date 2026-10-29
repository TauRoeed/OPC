# DRAFT — Phase-2b: CausE / BLOB hybrid & completeness

**Status:** DRAFT for approval. Do **not** launch confirmatory GPU jobs until §8 is filled.  
**Inherits:** CRN, residual world (R\*, γ∈{0,0.5}), seeds, SEOI, kill language from `PROTOCOL_LOCKED.md`.  
**Does not replace** Phase-1 identifying DiD \(D=V(\mathrm{OPC})-V(\mathrm{NLL})\). Phase-2 / 2b are **reference** contrasts.

---

## 0. Motivation

User ask: *what about testing CausE or BLOB additionally, both for a hybrid and a standalone?*

**Standalone is already done** (Phase-2 + blobcls lean DiD). What remains is (i) publishing those numbers cleanly, (ii) clarifying two BLOB meanings, (iii) defining *hybrid* without overclaiming, (iv) cheap follow-ups only.

---

## 1. What already exists (do not re-run)

### 1.1 Phase-2 standalone — OPC-mlp64 vs CausE/BLOB (seeds 300–309)

Tags: `nlres_p2_rstar_g{0,0.25,0.5}_{mlp64,refs}` · analysis: `phase2/` · wall ~3 h × 4 GPUs (~12 GPU-h).

Primary locked contrast \(D=V(\mathrm{OPC\textrm{-}mlp64})-V(\mathrm{CausE\textrm{-}cap\textrm{-}C\textrm{-}mlp64})\):

| γ | Mean D (pt) |
|---|---:|
| 0 | −1.46 |
| 0.25 | −0.87 |
| 0.5 | **+0.87** |

**DiD = +2.32** pt · 95% CI [+1.11, +3.54] · **9/10** signs.

Absolute greedy CTR % (selected):

| γ | OPC-mlp | CausE-C-mlp | CausE-C-lin | BLOB-l10-nq |
|---|---:|---:|---:|---:|
| 0 | 17.54 | 19.00 | 18.90 | 19.28 |
| 0.5 | 19.55 | 18.68 | 19.04 | **20.09** |

Refs DiDs (OPC-mlp − ref), same seeds:

| Contrast | DiD (pt) | 95% CI | Signs | Level at γ\* |
|---|---:|---|---|---|
| vs CausE-C-linear | +1.86 | [+1.01, +2.71] | 10/10 | OPC 19.55 > CausE 19.04 |
| vs CausE-T-mlp | +2.64 | [+1.64, +3.63] | 10/10 | OPC overtakes |
| vs BLOB-l10-nq | +1.20 | [+0.54, +1.85] | 8/10 | **OPC still trails** (19.55 vs 20.09) |

**Reading.** Matched-capacity OPC closes / reverses the gap vs CausE under R\* misspec. Published BLOB’s *level* remains higher; DiD only says OPC gains *relatively* more with γ. Do **not** claim “OPC beats BLOB.”

Selection note (from `REPORT.md`): best-of-20 Phase-2 DiD still +1.07 (9/10); level flip at γ\* is partly NLL-selection on CausE (~1.3 pt left on table vs OPC’s ~0.2).

### 1.2 Blobcls lean DiD — score-class capacity, not published BLOB

Tags: `nlres_blobcls_rstar_g{0,0.5}` · `policy_transform=blob` · methods opc vs nll · seeds 300–304 · wall ~5 min.

| γ | OPC (blob cls) | NLL (blob cls) | D |
|---|---:|---:|---:|
| 0 | 16.14% | 17.11% | −0.97 |
| 0.5 | 17.40% | 17.06% | +0.34 |

**DiD = +1.31** · CI [+0.12, +2.49] · **4/5** signs (underpowered vs 7/10 rule; same sign pattern as Phase-1 linear +1.53).

---

## 2. Two BLOB meanings (must stay distinct in prose)

| Name | Code path | Objective | What it tests |
|---|---|---|---|
| **BLOB-ELBO (published)** | `methods=blob` → `blob_trials` / `BlobBanditBatch.neg_elbo` | \(E_q[\mathrm{NLL}]+\mathrm{KL}(q\|\mathrm{prior})\) on source-anchored \(M,\kappa\); select val NLL | Prior-work baseline (Phase-2 refs) |
| **blob score-class** | `policy_transform=blob` on CFModel (`x^\top M a+\kappa_a`) | OPC raw-DR or logged NLL (same Optuna as Phase-1) | Matched-capacity DR vs NLL *inside* BLOB’s ranking class |

Calling both “BLOB” in a table without a qualifier is a review hazard. Prefer `BLOB-l10-nq` vs `blobcls` / `ScoreClass-blob`.

---

## 3. What “hybrid” can mean (three tiers)

| ID | Definition | Implementable now? | Claim allowed |
|---|---|---|---|
| **H-A Same-capacity NLL→DR** | `hybrid_seq_f*` on CFModel with `policy_transform∈{linear,blob,linear+mlp}` | **Yes** (trainer_trials only) | Empirical schedule on that class — same as linear hybrid |
| **H-B Native train + DR select** | Keep CausE/BLOB native loss; re-pick trial by val DR-LB (selection hybrid) | **Partial:** BLOB trials already log `val_dr_greedy(_low)`; CausE trials log oracle `*_value_greedy` but not full per-trial val DR — need cheap re-score or retrain with temper-all | “Selection swap,” **not** a joint estimator |
| **H-C True joint** | Mix CausE NLL / BLOB ELBO with DR in one train loop | **No** without new trainers; scale calibration + double-counting risk | Forbidden until separate protocol + reviews |

`hybrid_nll_frac` does **not** apply to `cause_trials` / `blob_trials` today. Do not pretend H-C exists.

---

## 4. Proposed Phase-2b arms (responsible)

### Tier 0 — analysis only (0 GPU; do first)

1. Publish Phase-2 table + blobcls DiD in paper (§ already drafted).  
2. BLOB **selection-swap**: for each seed/γ, pick trial by `val_dr_greedy_low` vs by `val_nll`; report ΔV. Uses existing `*_trials.csv`.  
3. CausE **oracle selection ceiling** only (best `*_value_greedy`): upper bound on selection harm — label as oracle, not DR-select.

### Tier 1 — cheap confirmatory / sensitivity (recommended if hybrid asked)

| Arm | Spec | Seeds | Est. wall | Why |
|---|---|---|---|---|
| **B1** blobcls hybrid | `methods=hybrid_seq_f0.5 opc nll` · `policy_transform=blob` · γ∈{0,0.5} · R\* | 300–304 (lean) or 300–309 | ~10–25 min (4 GPU) | Same H1 schedule as linear hybrid, inside score-class |
| **B2** blobcls power | extend blobcls opc/nll to seeds 305–309 | 5 more × 2 γ | ~5 min | Fix 4/5 → 10-seed signs |
| **B3** (optional) mlp hybrid | `hybrid_seq_f0.5` · `linear+mlp` · hidden 64 · γ∈{0,0.5} | 300–304 | ~15–20 min | Parallel to Phase-2 capacity; **not** CausE hybrid |

Primary descriptive endpoints for B1/B3: \(V(\mathrm{H})\), \(\Delta_{\mathrm{H|NLL}}\), \(\Delta_{\mathrm{H|OPC}}\); G_well at γ=0 as in hybrid protocol.

### Tier 2 — deferred / do not start

- CausE native + DR fine-tune (needs new loop).  
- BLOB ELBO + DR mixture / sequential ELBO→DR.  
- Claiming H-B “proves CausE/BLOB benefit from hybrid.”  
- Anime CausE (Phase-0b still collapsed).  
- Replacing Phase-1 NLL cell with CausE/BLOB.  
- Re-running full Phase-2 standalone (already complete).

---

## 5. GPU cost estimate

| Work | Approx GPU-h (4×GPU wall × 4) |
|---|---:|
| Phase-2 standalone (done) | ~12 |
| blobcls DiD lean (done) | ~0.3 |
| Tier 0 analysis | 0 |
| B1 lean hybrid blobcls | ~0.5–1 |
| B2 +5 seeds blobcls | ~0.3 |
| B3 lean mlp hybrid | ~1–1.5 |
| Full CausE/BLOB H-C (not proposed) | tens + eng cost |

**Recommendation:** Tier 0 now; B1+B2 only if hybrid-on-blobcls is wanted for the paper; skip B3 unless mlp story needs hybrid.

---

## 6. Estimands & claims (locked language)

**Allowed**

- “Under R\*, matched mlp OPC vs CausE-cap DiD = +2.32 (9/10).”  
- “Relative to frozen BLOB-l10-nq, OPC-mlp DiD = +1.20 but still trails in level at γ\*.”  
- “Inside ScoreClass-blob, DR vs NLL DiD = +1.31 (5 seeds, underpowered).”  
- If B1 run: “NLL→DR schedule on ScoreClass-blob …” (same caveats as linear hybrid).

**Forbidden**

- “OPC beats published BLOB.”  
- “CausE/BLOB hybrid” without specifying H-A/H-B/H-C.  
- Treating blobcls DiD as a test of the BLOB ELBO method.  
- Using Phase-2 as the identifying contrast instead of Phase-1 linear.

---

## 7. Kill / go for Tier 1

Mirror hybrid protocol lightly:

- **G_well (B1):** mean \(V(\mathrm{H})-V(\mathrm{NLL})\) at γ=0 ∈ [−0.5, +0.3] pt.  
- **G_pattern:** at γ=0.5, \(V(\mathrm{H})\geq \min(V(\mathrm{NLL}),V(\mathrm{OPC}))-0.3\).  
- **Abort B2 narrative** if extended signs &lt; 7/10 *and* DiD CI includes 0.

---

## 8. Lock checklist (fill before launch)

```text
approved_by:
date:
tier0_done: yes/no
run_B1: yes/no
run_B2: yes/no
run_B3: yes/no
seeds_B1:
f_star: 0.5 (inherit hybrid lock; do not retune on confirmatory)
git_sha:
```

---

## 9. Scripts

| Script | Role |
|---|---|
| `run_phase2.sh` | Standalone Phase-2 (done) |
| `run_blobcls_did.sh` | Score-class DiD (done) |
| `run_phase2b_blobcls_hybrid.sh` | **Prepared** Tier-1 B1 (+ optional B2); do not execute until §8 |
| `analyze_phase2.py` | Standalone tables |
| `analyze_phase2_selection_swap.py` | Tier-0 BLOB DR-select (to add when running Tier 0) |
