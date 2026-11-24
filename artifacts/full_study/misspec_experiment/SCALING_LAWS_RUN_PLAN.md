# Scaling laws — precise run plan (**v1.1 LOCKED**)

**Date.** 2026-11-23 (Europe/Rome).  
**Basis.** `PROTOCOL_DRAFT_scaling_laws.md` + MATH rundown §8.  
**Stance.** No N-curve claims until Phase 0 **A1–A4** pass. **No huge-N A3 until A3-0 decides.** Reuse existing data only where licensed below.

**Scrutiny (all Approve-with-changes → folded):**
- Math/ID: [Math/scaling scrutiny](bc-ea95e897-c6c7-50fd-aef8-e9c47b944263) · [Math ID review of plan](bc-c91c351f-3d04-5bf7-a631-5bd2f492578e)
- Protocol: [Protocol review of plan](bc-db9e30d5-45ff-5cba-a115-e8d7148d9312) · [Protocol scrutiny of scaling plan](bc-f1fe8c8f-41b9-5dd6-8075-9291562561d6)

**Runtime note.** A3a huge-N on GPU2 was **stopped** after late reviews blocked value-based F2 decisions; partial `run_scale_a3a_g0` seed100 is incomplete (trials only).

**A3-0 / pin decision (2026-11-23, Europe/Rome).** Strong lik oracle (20k steps, all users) on seeds 300–302 @ γ=0: **V_lin_lik ≈ 19.0–21.0 pt** vs recoverability pin **12.4–14.1 pt** (+~7 pt). **Replace the recoverability `V_lin_lik` pin.** Legacy NLL @100k (~18.5 pt) no longer beats the strong pin → **no huge-N A3 for F2.** Artifacts: `phase0_a3_0/lik_oracles_strong.csv`, `lik_pin_decision.json`.

**A4 / K2 (2026-11-23).** `scale_a4_rstar_g2` done (seeds 100–102, N∈{25k,400k}, 60 trials). Best-of-60 vs best-of-20 on `actual_reward_greedy`: **1 fail** — nll @25k seed100 (+1.51 pt). **K2 fires lightly** → Phase 1 use **`n_trials=60`** (esp. N≤25k). Artifact: `phase0_a4/a4_k2_decision.json`.

**Phase 1 launch (2026-11-23).** Stage‑1 probe started on GPU2: `run_scaling_phase1.sh` → tags `scale_p1_rstar_g{2,0,0.5}` then `scale_p1_wellspec`. N∈{25k,100k,400k}, seeds 100–104, methods {nll,opc,hybrid_seq_f0.5}, `n_trials=60`, nested+A2 pins. A4 cells kept as HPO evidence only (not spliced into nested curve). Log: `logs/scaling_phase1.log`.

---

## 0. Inventory

| Asset | Status | Reuse |
|---|---|---|
| `nlres_grid_ml_{rstar,softand,softthresh}_g*` | **40/40** @ N=100k | **External replication only**; A5 paired TOST per **validated cell** — never nested curve points |
| `nlres_cap_*` | **40/40** | **A6 DONE**; Phase-3 C anchors |
| Recoverability CSVs | present | Independent oracle pin for `V_ref`, `Δ_proj` |
| Hybrid multi-ds (GPU3) | running | Leave alone; do not block Phase 0 |
| `run_ml_warp_size_curve_n*` | wrong world | **Do not reuse** |

**Legacy 100k greedy (pt, n=40) — external anchors:**

| cell | NLL | OPC | Hy₀.₅ | note |
|---|---:|---:|---:|---|
| R*/ST γ0 | 18.54 | 17.32 | 18.36 | F2 broken vs V_lin_lik≈13 |
| R* γ0.5 | 19.12 | 19.08 | 20.10 | OPC−NLL≈0 → crossover-ID |
| R* γ2 | 17.08 | 20.35 | 20.48 | N_star may be <25k |
| ST γ0.5 / γ2 | … | … | … | Phase 2 only |

**Blockers:** A1 code missing; F2 pin broken; pin `CUDA_VISIBLE_DEVICES=2` (GPU0–1 vLLM, GPU3 hybrid).

---

## 0b. Identification objects

\[
D_\gamma(N)=V_{\mathrm{OPC}}(N)-V_{\mathrm{NLL}}(N),\quad
N_\star(\gamma)=\inf\{N:\mathbb{E}[D_\gamma(N)]\ge 0\}
\]
Require one monotone crossing; bootstrap seed-level N-vectors; if 0 not bracketed → report \(N_\star<N_{\min}\) or \(>N_{\max}\) (no extrapolated root).

\(V_{\mathrm{ref}}\) from **independent** oracle sim before fitting. If operational \(V_{\mathrm{NLL}}^\infty>V_{\mathrm{lin,lik}}\):
\(B_{\mathrm{NLL}}=V_{\mathrm{ref}}-V_{\mathrm{NLL}}^\infty\) (same pipeline as curve). No largest-N-as-\(V^\infty\) without plateau gate. If \(V^\infty>V_{\mathrm{ref}}\) → recompute ref, do not clamp \(B\to0\).

S3: decreasing \(N_\star\) when **CI upper** on slope is <0.

---

## 1. Phase 0 gates

### P0-A1 — Nested logged data (**CODE FIRST**)

One simulation per `(seed, world)` of pool size `reg_size + N_max + val_size`.
- **Fixed** regression/q-hat partition (default 50k) — same at every N; account explicitly in A2.
- **Fixed** val partition (20k) — identical across N.
- **One shuffle** of the train slice; train@\(N_k\) = **prefix** of length \(N_k\).
- Legacy independent-N path unchanged (flag off).

```
CUDA_VISIBLE_DEVICES=2 ... --nested-train-prefixes --val-size 20000 --shared-regression-size 50000
```

**Accept.** \(N_k\subset N_{k+1}\); val+reg identical; unit test + smoke.  
**Kill.** Cannot ship without breaking confirmatory CRN → stop.

### P0-A2 — Equal data accounting

Same train/val for {nll, opc, hybrid_f0.5}; document **+50k shared reg rows** for q-hat. Split-hash dump + code SHA in results.

### P0-A3-0 — Oracle / NLL diagnostics (**BEFORE any huge-N**; GPU minutes)

Late reviews **block** deciding F2 from greedy V at huge N. Run first (`run_scaling_a3_0.py`):

1. Re-fit `affine_bilinear` likelihood oracle on seed **300**, γ=0: stock (2k/15k) vs 10× steps / all users; record `flat`, greedy, fit objective.
2. Well-spec oracle check: `bias=none` ⇒ \(|V_{\mathrm{lin,val}}-V_{\mathrm{lin,lik}}|\lesssim 0.5\).
3. (Next slice) Production `nll` @100k seed 300 + exact logger-weighted CE vs oracle; epochs/lr stress at fixed N.

**Decision.**
- If 10× oracle lifts \(V_{\mathrm{lin,lik}}\) by >0.5 pt → **oracle under-optimized**; recompute pins; huge-N A3 **not** needed for F2.
- If learned CE < oracle CE → same.
- If learned CE > oracle CE while V stays high → non-monotone path; then A3a/b at huge N, decide on **population CE**, not V; floor from disjoint seeds / N≥2× fit max — never largest-N mean of the fit curve.

### P0-A3 — NLL floor (only if A3-0 says so)

| Track | Pipeline | Decision use |
|---|---|---|
| **A3a** | `nll` (val-NLL select); fixed/locked schedule | Population CE vs oracle; V secondary |
| **A3b** | Phase-1 HPO + same select | Operational floor for curves |

Prefer seeds **300–302** (have oracles) or compute oracles for 100–102 first. Plateau: \(|ΔV|\) last doubling <0.3 pt **and** paired 90% CI includes 0.

### P0-A4 — HPO 20 vs 60

N∈{25k,400k}, R*γ=2, {nll,opc}, seeds 100–102.  
**K2:** best-of-20 worse than best-of-60 by >0.5 pt greedy on either method → raise n_trials or report confound.

### P0-A5 — Legacy overlay (demoted)

Recollect nested 100k for every scaling seed. Legacy n=40 = external replication.  
Equivalence claim ⇒ **paired TOST** on NLL, OPC, hybrid, \(D\); full CI inside ±0.5 pt; **only cells that pass** may be cited as overlays.

### P0-A6 — Capacity DONE (40/40).

**Phase 1 gate:** A1∧A2∧A3(decision recorded)∧A4 must pass — not merely A1+A3.

---

## 2. Phase 1 — N-probe (after A1–A4)

| Knob | Value |
|---|---|
| Data | ml, linear |
| Methods | nll, opc, hybrid_seq_f0.5 |
| N | **6.25k, 12.5k**, 25k, 50k, 100k, 200k, 400k, 800k nested |
| Val / reg | 20k fixed / 50k fixed |
| Seeds | 100–104 (bracketing pilot) |
| Worlds | R*γ=0 · **R*γ=0.5** · R*γ=2 · well-spec `bias=none,γ=0` |
| HPO / select | **60** (A4 K2) / val-DR (=A3b) |
| Device | `CUDA_VISIBLE_DEVICES=2` + abort if GPU2 occupied |

**Kill thresholds (operational):**
| ID | Rule |
|---|---|
| K1 flat | Paired mean \(\|V(800\mathrm{k})-V(25\mathrm{k})\|<0.5\) pt on both nll and opc in a misspec world |
| K2 HPO | From A4 (>0.5 pt) |
| K3 selection | Select-on-NLL vs select-on-DR disagrees by >1.0 pt greedy at N≥400k (one seed ablation) |
| K4 well-spec | At N≥400k, mean \(D\) not within ±0.5 pt of 0 **and** OPC persistently ahead → fail (not “NLL≥OPC ∀N”) |
| K5 floor | A3b plateau gate failed / no seed-extension when triggered |
| K6 bracket | No world brackets \(D=0\) → report bounds only; no \(N_\star\) point estimate |

---

## 3. Phase 2+ — not yet

As protocol (seeds 300–309, γ×{R*,ST}, leave-largest-N-out; N×C; multi-ds/ESS).

---

## 4. Resources

| GPU | Role |
|---|---|
| 0–1 | vLLM — avoid |
| **2** | Scaling only (`CUDA_VISIBLE_DEVICES=2`); preflight abort if busy |
| 3 | Hybrid — do not steal |

---

## 5. Execution order (now)

1. **A1** ✅ nested flag + tests.  
2. **A3-0** ✅ strong lik pin; no huge-N A3 for F2.  
3. **A2** ✅ pins documented (`phase0_a2_data_accounting.md`).  
4. **A4** ✅ K2 → `n_trials=60`.  
5. **Phase 1 fill:** R\* γ∈{0,0.5,2} densify **DONE** (full 6.25k–400k ladder). Wellspec fill **IN FLIGHT** (`scale_p1_wellspec`, started 21:35 Asia/Jerusalem). 800k deferred.  
6. A5 demoted: never splice legacy / A4 into nested curves.  
7. **Rethink (2026-11-24):** SoftAND/ST N-ladders skipped. See `NOTE_ess_opc_usability.md` (+ phone PDF from `.tex`). N★: γ0∈(25,50]k; γ0.5∈(100,200]k; γ2 none ≤400k.

---

## 6. Non-goals

No SoftAND mid-γ for scaling; no free-fit asymptotes; no independent-N “laws”; no legacy-as-curve-points; no pausing hybrid; no Phase 1 before A1–A4.

---

## 7. Scrutiny ledger (closed)

| # | Math | Protocol | Locked resolution |
|---|---|---|---|
| 1 A1 | Agree | Change (fixed reg+val; train prefixes only) | One pool; fixed reg+val; train prefixes |
| 2 A3 | Block→split | Change (pilot + extension) | A3a/A3b; plateau CI; seed-extension |
| 3 A5 | Change | Change (paired; cell-restricted) | TOST; validated cells only |
| 4 P1 worlds | Change (+γ0.5, small N) | Agree defer ST | R* {0,0.5,2}+well-spec; +6.25/12.5k |
| 5 well-spec | Agree | Agree | bias=none,γ=0 |
| 6 GPU | Agree | Change (pin CVD=2) | Pin + preflight |
| 7 kills | Change | Block→need thresholds | K1–K6 with numeric rules |
