# Misspec-DR-v1 results (ml, confirmatory seeds 300–309)

Locked: R\*=rstar (s=1,m=4,psi=tanh), γ\*=0.5, feature cell=combined-high, n=100k, 20 trials, raw DR primary.

## T2 / T4 — primary go

| Check | Result |
|---|---|
| Phase-1 DiD D(γ\*)−D(0), D=OPC−NLL | **+1.53 pt**, CI [+0.97, +2.09], **10/10** |
| point ≥ 0.5 / CI LB > 0 / signs ≥7/10 | pass / pass / pass |
| NC1 DiD \|·\| ≤ 0.3 | **+0.20 pt**, CI [−0.08, +0.49] → pass |
| Kill: DR≈DM | no (OPC−DM DiD ≈ +1.17) |
| Kill: win vanishes under NLL→DR | no (OPC−NLL/DR ~ +3 pt) |

Phase-1 mean D: γ0 **−1.07**, γ0.25 **−0.70**, γ\* **+0.46**.

## T7 — Phase-2 mlp matched

D = OPC-mlp64 − CausE-cap-C-mlp64 (same `linear+mlp` / hidden 64; manifests confirmed).

| γ | Mean D |
|---|---|
| 0 | −1.46 pt |
| 0.25 | −0.87 pt |
| 0.5 | **+0.87 pt** |

**DiD = +2.32 pt**, CI [+1.11, +3.54], **9/10** signs.

Absolute at γ\* (tier-separated): OPC-mlp 19.55%, CausE-cap-C-mlp 18.68%, CausE-cap-C-linear 19.04%, BLOB-l10-nq 20.09%.

Refs DiD (OPC-mlp − ref): vs CausE-linear **+1.86** [+1.01, +2.71] 10/10; vs BLOB **+1.20** [+0.54, +1.85] 8/10 (OPC still trails BLOB in level). See `PROTOCOL_DRAFT_phase2b.md` / `paper/PHASE2_CAUSE_BLOB_ADDENDUM.md` for hybrid feasibility (standalone done; true CausE/BLOB joint hybrid not implemented).

### Selection (best-of-20)

| Estimand | Selected | Best-of-20 |
|---|---|---|
| Phase-2 DiD | +2.32 | **+1.07** (still ≥ SEOI, 9/10) |
| Phase-2 D(γ\*) | +0.87 | **−0.23** |

CausE NLL selection leaves ~1.3 pt on the table at γ\* vs OPC’s ~0.2 pt — the **level** flip at γ\* is partly selection; the **DiD** is not. Phase-1 sign(OPC−NLL) agrees selected vs best-of-20 in 10/10 (γ0) and 9/10 (γ\*).

## T3 — arm means (Phase 1, greedy CTR %)

| γ | OPC | NLL | DM | NLL/DR |
|---|---|---|---|---|
| 0 | 16.99 | 18.06 | 17.34 | 14.10 |
| 0.25 | 17.26 | 17.96 | 17.56 | 13.96 |
| 0.5 | 18.42 | 17.96 | 17.61 | 14.57 |

## Mechanism note

OPC absolute (greedy %): linear 16.99→18.42 (+1.43); mlp 17.54→19.55 (+2.01). Residual helps both; mlp gains more. Pattern matches Phase-0a Δ_proj rising with γ on combined-high.

## What we do **not** claim

OPC-mlp vs CausE-linear as primary (forbidden); BLOB beaten in level; anime; realism of residual.

## Sensitivities / Phase-3

Harmonic:0.1 DiD (seeds 300–304) **+1.84** [+0.42, +3.25] 5/5. IPS (q̂=0) loses.  
Phase-3 `logger_greedy_share=0.95` DiD **+2.45** [+1.86, +3.05] 10/10.

## Lean follow-up — BLOB-class × {DR, NLL}

Same residual world; `policy_transform=blob` = free M + κ on frozen logger vectors
(ScoreClass blob; not the published ELBO). Seeds 300–304, γ ∈ {0, 0.5}.

| γ | OPC (blob cls) | NLL (blob cls) | D |
|---|---|---|---|
| 0 | 16.14% | 17.11% | −0.97 |
| 0.5 | 17.40% | 17.06% | +0.34 |

**DiD = +1.31 pt**, CI [+0.12, +2.49], **4/5** signs.  
Pattern matches Phase-1 linear DiD; published BLOB’s ~20% level is not “the class under DR/NLL.”
Artifacts: `blobcls_did/`, tags `nlres_blobcls_rstar_g{0,0.5}`.
