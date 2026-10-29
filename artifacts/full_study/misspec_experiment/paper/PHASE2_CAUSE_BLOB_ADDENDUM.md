# Addendum — Phase-2 CausE / BLOB standalone (and blobcls DiD)

**Source artifacts:** `phase2/`, `blobcls_did/`, tags `nlres_p2_rstar_g*_mlp64|refs`, `nlres_blobcls_rstar_g{0,0.5}`.  
**Protocol:** `PROTOCOL_LOCKED.md` §Phase 2; hybrid follow-up draft `PROTOCOL_DRAFT_phase2b.md`.

## Setup

- World: ml, combined-high, R\* (\(s{=}1,m{=}4,\psi{=}\tanh\)), γ ∈ {0, 0.25, 0.5}, n=100k, 20 trials, confirmatory seeds 300–309.
- Matched tier: OPC + CausE-cap (ρ=0) under `policy_transform=linear+mlp`, hidden 64.
- Frozen refs: CausE-cap-affine ρ=0 + BLOB-l10-nq under linear transform (published BLOB ELBO path).
- Primary Phase-2 estimand: \(D(\gamma)=V(\mathrm{OPC\textrm{-}mlp64})-V(\mathrm{CausE\textrm{-}cap\textrm{-}C\textrm{-}mlp64})\), \(\mathrm{DiD}=D(0.5)-D(0)\).

## Table A — Matched mlp DiD (primary Phase-2)

| γ | \(D\) (pt) | sd | n |
|---:|---:|---:|---:|
| 0 | −1.46 | 0.44 | 10 |
| 0.25 | −0.87 | 0.37 | 10 |
| 0.5 | +0.87 | 1.50 | 10 |

**DiD = +2.32** pt · 95% CI **[+1.11, +3.54]** · signs **9/10**.

Per-seed DiD (pt): 300:+3.13, 301:+4.03, 302:+1.22, 303:+2.48, 304:−0.24, 305:+0.83, 306:+3.56, 307:+4.80, 308:+0.34, 309:+3.10.

## Table B — Absolute greedy CTR % (selected)

| γ | OPC-mlp | CausE-C-mlp | CausE-T-mlp | CausE-C-lin | CausE-T-lin | BLOB-l10-nq |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17.54 | 19.00 | 18.98 | 18.90 | 18.94 | 19.28 |
| 0.25 | 18.24 | 19.11 | 18.98 | 19.07 | 19.08 | 19.53 |
| 0.5 | 19.55 | 18.68 | 18.35 | 19.04 | 19.13 | **20.09** |

## Table C — Refs DiDs (OPC-mlp − ref)

| Contrast | DiD (pt) | 95% CI | Signs | Note at γ\* |
|---|---:|---|---|---|
| vs CausE-C-linear | +1.86 | [+1.01, +2.71] | 10/10 | OPC 19.55 > 19.04 |
| vs CausE-T-mlp | +2.64 | [+1.64, +3.63] | 10/10 | OPC overtakes |
| vs BLOB-l10-nq | +1.20 | [+0.54, +1.85] | 8/10 | OPC **still trails** 19.55 vs 20.09 |

## Table D — Blob score-class lean DiD (not published BLOB)

`policy_transform=blob`, methods OPC vs NLL, seeds 300–304.

| γ | OPC | NLL | \(D\) |
|---:|---:|---:|---:|
| 0 | 16.14% | 17.11% | −0.97 |
| 0.5 | 17.40% | 17.06% | +0.34 |

**DiD = +1.31** · CI [+0.12, +2.49] · **4/5** signs (underpowered vs locked 7/10 rule).

## Naming

- **BLOB-l10-nq** = published ELBO arm (`methods=blob`).  
- **blobcls** = ScoreClass \(x^\top Ma+\kappa\) under OPC/NLL (`policy_transform=blob`).  
Do not equate the two.

## Hybrid status

- Sequential NLL→DR (`hybrid_seq`) exists only on the CFModel path — usable for **blobcls** (H-A), not for CausE/BLOB native trainers without new code.
- Selection-swap (H-B): BLOB trials already store `val_dr_*`; CausE needs re-score. Not a train hybrid.
- True joint ELBO/NLL+DR (H-C): out of scope (see `PROTOCOL_DRAFT_phase2b.md`).

## Claims / non-claims

**Claim:** Under locked R\* misspec, matched-capacity OPC-mlp widens vs CausE-cap (DiD +2.32).  
**Non-claim:** OPC beats published BLOB in level; CausE/BLOB “hybrid” without specifying H-A/B/C; blobcls DiD as evidence about the BLOB ELBO method.
