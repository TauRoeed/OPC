# ESS ↔ OPC usability under misspec

**Updated:** 2026-11-24 ~21:35 Asia/Jerusalem  
**Scope:** MovieLens, linear policy, `select-weights clip:10`, nested prefixes, 5 seeds (100–104).

## One claim

OPC beats NLL when misspec is strong *relative to* \(N\) and the policy class — **not** because DR “accumulates more ESS.”

Define
\[
D(N)=100\cdot\bigl(V_{\mathrm{greedy}}(\mathrm{OPC})-V_{\mathrm{greedy}}(\mathrm{NLL})\bigr)
\]
(percentage points of CTR). \(D>0\) ⇒ OPC ahead.

## What we actually ran

| Piece | Status | Tag / note |
|---|---|---|
| R\* stage-1 \(N\in\{25,100,400\}\mathrm{k}\) | done | `scale_p1_rstar_g{0,0.5,2}`, `scale_p1_wellspec` |
| R\* fill \(N\in\{6.25,12.5,50,200\}\mathrm{k}\) | **3/4 worlds done** | same tags; wellspec fill **in flight** (started 21:35) |
| SoftAND / SoftThresh \(N\)-ladders | **skipped** | redundant for this claim |
| Cross-family γ-grid @ \(N=100\mathrm{k}\) | ~165/192 cells @40 seeds | `nlres_grid_*` (hybrid evidence) |
| Capacity confirm | done (30/30) | MLP width does not erase strong-\(R^\star\) edge |
| 800k nested ladder | deferred | would break \(N_{\max}=400\mathrm{k}\) prefix pool |

Shared fill knobs: `n_trials=60`, methods `{nll, opc, hybrid_seq_f0.5}`, nested pool with \(N_{\max}=400\mathrm{k}\).

## \(D(N)\) on the dense R\* ladder (mean ± SE, \(n=5\))

| \(N\) | \(R^\star\,\gamma{=}0\) | \(\gamma{=}0.5\) | \(\gamma{=}2\) | wellspec |
|---:|---:|---:|---:|---:|
| 6.25k | **+1.7** ±0.7 | **+2.9** ±0.5 | **+3.5** ±0.4 | *(fill pending)* |
| 12.5k | **+0.9** ±0.2 | **+2.0** ±0.5 | **+4.5** ±0.5 | — |
| 25k | +0.4 ±0.3 | **+1.7** ±0.5 | **+3.3** ±0.8 | **+6.8** ±0.9 |
| 50k | −0.2 ±0.2 | +0.9 ±0.3 | **+4.6** ±0.8 | — |
| 100k | −1.0 ±0.3 | +0.5 ±0.8 | **+4.8** ±1.2 | +1.8 ±0.5 |
| 200k | −1.7 ±0.1 | −0.1 ±0.2 | **+5.3** ±0.6 | — |
| 400k | −1.4 ±0.2 | −0.3 ±0.9 | **+5.6** ±1.3 | +0.3 ±0.1 |

**Crossover \(N^\star(\gamma)\)** = smallest \(N\) where mean \(D\le 0\):

- \(\gamma=0\): \(N^\star \in (25\mathrm{k},50\mathrm{k}]\)
- \(\gamma=0.5\): \(N^\star \in (100\mathrm{k},200\mathrm{k}]\)
- \(\gamma=2\): **no crossover ≤400k** (\(D\) *widens*)
- wellspec: \(D\downarrow 0\) from above (NLL catches the bilinear target)

Absolute levels at \(\gamma=2\), \(N=400\mathrm{k}\): OPC ≈21.0%, NLL ≈15.4% — NLL stalls; OPC keeps climbing.

## ESS is not the ranking flip

Winning-run `weights_ess` (OPC, mean over seeds):

| world | ESS @25k | @100k | @400k | ESS/\(N\) @400k |
|---|---:|---:|---:|---:|
| \(R^\star\,\gamma=0\) | 250 | 493 | 2008 | 0.50% |
| \(\gamma=0.5\) | 398 | 1159 | 1463 | 0.37% |
| \(\gamma=2\) | 254 | 712 | 1058 | 0.26% |
| wellspec | 2994 | ~13k | ~61k | ~15% |

Under misspec, ESS/\(N\) **collapses** while \(D(N)\) still crosses (mild γ) or grows (strong γ). Wellspec has far higher ESS *and* \(D\to 0\). So more logged rows ≠ “OPC wins via ESS”; the binding axis is **misspec vs class capacity**.

## Cross-family (hybrid γ-grid @ \(N=100\mathrm{k}\), ML)

Min γ with mean \(D>0\) (OPC/hybrid vs NLL):

- **\(R^\star\)**: γ ≳ 0.75
- **SoftThresh**: γ ≳ 2
- **SoftAND**: γ ≳ 4

Same qualitative story; family only shifts the γ threshold. Full SoftAND/ST \(N\)-ladders were skipped on purpose.

## Usability rule of thumb (linear class)

1. **Well-spec / absorbable misspec + large \(N\):** NLL and OPC **converge** (\(D\to 0\)).
2. **Slight misspec (\(R^\star\,\gamma\approx 0.5\)):** OPC helps at **small \(N\)**; gap shrinks and crosses near \(10^5\)–\(2\cdot 10^5\).
3. **Strong misspec vs the class (\(R^\star\,\gamma\gtrsim 2\)):** they **do not** converge ≤400k — NLL plateaus, OPC stays ahead. Extra MLP width at fixed \(N=100\mathrm{k}\) does **not** erase this.
4. Clipped / weight ESS does not grow like \(N\) under misspec — do not equate “more data” with “better DR selection.”

## Headline

**Small class + mild misspec → OPC helps early, then converges. Strong misspec → OPC stays ahead at large \(N\). ESS clipping limits OPE precision; it is not what flips the ranking.**

Companion PDF source: `NOTE_ess_opc_usability.tex` (phone-friendly layout).
