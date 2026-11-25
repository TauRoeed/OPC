# ESS ↔ OPC usability under misspec

**Updated:** 2026-11-25 ~08:15 Europe/Rome (09:15 Asia/Jerusalem)  
**Scope:** MovieLens, frozen \(d{=}32\) BPR emb, `select-weights clip:10`, nested prefixes, 5 seeds (100–104).

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
| R\* fill \(N\in\{6.25,12.5,50,200\}\mathrm{k}\) | **done (4/4)** | same tags |
| SoftAND / SoftThresh \(N\)-ladders | **skipped** | redundant for this claim |
| Cross-family γ-grid @ \(N=100\mathrm{k}\) | hybrid ~165/192 | `nlres_grid_*` |
| Capacity @ fixed \(N=100\mathrm{k}\) | done (30/30) | MLP width does not erase strong-\(R^\star\) edge *at fixed \(N\)* |
| Capacity × \(N\) Tier 1 | **done** | `scale_cap_rstar_g{0.5,2}_mlp256` @ {25,100,400}k — **G1_NO_STOP** |
| Capacity × \(N\) Tier 2 | **in flight** | densify 50/200k (`g0.5` done; `g2` running) + γ=0 DiD next |
| 800k nested ladder | deferred | would break \(N_{\max}=400\mathrm{k}\) prefix pool |

Shared fill knobs: `n_trials=60`, methods `{nll, opc, hybrid_seq_f0.5}`, nested pool with \(N_{\max}=400\mathrm{k}\). Capacity ladders: `{nll, opc}` only, `linear+mlp` H=256.

## \(D(N)\) on the dense R\* ladder — **linear** (mean ± SE, \(n=5\))

| \(N\) | \(R^\star\,\gamma{=}0\) | \(\gamma{=}0.5\) | \(\gamma{=}2\) | wellspec |
|---:|---:|---:|---:|---:|
| 6.25k | **+1.7** ±0.7 | **+2.9** ±0.5 | **+3.5** ±0.4 | **+9.9** ±1.0 |
| 12.5k | **+0.9** ±0.2 | **+2.0** ±0.5 | **+4.5** ±0.5 | **+10.6** ±0.5 |
| 25k | +0.4 ±0.3 | **+1.7** ±0.5 | **+3.3** ±0.8 | **+6.8** ±0.9 |
| 50k | −0.2 ±0.2 | +0.9 ±0.3 | **+4.6** ±0.8 | **+5.6** ±1.3 |
| 100k | −1.0 ±0.3 | +0.5 ±0.8 | **+4.8** ±1.2 | +1.8 ±0.5 |
| 200k | −1.7 ±0.1 | −0.1 ±0.2 | **+5.3** ±0.6 | +0.4 ±0.1 |
| 400k | −1.4 ±0.2 | −0.3 ±0.9 | **+5.6** ±1.3 | +0.3 ±0.1 |

**Crossover \(N^\star(\gamma)\)** (linear) = smallest \(N\) where mean \(D\le 0\):

- \(\gamma=0\): \(N^\star \in (25\mathrm{k},50\mathrm{k}]\)
- \(\gamma=0.5\): \(N^\star \in (100\mathrm{k},200\mathrm{k}]\)
- \(\gamma=2\): **no crossover ≤400k** (\(D\) *widens*)
- wellspec: \(D\downarrow 0\) from above

## Capacity × \(N\) — **mlp256** vs linear

Tags `scale_cap_rstar_g*_mlp256` (Tier1 + `g0.5` densify done; `g2` 50/200k still filling).

| \(N\) | γ=0.5 mlp256 | γ=0.5 linear | γ=2 mlp256 | γ=2 linear |
|---:|---:|---:|---:|---:|
| 25k | **+1.6** ±0.6 | **+1.7** | **+2.0** ±1.5 | **+3.3** |
| 50k | +0.0 ±0.8 | +0.9 | *(densify)* | **+4.6** |
| 100k | −0.4 ±0.6 | +0.5 | +1.2 ±1.1 | **+4.8** |
| 200k | −0.6 ±0.7 | −0.1 | *(densify)* | **+5.3** |
| 400k | **−2.9** ±1.5 | −0.3 | +0.9 ±0.8 | **+5.6** |

- Mild γ: mlp \(N^\star \in (25\mathrm{k},50\mathrm{k}]\) vs linear \((100\mathrm{k},200\mathrm{k}]\) — **≥1 ladder step earlier**.
- Strong γ: OPC edge **collapses** (+0.9 vs +5.6 at 400k) — capacity absorbs misspec for NLL; fixed-\(N{=}100\mathrm{k}\) confirm had understated this.
- **G1_NO_STOP** → Tier 2 (densify + γ=0 DiD); Tier 3 mlp64 after that. Plan: `PROTOCOL_LOCKED_capacity_x_N.md`.

## What is a “large dataset” here?

Not raw \(N\). On this catalog (\(|U|{=}6038\), \(|A|{=}3533\), frozen \(d{=}32\)):

| Scale @ \(N{=}400\mathrm{k}\) | linear (\(p{\approx}2112\)) | mlp256 (\(p{\approx}35\mathrm{k}\)) |
|---|---:|---:|
| \(N/p\) | ~190 | ~11 |
| ESS/\(N\) under \(R^\star\,\gamma{=}2\) | ~0.3% | (same clip regime) |
| rows / user | ~66 | ~66 |
| pair coverage \(N/(|U||A|)\) | ~1.9% | ~1.9% |

**Working definition:** \(N\) is “large” once \(N \gtrsim N^\star(\gamma,C)\).  
**Linear:** ~50k (γ=0), ~200k (γ=0.5); never ≤400k at γ=2.  
**mlp256:** “large” arrives sooner at mild γ; at strong γ the OPC–NLL gap nearly vanishes by 400k even though \(N/p\sim 11\).

**Generalizable?** Mechanism yes (ESS/\(N\) collapse ≠ ranking flip; binding axis = misspec vs class). Numbers no — do not export \(N^\star\) outside ML + this clip + frozen emb.

## Model / capacity

- **Policy:** frozen biased BPR \(d{=}32\) + per-side linear correction (≈2112 params), or `linear+mlp` H∈{64,256} (≈10.5k / 35k).
- **Nuisance \(\hat q\):** separate regression; **not** widened in capacity trials.
- Policy \(p\) does **not** scale with \(|U|,|A|\) (shared transforms).

## ESS is not the ranking flip

Winning-run `weights_ess` (OPC, mean over seeds):

| world | ESS @25k | @100k | @400k | ESS/\(N\) @400k |
|---|---:|---:|---:|---:|
| \(R^\star\,\gamma=0\) | 250 | 493 | 2008 | 0.50% |
| \(\gamma=0.5\) | 398 | 1159 | 1463 | 0.37% |
| \(\gamma=2\) | 254 | 712 | 1058 | 0.26% |
| wellspec | 2994 | ~13k | ~61k | ~15% |

Under misspec, ESS/\(N\) **collapses** while \(D(N)\) still crosses (mild γ) or grows (strong γ, linear). Wellspec has far higher ESS *and* \(D\to 0\). Ranking tracks **misspec vs class**, not ESS accumulation.

## Cross-family (hybrid γ-grid @ \(N=100\mathrm{k}\), ML)

Min γ with mean \(D>0\) (OPC/hybrid vs NLL):

- **\(R^\star\)**: γ ≳ 0.75
- **SoftThresh**: γ ≳ 2
- **SoftAND**: γ ≳ 4

Same qualitative story; family only shifts the γ threshold.

## Usability rule of thumb

1. **Well-spec / absorbable misspec + large \(N\):** NLL and OPC **converge** (\(D\to 0\)).
2. **Slight misspec (\(R^\star\,\gamma\approx 0.5\)):** OPC helps at **small \(N\)**; crossover depends on class — linear ~\(10^5\)–\(2\cdot 10^5\), mlp256 already ~50k.
3. **Strong misspec vs a small class (\(R^\star\,\gamma\gtrsim 2\), linear):** they **do not** converge ≤400k. **Wider policy class** can absorb much of that edge (mlp256 \(D(400\mathrm{k})\approx +0.9\)).
4. Clipped ESS does not grow like \(N\) under misspec — do not equate “more data” with “better DR selection.”

## Headline

**Usability is class-relative:** mild misspec → OPC helps early then converges (sooner with more capacity); strong misspec → OPC stays ahead at large \(N\) *only if* the class cannot absorb it. ESS clipping is not what flips the ranking.

Companion PDF: `NOTE_ess_opc_usability.tex`. Locked plan: `PROTOCOL_LOCKED_capacity_x_N.md`.
