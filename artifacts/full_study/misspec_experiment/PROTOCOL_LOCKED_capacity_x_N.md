# Locked plan — capacity × N (OPC usability)

**Locked:** 2026-11-25 (Europe/Rome)  
**Reviews:** GPT-5.6, Claude Opus 5.5, Gemini Neon (consensus below)  
**GPU:** 2 only · do not touch hybrid CPU workers

## Answers (consensus)

### What is a “large dataset” here?
Not raw \(N\). Use dimensionless scales on **this** setup (ML, frozen \(d{=}32\) BPR emb):

| Scale | linear @400k | mlp256 @400k |
|---|---:|---:|
| \(N/p\) (rows / policy params) | ~190 | ~11 |
| ESS/\(N\) under \(R^\star\,\gamma{=}2\) | ~0.3% | (same clip regime) |
| rows / user | ~66 | ~66 |
| pair coverage \(N/(|U||A|)\) | ~1.9% | ~1.9% |

**Working definition:** \(N\) is “large” once \(N \gtrsim N^\star(\gamma,C)\) — NLL’s bias floor dominates variance for that class.  
For **linear**: ~50k at γ=0, ~200k at γ=0.5; **never ≤400k** at γ=2.  
Same 400k is **not** large for mlp256 (\(N/p\sim 11\)).

Catalog: \(|U|{=}6038\), \(|A|{=}3533\), \(d{=}32\). Policy \(p\) does **not** scale with \(|U|,|A|\) (shared transforms).

### Generalizable?
**Mechanism yes, numbers no.**  
Claim that survives: ESS/\(N\) collapse does not track the sign of \(D(N)\); binding axis is misspec vs class.  
Do **not** claim universal \(N^\star\) or “OPC unusable at large \(N\)” outside ML + this clip + frozen emb.

### Model used / can capacity change it?
- **Policy:** frozen biased BPR \(d{=}32\) + per-side `GlobalLinearCorrection` \((I{+}D)x{+}b\) (≈2112 params), or `linear+mlp` H∈{64,256} (≈10.5k / 35k).
- **Nuisance \(\hat q\):** separate regression; **not** widened in capacity trials.
- At fixed \(N{=}100\mathrm{k}\), mlp256 **did not erase** \(R^\star\,\gamma{=}2\) edge (\(D_{\mathrm{opc}}\approx{+}2\)). SoftAND often has \(D_{\mathrm{opc}}{<}0\) (use hybrid/OPC carefully).
- Capacity can still **move \(N^\star\)** at mild γ (bias shrink vs estimation tax). That is what this plan tests.

## Tiered runs (free once locked)

**Shared knobs** = Phase-1 fill (`run_scaling_phase1_fill.sh` COMMON) except:
- `--policy-transform linear+mlp --mlp-hidden H`
- `--methods nll opc` (no hybrid — saves ~⅓)
- Always include `400000` in `--train-sizes` (nested pool lock)

**Reuse:** `scale_p1_rstar_g*` = linear curves. Never splice `nlres_cap_*` (seeds 300+) into nested ladders.

### Tier 1 — primary (run now)
| Tag | γ | H | \(N\) |
|---|---|---|---|
| `scale_cap_rstar_g0.5_mlp256` | 0.5 | 256 | 25k, 100k, 400k |
| `scale_cap_rstar_g2_mlp256` | 2 | 256 | 25k, 100k, 400k |

**G1 kill:** if γ=0.5 \(D_{\mathrm{mlp256}}\) stays within ±0.8 pp of linear at all 3 \(N\) **and** γ=2 \(D(400\mathrm{k})\ge +1.5\) without shrink 100→400k → **STOP** (capacity×N null/weak).

### Tier 1 result (2026-11-25 05:01 Asia/Jerusalem = 04:01 Europe/Rome) — **G1_NO_STOP**
Mean \(D\) (pp, \(n=5\)):

| \(N\) | γ=0.5 mlp256 | γ=0.5 linear | γ=2 mlp256 | γ=2 linear |
|---:|---:|---:|---:|---:|
| 25k | +1.56 | +1.68 | +2.04 | +3.33 |
| 100k | −0.37 | +0.45 | +1.19 | +4.78 |
| 400k | −2.87 | −0.26 | +0.93 | +5.55 |

- Mild γ: \(N^\star\) moves earlier (mlp already \(D{<}0\) at 100k); at 400k NLL wins harder.
- Strong γ: OPC edge **collapses** (+0.9 vs +5.6) — capacity absorbs misspec for NLL; \(D(400\mathrm{k}){<}1.5\).
- Fixed-\(N{=}100\mathrm{k}\) confirm still held a positive edge; the **ladder** shows capacity×\(N\) is not null.

→ **Tier 2 launched.**

### Tier 2 progress (2026-11-25 ~08:15 Europe/Rome)
- `g0.5` densify **done** (50k/200k filled). Dense ladder \(D\): 25k +1.6, **50k +0.0**, 100k −0.4, 200k −0.6, 400k −2.9 → \(N^\star\in(25\mathrm{k},50\mathrm{k}]\) (vs linear \((100\mathrm{k},200\mathrm{k}]\)).
- `g2` densify **running**; γ=0 DiD next.

### Tier 2 — only if capacity shifted curves (G1 did not kill)
- Densify bracketing \(N\) (50k/200k) on **both** moving tags (`--skip-completed`, same run-tags).
- Add `scale_cap_rstar_g0_mlp256` at {25k,100k,400k} (matched DiD).
- **G2:** if γ=0 \(N{=}25\mathrm{k}\) mlp256 hurts NLL **or** OPC greedy by >2 pp vs linear → flag \(N/p\) overparam, don’t read γ>0 as absorption.

Driver: `run_scaling_capacity_x_N_tier2.sh`

### Tier 3 — only if mlp256 moves \(N^\star\) ≥1 ladder step
- **Confirmed** for γ=0.5 after densify (\(N^\star\) moved ≥1 step).
- Launch after Tier 2 finishes: `scale_cap_rstar_g0.5_mlp64` at sentinel \(N\)s (dose-response), unless G2 flags overparam.

## Do NOT run
- SoftAND/ST \(N\)-ladders  
- mlp1024 / unfrozen emb / changing \(\hat q\) class in the same cells  
- 6.25k/12.5k for mlp256 (\(N/p\ll 1\))  
- 800k (breaks pool)  
- Multi-ds ladders before ML capacity×N is non-null  
- Hybrid arm on these ladders  

## Preflight
- Tiny-batch NLL overfit must pass for mlp256 (`probe_nll_overfit.py --tiny-batch` / existing pytest).
- Sign of \(D(100\mathrm{k})\) on seeds 100–104 should agree with `nlres_cap_*` direction (loose check).

## Driver
`artifacts/full_study/misspec_experiment/run_scaling_capacity_x_N.sh`
