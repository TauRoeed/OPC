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

### Tier 2 — only if G1 fires (shift / flip)
- Densify bracketing \(N\) (50k/200k) on the moving tag (`--skip-completed`).
- Add `scale_cap_rstar_g0_mlp256` at {25k,100k,400k} (matched DiD).
- **G2:** if γ=0 \(N{=}25\mathrm{k}\) mlp256 hurts NLL/OPC by >2 pp vs linear → flag \(N/p\) overparam, don’t read γ>0 as absorption.

### Tier 3 — only if mlp256 moves \(N^\star\) ≥1 ladder step
- `scale_cap_rstar_g0.5_mlp64` at the same sentinel \(N\)s (dose-response).

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
