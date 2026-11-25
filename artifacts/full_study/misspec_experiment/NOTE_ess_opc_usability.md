# ESS ↔ OPC usability under misspec

**Updated:** 2026-11-25 ~10:20 Europe/Rome  
**Scope:** MovieLens, frozen \(d{=}32\) BPR emb, `select-weights clip:10`, nested prefixes, seeds 100–104 (\(n{=}5\)).  
**Intervals:** mean and **95% Student-\(t\) CI** across seeds (not SE). Units = CTR percentage points.

## One claim

OPC beats NLL when misspec is strong *relative to* \(N\) and the policy class — **not** because DR “accumulates more ESS.”

\[
D(N)=V_{\mathrm{OPC}}(N)-V_{\mathrm{NLL}}(N)
\]
Positive \(D\) ⇒ OPC ahead. We report **absolute** \(V\) for both methods, not only \(D\).

## Why “OPC loses with more data” is misleading

**OPC’s absolute greedy CTR rises with \(N\) in every world.** What flips is the *gap*.

| World (linear, greedy) | OPC @6.25k → @400k | NLL @6.25k → @400k | What happens |
|---|---|---|---|
| mild \(R^\star\,\gamma{=}0.5\) | 14.3 → **19.3** | 11.4 → **19.5** | Both climb; NLL catches up → \(D\to 0^-\) |
| strong \(R^\star\,\gamma{=}2\) | 14.1 → **21.0** | 10.6 → **15.4** | NLL **stalls**; OPC keeps climbing → \(D\) stays large |
| wellspec | 28.9 → **29.6** | 19.0 → **29.4** | NLL learns the bilinear target; \(D\to 0^+\) |

So the paradox is not “more data hurts OPC.” It is:

1. **NLL optimizes likelihood of logged clicks** → converges to the KL projection of the logging law onto the policy class. When misspec is mild / absorbable, that projection is already a good ranking; large \(N\) just fits it well.
2. **OPC optimizes a clipped DR estimate of policy value** → helps most when the *value*-best policy in the class differs from the likelihood projection (large \(\Delta_{\mathrm{proj}}\)). When those two agree, DR selection noise + weight clipping can leave OPC a bit behind clean NLL.
3. **Clipped ESS/\(N\) collapses under misspec** (~0.3% at γ=2, 400k). Extra rows do **not** give OPC a proportional OPE-stability gift; NLL’s parametric fit *does* keep improving with \(N\) when the class can absorb the residual.
4. **mlp256 at mild γ** makes (1) stronger: more capacity → NLL absorbs more → at 400k NLL 22.9 vs OPC 20.1 (OPC still up from 17.8 @25k).

Intuition check: “OPC should be more stable with more data” refers to **estimator variance**. Ranking vs NLL is about **bias of the learning objective** vs the class. Stability ≠ winning the horse race once NLL’s bias floor is low.

Recoverability context (\(R^\star\)): value-oracle linear ≈24–35 pt vs lik-oracle ≈13–14 pt — large \(\Delta_{\mathrm{proj}}\) when γ is big; that is exactly when OPC’s edge persists.

## Greedy vs stochastic (“full”) value

| Metric | Field | Meaning |
|---|---|---|
| **Greedy** (primary) | `policy_rewards_greedy` | CTR of \(\arg\max_a\) scores — usual deployment metric in this repo |
| **Stochastic** | `policy_rewards` | CTR under the learned softmax policy |

Empirically at \(N{=}400\mathrm{k}\), \(R^\star\,\gamma{=}0.5\): OPC greedy≈stochastic (policy is nearly deterministic); NLL greedy≈19.5 but stochastic≈8.4 (diffuse softmax). Stochastic \(D\) is therefore huge and **mostly a sharpness artifact**, not a fair “OPC wins harder” claim. We report both; **decisions use greedy**.

## \(V\) and \(D\) — linear class, **greedy** (95% \(t\) CI)

### \(R^\star\,\gamma{=}0\)

| \(N\) | \(V_{\mathrm{OPC}}\) | \(V_{\mathrm{NLL}}\) | \(D\) |
|---:|---|---|---|
| 6.25k | 14.75 [13.74, 15.75] | 13.04 [10.75, 15.34] | +1.70 [−0.18, +3.59] |
| 12.5k | 15.60 [14.26, 16.94] | 14.72 [13.78, 15.67] | +0.87 [+0.34, +1.41] |
| 25k | 16.43 [14.86, 18.01] | 16.01 [15.14, 16.88] | +0.42 [−0.49, +1.33] |
| 50k | 16.98 [15.93, 18.04] | 17.21 [16.54, 17.88] | −0.23 [−0.81, +0.35] |
| 100k | 17.89 [16.92, 18.85] | 18.93 [18.42, 19.44] | −1.04 [−1.84, −0.24] |
| 200k | 18.12 [17.45, 18.78] | 19.83 [19.39, 20.27] | −1.71 [−2.00, −1.42] |
| 400k | 18.76 [17.73, 19.79] | 20.12 [19.54, 20.69] | −1.35 [−1.92, −0.79] |

### \(R^\star\,\gamma{=}0.5\)

| \(N\) | \(V_{\mathrm{OPC}}\) | \(V_{\mathrm{NLL}}\) | \(D\) |
|---:|---|---|---|
| 6.25k | 14.32 [13.17, 15.47] | 11.41 [9.51, 13.32] | +2.91 [+1.44, +4.38] |
| 12.5k | 15.29 [13.81, 16.76] | 13.32 [12.32, 14.32] | +1.97 [+0.56, +3.37] |
| 25k | 16.51 [14.15, 18.87] | 14.84 [13.83, 15.85] | +1.68 [+0.31, +3.04] |
| 50k | 17.53 [15.42, 19.65] | 16.67 [15.35, 18.00] | +0.86 [−0.08, +1.80] |
| 100k | 18.66 [15.66, 21.66] | 18.21 [16.25, 20.18] | +0.45 [−1.87, +2.76] |
| 200k | 19.18 [17.30, 21.07] | 19.25 [17.26, 21.23] | −0.06 [−0.69, +0.57] |
| 400k | 19.28 [15.71, 22.84] | 19.54 [17.51, 21.58] | −0.26 [−2.68, +2.15] |

### \(R^\star\,\gamma{=}2\)

| \(N\) | \(V_{\mathrm{OPC}}\) | \(V_{\mathrm{NLL}}\) | \(D\) |
|---:|---|---|---|
| 6.25k | 14.07 [12.15, 15.99] | 10.58 [9.39, 11.78] | +3.49 [+2.52, +4.45] |
| 12.5k | 15.99 [14.36, 17.62] | 11.48 [9.71, 13.25] | +4.51 [+3.14, +5.88] |
| 25k | 16.76 [13.93, 19.59] | 13.43 [10.76, 16.10] | +3.33 [+1.00, +5.66] |
| 50k | 18.45 [14.64, 22.26] | 13.84 [11.04, 16.64] | +4.62 [+2.40, +6.83] |
| 100k | 19.61 [15.49, 23.73] | 14.83 [12.34, 17.33] | +4.78 [+1.52, +8.04] |
| 200k | 20.80 [17.68, 23.91] | 15.54 [12.75, 18.34] | +5.26 [+3.66, +6.85] |
| 400k | 20.96 [15.75, 26.17] | 15.41 [13.30, 17.52] | +5.55 [+1.94, +9.16] |

### Wellspec (no residual)

| \(N\) | \(V_{\mathrm{OPC}}\) | \(V_{\mathrm{NLL}}\) | \(D\) |
|---:|---|---|---|
| 6.25k | 28.89 [28.35, 29.42] | 19.04 [16.35, 21.73] | +9.85 [+7.06, +12.63] |
| 12.5k | 29.10 [28.49, 29.72] | 18.51 [16.79, 20.22] | +10.59 [+9.13, +12.06] |
| 25k | 29.25 [28.54, 29.97] | 22.51 [20.42, 24.60] | +6.75 [+4.38, +9.12] |
| 50k | 29.22 [28.57, 29.86] | 23.66 [20.42, 26.90] | +5.56 [+1.91, +9.21] |
| 100k | 29.33 [28.89, 29.77] | 27.58 [26.49, 28.68] | +1.75 [+0.35, +3.14] |
| 200k | 29.04 [28.49, 29.60] | 28.63 [28.11, 29.16] | +0.41 [+0.10, +0.72] |
| 400k | 29.63 [29.14, 30.12] | 29.37 [28.99, 29.75] | +0.26 [−0.06, +0.58] |

**Crossover \(N^\star\)** (first \(N\) with mean \(D\le 0\), linear greedy): γ=0 ∈(25k,50k]; γ=0.5 ∈(100k,200k]; γ=2 none ≤400k.

## Stochastic value (same runs; secondary)

| World | \(N\) | \(V_{\mathrm{OPC}}^{\mathrm{stoch}}\) | \(V_{\mathrm{NLL}}^{\mathrm{stoch}}\) | \(D^{\mathrm{stoch}}\) |
|---|---:|---|---|---|
| γ=0.5 | 25k | 16.51 [14.15, 18.87] | 6.38 [5.11, 7.66] | +10.13 [+8.14, +12.11] |
| γ=0.5 | 100k | 18.66 [15.65, 21.66] | 8.00 [7.26, 8.75] | +10.65 [+8.29, +13.02] |
| γ=0.5 | 400k | 19.28 [15.71, 22.84] | 8.37 [7.82, 8.92] | +10.91 [+7.75, +14.06] |
| γ=2 | 25k | 16.73 [13.87, 19.59] | 7.09 [5.90, 8.28] | +9.65 [+7.53, +11.76] |
| γ=2 | 100k | 19.61 [15.49, 23.72] | 7.80 [7.04, 8.56] | +11.81 [+7.87, +15.74] |
| γ=2 | 400k | 20.95 [15.75, 26.15] | 8.00 [7.44, 8.55] | +12.95 [+7.84, +18.07] |
| well | 25k | 29.21 [28.56, 29.87] | 6.85 [4.27, 9.43] | +22.36 [+19.57, +25.16] |
| well | 100k | 29.30 [28.90, 29.70] | 9.61 [7.77, 11.45] | +19.69 [+17.56, +21.81] |
| well | 400k | 29.60 [29.15, 30.05] | 10.99 [10.69, 11.29] | +18.61 [+18.05, +19.17] |

## Capacity × \(N\) (mlp256) — greedy

| \(N\) | γ=0.5 OPC | γ=0.5 NLL | \(D\) | γ=2 OPC | γ=2 NLL | \(D\) |
|---:|---|---|---|---|---|---|
| 25k | 17.84 [15.52, 20.16] | 16.28 [14.85, 17.72] | +1.56 [−0.10, +3.21] | 17.91 [13.68, 22.13] | 15.87 [11.84, 19.89] | +2.04 [−2.04, +6.12] |
| 50k | 18.51 [15.39, 21.64] | 18.48 [16.29, 20.66] | +0.04 [−2.09, +2.17] | 18.99 [15.12, 22.85]† | 15.94 [15.43, 16.46]† | +3.04 [−0.88, +6.97]† |
| 100k | 19.52 [17.03, 22.01] | 19.88 [17.21, 22.56] | −0.37 [−2.14, +1.40] | 20.64 [15.96, 25.31] | 19.45 [15.21, 23.70] | +1.19 [−1.72, +4.09] |
| 200k | 20.14 [18.25, 22.04] | 20.73 [18.05, 23.42] | −0.59 [−2.53, +1.35] | 21.90 [19.19, 24.61]† | 19.77 [17.80, 21.75]† | +2.12 [−1.36, +5.61]† |
| 400k | 20.06 [15.57, 24.55] | 22.92 [20.46, 25.38] | −2.87 [−6.95, +1.21] | 23.94 [19.66, 28.23] | 23.01 [19.64, 26.39] | +0.93 [−1.23, +3.09] |

† γ=2 @50/200k: densify still filling (\(n{<}5\) on those rows when last frozen — check live tag). Tier1 sentinels @25/100/400k are \(n{=}5\).

## What is a “large dataset” here?

Not raw \(N\). ML catalog \(|U|{=}6038\), \(|A|{=}3533\), frozen \(d{=}32\):

| Scale @400k | linear \(p{\approx}2112\) | mlp256 \(p{\approx}35\mathrm{k}\) |
|---|---:|---:|
| \(N/p\) | ~190 | ~11 |
| ESS/\(N\) @\(R^\star\,\gamma{=}2\) | ~0.3% | same clip |
| rows/user | ~66 | ~66 |

\(N\) is “large” once \(N\gtrsim N^\star(\gamma,C)\). Mechanism (misspec vs class) generalizes; the numbers do not.

## Plan — can we learn the optimal policy? (ESS support / best achievable)

**Question.** Is \(\pi^\star\) (or the class value-oracle) even in the logging support enough to learn? If not, what is the best policy we can still certify from data?

### Step A — support diagnostics (no new training)
On nested prefixes for \(R^\star\) γ∈{0,0.5,2} + wellspec, seeds 100–104:
1. **Opt-mass:** \(m(N)=\mathbb{E}_{u\sim\mathrm{log}}[\pi_{\mathrm{log}}(a^\star(u)\mid u)]\) and top-\(k\) mass for \(a^\star=\) true greedy under \(q\), and under class value-oracle / lik-oracle actions.
2. **SnIPS / clipped ESS** for \(\pi^\star\), value-oracle, lik-oracle, and learned OPC/NLL policies under `clip:10` (and unclipped for diagnosis only).
3. **Overlap proxy:** fraction of users whose \(a^\star\) appears ≥1 in the logged prefix; coverage of top-1% value pairs.

Kill: if \(m(N)\) and clipped ESS for \(\pi^\star\) stay ~0 while value-oracle ESS is moderate → optimal is unsupported; “learn \(\pi^\star\)” is the wrong target.

### Step B — class ceilings (reuse recoverability + strong lik oracles)
Report side-by-side with learned \(V\): logger, \(V_{\mathrm{lin,value}}\), \(V_{\mathrm{lin,lik}}\), repair-mlp, ceiling. Already partly in `recoverability/`. Refresh with strong lik oracles from A3-0 where pins moved.

### Step C — best achievable *within confidence* (new, small)
For each \((N,\gamma,\mathrm{seed})\):
1. Build a finite candidate set: logger, NLL, OPC, hybrid (if present), lik-oracle projection, value-oracle projection (oracle fits, not “cheating” at eval — score with **OPE on held-out / crossfit folds only**).
2. For each candidate, compute DR (or SNIPS) value + **percentile bootstrap CI** on the same logged prefix (respect clip).
3. **Best achievable under confidence:** \(\hat\pi_{\mathrm{safe}}=\arg\max_\pi \{\mathrm{CI}_{-}(\pi):\mathrm{CI}_{-}\ge V_{\mathrm{logger}}\}\) (or unconstrained \(\arg\max \mathrm{CI}_{-}\)). Report \(V_{\mathrm{true}}(\hat\pi_{\mathrm{safe}})\) vs \(V_{\mathrm{true}}(\pi^\star)\) and vs OPC/NLL.

This answers: given only logged data + clip, what policy would a cautious DR selector pick, and how close is it to oracle?

### Step D — deliverable
Short note `NOTE_support_best_achievable.md` + CSV: support metrics × \(N\), safe-policy gap × \(N\), recommendation (when OPC’s edge is “real recoverability” vs “unsupported optimum”).

**Order:** A → B (tables) → C prototype on γ=2 @ {25k,100k,400k} only → expand if useful. Do not block Tier2/3 capacity ladders.

## Headline

**OPC does not degrade with \(N\); NLL catches absorbable misspec.** Strong misspec vs a small class → NLL stalls, OPC keeps rising. ESS clipping limits OPE precision; it is not what flips the ranking. Use **greedy** for decisions; stochastic value mainly shows NLL stays soft.

Companion PDF: `NOTE_ess_opc_usability.tex`. Capacity plan: `PROTOCOL_LOCKED_capacity_x_N.md`. Support plan: section above (to be spun into its own note when Step A runs).
