# Draft — logging support for π★ vs best achievable under confidence

**Status:** draft plan (from ESS/OPC usability follow-ups, 2026-11-25 Europe/Rome)  
**Does not block:** capacity×N Tier2/3 on GPU2

## Motivation
Absolute greedy tables show OPC **improves** with \(N\); relative losses vs NLL under mild misspec are NLL catching an absorbable projection. Separately: if \(\pi^\star\) has near-zero logging mass, “learn the optimum” is impossible under IPS/DR with clip — we should report the **best certifiable** policy instead.

## Step A — support (offline on existing worlds)
Worlds: `scale_p1_rstar_g{0,0.5,2}`, `scale_p1_wellspec`; \(N\in\{25,100,400\}\mathrm{k}\); seeds 100–104.

| Metric | Definition |
|---|---|
| opt-mass | \(\mathbb{E}_u[\pi_{\log}(a^\star(u)\mid u)]\) for \(a^\star\in\{\mathrm{true\ greedy},\ \mathrm{value\ oracle},\ \mathrm{lik\ oracle}\}\) |
| top-k mass | same for top-5 / top-50 by true \(q\) |
| coverage | frac users with \(a^\star\) logged ≥1 in prefix |
| clipped ESS | ESS of \(w=\pi/\pi_{\log}\) under `clip:10` for those π and for learned OPC/NLL |

**Kill:** opt-mass & clipped ESS for true \(\pi^\star\) ≈ 0 while value-oracle ESS is OK → target class oracle / safe set, not \(\pi^\star\).

## Step B — ceilings
Reuse `recoverability/` + strong lik oracles (A3-0). Table: logger, \(V_{\mathrm{lin,value}}\), \(V_{\mathrm{lin,lik}}\), repair mlp, learned OPC/NLL @ same \(N\).

## Step C — best under confidence
Candidates: logger, NLL, OPC, (hybrid), lik-oracle π, value-oracle π.  
Score with **crossfit DR + percentile bootstrap CI** on the logged prefix (clip:10).  
\(\hat\pi_{\mathrm{safe}}=\arg\max_\pi \mathrm{CI}_{-}(\pi)\) (optional: require \(\mathrm{CI}_{-}\ge V_{\mathrm{logger}}\)).  
Report true greedy \(V\) of \(\hat\pi_{\mathrm{safe}}\) vs oracles and vs OPC/NLL.

Prototype only: \(R^\star\,\gamma{=}2\) @ {25,100,400}k, 2–3 seeds, before full grid.

## Step D — writeup
`NOTE_support_best_achievable.md` + CSV. Link from ESS/OPC note.

## Non-goals
- Unclipped IPS as a deployment selector  
- Retraining with different clip in the same “fair ladder” cells  
- SoftAND/ST support ladders before R* prototype
