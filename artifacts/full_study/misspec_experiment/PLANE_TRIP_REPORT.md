# Misspec / OPC — plane report

**Updated:** 2026-11-26 Europe/Rome  
**PDF (phone):** `PLANE_TRIP_REPORT.pdf`  
**Reviews:** pedagogical + conciseness pass applied.

Companions: `NOTE_ess_opc_usability.*` (full ladders+Gini+CI), `PROTOCOL_LOCKED_capacity_x_N.md`, `MATH_AND_RESULTS_RUNDOWN.*`, `PROTOCOL_DRAFT_support_best_achievable.md`.

## Headlines

- OPC **absolute** greedy CTR rises with \(N\); relative “losses” under mild \(\gamma\) are NLL catching an absorbable projection.
- Strong \(\gamma=2\): clear OPC (\(D\approx+5.6\) @400k). Mild \(\gamma\): capacity moves \(N^\star\) earlier (lin≈200k → mlp64≈100k → mlp256≈50k).
- Use **greedy** + CI-excludes-0. Stochastic OPC wins ≈ sharpness (`logit_scale`), not IW Gini alone.
- Hybrid grid: **168/192** complete; SoftAND γ=100 clear OPC; SoftThresh mid-γ clear NLL — don’t pool families.

## Plan

1. Drain hybrid partials.  
2. Support / best-under-CI prototype (`PROTOCOL_DRAFT_support_best_achievable.md`).  
3. No SoftAND/ST \(N\)-ladders / mlp1024 until then.  
4. Writeup: greedy, abs \(V\)+\(D\)+CI, clear winners.
