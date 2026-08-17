# Critical Reflection Checklist

Self-review of methodology, assumptions, and direction.

## Core assumptions

1. **Semi-synthetic GT is valid** — BPR factors ≈ true preference; noisy copies ≈ learner state. Real world may not factorize this cleanly.
2. **Propensities are known / logged exactly** — production logs often have clipped, stale, or approximate scores.
3. **Binary bandit rewards** — CTR-style; ignores dwell, multi-objective, delayed feedback.
4. **Matched search budget** — OPC vs NoProp fair only if Optuna budgets and splits stay identical.
5. **Noise model covers “bias”** — structured embedding noise + uniform mix is not a full taxonomy of industrial biases.

## Methodology risks

- **Confounding ε label vs SNR** — same level can yield different measured SNR across datasets/modes; prefer reporting measured SNR ([noise_snr.md](noise_snr.md)).
- **Selection metric mismatch** — `ci_low` vs `r_hat` vs oracle actual reward can invert method ranking.
- **Reward-model misspecification** — SNDR quality tracks `q̂`; oracle/logscore ablations already hint at this.
- **Dataset coverage** — early results lean on `ml`/`anime`; Kuai adds short-video + random exposure but not yet in result tables.
- **Compute / size** — small train regimes may favor low-variance naive; large regimes may favor OPC — need explicit size × SNR grid.

## Research direction questions

1. Is the main contribution “when OPC helps” (regime map) or a new estimator?
2. Do different noise *types* need different corrections at matched SNR, or only severity matters?
3. Can we diagnose operating regime from observable stats (propensity ESS, SNR proxies, val gap) without GT?

## Testable hypothesis

**H1 (bad \(\hat q\) → naive can win):** Under strong logging bias and large \(n\), poorly specified \(\hat q\) can make propensity-aware OPC lose to naive/no-propensity on true reward (variance / bad DM term), even though naive is biased. See [regime_inference.md](regime_inference.md). Not a literature quote — experiment target.

## What random interactions give

- **Yes:** \(\hat\rho=\) rand_CTR, density regime, sample size \(n=\max(9604, 10|U|)\) for \(\varepsilon\approx0.01\).
- **No (alone):** “optimal CTR” / link ceiling `ctr`. Calibrate `ctr` only *given* scores so \(\mathbb{E}[q]\approx\hat\rho\).

## What to write in Limitations (§14)

- Semi-synthetic construction.
- Known propensities.
- Limited real OPE online A/B validation.
- Kuai not yet in main tables (until artifacts exist).
- Theory deferred.

## Status

Scaffold for ongoing critical review; update after SNR characterization and Kuai pilot.
