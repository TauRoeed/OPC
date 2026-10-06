# Representation-mismatch research: handoff and checkpoint (2026-10-05)

*A repository checkpoint before the next research phase: what exists, where it is, what it shows and what is open.
All results are development evidence. Nothing here is confirmatory.*

*Superseded as the entry point by `docs/handoff_20261006.md` (the BLOB phases, the training-objectives audit and a code
review). This document's sections on the simulator fix, the OPC configuration and the CausE variants still hold.*

## 1. Branch and commit map (verified from `origin` on 2026-10-05)

| branch | HEAD | contents | status |
|---|---|---|---|
| `origin/CRM` | `d5b4478` | the corrected simulator and the OPC revalidation (phases 0–5, the final decision report, Atlas v9 PDF) | complete and reviewed; shared with Roee; unchanged since 2026-10-04 |
| `origin/cause-baseline` | `d31a5dd` | the faithful native CausE: the port checked against the original TensorFlow graph, the ML-100K / ML-10M reproduction, and the M5 bounded comparison. It carries the simulator fix as `5c011a9` | archive; do not rewrite |
| `origin/cause-fair` | `386f3fd` | `CRM` plus a merge of `cause-baseline` (`ee4a09e`) plus the fair 25k CausE study (CausE-warm, CausE-capacity-matched, tuning, analysis, report, Atlas v10 PDF) | complete; stopped for review |
| `origin/representation-mismatch-next` | this checkpoint | `cause-fair` plus this handoff | the base for the next phase; **not merged into `CRM`** |
| `origin/main`, `origin/slim-OPC`, `origin/recogym`, `origin/pre-criteo`, `origin/first-fix`, `origin/cursor/*` | — | older lines of work | not part of this checkpoint |

- **Ancestry.**
  - `CRM` ⊂ `cause-fair` and `cause-baseline` ⊂ `cause-fair`; `cause-fair` ⊂ `representation-mismatch-next`.
  - `cause-baseline` does not contain `CRM` (the two met in the merge `ee4a09e`).
  - On `cause-fair` the simulator files are byte-identical to `CRM`'s fixed version.
- **Milestones.**
  - The corrected simulator and the OPC revalidation are on `CRM`, and through it on `cause-fair` and this branch.
  - The native CausE reproduction is on `cause-baseline`, and through the merge on `cause-fair` and this branch.
  - The fair 25k comparison is on `cause-fair` and this branch.
- **Local-only state** (by design, not pushed):
  - the run folders `artifacts/full_study/run_*`: 97, every one listed in `artifacts/full_study/run_registry.csv`;
  - the console logs (`*/logs/`);
  - the raw datasets and the generated BPR embeddings;
  - one stash from 2026-09-24 (`stash@{0}`, "issue1: reward features"). It is superseded by `b953d89` on `CRM`,
    which committed the same feature: 78 of its 88 added lines are verbatim on `CRM`, the rest reworded help text. It
    is kept untouched.

## 2. The simulator bug and its fix

- **Bug.** From `69fffab` (2026-09-24) to `c11b2b3`, the logger's random generator was the simulation's own stream.
  Each logged row's action reused the uniform draw that had picked its user. Each user therefore got a nearly fixed
  action (93–99% of its probability), while the stored propensity π0(a|u) was 6–25 times smaller than the
  probability that generated the row. Rewards were unaffected; every study run from 2026-09-26 was affected.
- **Fix.** `dbc401b` on `CRM` (`5c011a9` on `cause-baseline`).
  - Logged actions come from their own stream, `derive_seed(random_state, "logging_policy_actions")`.
  - The simulation refuses a policy that shares its generator.
- **Permanent tests.** `tests/test_logging_propensity_calibration.py` (15) and `tests/test_logging_rng_independence.py`
  (4). They check that logged rows are draws from the propensities they record (per user, jointly, and through the
  IPS identities). 13 of the 15 fail on the old code.
- **Rerun.** Every affected result was rerun and OPC was re-tuned on the corrected logs:
  `docs/simulator_fix_opc_revalidation_20261004.md`.

## 3. Current OPC configuration (the revalidated development default)

- **Objective.** Additive DR with the direct gradient. Training weights harmonic:0.1, w / (0.9 + 0.1w), at most 10.
- **Selection.** The 95% DR lower bound with clip:10 weights, on 20,000 warm validation rows.
- **Search.** lr 1e-4–2e-3, 5–30 epochs, lr decay 0.8–1. A learnable logit scale. The paired random sampler, 20
  trials per size.
- **Data and reward model.** Logger share 0.8. q̂ is a logistic regression on [x, a, x⊙a], fit on the policy's own
  rows and cross-fitted by user (5 folds).
- **Policy class.** One global linear correction per side on the logger's frozen vectors, (I + D)x + b, starting at
  the logger.
- **Flags.** `--policy-losses dr --opc-gradient direct --train-weights harmonic:0.1 --select-weights clip:10
  --learn-logit-scale --sampler random --lr-range 1e-4 2e-3 --epochs-range 5 30`. The CLI defaults still hold the
  older range.
- **Regime dependence.** With a misspecified q̂ (concat features), raw DR (`--train-weights none`) is the robust
  choice. With a well-specified q̂ it costs about 0.4–0.8 points.

## 4. The CausE variants

All three use CausE's released objective (cross-entropy, L2, and the L1 tie between control and treatment item rows;
momentum with linear decay), selection by validation NLL, and a fixed budget: (1 − ρ)N logger rows plus ρN uniform
rows.

| variant | starts from | capacity | ceiling | code |
|---|---|---|---|---|
| native CausE (prod-C, prod-T, avg) | random vectors | free user and item vectors, per-row biases | target best | `models/cause.py · CausEModel` |
| CausE-warm (C, T) | OPC's source vectors | the native capacity, all trainable | target best | `warm_start_` |
| CausE-capacity-matched (C, T) | the frozen source vectors | OPC's (I + D)x + b per side: users, control items, treatment items | OPC's linear-repair oracle | `CausELinModel` |

Run them with `--methods cause --cause-family native|warm|cap`. The design and the search spaces are in
`docs/cause_fair_comparison_25k.md` §1–§3.

## 5. Headline 25k findings (`docs/cause_fair_comparison_25k.md`)

Greedy value on the 24 biased worlds; brackets are 95% CIs paired by world. The worlds are ml, kuairand and anime
× warp / group / vector / combined high × seeds 100/101, with N = 25,000.

- **OPC − CausE-capacity-matched** is −0.41 [−0.66, −0.16] at ρ = 0 and −0.36 [−0.57, −0.15] at ρ = 0.25, with OPC
  higher in 4–6 of 24 worlds.
  - The lead sits under warp bias: −1.19 [−1.44, −0.95], 6/6 worlds. Group and vector bias are ties.
  - The stochastic value with fair tempering agrees (−0.33 to −0.40).
- **CausE-capacity-matched is flat in ρ**, within 0.05 points of ρ = 0, so randomized traffic adds nothing. It also
  beats DM-only by +0.94 [0.49, 1.38].
- **OPC beats CausE at native capacity.**
  - OPC − CausE-warm is +1.03 (C) and +2.40 (T).
  - Native CausE trails OPC by about 9 points.
  - CausE-warm − native is +8.3, the missing source representation; CausE-cap − CausE-warm adds +1.4.
- **Collection cost.** At ρ = 0.25, CausE gives up 932 clicks per world, 21% of the all-logger collection's clicks.

## 6. Status of the evidence

- **All of it is development evidence.**
  - Main grids on seeds 100/101 (3 datasets × 5 or 6 bias settings).
  - Tuning on seeds 200/201.
  - CausE at 25k only; OPC at 5k / 25k / 100k.
- **Nothing is confirmatory.** The confirmatory runs (fresh seeds, all six datasets, frozen protocol) have not been
  made.
- **Per-bias claims rest on 6 worlds.** Pooled claims rest on 24 paired worlds.

## 7. Current interpretation

- **Randomized traffic did not help.** The 25k fair comparison did NOT show that randomized CausE beats OPC.
  CausE-capacity-matched already beats OPC at ρ = 0, with no randomized rows and no propensities. Randomized traffic
  gives essentially no benefit at 25k and costs collection reward.
- **The lead is concentrated in the well-specified global-warp case.** Warp is a global linear distortion that the
  linear correction family can undo exactly.
- **The emerging comparison is between learning principles.** On one side, likelihood-based outcome adaptation in
  the policy's class. On the other, OPC's propensity-aware direct optimization of policy value. The 25k result does
  not separate CausE's objective from its selection rule (validation NLL vs the DR lower bound).
- **The current global linear correction is not the final OPC capacity.** It is a first, global layer. Planned
  capacity:
  - global correction;
  - group correction;
  - regional / neighborhood correction;
  - shrunk per-user and per-item vector corrections, where there is enough evidence.

## 8. Unresolved questions

- Is CausE-capacity-matched's lead due to its objective (click likelihood) or its selection (validation NLL)?
- Does the lead survive more data? OPC's share of the linear-repair oracle rises from 0.38 at 25k to 0.49 at 100k.
- Do randomized rows start to pay at a larger N (2–7 uniform rows per item at 100k and ρ = 0.25)?
- Under group and vector bias every method sits far below the target. How much of that is capacity, since the
  linear oracle itself repairs only part of it?
- OPC's DM-only arm is not the strongest no-propensity learner in OPC's class. Every OPC-vs-DM claim should be
  re-checked against an in-class likelihood learner.
- The weighting's regime dependence: harmonic:0.1 vs raw DR depends on how well q̂ is specified.
- Unneeded correction is not free: without bias, OPC, DM-only and CausE-capacity-matched lose 0.36–0.80 points
  against the logger's greedy value.

## 9. Roadmap (high level; not implemented)

Future experiments will compare, **under matched correction capacity**, three ways of learning from logged data:
1. ordinary outcome likelihood;
2. propensity / importance-weighted outcome likelihood;
3. direct OPC policy-value optimization.

They will do so while progressively introducing representation misspecification and richer correction structure:
global, then group, then regional / neighborhood, then shrunk per-user and per-item corrections.

- **Pending decisions** from `docs/cause_fair_comparison_25k.md` §10: a targeted CausE-cap run at 100k, and whether
  to separate objective from selection on the OPC side.
- **Later planned work:** the Bayesian Latent Organic Bandit (BLOB) and RecoGym.

## 10. Where to start

| what | where |
|---|---|
| simulator fix, OPC revalidation | `docs/simulator_fix_opc_revalidation_20261004.md`; `artifacts/full_study/opc_revalidation_20261004/` |
| CausE specification and native reproduction | `docs/cause_baseline.md`; `docs/cause_dev_report_20261004.md`; `artifacts/cause_repro/` |
| fair 25k comparison | `docs/cause_fair_comparison_25k.md`; `artifacts/full_study/cause_fair_25k/` (rebuild: `python -m training.analyze_cause_fair {tune,compare}`) |
| every run (code commit, purpose, pairing) | `artifacts/full_study/run_registry.csv` |
| code map | Code Atlas v10: `docs/opc_code_atlas.pdf` and the published page |
| earlier handoff (before the simulator fix) | `docs/roee_handoff_20260928.md` |
