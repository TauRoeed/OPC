# Code handoff for Roee: CRM from c072f9b to now (2026-09-23 → 2026-09-28)

## Baseline

- **Baseline: `c072f9b`** (2026-09-23 12:55, TauRoeed): "Document sndr default, fixed DR score clip, and batch schedule".
  - It is your latest commit under either author identity (TauRoeed, roeed) on any branch.
  - It is in CRM's history, and all 61 commits after it on CRM are Noam's.
  - There was no earlier handoff document to anchor to.
- **Head:** CRM as of this document. The last content commit before it is `c048e8f`. The handoff and report commits
  follow it; see `git log c072f9b..CRM`.
- **Size:**
  - 132 files changed (+14,942 / −2,678 lines); the core code in `training/`, `models/`, `utils/` and `BPR/` is 35
    of them (+6,162 / −1,932).
  - Tests went from 56 test functions in 6 files to 284 in 38 files. That is 424 collected tests with the GPU;
    with it hidden, 401 pass and 8 are skipped.

**In one paragraph.** The simulator was rebuilt around representation bias:
- clean BPR v2 vectors define a calibrated logistic click model;
- the logger, the reward model and the policy see warped, group-shifted or vector-perturbed copies;
- the logger is sharpened.

The study gained DM-only and tempered-logger baselines, a budget-fair cross-fitted reward model, a learnable logit
scale, paired "replay" sampling, run-stage tags and full seeded reproducibility. The OPC objective was examined and
changed: legacy minibatch SNDR depends on the batch size and was replaced as the working choice by DR,
differentiated directly, with Metelli harmonic weights (λ = 0.1). A short-final-batch bug was fixed. A true-reward
oracle measures what the repair class can recover (Stage 1), and was validated; learned recovery (Stage 2) and
logging support (Stage 3) were measured against it. A memory-aware worker planner fixes a silent WSL2 GPU-memory
spill. Results are in [representation_repair_experimental_report_20260928.md](representation_repair_experimental_report_20260928.md).

## Changes by purpose

Each item lists: what changed; why; what motivated it; the current behaviour; the effect on old results; commits;
paths.

### 1. Simulator: the representation-bias world

- **New world (`9c76ef2`).**
  - **What:** clean vectors define clicks q = sigmoid(α·z + b) on the standardized clean score. α puts each user's
    best item at 30% CTR, and b puts the reference policy at `--ctr-levels`.
  - **Bias:** users and items get warp, group or vector bias, each none / low / medium / high. The three types
    together keep 90 / 75 / 50% of the taste signal, calibrated per dataset and seed. The draws are nested, so the
    truth is identical across bias settings.
  - **Why:** the old ε-mix noise model and CTR-ceiling link were not a controlled representation mismatch.
  - **Current:** the only world (no legacy mode). `--logging-spread` replaces `--policy-temperature`, and
    `WorldCalibrationError` is raised on unreachable targets.
  - **Old results:** runs before `9c76ef2` used a different world and are not comparable.
  - **Paths:** `utils/representation_bias.py`, `utils/simulation_utils.py`, `training/characterize_world.py`,
    `docs/representation_bias.md`.
- **Popularity dial (`44fb306`).** The true score is x·a + β·b, using BPR's item bias b (`--pop-strength`, default
  0: taste only). The logger has its own β (`--logger-pop-strength`). Centering is off by default. With both
  weights at 0 there is no extra column, and runs are identical to taste-only.
- **Logger sharpness (`94404f8`, `e7115f5`).**
  - **Why:** the spread logger earned only 25–35% of what its own ranking could give, so most measured "gains" were
    plain sharpening.
  - **What:** `--logger-greedy-share` sharpens the logger until its CTR is that share of its greedy CTR. The truth
    does not change.
  - **Current:** 0.8. `off` reproduces the older worlds.
  - **Old results:** runs before `94404f8` used the spread logger.

### 2. Data and embeddings

- **Myket loader (`89dbc8f`).** It used row ids as users and users as items, which made the Myket embeddings
  meaningless. Any earlier Myket result is invalid.
- **Side metadata (`ff1644a`).** lastfm and msd get metadata only; interactions and factors are byte-identical.
- **BPR v2 (`edcdec6`, `ce827f5`).** A mini-batch trainer with item bias, early stopping, per-dataset sampling and
  test-set Recall / NDCG / MPR (`BPR/bpr_minibatch.py`, `BPR/evaluate.py`, `BPR/README.md`). It beats the old
  trainer everywhere; the old lastfm and msd embeddings were no better than popularity.
- **Default datasets (`1cd0235`, `36da894`, `7d7caef`).** `BPR.bpr_config.DEFAULT_DATASETS` = ml, myket, kuairec,
  kuairand, anime, msd; lastfm is opt-in. The representation-repair study used ml, kuairand and anime.

### 3. Policy class

- **Linear correction (`b8450c9`, `9105ac4`).**
  - **What:** `--policy-transform linear` (default) repairs both sides as (I + D)x + b, with D and b at zero, so
    the policy starts exactly at the logger.
  - **Why:** the old x + MLP(LayerNorm(x)) started at a random perturbation.
  - **Alternatives:** `linear+mlp` (the MLP sees the raw vector) and `mlp` (the old transform).
  - **Paths:** `models/models.py`.
- **Learnable logit scale (`2a29056`).** `--learn-logit-scale` gives every trained arm a sharpness s, with log s =
  30θ. It is off by default, **but the representation-repair study used it**.

### 4. Study arms and experimental protocol

- **Baselines (`2a29056`).** `--methods` adds `dm` (trained and selected on q̂ alone) and `tempered_logger` (the
  logger's logits × a searched scale). They share splits, reward model, selection and budget. The default is
  `opc no_propensity`.
- **Greedy values (`c1b6dfb`).** Every trial and summary logs the true CTR of each user's top item
  (`actual_reward_greedy`, `policy_rewards_greedy`), which compares rankings without sharpening.
- **Post-hoc tempering (`63a3cc1`).** `--post-temper` picks a trained policy's sharpness after training. Off by
  default and not used in the study.
- **Replay sampler (`7bc3fb7`, `20f8145`, `39657b7`).**
  - **What:** `--sampler random` is seeded random search with no warm start between sizes. Trial k has the same
    configuration and seed in every run with the same grid and seeds, whatever the objective. OPC, no-propensity
    and DM-only share OPC's stream (`seed_label="opc"`), so the arms are paired.
  - **Why:** TPE re-tunes each run independently and warm-starts each size from the previous size's best, so
    objectives and arms could not be compared trial by trial.
  - **Current:** the default is still `tpe`.
  - **Test:** a one-size run reproduces that size's trials of a multi-size run, so slices pair with main runs
    (`tests/test_replay_mode.py`).
- **Run stage (`7bc3fb7`).** `--stage development | confirmatory` is recorded everywhere. Everything so far is
  development.

### 5. OPC objective and normalization

- **Legacy minibatch SNDR (`b584edc`, `b5efc7d`).**
  - **The issue:** the loss divided the correction by the minibatch mean weight. Its objective therefore changed
    with the batch size, which Optuna searches: tuning the batch also tuned the estimator.
  - **Replacement:** `dr` = DM(q̂) + w(r − q̂), with no self-normalization. Averaged over equal-size batches its loss
    and gradient equal the full-data ones exactly, and it matches the DR selection estimator.
  - **Also kept:** `sndr --sn-scope global` divides by a full-data mean weight fixed per epoch. It is a
    stop-gradient, epoch-stale surrogate, not exact SNDR, and exact SNDR was not implemented.
  - **Where it stands:** legacy SNDR stays available (`--policy-losses sndr`).
  - **Evidence:** on identical configurations, legacy SNDR trailed `dr` by 0.03–0.32 points per trial, and global
    SNDR by 0.02–0.26, with 6 of 6 cells excluding 0 (report Table 7).
  - **Paths:** `models/custom_losses.py`, `docs/training_losses.md` §3.4.
- **Short final batch (`b5efc7d`).**
  - **The bug:** the DataLoader keeps the short last batch, and its mean counted like a full batch's. That
    upweighted its rows by b/rows (1.1×–4.8× in the grid).
  - **The fix:** per-example losses (dr, global SNDR/KL, DM, naive, IPW) scale it by rows/b
    (`training_utils.minibatch_loss`). Legacy SNDR and CRM keep one mean per batch.
  - **Old results:** runs before `b5efc7d` are "pre-fix" for DM-only and no-propensity (their last batch only) and
    for any `dr` run.
- **Working defaults (`f5cade9`).** OPC = `dr`, `--opc-gradient direct`, `--train-weights harmonic:0.1`, selection
  `clip:10`.
  - These are the working development choice, not the final paper choice.
  - The previous defaults reproduce with `--policy-losses sndr --sn-scope batch --opc-gradient log-trick
    --train-weights shrink:100`.
  - The H1 runner keeps sndr / log trick / shrink:100 pinned.
  - Paths: `training/run_full_study.py` (`STUDY_*`), `docs/decision_record_opc_objective_weighting.md`.

### 6. Gradient handling

- **`--opc-gradient {log-trick, direct}` (`3bed8de`).**
  - **The issue:** with a weight transform g, the log trick puts g(w) as a detached coefficient on ∇ log π. It
    therefore follows the gradient of DM + H(w)(r − q̂), with H(w) = ∫₀ʷ g(t)/t dt, not of the named estimate.
    For clip:M, H = w up to M, then M(1 + ln(w/M)); for shrink:λ, H = √λ·arctan(w/√λ).
  - **What:** `direct` differentiates the transformed estimate itself.
  - **Current:** the study default is `direct`, and it is required with harmonic weights.
  - **Also:** `--no-log-trick` never reached OPC in the full study; this is documented, not changed.
  - **Paths:** `models/custom_losses.py`, `tests/test_opc_gradient.py`.

### 7. Importance-weight transforms

- **One transform everywhere (`33f1260`).** `none` / `clip:M` / `shrink:λ` (Su et al. 2020) apply in training,
  selection and post-hoc estimates. Trials log the raw-weight ESS (`ess_raw`). `--log-select-weights` logs each
  trial's selection score under other transforms (`utils/importance_weights.py`).
- **Harmonic (`d40aaef`).** `harmonic:λ` is Metelli, Russo & Restelli (NeurIPS 2021): w / (1 − λ + λw), with
  w_λ ≤ 1/λ, checked against the paper and the authors' code. It requires `--opc-gradient direct`. An unknown mode
  now raises instead of silently using raw weights.
- **Defaults (`d7a434c`, `f5cade9`):**
  - training `harmonic:0.1`: working, provisional;
  - `shrink:100`: the prespecified comparison;
  - `none`: the reference;
  - selection `clip:10`: tuned on the old spread logger, interim.

### 8. Reward model and cross-fitting

- **Interaction features (`b953d89`).** q̂ is logistic on [x, a, x·a], where the old [x, a] ranked items the same
  way for every user. `--reward-features concat` is the old model and serves as the "misspecified q̂" test
  (`run_qhat_concat`). There, DM-only lost 1.7–9.1 points and OPC's change was within noise (report §F).
- **Budget-fair q̂ (`4f85d48`, `66fd303`).**
  - **What:** `--reward-data train` fits q̂ on each train size's own rows.
  - **Why:** the old default fit one q̂ on an extra 50k-row slice, which only the arms that use q̂ (DM-only, OPC)
    benefited from.
  - **Current:** the study default. `external` restores the old behaviour.
- **Cross-fitting (`bbbc9c2`).** `--crossfit-folds K`: the training losses take each user's q̂ from the fold model
  that never saw that user's rows. Selection uses the full model. The default is 5 with `train`.
- **Validation (`66fd303`).** `--val-size 20000` by default (DR standard error ≈ 0.5–0.6 CTR points).
- **Fixes:**
  - Post-hoc estimates now feed q̂ its own user vectors (`fdbb9b4`).
  - The single-class fallback now predicts p, not 0 (`a01e630`); it never triggered in study runs.
  - Vectorized q̂, bit-identical (`52a9394`).

### 9. Selection and validation

- **Unchanged:** the DR lower bound on validation, with clip:10 weights.
- **Diagnostics:**
  - every arm logs the selection variants (`23e634b`, `76a29e6`: the no-propensity forwarding fix);
  - OPC trials log the weight diagnostics `diag_*`: the largest weight, the shares above 1 / 10 / 100, and the DM
    and correction parts (`8d0fd21`).

### 10. Representation-repair experiments: oracle and recoverability machinery

- **Oracle repair (`af34cf0`).**
  - **What:** the learner's own class (linear repair ± learned scale) is trained on the true click model. It
    measures the structural limit, separating structural from statistical failure.
  - **Paths:** `training/oracle_repair.py`. `build_condition_world` is shared with the study, so the worlds are
    identical.
- **Tables (`20f8145`, `5b66e18`, `d43ef26`, `9925205`).** `training/analyze_recoverability.py` has CLI modes
  `stage1` / `stage2` / `followup`:
  - recoverability;
  - learned fraction of the oracle repair;
  - paired arm differences;
  - per-dataset views;
  - the gap decomposition.
- **Stage 1–3 runs and outputs (`fc07416`, `d7bc67b`, `1c85940`).**
  - Stage 1 is the oracle (36 worlds).
  - Stage 2 is four arms × 5k / 25k / 100k × 3 datasets × 6 bias settings × 2 seeds, plus the shrink:100 slice.
  - Stage 3 is the logger-share sweep 0.6 / 0.8 / 0.95.
- **Oracle validation (`67c3b4a`, `e7beafa`).**
  - **Why:** the scale-fixed class won at the top of its learning-rate grid in 35 of 36 worlds.
  - **What:** `training/oracle_validation.py` refits the same class past that edge, with 3× and 9× the budget and a
    27× tail check (`--refit`).
  - **Result:** the bounds rise by 0.005–0.010 on average, which is not material.
- **Report (this handoff).** `training/representation_report.py` builds every figure and table of the experimental
  report from the committed summaries.

### 11. Reproducibility and run metadata

- **One seed per condition (`9036687`).** `utils/seeding.py`: derived sub-seeds, a seeded Optuna sampler, seeded
  torch/numpy per trial, and deterministic cuDNN/cuBLAS. `--deterministic` (on) and `--cpu-threads` (4) are
  recorded; the thread count changes BLAS summation order.
- **Sampling (`0d1ffb6`, `69fffab`).** Logged-data sampling is identical on CPU and GPU for a seed. The draws
  changed once relative to older runs; the distribution did not.
- **Metadata:** every flag above goes to `run_meta.json`, the summaries and the manifest.
- **Metadata fix (`e57d171`).** Runs with `--methods dm` or `tempered_logger` recorded `bias_label` and
  `noise_level` as "tempered_logger". Folder names were right and the analyses read folder names. Old
  `run_meta.json` files keep the wrong label.
- **Run registry and summaries (`fc07416`).**
  - `artifacts/full_study/run_registry.csv` lists 43 runs, each with its code commit and notes, and is tracked.
  - The raw run folders stay local (`.git/info/exclude`).
  - The compact tables are in `artifacts/full_study/summaries_20260927/`.

### 12. Speed, GPU and parallelism

- **Performance (`980eb31`, `8b120e6`, `7d7caef`):**
  - exact policy values on the GPU (msd 1.6 h → 2 s);
  - batched DataLoader fetch;
  - trial scoring on the GPU (trials 1.5–7× faster).
  - They are equivalent to the old code to ~1e-5 and select the same trials.
  - Environment switches `OPC_EXACT_REWARD_DEVICE`, `OPC_SAMPLER_DEVICE` and `OPC_SCORING_DEVICE` force the CPU.
- **Worker planner (`a9ba1e9`, `f2e03b5`, `05222f8`).**
  - **The issue:** under WSL2, an oversubscribed GPU does not OOM. It spills into shared system memory, and
    everything silently slows about 50× (100% utilization at about 130 W).
  - **What:** the planner estimates each condition's peak: the training step at its largest batch, its dense q̂
    copies (3 with cross-fitting and a per-size refit), and 1.5 GiB. Each GPU then takes
    floor(0.75 × free / peak) workers, capped by `--max-workers`. anime gets 2 workers on a 48 GB card.
  - **Scheduling only.** `training/memory_budget.py`, README.

### 13. Correctness fixes (summary)

- `f188545`: `_probs_block` normalized the softmax per 8,192-item chunk. Only the `scripts/sim_dr_*` sweeps used
  it, on MovieLens, which fits in one chunk.
- `a01e630`: single-class fallback.
- `fdbb9b4`: post-hoc reward-model inputs.
- `76a29e6`: no-propensity selection logging.
- `b5efc7d`: short final batch.
- `3bed8de`: `--no-log-trick` no-op, documented.
- `e57d171`: bias label.
- `d40aaef`: unknown weight mode now raises.

### 14. Testing and safety checks

- **Regression tests for silent failures (`e02526e`).** They cover the q̂ fast paths against sklearn, that
  sampling matches the policy's probabilities, and seeded reproducibility. `pytest` is in `requirements.txt`.
- **One test per change.** Each change since then carries a test, checked by reintroducing the bug (mutation
  checks). Among them:
  - every objective's gradient against literal full-data autograd;
  - the log-trick H(w) closed forms;
  - the harmonic formula and its gradient bound;
  - short-batch weighting with the real DataLoader;
  - replayed and paired trials;
  - one-size vs multi-size pairing;
  - the oracle starting exactly at the logger;
  - the planner's q̂ copies against a live run;
  - the report rebuilding from committed tables.
- **Running the suite:** the full suite runs in about 3.5 minutes, both with the GPU and with
  `CUDA_VISIBLE_DEVICES=""`.

### 15. Analysis scripts, reports and docs

- **Loss and gradient reference:** `docs/training_losses.md` (§3.4 what each objective optimizes; §9 the objective
  and weighting evidence).
- **Decision record:** `docs/decision_record_opc_objective_weighting.md`.
- **Representation repair:** `docs/representation_repair_dev_20260927.md` (Stages 1–3) and
  `docs/representation_repair_followup_20260927.md` (validation, per-dataset views, gaps).
- **Experimental report:** `docs/representation_repair_experimental_report_20260928.md`, with figures in
  `artifacts/full_study/report_20260928/`.

## Commit timeline (secondary)

| day | commits | purpose |
|---|---|---|
| 09-23 | `89dbc8f` `857cd91` `ff1644a` `980eb31` | Myket fix, README, metadata, exact values on GPU |
| 09-24 | `9036687` `52a9394` `0d1ffb6` `f188545` `a01e630` `8b120e6` `e02526e` `69fffab` `a9ba1e9` `fdbb9b4` | reproducibility, speed-ups, correctness fixes, regression tests, memory cap, post-hoc inputs |
| 09-25 | `9c76ef2` `1cd0235` `edcdec6` `44fb306` `36da894` | the representation-bias world, BPR v2, popularity dial, default datasets |
| 09-26 | `7d7caef` `ce827f5` `b953d89` `33f1260` `b8450c9` `9105ac4` `8d0fd21` `94404f8` `d7a434c` `2a29056` `23e634b` `4f85d48` `76a29e6` `bbbc9c2` `66fd303` `e7115f5` `c1b6dfb` `63a3cc1` `b584edc` | GPU scoring; BPR metrics; interaction q̂; weight transforms; linear policy transform; logger sharpness; DM-only and tempered arms; budget-fair and cross-fitted q̂; greedy values; post-hoc tempering; `dr` |
| 09-27 | `b5efc7d` `7bc3fb7` `3bed8de` `d40aaef` `1c275dd` `f5cade9` `e57d171` `af34cf0` `20f8145` `5b66e18` `f2e03b5` `d43ef26` `24a5805` `39657b7` `05222f8` `fc07416` `d7bc67b` `e750081` `67c3b4a` | objective variants and short batch; replay and stage; gradient form; harmonic; decision record and working defaults; oracle, Stages 1–3; worker planner; registry and summaries; oracle validation |
| 09-28 | `e7beafa` `9925205` `1c85940` `c048e8f` (+ this handoff) | validation tail, follow-up tables, docs |

## What Roee needs to know before modifying the code

**Defaults: provisional vs intended.**
- **Provisional:** OPC = `dr` + `--opc-gradient direct` + `--train-weights harmonic:0.1` is the working development
  method, not the paper's choice. Selection `clip:10` was tuned on the old spread logger and is interim.
- **Study defaults:** logger share 0.8, `--reward-data train` with 5-fold cross-fitting, and `--val-size 20000`.

**CLI defaults that differ from what the studies used.**
- `--sampler tpe`. Paired studies need `--sampler random`.
- `--learn-logit-scale` is off. The study passed it.
- `--methods opc no_propensity`. DM-only and the tempered logger are opt-in.
- `--seeds 0 1 2`. Development used 100/101; confirmatory runs need fresh seeds.
- `--slim` is off, which means the full post-hoc evaluation, and slower.
- `--train-sizes` includes 50k.

**`_run_condition` has its own defaults for direct callers.** It defaults to `reward_data="external"`,
`crossfit_folds=0`, `sampler="tpe"` and `learn_logit_scale=False`, not the CLI's budget-fair setting. The trainer
API's own fallback weights are `shrink:100` (`DEFAULT_TRAIN_WEIGHTS`). Pass everything explicitly when calling
these directly.

**Paired studies.** Use `--sampler random` and the same grid and seeds; only then is trial k identical across runs
and across OPC / no-propensity / DM-only. One-size slices pair with multi-size runs.

**Old TPE runs.**
- Each size warm-starts from the previous size's best, so sizes are not independent.
- Arms use their own streams, so they are unpaired.
- Do not read trial-level objective or arm comparisons, or clean size trends, from them.

**Pre-fix and older runs** (`run_registry.csv` has each run's code commit):
- Before `9c76ef2`: a different world.
- Before `94404f8`: the spread logger.
- Before `b5efc7d`: short-batch upweighting (DM-only, no-propensity and `dr`).
- Before `f5cade9`: legacy SNDR, log trick, shrink:100.
- Before `9036687`: not reproducible.
- Before `69fffab`: different sampling draws.
- Before `e57d171`: wrong `bias_label` in `run_meta.json` when the DM or tempered arms ran.

**Issues that can silently distort an experiment:**
- **WSL2 GPU oversubscription** means a silent ~50× slowdown with no error. The planner prevents it, but it does
  not plan host RAM when GPUs are present; an anime worker holds about 14.5 GiB of RAM.
- **One non-finite gradient** (`clip_grad_norm_(error_if_nonfinite=True)`) fails the whole condition. It is loud
  rather than silent, but costs the condition.
- **Result dependencies:**
  - Results depend on `--cpu-threads`, through BLAS summation order; keep 4 for comparability.
  - `OPC_QHAT_MATERIALIZE_MAX_GB` switches q̂ between dense and lazy forms.
- **The no-propensity arm's selection score** is about 15 points below the truth. It only ranks its own trials; do
  not read it as an estimate.
- **The oracle bound is a lower bound** on the class optimum (validated to within about 0.01).
- **Linear+scale fits destabilize at long schedules,** as the learned scale runs away. The pooled bound keeps its
  best fit.

**Guards that exist.** Harmonic without `direct` raises. An unknown weight mode raises. Unreachable world targets
raise. Skip-completed checks that every requested method is present.

**Local state not in the repo.** A stash from 2026-09-24 ("issue1: reward features (interaction/concat) …"), which
looks superseded, and the ignored raw run folders.

**CausE and BLOB are not in this repository.** Any audits of them are external, standalone work. Nothing here
integrates or depends on them.
