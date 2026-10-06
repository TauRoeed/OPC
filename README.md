# OPC Repository

Offline policy comparison experiments with matrix-factorization embeddings.

**Start here (2026-10-06):** [`docs/handoff_20261006.md`](docs/handoff_20261006.md): the branch map, what is current,
the headline 25k results, the code review and its fixes, and the open decisions. What each compared arm optimizes:
[`docs/training_objectives_audit.md`](docs/training_objectives_audit.md). Rerunning, resuming and extending a run:
[below](#rerunning-resuming-and-extending-a-run-since-2026-10-06).

**Loss / Optuna details:** [`docs/training_losses.md`](docs/training_losses.md)  
**Research notes:** [`docs/research_workplan.md`](docs/research_workplan.md)  
**Simulator:** [`docs/representation_bias.md`](docs/representation_bias.md)  
**Simulator fix and OPC revalidation (current results; old vs corrected, 2026-10-04):** [`docs/simulator_fix_opc_revalidation_20261004.md`](docs/simulator_fix_opc_revalidation_20261004.md) (summaries, figures and tables: `artifacts/full_study/opc_revalidation_20261004/`)  
**Experimental report, historical (generated on the buggy simulator, 69fffab..c11b2b3; superseded by the revalidation):** [`docs/representation_repair_experimental_report_20260928.md`](docs/representation_repair_experimental_report_20260928.md) (figures and tables: `artifacts/full_study/report_20260928/`)  
**Code handoff since c072f9b (for Roee):** [`docs/roee_handoff_20260928.md`](docs/roee_handoff_20260928.md)  
**Representation repair, stage write-ups:** [`docs/representation_repair_dev_20260927.md`](docs/representation_repair_dev_20260927.md) · follow-up: [`docs/representation_repair_followup_20260927.md`](docs/representation_repair_followup_20260927.md)  
**Objective and weighting decision record:** [`docs/decision_record_opc_objective_weighting.md`](docs/decision_record_opc_objective_weighting.md)  
**Code Atlas (PDF snapshot of the published page, version 14, branch `handoff-20261006`, code as tested at d704f03; exported by `scripts/export_atlas_pdf.py`):** [`docs/opc_code_atlas.pdf`](docs/opc_code_atlas.pdf)  
**CausE baseline (specification, OPC mapping, budget protocol):** [`docs/cause_baseline.md`](docs/cause_baseline.md)  
**CausE vs OPC, bounded development comparison (report before scaling):** [`docs/cause_dev_report_20261004.md`](docs/cause_dev_report_20261004.md)  
**CausE vs the revalidated OPC, the fair 25k comparison (CausE-warm, CausE-capacity-matched):** [`docs/cause_fair_comparison_25k.md`](docs/cause_fair_comparison_25k.md)  
**Handoff and checkpoint before the next representation-mismatch phase (branch map, status, open questions, roadmap):** [`docs/representation_mismatch_handoff_20261005.md`](docs/representation_mismatch_handoff_20261005.md)  
**BLOB in the controlled environment (the published model, its reproduction, BLOB-supplied-source vs the likelihood learner, DM-only and OPC at 25k):** [`docs/blob_controlled_integration.md`](docs/blob_controlled_integration.md)  
**BLOB's catalog-size prior: the derivation, the pre-registered calibration and the 25k check of BLOB-Pnorm:** [`docs/blob_prior_calibration.md`](docs/blob_prior_calibration.md)  
**What BLOB-NQ, CausE-capacity-matched (ρ = 0) and OPC each optimize (audit of the executable objectives):** [`docs/training_objectives_audit.md`](docs/training_objectives_audit.md)  
**Handoff and status, 2026-10-06 (branch map, results, the code review and its fixes, verification, open decisions):** [`docs/handoff_20261006.md`](docs/handoff_20261006.md)

Main flow:
1. Fit/generate BPR artifacts (user/item factors + metadata arrays).
2. Run OPC vs no-propensity on simulated logged bandit data.
3. Analyze under `artifacts/full_study/...`.

## Current defaults (full study)

| Knob | Default |
|------|---------|
| OPC train loss (`--policy-losses`) | `dr` (DM + weighted correction, no self-normalization): the working development default since f5cade9 (2026-09-27), revalidated on corrected logs (2026-10-04: exact SNDR, `--sn-scope exact`, adds nothing over DR with harmonic weights; `docs/simulator_fix_opc_revalidation_20261004.md` §2.8). Not the final paper choice. Legacy `sndr --sn-scope batch` (the default before f5cade9) and `--sn-scope global` stay for reproducibility and diagnostics |
| OPC gradient (`--opc-gradient`) | `direct`: the named estimate is the objective optimized; `log-trick` (the default before f5cade9) reproduces older runs |
| Policy transform (`--policy-transform`) | `linear`: (I + D) x + b per side, starting exactly at the logger |
| No-propensity train | always `naive` (no IW, no DM/DR, no clip) |
| Optuna objective | `ci_low` = DR/naive mean − t·SE |
| Importance weights | `--train-weights` default `harmonic:0.1` (Metelli et al. 2021), re-tuned on corrected logs against raw, clip, Su shrinkage and harmonic grids (2026-10-04, revalidation §2.7): the working development default, not the final paper choice. `shrink:100` (Su et al. 2020) is a robustness alternative (within noise of the default on the whole Stage 2 grid), and `none` (raw DR) the unregularized reference. **Regime dependence:** with a misspecified reward model (`--reward-features concat`) harmonic:0.1 and shrink:100 lose about 4.5 points at 100k while raw DR loses nothing; raw DR costs 0.77 points with a well-specified q̂ (revalidation §3C). Use `--train-weights none` as well wherever q̂ may be misspecified. `--select-weights` (selection + post-hoc) keeps `clip:10` with the 95% lower bound (revalidated post hoc). Neither is Optuna-searched. `--policy-losses sndr --sn-scope batch --opc-gradient log-trick --train-weights shrink:100` reproduces the defaults before f5cade9; the H1 runner keeps those settings |
| Study arms (`--methods`) | `opc no_propensity`; opt-in baselines `dm` (policy trained and selected on `q̂` alone) and `tempered_logger` (the logger's logits × s, s chosen by the DR score); opt-in prior-work baselines `cause` (CausE: `--cause-family native\|warm\|cap`, one row per prediction and ρ) and `blob` (BLOB-supplied-source: `--blob-families nq mnq`, `--blob-variants released L<P0>`); see "Prior-work baselines" below |
| Logit scale (`--learn-logit-scale`) | off; on, every trained policy also learns s in softmax(s·u·a/T), starting at 1. The study configuration turns it on (revalidated: a fixed scale loses 0.2–1.3 points; post-hoc tempering on top adds nothing to the selected policy) |
| Policy search space (`--lr-range`, `--epochs-range`, `--lr-decay-range`, `--weight-decay-range`) | the code default keeps the older range, lr 1e-4–1e-3 (log-uniform), epochs 5–25, lr decay 0.8–1, no weight decay, so that older commands reproduce. The revalidated study configuration widens it to **lr 1e-4–2e-3, epochs 5–30** (revalidation §2.3; the old optimum sat at the range's edge); AdamW decay is available but off (§2.4). Every run records its space in `run_meta.json` (`search_space`) |
| Reward model `q̂` | `regression` on interaction features `[x, a, x⊙a]` (`--reward-features`; bias script often uses `logging_score`) |
| Reward-model data (`--reward-data`, `--crossfit-folds`) | `train`: fit on each train size's own training rows (the same budget as the policy), cross-fitted by user in 5 folds; `external` = a separate 50k-row slice (runs before 2026-09-26) |
| Validation (`--val-size`) | 20,000 logged rows at every train size; `0` = the older rule `clamp(0.15·n, 5000, –)` |
| Datasets (`--datasets`) | `ml myket kuairec kuairand anime msd`; lastfm is opt-in (a condition costs ~75× ml's; see Runtime estimate) |
| Representation bias (`--bias-configs`) | `low medium high` (all three types at that level) |
| Reference CTR (`--ctr-levels`) | 5% for the spread logger at medium bias; best item 30% |
| Logging temperature | the spread logger (clean logger over 50% of the catalog, `--logging-spread`) calibrates the click model; the actual logger is sharpened per condition to earn 80% of its own greedy CTR (`--logger-greedy-share 0.8`; `off` = the spread logger, as before 2026-09-26) |
| Batch sizes | `batch_schedule(train_size)` unless you pass `--optuna-batch-sizes`. The last minibatch of an epoch is usually short; for per-example additive losses (`dr`, `--sn-scope global`, `dm`, `naive`) it counts in proportion to its rows, so every row weighs the same (legacy SNDR keeps one mean per batch) |
| Optuna sampler (`--sampler`) | `tpe`; `random` = seeded random search without warm starts: the same trial configurations and seeds in every run of the grid, so runs that differ only in the objective are a paired (replayed) comparison |
| Run stage (`--stage`) | `development`, recorded in `run_meta.json`, the summaries and the manifest; `confirmatory` for the frozen method on fresh seeds and conditions |
| Skip finished cells | `--skip-completed` on (checks `summary_metrics.csv` only) |

**Revalidated study configuration (2026-10-04, corrected simulator).** The Stage 2 / Stage 3 reruns used, on top of the
defaults above (and `--stage development`, `--n-trials 20`, `--slim`):

```text
--policy-losses dr --opc-gradient direct --train-weights harmonic:0.1 --select-weights clip:10
--learn-logit-scale --sampler random --lr-range 1e-4 2e-3 --epochs-range 5 30
--methods opc dm no_propensity tempered_logger
(robustness alternatives: --train-weights shrink:100; --train-weights none where q̂ may be misspecified)
```

with the budget-fair reward model (`--reward-data train`, 5-fold cross-fitting, interaction features), 20,000
validation rows and logger share 0.8 (all defaults). Each run's exact flags, code commit and search space are in its
`run_meta.json` and in `artifacts/full_study/run_registry.csv`.

Offline clip pick: `scripts/sim_dr_score_clip_logging_score.py` → `artifacts/oom_smoke/dr_score_clip_logging_score`.

## Repository Layout

- `BPR/` - BPR training and artifact generation.
- `training/` - experiment runners and trainers.
- `models/` - model definitions and estimators.
- `utils/` - simulated world (representation bias), policy, SNR, plotting.
- `datasets/` - dataset files (MovieLens, etc.).
- `artifacts/` - generated outputs.
- `scripts/` - bias trial, runtime estimate, clip sweeps, profiling.
- `docs/` - paper outline, workplan, simulator/regime/bias notes.

## Quick Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or from scratch (venv + BPR + parallel study):

```bash
./scripts/run_from_scratch.sh
```

### Docker

```bash
docker build -t opc .
# GPU image example:
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t opc:gpu .

docker run --rm -it \
  --gpus all --shm-size=128g \
  -v "$PWD:/app" -w /app \
  --entrypoint bash \
  opc:gpu
```

Image entrypoint is `python`; pass `-m training...` as args, or override with `--entrypoint bash`.

## 1) BPR Fitting / Artifact Generation

Use `BPR/generate_artifacts.py`. Datasets auto-download on first load (`--download` default).
BPR v2 trains mini-batches with an item bias and stops early on held-out recall@20, then refits on
all interactions; settings live in `BPR/bpr_dataset_config.json` and are explained in
[`BPR/README.md`](BPR/README.md).

### Example: MovieLens 1M

```bash
python -m BPR.generate_artifacts \
  --dataset ml \
  --root datasets/ml-1m \
  --emb-dir BPR/embeddings
```

Fresh clone (all seven datasets; the launch scripts do the same). The embeddings are stored in
`BPR/embeddings`, so this is a one-off: about 2 hours in total, most of it lastfm (~1 h), msd
(~45 min) and anime (~15 min); the other four take under a minute each.

```bash
python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m
python -m BPR.generate_artifacts --dataset myket --root datasets/myket
python -m BPR.generate_artifacts --dataset kuairec --root datasets/kuairec
python -m BPR.generate_artifacts --dataset kuairand --root datasets/kuairand-pure
python -m BPR.generate_artifacts --dataset anime --root datasets/anime
python -m BPR.generate_artifacts --dataset lastfm --root datasets/lastfm/lastfm_360k.hdf5
python -m BPR.generate_artifacts --dataset msd --root datasets/msd/msd_taste_profile.hdf5
```

lastfm / msd also fetch side metadata (user profiles, artist genres and tags; ~1.2GB); see `BPR/README.md`.

Disable auto-download: add `--no-download`.  
Smoke test: `python -m BPR.smoke_test_loaders --include-large`.  
Test-set quality of the recipes (Recall@5/@20, NDCG, MPR): `python -m BPR.evaluate`; results in `BPR/README.md`.

### Important Arguments

- `--dataset`: `ml`, `myket`, `anime`, `lastfm`, `msd`, `kuairec`, `kuairand`.
- `--root`: dataset path (for `lastfm` / `msd`, the loader file path).
- `--emb-dir`: output directory for `.npy` artifacts.

### Produced Files

For each dataset `<name>`:

- `BPR/embeddings/<name>_user_factors.npy`
- `BPR/embeddings/<name>_item_factors.npy`
- `BPR/embeddings/<name>_item_bias.npy` (BPR v2 with the item bias)
- `BPR/embeddings/<name>_bpr_meta.json` (settings, validation curve, git commit; runs warn when it is missing or stale)
- `BPR/embeddings/<name>_item_metadata.npy`
- `BPR/embeddings/<name>_user_metadata.npy` (if available)
- `BPR/embeddings/<name>_user_interaction_counts.npy`

## 2) Run Experiments (OPC vs No-Propensity)

- Single process: `training/run_full_study.py`
- Parallel: `training/run_full_study_parallel.py`

OPC trains with `dr` (direct gradient, `harmonic:0.1` training weights) by default. Trial selection uses the DR 95%
lower bound with weights clipped at 10 (`--select-weights clip:10`). No-propensity stays pure naive.

### Simulated world

Each condition builds a world from the dataset's BPR vectors
([`docs/representation_bias.md`](docs/representation_bias.md)):

- **Truth.** The clean score is BPR's taste score `x·a`, plus `β·b_i` with its item bias when
  `--pop-strength β` is set (default 0: personal taste only). Clicks follow `sigmoid(α·z + b)`
  on the standardized clean score `z`: α puts each user's best item at 30% on average, and b
  puts the logger at medium bias at 5% CTR.
- **What the learner sees.** Biased copies of users and items, with three types applied in
  order: a global warp, a group offset (k-means clusters, or metadata groups) and a
  per-vector offset. Each type is `none` / `low` / `medium` / `high`. All three at
  low / medium / high keep 90 / 75 / 50 % of the signal (calibrated per dataset).
- **Logger.** `softmax(biased scores / T)`, with T set so the clean logger spreads over half
  the catalog. `--logger-pop-strength` gives the logger its own weight on the item bias
  (default: the truth's).

The truth is identical across bias configurations for a dataset and seed. Every
calibrated value is written to `run_meta.json → world`. Inspect a dataset with
`python -m training.characterize_world --datasets ml` (writes `artifacts/world/summary.csv` and `calibration.json`;
the worlds are built exactly as the study builds them, `--logger-greedy-share` included).

### Small Local Run

```bash
python -m training.run_full_study \
  --datasets ml \
  --bias-configs low medium high \
  --ctr-levels 0.05 \
  --seeds 0 1 \
  --train-sizes 5000 25000 \
  --n-trials 10 \
  --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study \
  --run-tag demo_ml
```

Batch comes from `batch_schedule` (omit `--batch-size` / `--optuna-batch-sizes` unless you want to override).

### Parallel Sweep

```bash
python -m training.run_full_study_parallel \
  --datasets ml kuairec \
  --bias-configs low high high/none/none none/none/high \
  --ctr-levels 0.05 \
  --seeds 0 1 2 \
  --train-sizes 5000 25000 50000 \
  --n-trials 20 \
  --max-workers 4 \
  --num-gpus 1 \
  --slim \
  --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study \
  --run-tag sweep_v1
```

Docker one-liner (4 GPUs / 120 workers example):

```bash
docker run --rm --gpus all --shm-size=128g \
  -v "$PWD:/app" -w /app opc:gpu \
  -m training.run_full_study_parallel \
  --datasets ml --bias-configs low medium high \
  --train-sizes 500000 1000000 2000000 5000000 10000000 \
  --val-size 100000 --seeds 0 1 2 3 4 --n-trials 15 \
  --reward-model logging_score --slim \
  --num-gpus 4 --max-workers 120 --require-cuda --skip-completed \
  --run-tag my_bias_run
```

### Bias type trial

Large ML bias grid (sndr + `logging_score` + fixed clip M=1): each bias type alone and all
three together, per level ([`docs/bias_axis_trial.md`](docs/bias_axis_trial.md)):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 MAX_WORKERS=120 \
  BIAS_LEVELS="low medium high" \
  ./scripts/run_bias_axis_comp_large_trial.sh
```

Set `IMAGE=opc:gpu` to launch inside Docker (script uses `nohup`).

### Runtime estimate

```bash
python -m scripts.estimate_study_runtime --train-size 1000000 --n-trials 20 --methods 2
```

Wall scales roughly with `n_trials × (train_size / batch)` using `batch_schedule`. Order-of-magnitude only (±2×).
The estimate counts logged samples, batch size and trials but not the catalog, so it is far too
low on msd and lastfm. Measured on one RTX 6000 Ada (48 GB), default study settings (train sizes
5k–100k, 20 trials per train size, both methods), one worker:

| Dataset | Per condition | Default study (3 bias levels × 10 seeds) | GPU per worker |
|---|---|---|---|
| ml | 5.5 min | 2.8 h | 2 GB |
| myket | 8.3 min | 4.1 h | 3 GB |
| kuairand | 8.0 min | 4.0 h | 4 GB |
| kuairec | 8.7 min | 4.4 h | 4 GB |
| anime | 8.3 min | 4.2 h | 7 GB |
| msd | 32 min | 16 h | 10 GB |
| lastfm (opt-in) | ~6.8 h | ~200 h | 15–46 GB |

Parallel workers divide the wall time. A trial trains the policy (SGD on mini-batches of logged
samples; each step takes a softmax over the whole catalog for every user in the batch, so memory
grows with batch × catalog), computes its exact value, and scores it on the logged train and
validation rows; the scoring runs on the GPU. On lastfm a batch of 8192 fills the 48 GB GPU and
runs ~5× slower per epoch than 4096.

### Useful Flags

- `--policy-losses dr` (default) — OPC train loss; `sndr` (with `--sn-scope batch` or `global`) for the legacy objectives;
  `sndr --sn-scope exact` is the gradient of the full-data SNDR ratio (its normalizers refreshed each epoch; needs
  `--opc-gradient direct`).
- `--lr-range LOW HIGH`, `--epochs-range LOW HIGH`, `--lr-decay-range LOW HIGH` — the policy search space shared by
  every trained arm (defaults 1e-4 1e-3, 5 25, 0.8 1; the revalidated study configuration uses `--lr-range 1e-4 2e-3
  --epochs-range 5 30`). `--weight-decay-range LOW HIGH` adds AdamW decay drawn per trial (needs `--sampler random`).
- `--policy-transform {linear,linear+mlp,mlp}` — how the learned policy corrects the biased vectors: `linear`
  (default) = (I + D) x + b per side, starting exactly at the logger; `mlp` = x + MLP(LayerNorm(x)) (older);
  `linear+mlp` = (I + D) x + b + MLP(x), no LayerNorm or dropout.
- `--train-weights`, `--select-weights` — importance-weight transform (`none`, `clip:M`, `shrink:λ` of Su et al.
  2020, `harmonic:λ` of Metelli et al. 2021 = w / (1 − λ + λw), trained with `--opc-gradient direct`) in the OPC
  training losses and in selection + post-hoc estimates; `--log-select-weights` logs other selection transforms
  per trial (tuning). `--train-weights none --select-weights clip:1` reproduces older runs.
- `--methods opc no_propensity dm tempered_logger` — adds the DM-only and tempered-logger baselines (same
  splits, reward model and selection weights); `--learn-logit-scale` lets every trained policy learn a logit scale;
  `--post-temper` instead chooses each trained policy's sharpness after training (logits × s, s on validation).
- `--sampler random` — replay mode: seeded random search with no warm start between train sizes, so trial k has
  the same configuration and the same trial seed in every run with the same grid and seeds. Run one objective per
  run tag (e.g. `--policy-losses dr`, `--policy-losses sndr --sn-scope global`) and compare trial by trial.
- `--stage {development,confirmatory}` (default `development`) — what the results are for, recorded with them.
- `--opc-gradient {log-trick,direct}` (default `direct`, since f5cade9 (2026-09-27)) — how OPC's loss is differentiated. The log trick
  follows the gradient of DM + H(w)(r − q̂) with H(w) = ∫₀ʷ g(t)/t dt. `direct` is the exact gradient of the
  transformed estimate DM + g(w)(r − q̂). They coincide for `--train-weights none`
  (`docs/training_losses.md` §3.4).
- `--logger-greedy-share` (default 0.8; `off` = the older spread logger) — logger sharpness: the logger earns this
  share of its own greedy CTR. Sharper loggers leave less room the logs can evaluate (see the simulator doc).
- `--bias-configs` — levels per condition: `medium` (all three types) or `warp/group/vector`, e.g. `high/none/low`.
- `--pop-strength` (default 0: clicks follow taste only), `--logger-pop-strength` (default: the same) — weight of BPR's item bias in the true score and in the logger's; needs `{dataset}_item_bias.npy`.
- `--bias-groups {cluster,metadata}`, `--env-centering` (default 0 = off), `--logging-spread`, `--best-ctr`, `--ctr-reference {logger,uniform}` — world calibration (see the simulator doc).
- `--reward-model {regression,logging_score,oracle}` — shared `q̂` for DM/DR/SNDR.
- `--reward-features {interaction,concat}` — the regression reward model's features: `[x, a, x⊙a]` (default; item
  rankings can differ between users) or `[x, a]` (the previous model: one item ranking for every user).
- `--reward-data {train,external}` — the reward model's data: each train size's own training rows (default; every arm
  uses only its n rows) or a separate 50k-row slice, the same at every train size (the older runs). `--crossfit-folds K`
  (default 5 with `train`, off with `external`) cross-fits it by user: each user's training `q̂` comes from a model fit
  without that user's rows. `--val-size` (default 20000; 0 = the older fraction rule) fixes the validation split.
- `--optuna-selection {ci_low,r_hat,actual_reward}` — what Optuna maximizes.
- `--methods opc no_propensity` — or one arm only (e.g. finish no-prop after OPC).
- `--slim` — log trial hyperparams; skip heavy post-hoc catalog eval.
- `--skip-completed` / `--no-skip-completed` — skip only if `summary_metrics.csv` exists (whole condition).
- `--val-size` or (`--val-frac`, `--val-min`, `--val-max`) / `--val-sizes` — validation sizing.
- `--policy-reward-mode mc --policy-reward-mc-sim 8` — faster approximate reward eval.
- `--num-gpus` / `--max-workers` — parallel only; OOM backoff on by default.
- `--memory-cap` (default) / `--no-memory-cap` — parallel/H1: run only as many workers as
  fit in the free memory each GPU reports at launch (RAM without a GPU). A condition's peak is
  estimated from its own budget:
  - the training step, 6 × largest batch × catalog × 4 bytes. The batch is the largest of the
    condition's Optuna choices (`--optuna-batch-sizes` replaces the schedule's), so smaller
    batches admit more workers;
  - the dense q̂ copies, users × catalog × 4 bytes each. There is one for the shared lookup, one
    more with cross-fitting, and one more when q̂ is refit per train size (the next size's lookup
    is built while the previous one is still held). None are counted above the
    dense-materialize limit;
  - 1.5 GiB per process.

  Each device takes floor(0.75 × free / peak) workers, over however many GPUs there are, and
  conditions are grouped by estimate. The `[memory]` log line shows, per device and dataset, the
  free memory, the estimated peak and the workers chosen. `--max-workers` stays the upper
  bound; results are unchanged. The calibration is anime at ≈ 12.5 GB per worker, which gives 2
  workers on a 48 GB card: under WSL2 an oversubscribed card spills into shared memory silently,
  with no OOM.

### Reproducibility

Each `--seeds` value is the single seed for its condition: data generation, splits,
Optuna sampler, model init, batch order and dropout are all derived from it
(`utils/seeding.py`). The same seed reproduces results bit-for-bit across serial,
parallel and H1 runners and regardless of run order; `--seeds 0 1 2 ...` gives
independent repeats for robustness.

- `--cpu-threads 4` (default) — fixed numpy/BLAS/torch threads; a different count gives
  slightly different floating-point results, so keep it fixed across runs you compare.
- `--deterministic` (default) / `--no-deterministic` — deterministic torch/cuDNN kernels.
- Exact equality also assumes the same GPU model and library versions.
- Logged data is identical on any machine for the same seed: actions are sampled by
  inverse CDF from numpy uniforms and float64 policy probabilities, on the GPU when
  available (`OPC_SAMPLER_DEVICE=cpu` forces CPU; same samples, slower). Training
  arithmetic still differs slightly between devices/GPU models.
- Trial scoring (each trained policy's DR / naive value on the logged train and validation rows)
  and exact policy values run on the GPU when available (`OPC_SCORING_DEVICE=cpu`,
  `OPC_EXACT_REWARD_DEVICE=cpu` force the CPU). CPU and GPU scores agree to ~1e-5 (relative).

### Batch schedule (default Optuna neighborhood)

| train_size ≤ | default batch | Optuna choices |
|-------------:|--------------:|----------------|
| 25k | 1024 | 512, 1024, 2048 |
| 100k | 4096 | 2048, 4096, 8192 |
| 500k | 8192 | 4096, 8192, 16384 |
| 2M | 16384 | 8192, 16384, 32768 |
| >2M (e.g. 5M/10M) | 163840 | 81920, 163840, 327680 |

Enqueued `--batch-size` values snap onto the current grid when needed.

### Prior-work baselines (CausE, BLOB) and their analyses

Both arms train on the same N logged rows and select on the same validation rows as OPC; neither uses propensities to
train or select (`docs/training_objectives_audit.md`). Development settings of the reported 25k grids (each run's exact
flags are in its `run_manifest.json` and `run_meta.json`, and in `artifacts/full_study/run_registry.csv`):

```bash
COMMON="--datasets ml kuairand anime --bias-configs none high/none/none none/high/none none/none/high high --seeds 100 101 --ctr-levels 0.05 --train-sizes 25000 --n-trials 20 --sampler random --stage development --slim --emb-dir BPR/embeddings --out-dir artifacts/full_study"
```

```bash
python -m training.run_full_study_parallel $COMMON --run-tag cause_cap_25k --methods cause --cause-family cap --cause-lr-range 3e-4 3e-2 --cause-epochs 30 100 300 --cause-l2 0 1e-7 1e-6 1e-5 1e-4 1e-3 --cause-cf 0 0.01 0.1 1 10 100 --cause-ties one_way symmetric --cause-bias-inits base_rate --cause-temper
```

```bash
python -m training.run_full_study_parallel $COMMON --run-tag blob_nq_25k --methods blob --blob-families nq --blob-lr-range 3e-3 1e-1 --blob-epochs 10 30 100 300 1000 --blob-wa-m -1 1 3 --blob-wb-m -6 -3 0 --blob-kappa-s 0.1 --blob-pick-diagnostics --save-policies
```

- **BLOB-Pnorm** (the catalog-normalized prior): add `--blob-variants L10` and extend `--blob-lr-range 3e-3 3e-1`
  (`docs/blob_prior_calibration.md` §8.1 has the exact commands).
- **Truth-trained class oracles:** `python -m training.class_oracles --datasets ml --seeds 100 101 --out <dir>
  --save-policies`.
- **Pick diagnostics of saved policies, and the comparison tables:** `python -m training.policy_diagnostics` and
  `python -m training.analyze_blob {tune,calib,compare}`; `python -m training.analyze_cause_fair {tune,compare}`. The
  exact invocations behind each committed table are in `artifacts/full_study/{cause_fair_25k,blob_controlled_25k,blob_prior_calibration}/README.md`.

## Outputs

Each condition:

`artifacts/full_study/run_<run-tag>/dataset=...__bias=...__ctr=...__seed=.../` (bias label: `medium`, or
`w-high.g-none.v-low` for mixed levels; `__mix=<a>` when `--logging-uniform-mix` mixes the logger)

Common files:

- `summary_metrics.csv` — one row per (label, train size): the method for OPC, no-propensity, DM-only and the tempered
  logger (they also hold the logger's row at train size 0), one per CausE prediction and rho, one per BLOB family and
  prior variant. `arm_config_key` hashes the settings that produced the row (the completion marker of
  `--skip-completed`).
- `opc_trials_long.csv` / `no_prop_trials_long.csv` / `dm_trials_long.csv` / `tempered_logger_trials_long.csv` — each
  arm's Optuna trials, one row per trial; `trials_long.csv` joins them (the arms the summary holds).
- `opc_runs_long.csv` / `no_prop_runs_long.csv` / … — per-run logs; `runs_long.csv` joins them.
- `<label>_trials.csv` for the prior-work arms (e.g. `causecap_c_r000_trials.csv`, `blob_nq_trials.csv`) — every trial.
- `*_selected_policy.npz` with `--save-policies` — each arm's selected policy vectors (`training/policy_diagnostics.py`).
- `run_meta.json` — exact parameters (includes `train_weights`, `select_weights`, policy losses), the calibrated
  `world`, and per label (`labels`) its arm, configuration key, code commit and time, with each key's settings
  (`arm_configs`).

At run root:

- `all_summary_metrics.csv` — merged summaries.
- `run_manifest.json` — the latest invocation's settings (arms, CausE / BLOB options when requested, code commit).
- `run_invocations.jsonl` — one line per invocation of the run tag: its manifest, the conditions it ran or skipped, and
  its failures.
- `failures.csv` — the latest invocation's failed conditions (removed when it had none).

Analyze: `python -m training.analyze_full_study --run-dir artifacts/full_study/run_<tag>`.

## Rerunning, resuming and extending a run (since 2026-10-06)

A run tag's condition folders accumulate rows from any number of invocations of either runner
(`training/run_state.py`):

- **`--skip-completed` (default)** runs only what a folder lacks. A requested arm is done when every label it writes has
  a summary row for every requested train size made with the same settings (`arm_config_key`); then it is skipped. A
  CausE arm missing some rhos runs only those; a BLOB arm missing a prior variant or family runs only that one.
- **Resume** an interrupted run with the same command: finished arms are skipped, an interrupted arm reruns, and its
  earlier attempt's trial rows are replaced, never repeated.
- **Extend** a run by adding another arm, CausE family, BLOB variant or rho under the same run tag: the new rows join
  the folder and every other row stays.
- **Rows made with other settings** for a requested label stop the invocation before anything runs, with a list of
  the differing settings. Use a new `--run-tag`, or `--no-skip-completed`, which reruns the requested arms and replaces
  their rows (other arms' rows are always kept).
- Rows written before 2026-10-06 carry no `arm_config_key` and count as matching.
- A CausE rho or a BLOB family added later gives exactly the rows of one run of the whole arm. A BLOB prior variant
  added later trains as it would alone. Variants trained in one invocation share float32 batches and agree with that
  to about 1e-6 relative, which can flip a near-tie of the selection.
- Trial logs written before the fix may repeat a resumed condition's trials. One run does:
  `run_reval_sndr_exact_harm0.1_s201`, two kuairand conditions, whose copies are identical. Every loader keeps one
  row per (method, train_size, run, trial_number) (`training.run_state.read_trials_long`).
