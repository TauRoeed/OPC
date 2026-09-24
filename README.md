# OPC Repository

Offline policy comparison experiments with matrix-factorization embeddings.

**Loss / Optuna details:** [`docs/training_losses.md`](docs/training_losses.md)  
**Research notes:** [`docs/research_workplan.md`](docs/research_workplan.md)  
**Simulator:** [`docs/representation_bias.md`](docs/representation_bias.md)

Main flow:
1. Fit/generate BPR artifacts (user/item factors + metadata arrays).
2. Run OPC vs no-propensity on simulated logged bandit data.
3. Analyze under `artifacts/full_study/...`.

## Current defaults (full study)

| Knob | Default |
|------|---------|
| OPC train loss (`--policy-losses`) | `sndr` (pure SNDR; no KL/CRM) |
| No-propensity train | always `naive` (no IW, no DM/DR, no clip) |
| Optuna objective | `ci_low` = DR/naive mean − t·SE |
| OPC DR score IW clip | fixed `M=1` (`DEFAULT_DR_SCORE_CLIP_M`); **not** Optuna-searched |
| Reward model `q̂` | `regression` (bias script often uses `logging_score`) |
| Representation bias (`--bias-configs`) | `low medium high` (all three types at that level) |
| Reference CTR (`--ctr-levels`) | 5% for the logger at medium bias; best item 30% |
| Logging temperature | calibrated: clean logger over 50% of the catalog (`--logging-spread`) |
| Batch sizes | `batch_schedule(train_size)` unless you pass `--optuna-batch-sizes` |
| Skip finished cells | `--skip-completed` on (checks `summary_metrics.csv` only) |

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

### Example: MovieLens 1M

```bash
python -m BPR.generate_artifacts \
  --dataset ml \
  --root datasets/ml-1m \
  --emb-dir BPR/embeddings
```

Fresh clone:

```bash
python -m BPR.generate_artifacts --dataset anime --root datasets/anime
python -m BPR.generate_artifacts --dataset myket --root datasets/myket
python -m BPR.generate_artifacts --dataset lastfm --root datasets/lastfm/lastfm_360k.hdf5
python -m BPR.generate_artifacts --dataset msd --root datasets/msd/msd_taste_profile.hdf5
```

lastfm / msd also fetch side metadata (user profiles, artist genres and tags; ~1.2GB); see `BPR/README.md`.

Disable auto-download: add `--no-download`.  
Smoke test: `python -m BPR.smoke_test_loaders --include-large`.

### Important Arguments

- `--dataset`: `ml`, `myket`, `anime`, `lastfm`, `msd`, `kuairec`, `kuairand`.
- `--root`: dataset path (for `lastfm` / `msd`, the loader file path).
- `--emb-dir`: output directory for `.npy` artifacts.

### Produced Files

For each dataset `<name>`:

- `BPR/embeddings/<name>_user_factors.npy`
- `BPR/embeddings/<name>_item_factors.npy`
- `BPR/embeddings/<name>_item_metadata.npy`
- `BPR/embeddings/<name>_user_metadata.npy` (if available)
- `BPR/embeddings/<name>_user_interaction_counts.npy`

## 2) Run Experiments (OPC vs No-Propensity)

- Single process: `training/run_full_study.py`
- Parallel: `training/run_full_study_parallel.py`

OPC trains with `sndr` by default. Trial selection uses DR with IW clipped at `M=1`. No-propensity stays pure naive.

### Simulated world

Each condition builds a world from the dataset's BPR vectors
([`docs/representation_bias.md`](docs/representation_bias.md)):

- **Truth.** The clean vectors are centered (80% of the mean removed, so scores are not just
  popularity). Clicks follow `sigmoid(α·z + b)` on the standardized clean score `z`: α puts
  each user's best item at 30% on average, and b puts the logger at medium bias at 5% CTR.
- **What the learner sees.** Biased copies of users and items, with three types applied in
  order: a global warp, a group offset (k-means clusters, or metadata groups) and a
  per-vector offset. Each type is `none` / `low` / `medium` / `high`. All three at
  low / medium / high keep 90 / 75 / 50 % of the signal (calibrated per dataset).
- **Logger.** `softmax(biased scores / T)`, with T set so the clean logger spreads over half
  the catalog.

The truth is identical across bias configurations for a dataset and seed. Every
calibrated value is written to `run_meta.json → world`. Inspect a dataset with
`python -m training.characterize_world --datasets ml`.

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
  --datasets ml anime \
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
  --policy-losses sndr --reward-model logging_score --slim \
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

### Useful Flags

- `--policy-losses sndr` (default) — OPC train; DR selection still clips IW at M=1.
- `--bias-configs` — levels per condition: `medium` (all three types) or `warp/group/vector`, e.g. `high/none/low`.
- `--bias-groups {cluster,metadata}`, `--env-centering`, `--logging-spread`, `--best-ctr`, `--ctr-reference {logger,uniform}` — world calibration (see the simulator doc).
- `--reward-model {regression,logging_score,oracle}` — shared `q̂` for DM/DR/SNDR.
- `--optuna-selection {ci_low,r_hat,actual_reward}` — what Optuna maximizes.
- `--methods opc no_propensity` — or one arm only (e.g. finish no-prop after OPC).
- `--slim` — log trial hyperparams; skip heavy post-hoc catalog eval.
- `--skip-completed` / `--no-skip-completed` — skip only if `summary_metrics.csv` exists (whole condition).
- `--val-size` or (`--val-frac`, `--val-min`, `--val-max`) / `--val-sizes` — validation sizing.
- `--policy-reward-mode mc --policy-reward-mc-sim 8` — faster approximate reward eval.
- `--num-gpus` / `--max-workers` — parallel only; OOM backoff on by default.
- `--memory-cap` (default) / `--no-memory-cap` — parallel/H1: run only as many workers as
  fit in free GPU memory (RAM without a GPU), from each condition's estimated peak
  (≈ 6 × largest Optuna batch × catalog × 4 bytes + dense q̂ + 1.5 GB); conditions are
  grouped by size. `--max-workers` stays the upper bound; results are unchanged.

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

### Batch schedule (default Optuna neighborhood)

| train_size ≤ | default batch | Optuna choices |
|-------------:|--------------:|----------------|
| 25k | 1024 | 512, 1024, 2048 |
| 100k | 4096 | 2048, 4096, 8192 |
| 500k | 8192 | 4096, 8192, 16384 |
| 2M | 16384 | 8192, 16384, 32768 |
| >2M (e.g. 5M/10M) | 163840 | 81920, 163840, 327680 |

Enqueued `--batch-size` values snap onto the current grid when needed.

## Outputs

Each condition:

`artifacts/full_study/run_<run-tag>/dataset=...__bias=...__ctr=...__seed=.../` (bias label: `medium`, or `w-high.g-none.v-low` for mixed levels)

Common files:

- `summary_metrics.csv` — per-method summary (also the skip-completed marker).
- `opc_trials_long.csv` / `no_prop_trials_long.csv` — Optuna trial logs.
- `opc_runs_long.csv` / `no_prop_runs_long.csv` — per-run logs.
- `run_meta.json` — exact parameters (includes `dr_score_clip_m`, policy losses) and the calibrated `world`.

At run root:

- `all_summary_metrics.csv` — merged summaries.
- `run_manifest.json` — global manifest.
- `failures.csv` — failed conditions (if any).

Analyze: `python -m training.analyze_full_study --run-dir artifacts/full_study/run_<tag>`.
