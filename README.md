# OPC Repository

Offline policy comparison experiments with matrix-factorization embeddings.

**Research continuation:** see [`docs/research_workplan.md`](docs/research_workplan.md) (SNR characterization, regimes, Kuai datasets, bias/reflection notes).

Main flow:
1. Fit/generate BPR artifacts (user/item factors + metadata arrays).
2. Run OPC vs no-propensity experiments on simulated logged bandit data.
3. Analyze outputs under `artifacts/full_study/...`.

## Repository Layout

- `BPR/` - BPR training and artifact generation.
- `training/` - experiment runners and trainer implementations.
- `models/` - model definitions and estimators.
- `utils/` - simulation, policy, noise/SNR, and plotting helpers.
- `datasets/` - dataset files (MovieLens, etc.).
- `artifacts/` - generated outputs.
- `docs/` - paper outline, workplan, SNR/regime/bias notes.

## Quick Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch optuna matplotlib tqdm
```

If you already have an environment with these packages, skip setup.

Or run everything from scratch (venv + all BPR + parallel study):

```bash
./scripts/run_from_scratch.sh
```

## 1) BPR Fitting / Artifact Generation

Use `BPR/generate_artifacts.py` to fit BPR and export all arrays needed by experiments.

Datasets auto-download on first load when files are missing (`--download` is default). You can pass only a target folder/path; large music datasets (`lastfm`, `msd`) are fetched automatically too.

### Example: MovieLens 1M

```bash
python -m BPR.generate_artifacts \
  --dataset ml \
  --root datasets/ml-1m \
  --emb-dir BPR/embeddings
```

Fresh clone (no local data yet):

```bash
python -m BPR.generate_artifacts --dataset anime --root datasets/anime
python -m BPR.generate_artifacts --dataset myket --root datasets/myket
python -m BPR.generate_artifacts --dataset lastfm --root datasets/lastfm/lastfm_360k.hdf5
python -m BPR.generate_artifacts --dataset msd --root datasets/msd/msd_taste_profile.hdf5
```

Disable auto-download:

```bash
python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m --no-download
```

Smoke test all loaders:

```bash
python -m BPR.smoke_test_loaders --include-large
```

### Important Arguments

- `--dataset`: one of `ml`, `myket`, `anime`, `lastfm`, `msd`, `kuairec`, `kuairand`.
- `--root`: dataset path (for `lastfm` / `msd`, use the dataset file path expected by loader).
- `--emb-dir`: output directory for generated `.npy` artifacts.

### Produced Files

For each dataset `<name>`:

- `BPR/embeddings/<name>_user_factors.npy`
- `BPR/embeddings/<name>_item_factors.npy`
- `BPR/embeddings/<name>_item_metadata.npy`
- `BPR/embeddings/<name>_user_metadata.npy` (if available)
- `BPR/embeddings/<name>_user_interaction_counts.npy`

## 2) Run Experiments (OPC vs No-Propensity)

Main script: `training/run_full_study.py`

### Small Local Run (single process)

```bash
python -m training.run_full_study \
  --datasets ml \
  --noise-modes kmeans_templates \
  --noise-axes combined \
  --noise-levels low medium high \
  --ctr-levels 0.05 \
  --seeds 0 1 \
  --train-sizes 5000 25000 \
  --n-trials 10 \
  --num-runs 2 \
  --batch-size 2048 \
  --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study \
  --run-tag demo_ml
```

### Parallel Sweep (faster)

```bash
python -m training.run_full_study_parallel \
  --datasets ml anime \
  --noise-modes kmeans_templates \
  --noise-axes combined \
  --noise-levels low high \
  --ctr-levels 0.05 \
  --seeds 0 1 2 \
  --train-sizes 5000 25000 50000 \
  --n-trials 20 \
  --num-runs 1 \
  --max-workers 4 \
  --batch-size 2048 \
  --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study \
  --run-tag sweep_v1
```

### Useful Flags

- `--slim`: keeps trial logs but skips heavy post-hoc evaluation pass.
- `--skip-completed` / `--no-skip-completed`: control reruns for finished conditions.
- `--policy-reward-mode mc --policy-reward-mc-sim 8`: faster approximate reward evaluation.
- `--val-size` or (`--val-frac`, `--val-min`, `--val-max`): validation sizing strategy.

## Outputs

Each run writes into:

`artifacts/full_study/run_<run-tag>/dataset=...__noise=...__axis=...__level=...__ctr=...__seed=.../`

Common files:

- `summary_metrics.csv` - per-method summary for that condition.
- `trials_long.csv` - all Optuna trial logs (merged OPC + no-prop).
- `runs_long.csv` - per-run logs.
- `run_meta.json` - exact parameters used.

At run root:

- `all_summary_metrics.csv` - merged summaries across completed conditions.
- `run_manifest.json` - global manifest.
- `failures.csv` - failed conditions (if any).
