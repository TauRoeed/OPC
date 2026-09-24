# Bias Trials — Two Protocols

Two **separate** trials. What each bias type means: [bias_examples.md](bias_examples.md).
Mechanics: [representation_bias.md](representation_bias.md).

| # | Goal | Train | Val | Bias |
|---|------|-------|-----|------|
| **1** | Min usable **val size** | fixed **1M** | **sweep** | fixed `low` (all three types) — no type grid |
| **2** | Bias effects at scale | **0.5M → 10M** | **fixed** (from trial 1) | each type alone and all three, per level |

Run trial 1 → pick `FIXED_VAL` → trial 2.

---

## Knobs

`--bias-configs` takes one level for all three types (`none` / `low` / `medium` / `high`) or
`warp/group/vector` levels. Bias always applies to users **and** items.

| Level | Signal kept (all three types) | Each type alone (typical) |
|-------|-------------------------------|---------------------------|
| low | 0.90 | ≈ 0.97 |
| medium | 0.75 | ≈ 0.94 |
| high | 0.50 | ≈ 0.88 |

ε per type and level is calibrated per dataset and recorded in `run_meta.json → world`.

---

## Trial 1 — Min val (no bias-type sweep)

```bash
./scripts/run_bias_min_val_trial.sh
```

**One-liner (host):**

```bash
python -m training.run_full_study_parallel --datasets ml --bias-configs low --ctr-levels 0.05 --train-sizes 1000000 --val-sizes 10000 25000 50000 100000 200000 --seeds 0 1 2 3 4 --n-trials 20 --num-gpus 2 --max-workers 16 --require-cuda --skip-completed --emb-dir BPR/embeddings --out-dir artifacts/full_study --run-tag bias_min_val_tr1m_t20_s5
```

**One-liner (docker):**

```bash
docker run --rm --gpus all --shm-size=128g -v "$PWD:/app" -w /app opc:gpu -m training.run_full_study_parallel --datasets ml --bias-configs low --ctr-levels 0.05 --train-sizes 1000000 --val-sizes 10000 25000 50000 100000 200000 --seeds 0 1 2 3 4 --n-trials 20 --num-gpus 2 --max-workers 16 --require-cuda --skip-completed --emb-dir BPR/embeddings --out-dir artifacts/full_study --run-tag bias_min_val_tr1m_t20_s5
```

**Grid:** 5 vals × 5 seeds = **25 conditions**.

Analyze → smallest val where reward/SE plateau → set `FIXED_VAL`.

---

## Trial 2 — Bias types @ large train

```bash
FIXED_VAL=100000 ./scripts/run_bias_axis_comp_large_trial.sh
```

- Train: `500000 1000000 2000000 5000000 10000000`
- Per level in `BIAS_LEVELS` (default `low medium high`): `L/none/none` (warp),
  `none/L/none` (group), `none/none/L` (vector) and `L` (all three)
- **Grid:** 12 configurations × 5 seeds = **60 conditions**

Tag: `bias_types_sndr_logscore_clip1_tr500k_10m_v100k_t20_s5`

```bash
python -m training.analyze_full_study \
  --run-dir artifacts/full_study/run_bias_types_sndr_logscore_clip1_tr500k_10m_v100k_t20_s5
```

Group bias follows metadata instead of clusters with `--bias-groups metadata` (add it to the
script's `ARGS`).

---

## Scripts

| Script | Role |
|--------|------|
| `scripts/run_bias_min_val_trial.sh` | Trial 1 |
| `scripts/run_bias_axis_comp_large_trial.sh` | Trial 2 |
| `scripts/run_bias_axis_comp_trial.sh` | → trial 2 |
| `scripts/run_bias_axes_val_sweep.sh` | Follow-up (val 20k/50k, seeds 0–9, each type at high) |

Docker: `IMAGE=opc:gpu …`
