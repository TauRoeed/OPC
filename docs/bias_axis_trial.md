# Bias / Noise Trials — Two Protocols

Two **separate** trials. Real-world noise meanings: [bias_examples.md](bias_examples.md).

| # | Goal | Train | Val | Noise |
|---|------|-------|-----|-------|
| **1** | Min usable **val size** | fixed **1M** | **sweep** | **none** (fixed `combined`/`combined`/`low` — no type grid) |
| **2** | Noise effects at scale | **0.5M → 10M** | **fixed** (from trial 1) | full axis × component × level |

Run trial 1 → pick `FIXED_VAL` → trial 2.

Axis = **where** (`context` / `action` / `combined`). Component = **what** (`linear` / `cluster` / `metadata`). Older bug that remapped context→linear / action→cluster is fixed.

---

## Knobs (trial 2)

| Knob | Flag | Meaning |
|------|------|---------|
| **Axis** | `--noise-axes` | `context` / `action` / `combined` |
| **Component** | `--noise-components` | `linear` / `cluster` / `metadata` / `combined` |

Severity (component columns; isolate one → other ε = 0):

| Level | linear ε1 | cluster ε2 | metadata ε_meta |
|-------|-----------|------------|-----------------|
| low | 0.05 | 0.05 | 0.00 |
| medium | 0.10 | 0.15 | 0.05 |
| high | 0.20 | 0.25 | 0.10 |
| extreme | 0.35 | 0.40 | 0.20 |
| brutal | 0.50 | 0.50 | 0.30 |
| catastrophic | 0.70 | 0.70 | 0.45 |

---

## Trial 1 — Min val (no noise-type sweep)

```bash
./scripts/run_bias_min_val_trial.sh
```

**One-liner (host):**

```bash
python -m training.run_full_study_parallel --datasets ml --noise-axes combined --noise-components combined --noise-levels low --ctr-levels 0.05 --train-sizes 1000000 --val-sizes 10000 25000 50000 100000 200000 --seeds 0 1 2 3 4 --n-trials 20 --num-gpus 2 --max-workers 16 --require-cuda --skip-completed --emb-dir BPR/embeddings --out-dir artifacts/full_study --run-tag bias_min_val_tr1m_t20_s5
```

**One-liner (docker):**

```bash
docker run --rm --gpus all --shm-size=128g -v "$PWD:/app" -w /app opc:gpu -m training.run_full_study_parallel --datasets ml --noise-axes combined --noise-components combined --noise-levels low --ctr-levels 0.05 --train-sizes 1000000 --val-sizes 10000 25000 50000 100000 200000 --seeds 0 1 2 3 4 --n-trials 20 --num-gpus 2 --max-workers 16 --require-cuda --skip-completed --emb-dir BPR/embeddings --out-dir artifacts/full_study --run-tag bias_min_val_tr1m_t20_s5
```

**Grid:** 5 vals × 5 seeds = **25 conditions**.

Analyze → smallest val where reward/SE plateau → set `FIXED_VAL`.

---

## Trial 2 — Noise types @ large train

```bash
FIXED_VAL=100000 ./scripts/run_bias_axis_comp_large_trial.sh
```

- Train: `500000 1000000 2000000 5000000 10000000`
- Axes: `context action` · comps: `linear cluster metadata` · levels: `low`…`brutal` (add `catastrophic` via `NOISE_LEVELS=…` if wanted)
- **Grid:** 2 × 3 × 5 × 5 seeds = **150 conditions**

Tag: `bias_axis_comp_tr500k_10m_l5_t20_s5`

```bash
python -m training.analyze_full_study \
  --run-dir artifacts/full_study/run_bias_axis_comp_tr500k_10m_l5_t20_s5
```

---

## Scripts

| Script | Role |
|--------|------|
| `scripts/run_bias_min_val_trial.sh` | Trial 1 |
| `scripts/run_bias_axis_comp_large_trial.sh` | Trial 2 |
| `scripts/run_bias_axis_comp_trial.sh` | → trial 2 |
| `scripts/run_bias_axes_val_sweep.sh` | Legacy follow-up (val 20k/50k, seeds 0–9, hard levels incl. catastrophic) |

Docker: `IMAGE=opc:gpu …`
