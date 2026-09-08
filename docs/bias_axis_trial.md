# Bias-Axis Trial — Per-Noise-Type Learning Study

**Question.** Starting from the same clean BPR embeddings, how does **each bias type** (corruption channel) affect **policy learning** as we add more logged data?

**Design.** Run one study where each condition activates **exactly one noise axis** at a time, sweep **five severity levels** (`low` → `brutal`), and compare **OPC** vs **no propensity** on true policy reward. A `combined` arm is included only as a bundled reference, not as the primary comparison unit.

**One-liner.** Isolate context / action / metadata embedding bias (plus combined reference), stress-test severity, measure learning curves and OPC lift.

---

## Launch

```bash
cd /home/roee/Documents/git-repos/OPC

python -m training.run_full_study \
  --datasets ml \
  --noise-axes context action metadata combined \
  --noise-levels low medium high extreme brutal \
  --ctr-levels 0.05 \
  --train-sizes 5000 25000 50000 100000 \
  --seeds 0 1 2 3 4 \
  --n-trials 20 \
  --run-tag bias_axes_l5_t20_s5
```

Outputs: `artifacts/full_study/run_bias_axes_l5_t20_s5/`

**Grid size:** 4 axes × 5 levels × 5 seeds = **100 conditions** (each runs OPC + no propensity, 4 train sizes, 20 Optuna trials per method).

Parallel launch (same grid, multi-GPU): use `training.run_full_study_parallel` with the same flags.

---

## Expanded follow-up (fixed val + more seeds + harder noise)

First run used `val_frac=0.15` (no fixed val). Follow-up holds **fixed** validation sizes and stresses the high end of the noise ladder.

```bash
./scripts/run_bias_axes_val_sweep.sh
# or Docker: IMAGE=opc:gpu ./scripts/run_bias_axes_val_sweep.sh
# smoke:    SMOKE=1 ./scripts/run_bias_axes_val_sweep.sh
```

Equivalent flags:

```bash
python -m training.run_full_study_parallel \
  --datasets ml \
  --noise-axes context action metadata combined \
  --noise-levels high extreme brutal catastrophic \
  --ctr-levels 0.05 \
  --train-sizes 5000 25000 50000 100000 \
  --val-sizes 20000 50000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-trials 20 \
  --slim --skip-completed --require-cuda \
  --run-tag bias_axes_val20_50_s10_hard
```

| Knob | Value | vs first run |
|------|-------|--------------|
| Validation | **fixed** `--val-sizes 20000 50000` | was `val_frac=0.15` |
| Seeds | **0–9** (10) | was 0–4 (5) |
| Noise levels | `high` `extreme` `brutal` **`catastrophic`** | drops low/medium; adds beyond brutal |
| Layout | `run_…/val_20000/`, `run_…/val_50000/` | one flat run dir before |

**Grid size:** 4 axes × 4 levels × 10 seeds × 2 vals = **320 conditions**.

New severity row (single-axis ε on the named channel):

| Level | context ε | action ε | metadata ε |
|-------|-----------|----------|------------|
| catastrophic | 0.70 | 0.70 | 0.45 |

Combined arm uses `(0.70, 0.70, 0.45)`.

---

## What is held fixed

| Knob | Value | Notes |
|------|-------|-------|
| Dataset | `ml` | Uses existing BPR factors under `BPR/embeddings/` |
| Clean embeddings | frozen GT | True reward always uses clean `emb_x`, `emb_a` |
| Noise mode | `kmeans_templates` | Cluster-template corruption (linear + k-means mix) |
| CTR | `0.05` | Single logging-rate setting; no CTR sweep in this trial |
| Train sizes | 5k, 25k, 50k, 100k | Inner learning curve per condition |
| Validation | `val_frac=0.15`, `val_min=5000` | Default from `run_full_study` |
| Seeds | 0–4 | 5 repeats per (axis, level) cell |
| Optuna trials | 20 per method | Tag suffix `t20` |
| Methods | `opc`, `no_propensity` | Paired comparison every cell |
| Policy eval | `exact` | True expected reward on clean environment |
| Policy temperature | `1.0` | Default softmax for logging/eval |
| Logging uniform mix | `0.0` | No HurtLog in this trial (embedding bias only) |
| Reward model | `regression` | Shared `q_hat` on noisy embeddings |
| OPC policy loss | `kl_crm` | Default SNDR+KL+CRM stack |

---

## What is swept

| Axis | Values | Meaning |
|------|--------|---------|
| `noise_axis` | `context`, `action`, `metadata`, `combined` | **Which channel** is corrupted; only one active per condition folder |
| `noise_level` | `low`, `medium`, `high`, `extreme`, `brutal`, `catastrophic` | **How much** corruption on that channel |
| `seed` | 0–4 | Stochasticity in noise templates + data draws |
| `train_size` | 5k … 100k | Amount of logged trajectories for learning |

### Noise axes (bias types)

Definitions match [`utils/noise_levels.py`](../utils/noise_levels.py) and real-world mappings in [`bias_examples.md`](bias_examples.md).

| `noise_axis` | What the learner sees | True reward uses |
|--------------|----------------------|------------------|
| `context` | Noisy **user** factors `our_x` | Clean `emb_x` |
| `action` | Noisy **item** factors `our_a` | Clean `emb_a` |
| `metadata` | Noisy **metadata** projection only | Clean embeddings + clean reward |
| `combined` | Context + action + metadata mix (bundled ε table) | Clean reward |

Logging policy and propensity scores are computed from the **noisy** representation the learner sees. Evaluation reward is always on the **clean** simulator.

### Severity table (ε per level)

**Single-axis runs** (`context` / `action` / `metadata`): only the named channel gets non-zero ε.

| Level | context ε | action ε | metadata ε |
|-------|-----------|----------|------------|
| low | 0.05 | 0.05 | 0.00 |
| medium | 0.10 | 0.15 | 0.05 |
| high | 0.20 | 0.25 | 0.10 |
| extreme | 0.35 | 0.40 | 0.20 |
| brutal | 0.50 | 0.50 | 0.30 |
| catastrophic | 0.70 | 0.70 | 0.45 |

**Combined arm** uses the bundled triple `(eps1, eps2, eps_meta)` from the same file, e.g. medium → `(0.10, 0.15, 0.05)`.

Embedding mix (all axes):

```text
X̃ = (1 − Σε) · X  +  ε_linear · N_linear  +  ε_cluster · N_cluster  +  ε_meta · N_meta
```

See [`noise_snr.md`](noise_snr.md) for SNR metrics logged in `run_meta.json`.

---

## Simulation flow (per condition)

1. **Load clean BPR embeddings** for MovieLens (`ml_user_factors.npy`, `ml_item_factors.npy`, plus metadata arrays for the metadata axis).
2. **Apply noise** for the chosen `noise_axis` and `noise_level` → `our_x`, `our_a` (and metadata corruption when relevant).
3. **Build logged bandit data** at `ctr=0.05`; rewards drawn from clean `q*(u,a)`.
4. **Fit shared regression `q_hat`** once per condition on a fixed regression slice.
5. **Optuna search** (20 trials) separately for OPC and no propensity at each `train_size`.
6. **Select winning hyperparameters** per (`train_size`, method); record true `policy_rewards` and OPE diagnostics.

Condition folder naming:

```text
dataset=ml__noise=kmeans_templates__axis=<axis>__level=<level>__ctr=0.05__seed=<seed>/
```

Key artifacts per folder: `summary_metrics.csv`, `runs_long.csv`, `opc_trials_long.csv`, `no_prop_trials_long.csv`, `run_meta.json`.

---

## Primary outcomes

| Metric | Definition | Use |
|--------|------------|-----|
| `policy_rewards` | True expected reward of the **selected** policy | Main learning curve vs `train_size` |
| `delta` | `opc − no_propensity` on `policy_rewards` | Does correction help under this bias? |
| Oracle best `actual_reward` | Max over 20 trials | Headroom if HPO were perfect |
| `context_delta`, `action_delta` | Policy drift off logging distribution | Links bias type to distributional shift |
| `conv_dr` vs `policy_rewards` | OPE calibration | Is selection trustworthy? |

**Learning effect (main readout):** for each `noise_axis`, plot mean `policy_rewards` vs `train_size`, faceted by `noise_level`, with seed error bars. Compare OPC vs no propensity curves and ask whether higher levels (`extreme`, `brutal`) flatten or invert the learning gain.

**Cross-axis readout:** at fixed (`train_size=100k`, `ctr=0.05`), compare mean `delta` across axes and levels — which bias type hurts OPC most?

---

## Analysis

After the run completes:

```bash
python -m training.analyze_full_study \
  --run-dir artifacts/full_study/run_bias_axes_l5_t20_s5 \
  --summary-csv artifacts/full_study/run_bias_axes_l5_t20_s5/all_summary_metrics.csv
```

If `all_summary_metrics.csv` is missing at run root, merge per-condition summaries first (same pattern as other full-study runs).

**Figures to inspect** (under `figures/`):

| Figure pattern | Question |
|----------------|----------|
| `selected_policy_reward_*_ctr_0.05.png` | Learning curves per axis, panels low→brutal |
| `delta_*_ctr_0.05.png` | OPC lift vs train size per axis/level |
| `oracle_best_actual_*_ctr_0.05.png` | HPO ceiling vs selected policy |
| `axis_comparison/` (from `plot_ablation_ctr_axis`) | Side-by-side axes at fixed CTR |
| `calibration_*.png` | OPE trustworthiness |

Optional SNR characterization (no training):

```bash
python -m training.characterize_noise_snr \
  --datasets ml \
  --noise-axes context action metadata combined \
  --noise-levels low medium high extreme brutal \
  --out-dir artifacts/noise_snr/bias_axes_l5
```

Join measured `snr_db` to learning outcomes via `run_meta.json`.

---

## Relation to earlier runs

| Run | Overlap |
|-----|---------|
| `run_ml_sweep_t20_s5` | Same dataset/methods/seeds; swept `ctr` and only `low/medium/high`; `action` incomplete |
| This trial | Single `ctr=0.05`; adds `extreme` + `brutal`; runs all four axes in one tag |

You may reuse completed cells from `run_ml_sweep_t20_s5` for `context`/`combined` at `low/medium/high` if protocols match, or treat this run as a clean standalone protocol.

---

## Expected claims (if hypotheses hold)

1. **Monotone damage:** higher `noise_level` → lower asymptotic `policy_rewards` and/or flatter learning curves.
2. **Axis specificity:** `action` bias may hurt more when item geometry drives logging; `context` when user cold-start dominates; `metadata` when side features are wrong.
3. **OPC vs naive:** `delta` shrinks or goes negative at `extreme`/`brutal` if correction variance dominates (related to H1, but here `q_error=0`).
4. **Combined reference:** bundled noise at level L is **not** equal to single-axis level L; compare using the ε table, not label alone.

---

## Out of scope (this trial)

- CTR sweep (`0.02`, `0.1`) — separate follow-up
- Logging-policy bias (`--logging-uniform-mix`, HurtLog) — see [`regimes.md`](regimes.md)
- Multi-dataset (`anime`, `kuairec`) — extend after `ml` results
- `q_error` / bad reward model — covered by H1 study

---

## Smoke test

Small grid before the full 100-cell run:

```bash
python -m training.run_full_study \
  --datasets ml \
  --noise-axes context metadata \
  --noise-levels low high \
  --ctr-levels 0.05 \
  --train-sizes 5000 25000 \
  --seeds 0 1 \
  --n-trials 5 \
  --run-tag bias_axes_smoke
```
