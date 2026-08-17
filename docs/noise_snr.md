# Noise / SNR Characterization

## Goal

Compare noise types systematically:

1. Relative to one another (linear / cluster / metadata).
2. Relative to measured SNR (not only ε labels).
3. Relative to correction magnitude (policy residual / OPC lift).

## Embedding mix

```text
X̃ = (1 − Σε_i) X + Σ ε_i N_i
```

- `linear_transform` — global linear warp + Gaussian (`eps1`)
- `random_centroids` / `kmeans_templates` — cluster templates (`eps2`)
- `metadata_projection` — side-info projection (`eps_meta`)

Levels and axes: shared source `utils/noise_levels.py` (same tables as the full study).

## Metrics (`utils/noise_snr.py`)

| Metric | Definition |
|--------|------------|
| `snr_db` | `10 log10(‖X‖²_F / ‖X̃−X‖²_F)` |
| `cosine_retention` | mean row cosine(`X`, `X̃`) |
| `rmse` | `sqrt(mean((X̃−X)²))` |
| `signal_frac` | `1 − Σε` (nominal mix weight on GT) |

Report separately for **action** (`emb_a` vs `our_a`) and **context** (`emb_x` vs `our_x`).

Per-component isolation: mix one noise source at a time (other ε = 0) and recompute the same metrics.

## Protocol

```bash
python -m training.characterize_noise_snr \
  --datasets ml \
  --noise-modes kmeans_templates \
  --noise-levels low medium high extreme brutal \
  --out-dir artifacts/noise_snr
```

Outputs: `summary.csv`, SNR/cosine vs level plots.

## Link to correction

Experiments attach an `snr` block to `run_meta.json` after env build. Later analysis should relate:

- OPC − no-propensity reward lift
- `action_delta` / `context_delta` (trainer)
- measured `snr_db` / `cosine_retention`

**Open question:** do different noise types need different correction behaviors at matched SNR?

## Logging bias (separate from embedding SNR)

`--logging-uniform-mix` and `--policy-temperature` damage propensities without changing embedding SNR. Treat as a distinct axis (see [regimes.md](regimes.md) hurtlog).
