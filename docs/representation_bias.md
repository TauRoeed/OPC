# Simulated world: representation bias

Code: `utils/representation_bias.py` (world), `utils/simulation_utils.py` (`SyntheticBanditEnv`,
`generate_dataset`). Real-world meaning of each bias type: [bias_examples.md](bias_examples.md).

The world has two copies of every user and item vector:

- **clean** vectors define the truth (the click model). Only the environment and the
  `oracle` reward model see them.
- **biased** vectors (`our_x`, `our_a`) are what the logger, the reward model and the learned
  policy see. Everything the learner knows comes from these.

## Build order (per dataset and seed)

1. **Centering.** `x' = x − λ·mean(x)` for users and for items, with λ = `--env-centering`
   (default 0.8). Raw BPR scores are dominated by item popularity (a shared mean
   direction). Centering removes most of it, so scores become personal. The centered
   vectors are the clean world.
2. **Representation bias**, applied to both sides in this order. Each type mixes the
   current vectors with a target, `x ← (1 − ε)·x + ε·target`, and then rescales them back
   to the clean RMS. Targets are scale-matched to the clean vectors.
   - **warp:** one random linear map for the whole side: `target = x_clean @ W`.
   - **group:** one Gaussian offset per group: `target = template[group(x)]`. Groups are
     k-means clusters of the clean vectors (k = √n, clipped to 8–64). With
     `--bias-groups metadata`, groups are k-means clusters of the metadata arrays
     instead; a side without metadata falls back to clusters, and a message says so.
   - **vector:** one Gaussian offset per user or item.
3. **Levels.** Each type is `none`, `low`, `medium` or `high`. ε is calibrated per dataset
   so that **all three types at level L together** keep 90 / 75 / 50 % of the signal.
   Signal kept is the mean over users of the correlation, across items, between biased
   and clean scores. Each type then keeps a share κ_L on its own (about 0.97 / 0.94 / 0.88).
   Mixed configurations such as `high/none/low` land in between.
4. **Logging temperature T.** The clean softmax logger spreads over `--logging-spread`
   of the catalog (default 0.5). Spread is the effective number of items, `exp(entropy)`,
   averaged over users. The logger in every run is `softmax(our_x · our_a / T)`, optionally
   mixed with uniform (`--logging-uniform-mix`). The learned policy uses the same T, so at
   initialization it equals the logger.
5. **Click model.** `q(u, a) = sigmoid(α·z(u, a) + b)`, where z is the clean score
   standardized over all user-item pairs:
   - α is set so the best item of each user averages `--best-ctr` (default 30%);
   - b is set so the **reference policy** has CTR `--ctr-levels` (default 5%). The
     reference policy is the logger at medium/medium/medium bias (`--ctr-reference logger`,
     default) or the uniform random policy (`--ctr-reference uniform`; H1 uses this).

   In code this is `SyntheticBanditEnv(scale=α/sd, offset=b − α·mean/sd)`.

All draws depend only on the seed. The levels are therefore nested (the same bias
directions at growing ε), and the truth (centered vectors, α, b, T, user prior) is identical
across bias configurations for a given dataset and seed. Calibration runs in float64 numpy,
so it gives the same result with or without a GPU. It uses samples:
- 50k users and items for signal kept (all of them on smaller catalogs);
- 300 users for T;
- 1000 users drawn from the user prior × 2048 items for α and b.

It takes 3–20 s per dataset and is cached per process. Targets that cannot be met stop the
run with a `WorldCalibrationError` that names the missed target.

## Flags (all runners)

| Flag | Default | Meaning |
|------|---------|---------|
| `--bias-configs` | `low medium high` | one level for all three types, or `warp/group/vector` levels, e.g. `high/none/low` |
| `--bias-groups` | `cluster` | groups for the group bias: `cluster` or `metadata` |
| `--env-centering` | 0.8 | λ, share of the mean vector removed |
| `--logging-spread` | 0.5 | clean logger's effective items / catalog |
| `--best-ctr` | 0.30 | best item per user, averaged over users |
| `--ctr-levels` | 0.05 | target CTR of the reference policy |
| `--ctr-reference` | `logger` | `logger` (at medium bias) or `uniform` (not on H1, which always uses uniform) |

Run folders are named `dataset=<ds>__bias=<label>__ctr=<ctr>__seed=<seed>[__val=<n>]`. The
label is the level when all three types share it (`medium`); otherwise it is
`w-high.g-none.v-low`. Summary CSVs keep the analysis columns: `noise_mode=representation_bias`,
`noise_axis=both`, `noise_level=<label>`, plus `bias_warp`, `bias_group`, `bias_vector`,
`signal_kept` and `logging_temperature`.

`run_meta.json → world` records everything calibrated:
- ε per type and level, κ per level, and the signal kept per level and for this run;
- T and the clean logger's effective items;
- α, b, scale and offset;
- the reference, uniform, best-item and logging CTRs;
- the group source and counts;
- per-side RMS and cosine to clean.

`run_meta.json → snr` holds vector-level SNR and cosine per side (`utils/noise_snr.py`).

## Characterize a dataset

```bash
python -m training.characterize_world --datasets ml myket --bias-configs all
```

This prints and writes `summary.csv` (signal kept, logging CTR, best/logging ratio, cosine
and SNR per configuration) and `calibration.json` under `artifacts/world/`. With `all`, it
covers the 64 combinations of four levels for three types.

## Calibrated values (seed 0, defaults)

From `python -m training.characterize_world --datasets ml myket anime kuairec kuairand lastfm msd`
(1 min 53 s for all seven, 4 GB peak RAM). Every target is met: signal kept 0.90 / 0.75 / 0.50,
clean logger over 50% of the catalog, logger at medium bias 5.0%, best item 30.0%.

| Dataset | Users × items | T | α | κ low / med / high | Uniform CTR | Logging CTR none / low / medium / high |
|---|---|---|---|---|---|---|
| ml | 6,038 × 3,533 | 0.723 | 0.827 | 0.972 / 0.937 / 0.874 | 2.8% | 9.2% / 6.5% / 5.0% / 3.9% |
| myket | 10,000 × 7,988 | 0.426 | 0.598 | 0.976 / 0.952 / 0.914 | 3.2% | 12.2% / 6.9% / 5.0% / 4.0% |
| anime | 73,417 × 10,803 | 0.414 | 0.494 | 0.960 / 0.918 / 0.864 | 3.5% | 11.0% / 7.2% / 5.0% / 4.1% |
| kuairec | 7,175 × 10,611 | 0.998 | 0.931 | 0.978 / 0.954 / 0.912 | 2.9% | 9.1% / 6.4% / 5.0% / 4.0% |
| kuairand | 27,111 × 7,579 | 0.863 | 0.548 | 0.970 / 0.933 / 0.867 | 3.4% | 8.3% / 6.2% / 5.0% / 4.2% |
| lastfm | 358,868 × 292,385 | 0.264 | 0.088 | 0.960 / 0.934 / 0.890 | 4.6% | 18.1% / 7.4% / 5.0% / 4.7% |
| msd | 1,019,318 × 42,053 | 0.342 | 0.582 | 0.969 / 0.941 / 0.904 | 2.3% | 12.8% / 8.2% / 5.0% / 3.4% |

One type alone at `high` (ε, signal kept, logging CTR):

| Dataset | warp | group | vector |
|---|---|---|---|
| ml | 0.43, 0.87, 6.0% | 0.44, 0.87, 6.2% | 0.43, 0.88, 6.3% |
| myket | 0.47, 0.92, 5.6% | 0.40, 0.91, 7.5% | 0.40, 0.91, 8.2% |
| anime | 0.57, 0.86, 6.4% | 0.49, 0.86, 6.6% | 0.47, 0.86, 6.6% |
| kuairec | 0.43, 0.91, 6.2% | 0.41, 0.91, 6.4% | 0.41, 0.91, 6.6% |
| kuairand | 0.41, 0.87, 5.9% | 0.43, 0.87, 5.8% | 0.41, 0.87, 5.9% |
| lastfm | 1.00, 0.92, 5.7% | 0.44, 0.89, 9.0% | 0.44, 0.89, 8.2% |
| msd | 0.55, 0.90, 5.2% | 0.36, 0.90, 9.5% | 0.40, 0.90, 8.9% |

### How personal is the clean world?

Centering removes the shared mean, but not variance along a shared direction. The table
uses 500 random users per dataset. It shows the share of each user's score variation
(across items) that lies along the top item principal component, and the mean overlap
between two random users' top-20 items:

| Dataset | top-PC share raw → centered | top-20 overlap raw → centered | users sharing the top-1 item, centered |
|---|---|---|---|
| ml | 0.90 → 0.52 | 0.33 → 0.10 | 11% |
| myket | 0.95 → 0.68 | 0.64 → 0.21 | 33% |
| kuairec | 0.73 → 0.31 | 0.23 → 0.12 | 18% |
| kuairand | 0.66 → 0.33 | 0.22 → 0.09 | 15% |
| anime | 1.00 → 1.00 | 0.91 → 0.54 | 61% |
| lastfm | 1.00 → 1.00 | 0.97 → 0.65 | 67% |
| msd | 1.00 → 1.00 | 0.84 → 0.47 | 17% |

On ml, myket, kuairec and kuairand, centering makes rankings personal. On anime, lastfm and
msd, BPR put almost all item variance on one popularity-like direction (98% of it).

- Most users (84–89%) rank items along that axis in the same order, and the rest in
  reverse, so these worlds stay mostly non-personalized.
- A linear warp barely changes such rankings. On lastfm even ε = 1 keeps 0.92 of the signal,
  so the calibration takes the high level from the other two types.
- lastfm's click model is flat for most items (α = 0.088, random policy already at 4.6%);
  its best-item target is reached through a heavy tail of top items.
