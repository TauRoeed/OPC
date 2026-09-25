# Simulated world: representation bias

Code: `utils/representation_bias.py` (world), `utils/simulation_utils.py` (`SyntheticBanditEnv`,
`generate_dataset`). Real-world meaning of each bias type: [bias_examples.md](bias_examples.md).

The world has two copies of every user and item vector:

- **clean** vectors define the truth (the click model). Only the environment and the
  `oracle` reward model see them.
- **biased** vectors (`our_x`, `our_a`) are what the logger, the reward model and the learned
  policy see. Everything the learner knows comes from these.

## Build order (per dataset and seed)

1. **Clean vectors.** BPR's user and item factors `x`, `a` (taste) and its item bias `b_i`
   (popularity; `{dataset}_item_bias.npy`, see [BPR/README.md](../BPR/README.md)). The true
   score is

   ```text
   s(u, i) = x_u · a_i + β_true · b_i        β_true = --pop-strength (default 0)
   ```

   By default clicks follow personal taste only. `--pop-strength 1` puts popularity back as
   BPR learned it. The popularity term is carried as one extra column, users `[x, 1]` and
   items `[a, β·b]`, so every dot-product path (policies, exact values, samplers, the
   oracle) works unchanged. With both popularity weights 0 there is no extra column.
   `--env-centering λ` (default 0 = off) removes a share of the mean vector,
   `x ← x − λ·mean(x)` (users and items), for experiments that also remove the
   popularity-like direction all users share in the taste vectors.
2. **Representation bias**, applied to the taste part of both sides in this order. Each type
   mixes the current vectors with a target, `x ← (1 − ε)·x + ε·target`, and then rescales
   them back to the clean RMS. Targets are scale-matched to the clean vectors.
   - **warp:** one random linear map for the whole side: `target = x_clean @ W`.
   - **group:** one Gaussian offset per group: `target = template[group(x)]`. Groups are
     k-means clusters of the clean vectors (k = √n, clipped to 8–64). With
     `--bias-groups metadata`, groups are k-means clusters of the metadata arrays
     instead; a side without metadata falls back to clusters, and a message says so.
   - **vector:** one Gaussian offset per user or item.

   The logger adds its own popularity term: it scores `x̃_u · ã_i + β_log · b_i`, with
   β_log = `--logger-pop-strength` (default β_true). Above β_true the logger over-exposes
   popular items; below it, it under-exposes them. The truth does not depend on β_log.
3. **Levels.** Each type is `none`, `low`, `medium` or `high`. ε is calibrated per dataset
   so that **all three types at level L together** keep 90 / 75 / 50 % of the taste signal.
   Signal kept is the mean over users of the correlation, across items, between biased
   and clean taste scores. Each type then keeps a share κ_L on its own (about 0.97 / 0.94 / 0.88).
   Mixed configurations such as `high/none/low` land in between.
4. **Logging temperature T.** The clean softmax logger spreads over `--logging-spread`
   of the catalog (default 0.5). Spread is the effective number of items, `exp(entropy)`,
   averaged over users, of `softmax(s(u, ·) / T)`. The logger in every run is
   `softmax(our_x · our_a / T)`, optionally mixed with uniform (`--logging-uniform-mix`).
   The learned policy uses the same T, so at initialization it equals the logger.
5. **Click model.** `q(u, a) = sigmoid(α·z(u, a) + b)`, where z is the clean score s
   standardized over all user-item pairs:
   - α is set so the best item of each user averages `--best-ctr` (default 30%);
   - b is set so the **reference policy** has CTR `--ctr-levels` (default 5%). The
     reference policy is the logger at medium/medium/medium bias with the truth's
     popularity weight (`--ctr-reference logger`, default) or the uniform random policy
     (`--ctr-reference uniform`; H1 uses this).

   In code this is `SyntheticBanditEnv(scale=α/sd, offset=b − α·mean/sd)`.

All draws depend only on the seed. The levels are therefore nested (the same bias
directions at growing ε), and the truth (clean vectors, α, b, T, user prior) is identical
across bias configurations and logger popularity weights for a given dataset and seed.
Calibration runs in float64 numpy, so it gives the same result with or without a GPU. It
uses samples:
- 50k users and items for signal kept (all of them on smaller catalogs);
- 300 users for T;
- 1000 users drawn from the user prior × 2048 items for α and b.

It takes 3–20 s per dataset and is cached per process. Targets that cannot be met stop the
run with a `WorldCalibrationError` that names the missed target.

## Popularity in the models

Only when a popularity weight is positive (otherwise the runs are the taste-only ones):

- **Learned policy** (`CFModel`): `softmax((f(x_u) · g(a_i) + w · b_i) / T)`. The transforms f, g
  act on the taste part; w is one learnable weight on b, starting at β_log, so at
  initialization the policy still equals the logger. The weight is reported as `pop_weight`:
  per trial in the trials CSVs, and in the summary (the logger's at train size 0, the
  selected policy's otherwise).
- **Fitted reward model** (`--reward-model regression`): gets BPR's raw b as an item feature, in
  place of the logger's β_log·b (0 when the logger ignores popularity).
- `logging_score` uses the logger's view (β_log·b); `oracle` uses the truth (β_true·b).
- The legacy neighbourhood trainer computes its similarities on taste vectors only.

## Flags (all runners)

| Flag | Default | Meaning |
|------|---------|---------|
| `--bias-configs` | `low medium high` | one level for all three types, or `warp/group/vector` levels, e.g. `high/none/low` |
| `--bias-groups` | `cluster` | groups for the group bias: `cluster` or `metadata` |
| `--pop-strength` | 0 | β_true, weight of BPR's item bias in the true score (needs `{dataset}_item_bias.npy`) |
| `--logger-pop-strength` | = `--pop-strength` | β_log, the logger's weight on the item bias |
| `--env-centering` | 0 | λ, share of the mean vector removed (0 = off) |
| `--logging-spread` | 0.5 | clean logger's effective items / catalog |
| `--best-ctr` | 0.30 | best item per user, averaged over users |
| `--ctr-levels` | 0.05 | target CTR of the reference policy |
| `--ctr-reference` | `logger` | `logger` (at medium bias) or `uniform` (not on H1, which always uses uniform) |

Run folders are named `dataset=<ds>__bias=<label>__ctr=<ctr>__seed=<seed>[<world>][__val=<n>]`.
The label is the level when all three types share it (`medium`); otherwise it is
`w-high.g-none.v-low`. `<world>` lists the world options that differ from the defaults, e.g.
`__pop=1__logpop=2` or `__center=0.8__groups=metadata` (tags `pop`, `logpop`, `center`,
`spread`, `best`, `groups`, `ref`), so runs of different worlds never share a folder; runs
with default options keep the plain names. Summary CSVs keep the analysis columns:
`noise_mode=representation_bias`, `noise_axis=both`, `noise_level=<label>`, plus `bias_warp`,
`bias_group`, `bias_vector`, `signal_kept`, `logging_temperature`, `pop_strength` and
`logger_pop_strength`.

`run_meta.json → world` records everything calibrated:
- ε per type and level, κ per level, and the signal kept per level and for this run;
- T and the clean logger's effective items;
- α, b, scale and offset;
- the reference, uniform, best-item and logging CTRs;
- the popularity weights (`pop_strength`, `logger_pop_strength`) and `popularity`: whether the
  item bias is used, its sd, and the share of a user's clean-score variance the popularity
  term carries;
- the group source and counts;
- per-side RMS and cosine to clean (taste parts).

`run_meta.json → snr` holds vector-level SNR and cosine per side (`utils/noise_snr.py`, taste
parts).

## Characterize a dataset

```bash
python -m training.characterize_world --datasets ml myket --bias-configs all
```

This prints and writes `summary.csv` (signal kept, logging CTR, best/logging ratio, cosine
and SNR per configuration) and `calibration.json` under `artifacts/world/`. With `all`, it
covers the 64 combinations of four levels for three types.

## Calibrated values (seed 0, defaults)

From `python -m training.characterize_world --datasets ml myket kuairec kuairand anime lastfm msd`
on the BPR v2 embeddings (1 min 49 s for all seven, 4 GB peak RAM). Every target is met:
signal kept 0.90 / 0.75 / 0.50, clean logger over 50% of the catalog, logger at medium bias
5.0%, best item 30.0%.

| Dataset | Users × items | T | α | κ low / med / high | Uniform CTR | Logging CTR none / low / medium / high |
|---|---|---|---|---|---|---|
| ml | 6,038 × 3,533 | 2.01 | 0.936 | 0.968 / 0.924 / 0.847 | 2.7% | 7.9% / 6.3% / 5.0% / 3.9% |
| myket | 10,000 × 7,988 | 1.52 | 0.887 | 0.968 / 0.923 / 0.840 | 2.8% | 7.4% / 6.1% / 5.0% / 4.0% |
| kuairec | 7,175 × 10,611 | 1.66 | 0.887 | 0.973 / 0.942 / 0.887 | 3.1% | 8.5% / 6.2% / 5.0% / 4.1% |
| kuairand | 27,111 × 7,579 | 1.08 | 0.746 | 0.967 / 0.920 / 0.833 | 2.8% | 8.2% / 6.3% / 5.0% / 3.9% |
| anime | 73,417 × 10,803 | 2.58 | 0.755 | 0.970 / 0.929 / 0.851 | 2.8% | 7.5% / 6.1% / 5.0% / 4.0% |
| lastfm | 358,868 × 292,385 | 1.72 | 0.539 | 0.970 / 0.930 / 0.859 | 3.5% | 7.1% / 5.9% / 5.0% / 4.3% |
| msd | 1,019,318 × 42,053 | 1.33 | 0.630 | 0.965 / 0.909 / 0.800 | 2.7% | 6.6% / 5.9% / 5.0% / 3.9% |

One type alone at `high` (ε, signal kept, logging CTR):

| Dataset | warp | group | vector |
|---|---|---|---|
| ml | 0.43, 0.85, 5.9% | 0.41, 0.85, 5.8% | 0.40, 0.85, 5.8% |
| myket | 0.39, 0.84, 5.5% | 0.38, 0.84, 5.6% | 0.38, 0.84, 5.7% |
| kuairec | 0.45, 0.89, 5.9% | 0.43, 0.89, 6.1% | 0.42, 0.89, 6.3% |
| kuairand | 0.39, 0.83, 5.5% | 0.40, 0.83, 5.7% | 0.39, 0.83, 5.9% |
| anime | 0.36, 0.85, 5.7% | 0.41, 0.85, 5.7% | 0.39, 0.85, 5.7% |
| lastfm | 0.41, 0.86, 5.4% | 0.40, 0.86, 5.6% | 0.40, 0.86, 5.6% |
| msd | 0.36, 0.80, 5.4% | 0.36, 0.80, 5.2% | 0.36, 0.80, 5.2% |

### Popularity on

With `--pop-strength 1` (truth and logger weigh BPR's item bias as BPR learned it) every target
is still met. The popularity term then carries this share of a user's clean-score variance:
ml 4%, myket 5%, kuairec 6%, kuairand 14%, anime 7%, lastfm 8%, msd 48%.

The logger's weight moves the logging CTR (ml / kuairand, medium bias):

| Truth β_true | Logger β_log | Logging CTR |
|---|---|---|
| 1 | 0 (ignores popularity) | 4.3% / 3.9% |
| 1 | 1 (default) | 5.0% / 5.0% |
| 1 | 3 (over-exposes) | 6.4% / 8.5% |
| 0 | 3 | 7.0% / 9.7% |

Over-exposing popular items raises the logger's CTR even when the truth ignores popularity:
b correlates with the taste direction all users share (next section), so popular items are
liked on average, and the extra term also makes the logger sharper.

### How personal is the clean world?

500 random users per dataset. Top-20 overlap is the mean share of top-20 items that two
random users have in common; "∩ popular" is the share of a user's top-20 among the 20 items
with the largest b. At random both would be 20 / items (0.007% on lastfm to 0.6% on ml).

| Dataset | top-20 overlap: taste / taste + b | ∩ popular: taste / taste + b | corr(x̄·a, b) |
|---|---|---|---|
| ml | 0.12 / 0.15 | 0.10 / 0.12 | 0.88 |
| myket | 0.19 / 0.27 | 0.15 / 0.22 | 0.75 |
| kuairec | 0.20 / 0.19 | 0.01 / 0.02 | 0.83 |
| kuairand | 0.07 / 0.15 | 0.11 / 0.23 | 0.73 |
| anime | 0.06 / 0.14 | 0.13 / 0.23 | 0.91 |
| lastfm | 0.02 / 0.06 | 0.02 / 0.09 | 0.64 |
| msd | 0.01 / 0.10 | 0.00 / 0.20 | 0.52 |

By default (β_true = 0) clicks follow taste only, and every dataset is personal. The taste
vectors still share one direction, the mean user x̄: its item scores x̄·a correlate with b,
so part of what everyone likes stays in the truth. This is kept on purpose;
`--env-centering 0.8` removes most of it (top-20 overlap 0.00–0.06, ∩ popular 0.00–0.04).

With the v1 embeddings (no item bias, 17–51 updates per user on the large datasets), anime,
lastfm and msd were popularity-dominated: 98% of the item variance lay on one direction, and
top-20 overlap stayed at 0.47–0.65 even after centering. BPR v2 puts popularity in b
([BPR/README.md](../BPR/README.md)).
