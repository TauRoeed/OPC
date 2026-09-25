# BPR Artifact Generation

Use `BPR/generate_artifacts.py` to create all dataset artifacts in one pass:

- user/item BPR factors, the item bias, and a meta file recording how they were trained
- item metadata matrix
- optional user metadata matrix (only when standalone user metadata exists)
- user interaction counts

## BPR v2 (`BPR/bpr_minibatch.py`)

```
score(u, i) = b_i + x_u · a_i                        (b_i with item_bias; 32 dimensions)
loss(u,i,j) = −ln σ(score(u, i) − score(u, j)) + (λ/2)(‖x_u‖² + ‖a_i‖² + ‖a_j‖²) + (λ_b/2)(b_i² + b_j²)
```

Each triple is a user u, an item i they liked and an item j they did not.

- **Sampling.** Liked items per interaction (`sampling: interaction`, every interaction equally
  likely) or per user (`user`). Negatives are uniform over the catalog (`negatives: uniform`) or
  proportional to popularity (`popularity`, n_j ** `negative_gamma`); a negative the user liked
  is redrawn.
- **Updates.** Mini-batches of `batch_size` triples; the per-triple gradients of each touched row
  are summed and the row takes one Adagrad step (`learning_rate`), so a popular item drawn many
  times in a batch does not take one huge step. Numpy only: the same seed gives the same vectors
  on any machine, with or without a GPU (`--cpu-threads` fixes the BLAS threads).
- **Early stopping.** One random liked item is held out for each user with at least 3 liked items.
  After every epoch (one pass over the training interactions) recall@20 and NDCG@20 are measured
  on up to 5,000 of those users, ranking only items they did not like in training. Training stops
  after `patience` epochs without a 0.1% gain, and by default refits on all interactions for the
  best number of epochs (`refit`). With nobody to hold out, it trains `epochs` fixed epochs.

Defaults (`BPR/bpr_dataset_config.json`): 32 factors, item bias, Adagrad step 0.1,
λ = λ_b = 1e-4, batch 8,192, early stopping with patience 5 and at most 100 epochs, refit, seed 0.
The negatives and the sampling are chosen per dataset (below). Every setting can be overridden on
the command line (`--no-item-bias`, `--negatives popularity`, `--sampling user`, `--learning-rate`, ...).

### How the per-dataset settings were chosen

All datasets use uniform negatives, so popularity is learned by the item bias and can be set in
the simulator. The sampling is the one with the higher held-out recall@20 (seed 0, same split);
popularity negatives were also tried with both samplings. "Popularity ranking" ranks every item by its number of likes; "old" is the previous
trainer and settings on that split.

| dataset | popularity ranking | old | best, uniform negatives | best, popularity negatives | chosen |
|---------|-------------------:|----:|------------------------:|---------------------------:|--------|
| ml | 0.147 | 0.207 | **0.288** (per user) | 0.235 | uniform, per user |
| myket | 0.130 | 0.149 | **0.174** (per interaction) | 0.125 | uniform, per interaction |
| kuairec | 0.081 | 0.125 | **0.131** (per user) | 0.113 | uniform, per user |
| kuairand | 0.087 | 0.122 | **0.133** (per interaction) | 0.109 | uniform, per interaction |
| anime | 0.155 | 0.167 | **0.358** (per user) | 0.343 | uniform, per user |
| lastfm | 0.055 | 0.055 | 0.174 (per interaction) | **0.197** (per user) | uniform, per interaction |
| msd | 0.089 | 0.091 | **0.164** (per user) | 0.145 | uniform, per user |

The old embeddings on lastfm and msd were no better than popularity ranking. Popularity negatives
rank best only on lastfm (+13% recall@20), but there the item bias stops tracking popularity (rank
correlation with the number of likes 0.07, against 0.88–0.99 with uniform negatives), which would
leave the simulator's popularity setting nothing to act on; lastfm therefore uses uniform
negatives like the rest.

```bash
python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m
python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m --negatives popularity --emb-dir BPR/embeddings/popneg
```

The item bias is written for the simulator's popularity term, which the study code does not use
yet: until it does, generate v2 embeddings into a separate `--emb-dir` rather than over the ones a
study uses.

### Data prep per dataset

| dataset | an item counts as liked when the user... |
|---------|------------------------------------------|
| ml | rated the movie 4 or 5 |
| myket | installed the app |
| anime | rated the title 7 or more, or watched it without rating it |
| kuairec | watched the video for at least twice its length (`watch_ratio >= 2`) |
| kuairand | clicked the video (`is_click`) |
| lastfm, msd | played the artist at least once |

The previous trainer (`BPR/bpr.py`, one SGD step per sampled triple in a Python loop, a fixed
step budget, no bias) is kept for comparisons only.

## Side metadata (lastfm / msd)

The HDF5 interaction files carry no features, so the loaders join optional side
files (auto-downloaded unless `--no-download`; skipped with a warning if missing):

| dataset | metadata | source (saved under) |
|---------|----------|----------------------|
| lastfm | user gender, age (10–80, else median), country (top 20 + other) | Last.fm-360K release, `datasets/lastfm/` (~543MB archive) |
| lastfm | artist tags (top 50), matched by lowercase name to MSD artists | MSD Last.fm tags, `datasets/msd/` |
| msd | artist majority tagtraum genre (15) + tags (top 50) | `unique_tracks.txt`, `msd_tagtraum_cd2.cls`, `lastfm_tags.db` (~650MB), `datasets/msd/` |

lastfm reads the MSD side files from the sibling `msd/` folder of its HDF5.

## Output files

For `<dataset>` in `{ml,myket,anime,lastfm,msd,kuairec,kuairand}`:

- `BPR/embeddings/<dataset>_user_factors.npy`
- `BPR/embeddings/<dataset>_item_factors.npy`
- `BPR/embeddings/<dataset>_item_bias.npy` (with the item bias; removed when training without it)
- `BPR/embeddings/<dataset>_bpr_meta.json`: settings, data prep, a data fingerprint, the
  validation curve and best epoch, runtime and the git commit. Study runs copy it into
  `run_meta.json` (`bpr`) and warn when it is missing or its settings differ from the config.
- `BPR/embeddings/<dataset>_item_metadata.npy`
- `BPR/embeddings/<dataset>_user_metadata.npy` (only when available)
- `BPR/embeddings/<dataset>_user_interaction_counts.npy`

These files are git-ignored: each machine generates them once.

## Alignment guarantees

- Item metadata rows are aligned to BPR item factor rows.
- User metadata rows are aligned to BPR user factor rows when user metadata exists.
- Metadata dimensions are dataset-specific and may differ across datasets.
