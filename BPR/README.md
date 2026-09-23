# BPR Artifact Generation

Use `BPR/generate_artifacts.py` to create all dataset artifacts in one pass:

- user/item BPR factors
- item metadata matrix
- optional user metadata matrix (only when standalone user metadata exists)
- user interaction counts

## Per-dataset params

Defaults come from `BPR/bpr_dataset_config.json` (matches `BPR/datasets.ipynb`). CLI flags override.

| dataset | lr | reg | epochs | samples/epoch | data prep |
|---------|-----|-----|--------|---------------|-----------|
| ml | 0.05 | 1e-4 | 30 | 100k | rating ≥ 4 |
| myket | 0.05 | 1e-4 | 30 | 150k | raw user ids |
| lastfm | 0.1 | 1e-5 | 30 | 250k | implicit |
| msd | 0.1 | 1e-5 | 50 | 350k | implicit |
| anime | 0.1 | 1e-5 | 15 | 250k | min rating 7 |

```bash
python -m BPR.generate_artifacts --dataset ml --root datasets/ml-1m
python -m BPR.generate_artifacts --dataset ml --config path/to/custom.json --epochs 10
```

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

For `<dataset>` in `{ml,myket,anime,lastfm,msd}`:

- `BPR/embeddings/<dataset>_user_factors.npy`
- `BPR/embeddings/<dataset>_item_factors.npy`
- `BPR/embeddings/<dataset>_item_metadata.npy`
- `BPR/embeddings/<dataset>_user_metadata.npy` (only when available)
- `BPR/embeddings/<dataset>_user_interaction_counts.npy`

## Alignment guarantees

- Item metadata rows are aligned to BPR item factor rows.
- User metadata rows are aligned to BPR user factor rows when user metadata exists.
- Metadata dimensions are dataset-specific and may differ across datasets.
