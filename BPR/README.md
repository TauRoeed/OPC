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
