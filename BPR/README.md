# BPR Artifact Generation

Use `BPR/generate_artifacts.py` to create all dataset artifacts in one pass:

- user/item BPR factors
- item metadata matrix
- optional user metadata matrix (only when standalone user metadata exists)
- user interaction counts

## Example

```bash
python -m BPR.generate_artifacts \
  --dataset ml \
  --root /home/roee/Documents/git-repos/OPC/datasets/ml-1m \
  --emb-dir /home/roee/Documents/git-repos/OPC/BPR/embeddings
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
