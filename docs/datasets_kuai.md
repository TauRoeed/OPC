# KuaiRec / KuaiRand Datasets

## Why these datasets

- **KuaiRec** — near-fully observed short-video matrix; good for denser BPR GT embeddings and controlled semi-synthetics.
- **KuaiRand-Pure** — includes **random exposure** logs; natural fit for off-policy / propensity stories (OPC thesis).

We use **KuaiRand-Pure** only (~194MB), not 1K/27K.

## Download

Zenodo:

- KuaiRec: `https://zenodo.org/records/18164998/files/KuaiRec.zip`
- KuaiRand-Pure: `https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz`

Auto via loaders:

```bash
python -m BPR.generate_artifacts \
  --dataset kuairec \
  --root datasets/kuairec \
  --emb-dir BPR/embeddings

python -m BPR.generate_artifacts \
  --dataset kuairand \
  --root datasets/kuairand-pure \
  --emb-dir BPR/embeddings
```

## Positive definitions

| Dataset | Code | Default positive |
|---------|------|------------------|
| KuaiRec | `kuairec` | `watch_ratio >= 2.0` on `big_matrix.csv` |
| KuaiRand | `kuairand` | `is_click == 1` (fallback `long_view`) on concat of all `log_*.csv` |

Overrides: `BPR/bpr_dataset_config.json` → `data.watch_ratio_min`, `data.matrix`, `data.positive_col`.

## Metadata

- KuaiRec: item `feat` tags; user `onehot_feat*`.
- KuaiRand: item `video_type` / `upload_type` / `tag` / `video_duration`; user `onehot_feat*`.

## Tests

Offline fixtures (no network):

```bash
python -m unittest tests.test_kuai_loaders
```

## Status

Loaders + BPR wiring done. Full BPR fit / OPC study on Kuai **not** run in this phase (large).
