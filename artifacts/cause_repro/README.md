# CausE reproduction with the PyTorch port (development)

These files hold the port's results on CausE's own protocol, compared with the standalone audit's runs of the unmodified TensorFlow
code (`~/code/CausE/repro/runs/_results.jsonl`, audit commit f536ea6). The split is the audit's reconstructed SKEW split, ported to
`training/cause_protocol.py`. The recipe is the released one: plain SGD at lr 1.0, batch 512, d = 50.

| file | contents |
|---|---|
| `ml100k_s0_configs.json`, `ml100k_s0_port.csv` | ML-100K split 0: CausE-prod (C and T), CausE-avg and SP2V at 1–300 epochs and tie strengths 0–100; seeds 0, 1, 2 |
| `compare_ml100k_s0/port_vs_tf.{csv,md}` | each configuration matched to its TF run; agreement within 0.5 lift point and 0.005 AUC |
| `ml100k_avg_emul_configs.json`, `ml100k_s0_avg_emul.csv` | CausE-avg with the TF rounding artifact emulated (`emulate_tf_pooled_rounding`) |
| `ml100k_fig1_configs.json`, `ml100k_fig1_port.csv`, `compare_ml100k_fig1/` | Fig. 1 protocol: randomized share 0–15% of all events; seeds 0, 1 |
| `ml10m_configs.json`, `ml10m_s0_port.csv` | ML-10M: the released default (1 epoch) and the README configuration (10 epochs) |

Rebuild one file, for example:

```bash
python -m training.cause_protocol --dataset ml100k --split-seed 0 --seeds 0 1 2 --configs artifacts/cause_repro/ml100k_s0_configs.json --out artifacts/cause_repro/ml100k_s0_port.csv
python -m training.cause_repro_compare --ours artifacts/cause_repro/ml100k_s0_port.csv --out artifacts/cause_repro/compare_ml100k_s0
```

The data need the audit's raw MovieLens files (`~/code/CausE/repro/.local/raw`). Every run is a development run. The interpretation
is in `docs/cause_baseline.md` §10.
