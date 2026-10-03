"""Match the PyTorch port's protocol results with the audit's TensorFlow runs, configuration by configuration.

The audit (~/code/CausE/repro/runs/_results.jsonl) ran the authors' unmodified code once per configuration
(seed 123). Our runs use several seeds. A configuration agrees when our mean is within 0.5 lift point
(MSE and NLL) and 0.005 AUC of the TF run, the criterion the audit proposed; for reference, the audit's
seed/split sd on ML-100K is 0.1-0.7 MSE-lift points.

Usage: python -m training.cause_repro_compare --ours artifacts/cause_repro/*.csv --out artifacts/cause_repro/compare
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_RESULTS = Path(os.path.expanduser("~/code/CausE/repro/runs/_results.jsonl"))
TOL = {"lift_mse": 0.5, "lift_nll": 0.5, "auc": 0.005}
# our method -> (audit kind, prediction key, data tag, tie forced to 0)
METHOD_MAP = {"CausE-prod-C": ("prod", "C", "adapt_2i", False), "CausE-prod-T": ("prod", "T", "adapt_2i", False),
              "CausE-avg": ("avg", "A", "adapt_0", False), "SP2V-no": ("avg", "A", "adapt_no", True),
              "SP2V-blend": ("avg", "A", "adapt_blend", True), "SP2V-test": ("avg", "A", "adapt_test", True)}
DATASET_DIR = {("ml100k", 0.10, 0): "ml100k_skew_s0", ("ml100k", 0.10, 1): "ml100k_skew_s1",
               ("ml100k", 0.15, 0): "ml100k_fig1_s0", ("ml10m", 0.10, 0): "ml10m_skew_s0"}


def _flag(flags: str, name: str, default):
    m = re.search(rf"--{name}\s+(\S+)", flags)
    return type(default)(m.group(1)) if m else default


def audit_table(path: Path = AUDIT_RESULTS) -> pd.DataFrame:
    rows = []
    for line in open(path):
        r = json.loads(line)
        flags = r.get("flags", "")
        kind = r.get("kind")
        if kind not in ("prod", "avg") or not r.get("C", r.get("A")):
            continue
        data_dir = _flag(flags, "data_set", "?").split("/")[0]
        tag = _flag(flags, "adapt_stat", "adapt_2i" if kind == "prod" else "adapt_0")
        level = None
        m = re.match(r"(adapt_\w+?)_st(\d{3})$", tag)
        if m:
            tag, level = m.group(1), int(m.group(2)) / 1000.0
        base = {"run": r["run"], "data_dir": data_dir, "data_tag": tag, "st_level": level,
                "epochs": _flag(flags, "num_epochs", 1), "l2_pen": _flag(flags, "l2_pen", 0.0),
                "cf_pen": _flag(flags, "cf_pen", 1.0), "lr": _flag(flags, "learning_rate", 1.0),
                "dim": _flag(flags, "embedding_size", 50), "kind": kind}
        for pred in ("C", "T", "A"):
            if pred in r and isinstance(r[pred], dict) and "mse_lift" in r[pred]:
                rows.append({**base, "pred": pred, "tf_lift_mse": r[pred]["mse_lift"], "tf_lift_nll": r[pred]["nll_lift"],
                             "tf_auc": r[pred]["auc"], "tf_alpha": (r.get("ckpt") or {}).get("alpha")})
    return pd.DataFrame(rows)


def compare(ours: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "split_seed", "st_train_frac", "st_level", "method", "epochs", "l2_pen", "cf_pen", "lr", "dim"]
    ours = ours.copy()
    if "st_train_frac" not in ours:
        ours["st_train_frac"] = 0.10
    ours["st_level"] = ours.get("st_level")
    for k, g in ours.groupby(keys, dropna=False):
        cfg = dict(zip(keys, k))
        kind, pred, tag, no_tie = METHOD_MAP[cfg["method"]]
        data_dir = DATASET_DIR.get((cfg["dataset"], round(float(cfg["st_train_frac"]), 2), int(cfg["split_seed"])))
        level = None if pd.isna(cfg["st_level"]) else float(cfg["st_level"])
        cf = 0.0 if no_tie else float(cfg["cf_pen"])
        sel = audit[(audit["data_dir"] == data_dir) & (audit["kind"] == kind) & (audit["pred"] == pred)
                    & (audit["data_tag"] == tag) & (audit["epochs"] == int(cfg["epochs"]))
                    & np.isclose(audit["l2_pen"], float(cfg["l2_pen"])) & np.isclose(audit["cf_pen"], cf)
                    & np.isclose(audit["lr"], float(cfg["lr"])) & (audit["dim"] == int(cfg["dim"]))]
        sel = sel[sel["st_level"].isna()] if level is None else sel[np.isclose(sel["st_level"].fillna(-1), level)]
        row = {**cfg, "n_seeds": len(g), "audit_run": sel["run"].iloc[0] if len(sel) else None}
        for m in ("lift_mse", "lift_nll", "auc", "alpha"):
            row[f"ours_{m}"] = float(g[m].mean())
            row[f"ours_{m}_sd"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
        for m in ("lift_mse", "lift_nll", "auc"):
            tf = float(sel[f"tf_{m}"].iloc[0]) if len(sel) else np.nan
            row[f"tf_{m}"] = tf
            row[f"diff_{m}"] = row[f"ours_{m}"] - tf
        row["tf_alpha"] = sel["tf_alpha"].iloc[0] if len(sel) else None
        row["agrees"] = bool(len(sel)) and all(abs(row[f"diff_{m}"]) <= TOL[m] for m in TOL)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    def cell(v):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ours = pd.concat([pd.read_csv(p) for p in args.ours], ignore_index=True)
    table = compare(ours, audit_table())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "port_vs_tf.csv", index=False)
    cols = ["dataset", "st_level", "method", "epochs", "l2_pen", "cf_pen", "n_seeds", "ours_lift_mse", "ours_lift_mse_sd",
            "tf_lift_mse", "ours_lift_nll", "tf_lift_nll", "ours_auc", "tf_auc", "agrees"]
    (out / "port_vs_tf.md").write_text(markdown_table(table[cols].round(4)))
    matched = table["audit_run"].notna()
    print(f"{int(matched.sum())} configurations matched to TF runs; {int(table.loc[matched, 'agrees'].sum())} agree "
          f"within {TOL}")
    print(table.loc[matched, cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
