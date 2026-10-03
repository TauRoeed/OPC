"""CausE's own experimental protocol (MovieLens SKEW split) and the released evaluation, for the reproduction.

The authors never released the script that builds their data files (upstream issues #1, #2, #5). This module is
a port of the standalone audit's reconstruction (~/code/CausE/repro/scripts/gen_dataset_reconstructed.py at
f536ea6); it makes the same numpy RandomState calls in the same order, so with the same raw data and seed it
gives the audit's files row for row (tests/test_cause_protocol.py). Its documented deviations D1-D6 hold here:
    D1 acceptance a_j = min(cap, kappa / c_j), kappa solved so the uniform sample has the stated size;
    D2 popularity = number of ratings; D3 uniform sample = 20% test + 10% S_t train + 5% S_t validation,
       complement = 60% S_c train + 5% S_c validation; D4 one seeded global order for all training files;
    D5 raw user ids, contiguous item ids 1..n_items with id 0 free; D6 per-event Bernoulli acceptance.
Paper (v6 §4.1.4): five-star ratings -> 1, else 0; a uniform-exposure test set by popularity-inverse acceptance
capped at 0.9; train = 60% from pi_c + 10% from pi_t, validation from pi_c, test 20% from pi_t.

Evaluation follows src/utils.py: 30 bootstrap resamples of 80% of the test set (``random.seed(2 * i)``),
MSE and NLL lift over the "average predictor" and AUC. The released average predictor feeds the test odds
clicks/non-clicks as a probability (audit V5); ``lift_*_rate`` uses the click rate instead.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.cause import CausEModel, epoch_batches, fit_cause, predict_logits

TF_LOG_LOSS_EPS = 1e-7  # tf.losses.log_loss epsilon
SKEW_DEFAULTS = {"cap": 0.9, "test_frac": 0.20, "st_train_frac": 0.10, "valid_test_frac": 0.05, "valid_train_frac": 0.05}
# data tag -> (S_c rows, S_t rows, how S_t product ids are written)
TAGS = {"adapt_0": ("sc", "st", "zero"), "adapt_2i": ("sc", "st", "twin"), "adapt_no": ("sc", None, "real"),
        "adapt_blend": ("sc", "st", "real"), "adapt_test": (None, "st", "real")}


def load_ratings(dataset: str, raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = str(raw_dir)
    if dataset == "ml100k":
        return pd.read_csv(os.path.join(raw_dir, "ml-100k", "u.data"), sep="\t", header=None, names=["u", "i", "r", "t"])
    if dataset == "ml10m":
        with open(os.path.join(raw_dir, "ml-10M100K", "ratings.dat"), "rb") as f:
            b = f.read().replace(b"::", b",")
        return pd.read_csv(io.BytesIO(b), header=None, names=["u", "i", "r", "t"])
    raise ValueError(dataset)


def solve_kappa(counts: np.ndarray, target_frac: float, cap: float) -> float:
    """kappa with sum_j c_j * min(cap, kappa / c_j) = target_frac * sum_j c_j (bisection, as the audit)."""
    total = counts.sum()
    lo, hi = 0.0, float(counts.max()) / cap * 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        acc = (counts * np.minimum(cap, mid / counts)).sum() / total
        if acc < target_frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_skew_split(dataset: str, raw_dir, seed: int = 0, st_levels=(), **fractions) -> dict:
    """The reconstructed SKEW split: index sets of every part, the global training order and the id spaces."""
    f = {**SKEW_DEFAULTS, **fractions}
    rng = np.random.RandomState(int(seed))
    df = load_ratings(dataset, raw_dir)
    n = len(df)
    u = df["u"].values.astype(np.int64)
    y = (df["r"].values >= 5).astype(np.int64)
    uniq_items, item_idx = np.unique(df["i"].values, return_inverse=True)
    i = item_idx.astype(np.int64) + 1
    n_items = len(uniq_items)
    counts = np.bincount(i, minlength=n_items + 1)[1:].astype(np.float64)
    uniform_frac = f["test_frac"] + f["st_train_frac"] + f["valid_test_frac"]
    kappa = solve_kappa(counts, uniform_frac, f["cap"])
    accept_item = np.minimum(f["cap"], kappa / counts)
    in_uniform = rng.uniform(size=n) < accept_item[i - 1]
    U = np.where(in_uniform)[0]
    C = np.where(~in_uniform)[0]
    U = U[rng.permutation(len(U))]
    C = C[rng.permutation(len(C))]
    n_test = int(round(len(U) * f["test_frac"] / uniform_frac))
    n_vt = int(round(len(U) * f["valid_test_frac"] / uniform_frac))
    test_idx, vt_idx, st_idx = U[:n_test], U[n_test:n_test + n_vt], U[n_test + n_vt:]
    n_vc = int(round(len(C) * f["valid_train_frac"] / (1.0 - uniform_frac)))
    vc_idx, sc_idx = C[:n_vc], C[n_vc:]
    global_order = np.concatenate([sc_idx, st_idx])
    global_order = global_order[rng.permutation(len(global_order))]
    levels = {}
    for lv in st_levels:
        k = int(round(len(st_idx) * float(lv) / f["st_train_frac"]))
        if k > len(st_idx):
            raise ValueError("level exceeds the pi_t pool")
        levels[float(lv)] = st_idx[:k]  # nested prefixes of one permutation
    return {"dataset": dataset, "seed": int(seed), "u": u, "i": i, "y": y, "n_items": n_items,
            "num_products": n_items + 1, "num_users": int(u.max()) + 1, "kappa": float(kappa), "accept_item": accept_item,
            "in_uniform": in_uniform, "test": test_idx, "valid_test": vt_idx, "valid_train": vc_idx, "sc": sc_idx,
            "st": st_idx, "global_order": global_order, "st_levels": levels, "fractions": f}


def tag_rows(split: dict, tag: str, st_rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(users, product ids, labels) of a training file in the released conventions, in the global order.

    ``st_rows`` overrides the S_t rows (the nested injection levels of the Fig. 1 protocol)."""
    sc_key, st_key, mode = TAGS[tag]
    n = len(split["y"])
    keep = np.zeros(n, dtype=bool)
    if sc_key:
        keep[split[sc_key]] = True
    if st_key is None:
        st = np.array([], dtype=np.int64)
    else:
        st = split[st_key] if st_rows is None else np.asarray(st_rows, dtype=np.int64)
    keep[st] = True
    is_st = np.zeros(n, dtype=bool)
    is_st[st] = True
    rows = split["global_order"][keep[split["global_order"]]]
    ti = split["i"][rows].copy()
    st_mask = is_st[rows]
    if mode == "zero":
        ti[st_mask] = 0
    elif mode == "twin":
        ti[st_mask] = ti[st_mask] + split["num_products"]
    return split["u"][rows], ti, split["y"][rows]


def eval_rows(split: dict, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = split[part]
    return split["u"][idx], split["i"][idx], split["y"][idx]


def _tf_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(-y * np.log(p + TF_LOG_LOSS_EPS) - (1 - y) * np.log(1 - p + TF_LOG_LOSS_EPS)))


def upstream_bootstrap_metrics(logits: np.ndarray, labels: np.ndarray, n_boot: int = 30) -> dict:
    """src/utils.compute_bootstraps: 30 resamples of 80% (random.seed(2i)); lifts in %, as printed upstream."""
    from sklearn import metrics

    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(y)
    odds = float(np.count_nonzero(y)) / float(len(np.where(y == 0)[0]))  # compute_empircal_cr
    rate = float(y.mean())
    out = {k: [] for k in ("mse", "nll", "auc", "lift_mse", "lift_nll", "lift_mse_rate", "lift_nll_rate")}
    for b in range(n_boot):
        random.seed(b * 2)
        ids = np.array([random.randint(0, n - 1) for _ in range(int(n * 0.8))])
        zb, yb = z[ids], y[ids]
        pb = 1.0 / (1.0 + np.exp(-zb))
        mse = float(np.mean((yb - pb) ** 2))
        nll = float(np.mean(np.maximum(zb, 0) - zb * yb + np.log1p(np.exp(-np.abs(zb)))))
        ap_mse, ap_nll = float(np.mean((yb - odds) ** 2)), _tf_log_loss(yb, np.full_like(yb, odds))
        r_mse, r_nll = float(np.mean((yb - rate) ** 2)), _tf_log_loss(yb, np.full_like(yb, rate))
        out["mse"].append(mse)
        out["nll"].append(nll)
        out["auc"].append(float(metrics.roc_auc_score(y_true=yb.astype(int), y_score=pb)))
        out["lift_mse"].append((ap_mse - mse) / ap_mse)
        out["lift_nll"].append((ap_nll - nll) / ap_nll)
        out["lift_mse_rate"].append((r_mse - mse) / r_mse)
        out["lift_nll_rate"].append((r_nll - nll) / r_nll)
    res = {}
    for k, v in out.items():
        scale = 100.0 if k.startswith("lift") else 1.0
        res[k] = float(np.mean(v)) * scale
        res[k + "_sd"] = float(np.std(v)) * scale
    return res


def protocol_model(split: dict, model: str, *, dim: int, generator: torch.Generator,
                   emulate_tf_pooled_rounding: bool = False) -> CausEModel:
    """``prod`` (CausalProd2Vec2i), ``avg`` (CausalProd2Vec) or ``sp2v`` (CausalProd2Vec with cf_pen = 0)."""
    n_users, n_p = split["num_users"], split["num_products"]
    if model == "prod":
        return CausEModel(n_users, 2 * n_p, dim, variant="prod", tie_offset=n_p, generator=generator)
    return CausEModel(n_users, n_p, dim, variant="avg", pooled_row=0, generator=generator,
                      emulate_tf_pooled_rounding=emulate_tf_pooled_rounding)


DATA_TAG = {"prod": "adapt_2i", "avg": "adapt_0", "sp2v_no": "adapt_no", "sp2v_blend": "adapt_blend", "sp2v_test": "adapt_test"}


def run_config(split: dict, method: str, *, epochs: int, lr: float = 1.0, l2_pen: float = 0.0, cf_pen: float = 1.0,
               dim: int = 50, batch_size: int = 512, optimizer: str = "sgd", symmetric: bool = False, seed: int = 0,
               st_rows=None, device="cpu", emulate_tf_pooled_rounding: bool = False) -> list[dict]:
    """Train one released-recipe configuration and evaluate it like the upstream scripts (one row per prediction)."""
    model_kind = "prod" if method == "prod" else ("avg" if method == "avg" else "sp2v")
    tag = DATA_TAG[method]
    users, prods, labels = tag_rows(split, tag, st_rows=st_rows)
    gen = torch.Generator().manual_seed(int(seed))
    model = protocol_model(split, model_kind, dim=dim, generator=gen, emulate_tf_pooled_rounding=emulate_tf_pooled_rounding)
    t0 = time.time()
    info = fit_cause(model, users, prods, labels, epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr=lr,
                     l2_pen=l2_pen, cf_pen=0.0 if model_kind == "sp2v" else cf_pen, symmetric=symmetric,
                     seed=int(seed) + 1, device=device)
    seconds = time.time() - t0
    tu, ti, ty = eval_rows(split, "test")
    rows = []
    preds = [("C", 0), ("T", split["num_products"])] if model_kind == "prod" else [("", 0)]
    for side, offset in preds:
        z = predict_logits(model, tu, ti + offset)
        vals = {}
        for part in ("valid_train", "valid_test"):
            vu, vi, vy = eval_rows(split, part)
            vz = predict_logits(model, vu, vi + offset)
            vals[f"{part}_nll"] = float(np.mean(np.maximum(vz, 0) - vz * vy + np.log1p(np.exp(-np.abs(vz)))))
        name = {"prod": f"CausE-prod-{side}", "avg": "CausE-avg"}.get(method, method.replace("sp2v_", "SP2V-"))
        rows.append({"method": name, "data_tag": tag, "epochs": epochs, "lr": lr, "l2_pen": l2_pen,
                     "cf_pen": cf_pen if model_kind != "sp2v" else 0.0, "dim": dim, "optimizer": optimizer,
                     "symmetric": symmetric, "emulate_tf_pooled_rounding": bool(emulate_tf_pooled_rounding),
                     "seed": seed, "n_train": int(len(labels)), "alpha": info["alpha"],
                     "finite": info["finite"], "seconds": seconds, **vals, **upstream_bootstrap_metrics(z, ty)})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Reproduce CausE's MovieLens SKEW experiments with the PyTorch port.")
    ap.add_argument("--dataset", default="ml100k", choices=["ml100k", "ml10m"])
    ap.add_argument("--raw-dir", default=os.path.expanduser("~/code/CausE/repro/.local/raw"))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", required=True, help="JSON list of run_config kwargs, each with 'method'")
    ap.add_argument("--st-train-frac", type=float, default=SKEW_DEFAULTS["st_train_frac"],
                    help="size of the pi_t training pool (the audit's Fig. 1 split uses 0.15)")
    ap.add_argument("--st-levels", type=float, nargs="+", default=None,
                    help="Fig. 1: inject these fractions of all events as pi_t rows (nested), instead of the whole pool")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = build_skew_split(args.dataset, args.raw_dir, seed=args.split_seed, st_levels=tuple(args.st_levels or ()),
                             st_train_frac=args.st_train_frac)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = json.loads(Path(args.configs).read_text()) if os.path.exists(args.configs) else json.loads(args.configs)
    rows = []
    for level in (args.st_levels or [None]):
        st_rows = None if level is None else split["st_levels"][float(level)]
        for cfg in configs:
            for s in args.seeds:
                for r in run_config(split, **{**cfg, "seed": s}, st_rows=st_rows, device=device):
                    rows.append({"dataset": args.dataset, "split_seed": args.split_seed, "st_train_frac": args.st_train_frac,
                                 "st_level": level, **r})
                    print({k: r[k] for k in ("method", "epochs", "l2_pen", "cf_pen", "seed", "lift_mse", "lift_nll", "auc",
                                             "alpha")}, {"st_level": level}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
