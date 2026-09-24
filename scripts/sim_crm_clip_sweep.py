#!/usr/bin/env python3
"""Sweep CRM clip_m (training) vs true V(π) on ML noise — find best clip for OPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from models.custom_losses import KLCRMPolicyLoss
from models.models import CFModel, RegressionModel, SingleMLPTransform
from training.trainer_trials import (
    RegressionScoresLookup,
    _build_regression_logged_split,
    _policy_reward_from_embeddings,
    fit_shared_regression_bundle,
)
from training.training_utils import train as train_cf
from utils.simulation_utils import CustomCFDatasetPS, ensure_exact_env_q_cache, generate_dataset


def _build_dataset(emb_dir: Path, *, bias: str, seed: int, ctr: float) -> dict:
    """ML world at one representation-bias configuration (utils/representation_bias.py)."""
    emb_x = np.load(emb_dir / "ml_user_factors.npy")
    emb_a = np.load(emb_dir / "ml_item_factors.npy")
    return generate_dataset({"bias": bias, "ctr": float(ctr)}, seed=seed, emb_a=emb_a, emb_x=emb_x)


def _train_once(
    dataset: dict,
    train_data: dict,
    scores_lookup: RegressionScoresLookup,
    *,
    clip_m: float,
    crm_lambda: float,
    kl_gamma: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    """Return (true_v, init_v) after training with given clip_m."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    init_v = float(_policy_reward_from_embeddings(dataset, our_x, our_a))

    model = CFModel(
        int(dataset["n_users"]),
        int(dataset["n_actions"]),
        int(dataset["emb_dim"]),
        initial_user_embeddings=torch.as_tensor(our_x, device=device, dtype=torch.float32),
        initial_actions_embeddings=torch.as_tensor(our_a, device=device, dtype=torch.float32),
        user_transform=SingleMLPTransform(int(dataset["emb_dim"])),
        action_transform=SingleMLPTransform(int(dataset["emb_dim"])),
        temperature=float(dataset.get("policy_temperature", 1.0)),
    ).to(device)

    cf = CustomCFDatasetPS(
        train_data["x_idx"],
        train_data["a"],
        train_data["r"],
        train_data["pscore"],
    )
    loader = DataLoader(
        cf,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    criterion = KLCRMPolicyLoss(
        gamma=float(kl_gamma),
        clip_m=float(clip_m),
        crm_lambda=float(crm_lambda),
        use_log_trick=True,
        propensity_mode="logged",
        iw_mode="clip",
    ).to(device)
    train_cf(
        model,
        loader,
        scores_lookup,
        criterion,
        num_epochs=int(epochs),
        lr=float(lr),
        lr_decay=0.95,
        device=str(device),
    )
    model.eval()
    ux, ua = model.get_params()
    true_v = float(
        _policy_reward_from_embeddings(
            dataset,
            ux.detach().cpu().numpy(),
            ua.detach().cpu().numpy(),
        )
    )
    return true_v, init_v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/oom_smoke/crm_clip_sweep"))
    p.add_argument("--n-train", type=int, default=50_000)
    p.add_argument("--reg-size", type=int, default=25_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--kl-gamma", type=float, default=0.05)
    p.add_argument("--crm-lambda", type=float, default=0.1)
    p.add_argument("--ctr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument(
        "--clips",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 1e9],
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bias applies to users and items; group = the old cluster cells, warp = the old linear cells
    cells = [
        ("group_high", "none/high/none"),
        ("warp_high", "high/none/none"),
    ]
    # 1e9 ≈ unclipped (raw IW in CRM variance term)
    clips = [float(c) if float(c) < 1e8 else float("inf") for c in args.clips]

    rows = []
    for cell_name, bias in cells:
        for seed in args.seeds:
            print(f"=== {cell_name} seed={seed} ===", flush=True)
            ds = _build_dataset(
                args.emb_dir,
                bias=bias,
                seed=int(seed),
                ctr=float(args.ctr),
            )
            ensure_exact_env_q_cache(ds)
            split = _build_regression_logged_split(
                ds,
                ds["our_x"],
                ds["our_a"],
                int(args.n_train),
                int(max(5_000, args.n_train // 5)),
                0,
                regression_size=int(args.reg_size),
            )
            bundle = fit_shared_regression_bundle(
                ds,
                split["reg_data"],
                reward_model="regression",
                materialize_qhat="always",
            )
            scores = RegressionScoresLookup(
                bundle["regression_model"],
                bundle["user_context"],
                device,
                q_hat_all=bundle.get("q_hat_all"),
            )
            for clip_m in clips:
                label = "inf" if not np.isfinite(clip_m) else f"{clip_m:g}"
                print(f"  clip={label}", flush=True)
                true_v, init_v = _train_once(
                    ds,
                    split["train_data"],
                    scores,
                    clip_m=1e12 if not np.isfinite(clip_m) else float(clip_m),
                    crm_lambda=float(args.crm_lambda),
                    kl_gamma=float(args.kl_gamma),
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    batch_size=int(args.batch_size),
                    device=device,
                    seed=int(seed) + 17,
                )
                rows.append(
                    {
                        "cell": cell_name,
                        "seed": int(seed),
                        "clip_m": None if not np.isfinite(clip_m) else float(clip_m),
                        "clip_label": label,
                        "true_v": true_v,
                        "init_v": init_v,
                        "lift": true_v - init_v,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "crm_clip_sweep.csv", index=False)

    # Aggregate: mean lift across cells×seeds (higher better)
    summary = (
        df.groupby("clip_label", dropna=False)
        .agg(
            mean_true_v=("true_v", "mean"),
            mean_lift=("lift", "mean"),
            std_lift=("lift", "std"),
            n=("lift", "count"),
        )
        .reset_index()
        .sort_values("mean_lift", ascending=False)
    )
    # Order numeric clips for readability in secondary table
    summary.to_csv(out / "crm_clip_summary.csv", index=False)
    best = summary.iloc[0]
    rec = {
        "winner_clip_label": str(best["clip_label"]),
        "winner_clip_m": None
        if str(best["clip_label"]) == "inf"
        else float(best["clip_label"]),
        "mean_lift": float(best["mean_lift"]),
        "mean_true_v": float(best["mean_true_v"]),
        "note": (
            "CRM clip_m only affects the CRM variance term in kl_crm; "
            "SNDR/DR surrogate and Optuna selection use unclipped IW. "
            "Winner maximizes mean true_v lift vs logging init."
        ),
    }
    (out / "recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("recommendation:", rec)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
