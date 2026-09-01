#!/usr/bin/env python3
"""Compare train-loop wall time: lazy q_hat vs materialized q_hat_all (same math)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.custom_losses import SNDRPolicyLoss
from models.models import CFModel, SingleMLPTransform
from training.trainer_trials import (
    RegressionScoresLookup,
    _build_regression_logged_split,
    fit_shared_regression_bundle,
    materialize_regression_qhat_all,
)
from training.training_utils import run_train_loop
from utils.simulation_utils import CustomCFDatasetPS, generate_dataset


def _load_ml_dataset(emb_dir: Path, seed: int) -> dict:
    emb_x = np.load(emb_dir / "ml_user_factors.npy")
    emb_a = np.load(emb_dir / "ml_item_factors.npy")
    meta_a = np.load(emb_dir / "ml_item_metadata.npy")
    meta_x = np.load(emb_dir / "ml_user_metadata.npy")
    params = {
        "n_users": int(emb_x.shape[0]),
        "n_actions": int(emb_a.shape[0]),
        "emb_dim": int(emb_x.shape[1]),
        "n_clusters": 32,
        "eps1": 0.05,
        "eps2": 0.05,
        "eps_meta": 0.0,
        "sigma1": 1.0,
        "sigma2": 1.0,
        "sigma_meta": 1.0,
        "ctr": 0.05,
        "policy_temperature": 2.0,
        "logging_uniform_mix": 0.0,
    }
    return generate_dataset(
        params,
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        metadata_a=meta_a,
        metadata_x=meta_x,
        store_original=True,
    )


def _make_train_loader(train_data: dict, batch_size: int):
    cf = CustomCFDatasetPS(
        train_data["x_idx"],
        train_data["a"],
        train_data["r"],
        train_data["pscore"],
    )
    return DataLoader(
        cf,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )


def _make_model(dataset: dict, device: torch.device) -> CFModel:
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
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
    return model


def _time_train_loop(
    *,
    label: str,
    model: CFModel,
    train_loader: DataLoader,
    scores_lookup: RegressionScoresLookup,
    device: torch.device,
    epochs: int,
    lr: float,
) -> float:
    criterion = SNDRPolicyLoss(propensity_mode="logged").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(epochs):
        run_train_loop(
            model,
            train_loader,
            optimizer,
            scores_lookup,
            criterion,
            device=str(device),
            check_nan=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_batches = len(train_loader) * epochs
    print(
        f"{label}: {elapsed:.3f}s total | {n_batches} batches | "
        f"{elapsed / max(n_batches, 1) * 1000:.2f} ms/batch",
        flush=True,
    )
    return elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--train-size", type=int, default=25_000)
    p.add_argument("--val-size", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-model", default="oracle")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    dataset = _load_ml_dataset(args.emb_dir, args.seed)
    split = _build_regression_logged_split(
        dataset,
        dataset["our_x"],
        dataset["our_a"],
        int(args.train_size),
        int(args.val_size),
        run_idx=0,
        split_seed=12345,
        regression_size=5_000,
    )
    train_data = split["train_data"]

    bundle_lazy = fit_shared_regression_bundle(
        dataset,
        split["reg_data"],
        reward_model=str(args.reward_model),
        materialize_qhat="never",
    )
    bundle_dense = fit_shared_regression_bundle(
        dataset,
        split["reg_data"],
        reward_model=str(args.reward_model),
        materialize_qhat="always",
    )

    # Sanity: lazy vs dense q_hat match on random users.
    users = np.random.default_rng(0).choice(
        int(dataset["n_users"]), size=128, replace=True
    )
    lazy_lu = RegressionScoresLookup(
        bundle_lazy["regression_model"],
        bundle_lazy["user_context"],
        device,
        q_hat_all=None,
    )
    dense_lu = RegressionScoresLookup(
        bundle_dense["regression_model"],
        bundle_dense["user_context"],
        device,
        q_hat_all=bundle_dense["q_hat_all"],
    )
    q_lazy = lazy_lu.qhat_rows_numpy(users)
    q_dense = dense_lu.qhat_rows_numpy(users)
    max_abs = float(np.max(np.abs(q_lazy - q_dense)))
    print(f"q_hat max|lazy-dense| = {max_abs:.3e}", flush=True)
    if max_abs > 1e-5:
        raise SystemExit("q_hat mismatch between lazy and dense cache")

    train_loader = _make_train_loader(train_data, int(args.batch_size))
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    print(
        f"catalog {n_users} users x {n_actions} actions | "
        f"train={args.train_size} batch={args.batch_size} epochs={args.epochs}",
        flush=True,
    )

    # One-time materialize cost (not counted in train loop).
    t_mat = time.perf_counter()
    _ = materialize_regression_qhat_all(
        bundle_lazy["regression_model"],
        bundle_lazy["user_context"],
    )
    mat_s = time.perf_counter() - t_mat
    print(f"one-time materialize (oracle/regression): {mat_s:.3f}s", flush=True)

    model_lazy = _make_model(dataset, device)
    t_lazy = _time_train_loop(
        label="lazy q_hat",
        model=model_lazy,
        train_loader=train_loader,
        scores_lookup=lazy_lu,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
    )

    model_dense = _make_model(dataset, device)
    t_dense = _time_train_loop(
        label="cached q_hat_all",
        model=model_dense,
        train_loader=train_loader,
        scores_lookup=dense_lu,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
    )

    speedup = t_lazy / max(t_dense, 1e-9)
    print(f"speedup (lazy/cached): {speedup:.2f}x", flush=True)


if __name__ == "__main__":
    main()
