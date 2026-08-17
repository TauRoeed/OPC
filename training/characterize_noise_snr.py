"""Characterize embedding noise via SNR / cosine / RMSE sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.noise_levels import VALID_NOISE_AXES, VALID_NOISE_LEVELS, noise_eps
from utils.noise_snr import dataset_snr_report, isolate_component_metrics
from utils.simulation_utils import (
    generate_kmeans_cluster_template_noise,
    generate_linear_transform_noise,
    generate_metadata_projection_noise,
    generate_random_cluster_template_noise,
    generate_dataset,
)


def _load_optional(path: Path):
    if path.exists():
        return np.load(path)
    return None


def _dataset_paths(emb_dir: Path, dataset_name: str):
    return (
        emb_dir / f"{dataset_name}_user_factors.npy",
        emb_dir / f"{dataset_name}_item_factors.npy",
        emb_dir / f"{dataset_name}_user_metadata.npy",
        emb_dir / f"{dataset_name}_item_metadata.npy",
    )


def _build_noise_vecs(
    emb: np.ndarray,
    *,
    noise_mode: str,
    n_clusters: int,
    seed: int,
    metadata: np.ndarray | None,
    include_meta: bool,
):
    linear = generate_linear_transform_noise(emb, seed=seed, sigma=1.0)
    if noise_mode == "kmeans_templates":
        cluster = generate_kmeans_cluster_template_noise(
            emb, n_clusters=n_clusters, seed=seed + 73, sigma=1.0
        )
    elif noise_mode == "random_centroids":
        cluster = generate_random_cluster_template_noise(
            emb, n_clusters=n_clusters, seed=seed + 73, sigma=1.0
        )
    else:
        raise ValueError(f"Unsupported noise_mode={noise_mode!r}")
    names = ["linear", "cluster"]
    vecs = [linear, cluster]
    if include_meta and metadata is not None:
        vecs.append(
            generate_metadata_projection_noise(
                metadata, out_dim=emb.shape[1], seed=seed + 131, sigma=1.0
            )
        )
        names.append("metadata")
    return names, vecs


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str):
    if df.empty:
        return
    level_order = [lv for lv in VALID_NOISE_LEVELS if lv in set(df["noise_level"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (dataset, axis), grp in df.groupby(["dataset", "noise_axis"]):
        ys = []
        for lv in level_order:
            sub = grp[grp["noise_level"] == lv]
            ys.append(float(sub[metric].mean()) if not sub.empty else np.nan)
        ax.plot(level_order, ys, marker="o", label=f"{dataset}/{axis}")
    ax.set_xlabel("noise_level")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def characterize_one(
    *,
    dataset_name: str,
    emb_dir: Path,
    noise_mode: str,
    noise_axis: str,
    noise_level: str,
    seed: int,
    ctr: float,
) -> list[dict]:
    user_path, item_path, user_meta_path, item_meta_path = _dataset_paths(
        emb_dir, dataset_name
    )
    if not user_path.exists() or not item_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings for {dataset_name}: {user_path} / {item_path}"
        )
    emb_x = np.load(user_path)
    emb_a = np.load(item_path)
    metadata_x = _load_optional(user_meta_path)
    metadata_a = _load_optional(item_meta_path)
    eps1, eps2, eps_meta = noise_eps(noise_level, noise_axis)

    if float(eps_meta) > 0 and metadata_x is None and metadata_a is None:
        eps_meta = 0.0

    n_clusters = max(8, min(64, int(np.sqrt(emb_a.shape[0]))))
    params = {
        "n_users": int(emb_x.shape[0]),
        "n_actions": int(emb_a.shape[0]),
        "emb_dim": int(emb_x.shape[1]),
        "n_clusters": int(n_clusters),
        "eps1": float(eps1),
        "eps2": float(eps2),
        "eps_meta": float(eps_meta),
        "sigma1": 1.0,
        "sigma2": 1.0,
        "sigma_meta": 1.0,
        "noise_mode": noise_mode,
        "noise_axis": noise_axis,
        "ctr": float(ctr),
    }
    dataset = generate_dataset(
        params=params,
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        metadata_a=metadata_a,
        metadata_x=metadata_x,
        store_original=True,
    )
    report = dataset_snr_report(
        dataset, eps1=float(eps1), eps2=float(eps2), eps_meta=float(eps_meta)
    )

    rows: list[dict] = [
        {
            "dataset": dataset_name,
            "noise_mode": noise_mode,
            "noise_axis": noise_axis,
            "noise_level": noise_level,
            "seed": int(seed),
            "component": "full_mix",
            "side": "action",
            "eps1": float(eps1),
            "eps2": float(eps2),
            "eps_meta": float(eps_meta),
            **{f"action_{k}": v for k, v in report["action"].items()},
            **{f"context_{k}": v for k, v in report["context"].items()},
            "snr_db": report["action"]["snr_db"],
            "cosine_retention": report["action"]["cosine_retention"],
            "rmse": report["action"]["rmse"],
            "signal_frac": report["action"].get("signal_frac", np.nan),
            "snr_db_mean": report["snr_db_mean"],
            "cosine_mean": report["cosine_mean"],
        },
        {
            "dataset": dataset_name,
            "noise_mode": noise_mode,
            "noise_axis": noise_axis,
            "noise_level": noise_level,
            "seed": int(seed),
            "component": "full_mix",
            "side": "context",
            "eps1": float(eps1),
            "eps2": float(eps2),
            "eps_meta": float(eps_meta),
            "snr_db": report["context"]["snr_db"],
            "cosine_retention": report["context"]["cosine_retention"],
            "rmse": report["context"]["rmse"],
            "signal_frac": report["context"].get("signal_frac", np.nan),
            "snr_db_mean": report["snr_db_mean"],
            "cosine_mean": report["cosine_mean"],
        },
    ]

    include_meta = float(eps_meta) > 0.0
    a_names, a_vecs = _build_noise_vecs(
        emb_a,
        noise_mode=noise_mode,
        n_clusters=n_clusters,
        seed=seed,
        metadata=metadata_a,
        include_meta=include_meta and metadata_a is not None,
    )
    a_eps = [eps1, eps2] + ([eps_meta] if "metadata" in a_names else [])
    for name, metrics in isolate_component_metrics(
        emb_a, a_vecs, a_eps, component_names=a_names
    ).items():
        rows.append(
            {
                "dataset": dataset_name,
                "noise_mode": noise_mode,
                "noise_axis": noise_axis,
                "noise_level": noise_level,
                "seed": int(seed),
                "component": name,
                "side": "action",
                "eps1": float(eps1),
                "eps2": float(eps2),
                "eps_meta": float(eps_meta),
                "snr_db": metrics["snr_db"],
                "cosine_retention": metrics["cosine_retention"],
                "rmse": metrics["rmse"],
                "signal_frac": metrics.get("signal_frac", np.nan),
                "component_eps": metrics.get("eps", np.nan),
            }
        )

    x_names, x_vecs = _build_noise_vecs(
        emb_x,
        noise_mode=noise_mode,
        n_clusters=n_clusters,
        seed=seed + 1,
        metadata=metadata_x,
        include_meta=include_meta and metadata_x is not None,
    )
    x_eps = [eps1, eps2] + ([eps_meta] if "metadata" in x_names else [])
    for name, metrics in isolate_component_metrics(
        emb_x, x_vecs, x_eps, component_names=x_names
    ).items():
        rows.append(
            {
                "dataset": dataset_name,
                "noise_mode": noise_mode,
                "noise_axis": noise_axis,
                "noise_level": noise_level,
                "seed": int(seed),
                "component": name,
                "side": "context",
                "eps1": float(eps1),
                "eps2": float(eps2),
                "eps_meta": float(eps_meta),
                "snr_db": metrics["snr_db"],
                "cosine_retention": metrics["cosine_retention"],
                "rmse": metrics["rmse"],
                "signal_frac": metrics.get("signal_frac", np.nan),
                "component_eps": metrics.get("eps", np.nan),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Characterize noise via SNR metrics.")
    parser.add_argument("--datasets", nargs="+", default=["ml"])
    parser.add_argument(
        "--noise-modes", nargs="+", default=["kmeans_templates", "random_centroids"]
    )
    parser.add_argument(
        "--noise-axes", nargs="+", default=["combined"], choices=list(VALID_NOISE_AXES)
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        default=list(VALID_NOISE_LEVELS),
        choices=list(VALID_NOISE_LEVELS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ctr", type=float, default=0.05)
    parser.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/noise_snr"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dataset_name in args.datasets:
        for noise_mode in args.noise_modes:
            for noise_axis in args.noise_axes:
                for noise_level in args.noise_levels:
                    for seed in args.seeds:
                        print(
                            f"{dataset_name} {noise_mode} {noise_axis} "
                            f"{noise_level} seed={seed}",
                            flush=True,
                        )
                        rows.extend(
                            characterize_one(
                                dataset_name=dataset_name,
                                emb_dir=Path(args.emb_dir),
                                noise_mode=noise_mode,
                                noise_axis=noise_axis,
                                noise_level=noise_level,
                                seed=int(seed),
                                ctr=float(args.ctr),
                            )
                        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    full = df[df["component"] == "full_mix"].copy()
    action_full = full[full["side"] == "action"]
    _plot_metric(action_full, "snr_db", out_dir / "snr_db_vs_level.png", "snr_db (action)")
    _plot_metric(
        action_full,
        "cosine_retention",
        out_dir / "cosine_vs_level.png",
        "cosine_retention (action)",
    )

    comps = df[df["component"] != "full_mix"].copy()
    if not comps.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (comp, side), grp in comps.groupby(["component", "side"]):
            if side != "action":
                continue
            level_order = [lv for lv in VALID_NOISE_LEVELS if lv in set(grp["noise_level"])]
            ys = [
                float(grp[grp["noise_level"] == lv]["snr_db"].mean())
                if not grp[grp["noise_level"] == lv].empty
                else np.nan
                for lv in level_order
            ]
            ax.plot(level_order, ys, marker="o", label=comp)
        ax.set_xlabel("noise_level")
        ax.set_ylabel("snr_db (isolated, action)")
        ax.set_title("Per-component SNR")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "snr_by_component.png", dpi=120)
        plt.close(fig)

    print(f"Plots under {out_dir}")


if __name__ == "__main__":
    main()
