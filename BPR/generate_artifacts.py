import argparse
import json
from pathlib import Path
from typing import Any

from BPR.bpr_config import build_bpr_meta, bpr_meta_path, load_bpr_dataset_config, resolve_bpr_params
from BPR.bpr_minibatch import NEGATIVES, SAMPLING, BPRConfig, MiniBatchBPR
from utils.seeding import DEFAULT_CPU_THREADS, pin_cpu_threads
from BPR.dataload import (
    build_and_save_metadata_artifacts,
    build_csr_from_interactions,
    load_anime_dfs,
    load_artistwise_dfs,
    load_kuairand,
    load_kuairec,
    load_movielens_1m,
    load_myket,
    save_user_interaction_counts,
)


def _dataset_bundle(
    dataset: str,
    root: str,
    *,
    data_cfg: dict[str, Any],
    download: bool = True,
):
    if dataset == "ml":
        ratings, users, items = load_movielens_1m(root, download=download)
        rating_min = data_cfg.get("rating_min")
        interactions = ratings
        if rating_min is not None:
            interactions = interactions[interactions["rating"] >= float(rating_min)]
        interactions = interactions[["user_id", "movie_id"]]
        data = build_csr_from_interactions(
            interactions=interactions,
            user_col="user_id",
            item_col="movie_id",
            value_col=None,
            item_info=items.rename(columns={"movie_id": "item_id"}),
        )
    elif dataset == "myket":
        ratings, users, items = load_myket(root, download=download)
        if "category" not in items.columns and "categories" in items.columns:
            items = items.rename(columns={"categories": "category"})
        data = build_csr_from_interactions(
            interactions=ratings[["user_id", "item_id"]],
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
            assume_users_are_indices=bool(
                data_cfg.get("assume_users_are_indices", False)
            ),
        )
    elif dataset == "anime":
        min_rating = float(data_cfg.get("min_rating", 7.0))
        ratings, users, items = load_anime_dfs(
            root, min_rating=min_rating, download=download
        )
        data = build_csr_from_interactions(
            interactions=ratings[["user_id", "item_id"]],
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
        )
    elif dataset in {"lastfm", "msd"}:
        ratings, users, items = load_artistwise_dfs(
            root, download=download, dataset=dataset
        )
        value_col = "rating" if data_cfg.get("use_rating_values", False) else None
        data = build_csr_from_interactions(
            interactions=ratings[["user_id", "item_id"]]
            if value_col is None
            else ratings,
            user_col="user_id",
            item_col="item_id",
            value_col=value_col,
            item_info=items,
        )
    elif dataset == "kuairec":
        ratings, users, items = load_kuairec(
            root,
            watch_ratio_min=float(data_cfg.get("watch_ratio_min", 2.0)),
            matrix=str(data_cfg.get("matrix", "big")),
            download=download,
        )
        data = build_csr_from_interactions(
            interactions=ratings[["user_id", "item_id"]],
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
        )
    elif dataset == "kuairand":
        ratings, users, items = load_kuairand(
            root,
            positive_col=str(data_cfg.get("positive_col", "is_click")),
            download=download,
        )
        data = build_csr_from_interactions(
            interactions=ratings[["user_id", "item_id"]],
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    return ratings, users, data


def main():
    parser = argparse.ArgumentParser(
        description="Generate BPR embeddings (BPR v2: mini-batch, item bias, early stopping) and metadata artifacts."
    )
    parser.add_argument(
        "--dataset",
        choices=["ml", "myket", "anime", "lastfm", "msd", "kuairec", "kuairand"],
        required=True,
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root path (directory or HDF5 path for lastfm/msd).",
    )
    parser.add_argument("--emb-dir", default="BPR/embeddings")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to bpr_dataset_config.json (default: BPR/bpr_dataset_config.json).",
    )
    g = parser.add_argument_group("BPR settings (override the dataset config)")
    g.add_argument("--factors", type=int, default=None)
    g.add_argument("--item-bias", action=argparse.BooleanOptionalAction, default=None,
                   help="Learn a per-item bias: score = b_i + x_u . a_i.")
    g.add_argument("--negatives", choices=list(NEGATIVES), default=None,
                   help="Negative items: uniform, or proportional to popularity ** --negative-gamma.")
    g.add_argument("--negative-gamma", type=float, default=None)
    g.add_argument("--sampling", choices=list(SAMPLING), default=None,
                   help="Liked items drawn per interaction (standard) or per user.")
    g.add_argument("--learning-rate", type=float, default=None, help="Adagrad step size.")
    g.add_argument("--regularization", type=float, default=None)
    g.add_argument("--bias-regularization", type=float, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None,
                   help="Stop on held-out recall@20 (one liked item per user held out).")
    g.add_argument("--max-epochs", type=int, default=None)
    g.add_argument("--patience", type=int, default=None)
    g.add_argument("--refit", action=argparse.BooleanOptionalAction, default=None,
                   help="After early stopping, retrain on all interactions for the best number of epochs.")
    g.add_argument("--epochs", type=int, default=None, help="Fixed epochs when early stopping is off.")
    g.add_argument("--seed", type=int, default=None, help="Maps to random_state.")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing dataset files before loading (default: true).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="CPU threads for numpy/BLAS (fixed so the same seed gives the same vectors).",
    )
    args = parser.parse_args()
    pin_cpu_threads(args.cpu_threads)

    ds_cfg = load_bpr_dataset_config(args.dataset, config_path=args.config)
    settings = resolve_bpr_params(
        args.dataset,
        config_path=args.config,
        overrides={
            "factors": args.factors,
            "item_bias": args.item_bias,
            "negatives": args.negatives,
            "negative_gamma": args.negative_gamma,
            "sampling": args.sampling,
            "learning_rate": args.learning_rate,
            "regularization": args.regularization,
            "bias_regularization": args.bias_regularization,
            "batch_size": args.batch_size,
            "early_stopping": args.early_stopping,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "refit": args.refit,
            "epochs": args.epochs,
            "random_state": args.seed,
        },
    )

    ratings, users_df, interaction_data = _dataset_bundle(
        args.dataset,
        args.root,
        data_cfg=ds_cfg["data"],
        download=args.download,
    )
    X = interaction_data.X
    print(f"{args.dataset}: {X.shape[0]:,} users, {X.shape[1]:,} items, {X.nnz:,} interactions")
    print(f"{args.dataset} BPR settings: {settings}")

    model = MiniBatchBPR(BPRConfig(**settings)).fit(X)

    emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    user_emb_path = emb_dir / f"{args.dataset}_user_factors.npy"
    item_emb_path = emb_dir / f"{args.dataset}_item_factors.npy"
    bias_path = emb_dir / f"{args.dataset}_item_bias.npy"
    model.save_embeddings(str(user_emb_path), str(item_emb_path), str(bias_path))
    if model.item_bias is None and bias_path.exists():
        bias_path.unlink()  # never leave a bias from an earlier run next to vectors trained without one
    meta = build_bpr_meta(args.dataset, model, ds_cfg["data"], X)
    meta_path = bpr_meta_path(emb_dir, args.dataset)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    val = meta["validation"]
    if val:
        print(f"{args.dataset}: best epoch {meta['best_epoch']}, held-out recall@{val['k']} {val['recall']:.4f}, "
              f"NDCG@{val['k']} {val['ndcg']:.4f} ({val['users']:,} users)")

    item_meta, user_meta, paths = build_and_save_metadata_artifacts(
        args.dataset,
        output_dir=str(emb_dir),
        interaction_data=interaction_data,
        users_df=users_df,
    )
    print(f"{args.dataset}: item metadata shape={item_meta.shape}")
    if user_meta is None:
        print(f"{args.dataset}: no standalone user metadata; skipped user metadata file")
    else:
        print(f"{args.dataset}: user metadata shape={user_meta.shape}")

    save_user_interaction_counts(
        str(emb_dir / f"{args.dataset}_user_interaction_counts.npy"),
        ratings,
        interaction_data.user2idx,
        user_col="user_id",
    )

    saved = [user_emb_path, item_emb_path] + ([bias_path] if model.item_bias is not None else []) + [meta_path]
    print(f"Saved: {', '.join(str(p) for p in saved)}")
    print(f"Saved metadata: {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
