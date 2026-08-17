import argparse
from pathlib import Path
from typing import Any

from BPR.bpr import BayesianPersonalizedRanking
from BPR.bpr_config import load_bpr_dataset_config, resolve_bpr_params
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
        description="Generate BPR embeddings and metadata artifacts."
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
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument("--mode", choices=["samples", "per_user"], default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Maps to random_state.")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing dataset files before loading (default: true).",
    )
    args = parser.parse_args()

    ds_cfg = load_bpr_dataset_config(args.dataset, config_path=args.config)
    bpr_params = resolve_bpr_params(
        args.dataset,
        config_path=args.config,
        overrides={
            "factors": args.factors,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "regularization": args.regularization,
            "mode": args.mode,
            "samples_per_epoch": args.samples_per_epoch,
            "random_state": args.seed,
        },
    )

    ratings, users_df, interaction_data = _dataset_bundle(
        args.dataset,
        args.root,
        data_cfg=ds_cfg["data"],
        download=args.download,
    )

    print(f"{args.dataset} BPR params: {bpr_params}")

    model = BayesianPersonalizedRanking(
        factors=int(bpr_params["factors"]),
        learning_rate=float(bpr_params["learning_rate"]),
        regularization=float(bpr_params["regularization"]),
        epochs=int(bpr_params["epochs"]),
        random_state=int(bpr_params["random_state"]),
        mode=str(bpr_params["mode"]),
        samples_per_epoch=int(bpr_params["samples_per_epoch"]),
    )
    model.fit(interaction_data.X)

    emb_dir = Path(args.emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    user_emb_path = emb_dir / f"{args.dataset}_user_factors.npy"
    item_emb_path = emb_dir / f"{args.dataset}_item_factors.npy"
    model.save_embeddings(str(user_emb_path), str(item_emb_path))

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

    print(f"Saved embeddings: {user_emb_path}, {item_emb_path}")
    print(f"Saved metadata: {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
