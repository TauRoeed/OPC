import argparse
from pathlib import Path

from BPR.bpr import BayesianPersonalizedRanking
from BPR.dataload import (
    build_and_save_metadata_artifacts,
    build_csr_from_interactions,
    load_anime_dfs,
    load_artistwise_dfs,
    load_movielens_1m,
    load_myket,
    save_user_interaction_counts,
)


def _dataset_bundle(dataset: str, root: str, *, download: bool = True):
    if dataset == "ml":
        ratings, users, items = load_movielens_1m(root, download=download)
        data = build_csr_from_interactions(
            interactions=ratings,
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
            interactions=ratings,
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
            assume_users_are_indices=True,
        )
    elif dataset == "anime":
        ratings, users, items = load_anime_dfs(root, download=download)
        data = build_csr_from_interactions(
            interactions=ratings,
            user_col="user_id",
            item_col="item_id",
            value_col=None,
            item_info=items,
        )
    elif dataset in {"lastfm", "msd"}:
        ratings, users, items = load_artistwise_dfs(
            root, download=download, dataset=dataset
        )
        data = build_csr_from_interactions(
            interactions=ratings,
            user_col="user_id",
            item_col="item_id",
            value_col="rating",
            item_info=items,
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    return ratings, users, data


def main():
    parser = argparse.ArgumentParser(description="Generate BPR embeddings and metadata artifacts.")
    parser.add_argument("--dataset", choices=["ml", "myket", "anime", "lastfm", "msd"], required=True)
    parser.add_argument("--root", required=True, help="Dataset root path (directory or HDF5 path for lastfm/msd).")
    parser.add_argument("--emb-dir", default="BPR/embeddings")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--mode", choices=["samples", "per_user"], default="samples")
    parser.add_argument("--samples-per-epoch", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing dataset files before loading (default: true).",
    )
    args = parser.parse_args()

    ratings, users_df, interaction_data = _dataset_bundle(
        args.dataset, args.root, download=args.download
    )

    model = BayesianPersonalizedRanking(
        factors=args.factors,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        epochs=args.epochs,
        random_state=args.seed,
        mode=args.mode,
        samples_per_epoch=args.samples_per_epoch,
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
