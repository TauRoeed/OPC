"""Smoke test for dataset auto-download + loaders."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from BPR.dataload import (
    load_anime_dfs,
    load_artistwise_dfs,
    load_movielens_1m,
    load_myket,
)
from BPR.dataset_download import DEFAULT_PATHS


def _check_loader(name: str, root: Path, loader, **kwargs):
    ratings, users, items = loader(str(root), download=True, **kwargs)
    assert len(ratings) > 0, f"{name}: empty ratings"
    assert len(users) > 0, f"{name}: empty users"
    assert len(items) > 0, f"{name}: empty items"
    print(
        f"{name}: ok "
        f"ratings={len(ratings):,} users={len(users):,} items={len(items):,}"
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test dataset loaders.")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also download/load LastFM and MSD (~700MB).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary download directories.",
    )
    args = parser.parse_args()

    temp_root = Path(tempfile.mkdtemp(prefix="opc_dataset_smoke_"))
    print(f"Temp smoke dir: {temp_root}")
    try:
        _check_loader("ml", temp_root / "ml-1m", load_movielens_1m)
        _check_loader("anime", temp_root / "anime", load_anime_dfs)
        _check_loader("myket", temp_root / "myket", load_myket)

        if args.include_large:
            _check_loader(
                "lastfm",
                DEFAULT_PATHS["lastfm"],
                load_artistwise_dfs,
                dataset="lastfm",
            )
            _check_loader(
                "msd",
                DEFAULT_PATHS["msd"],
                load_artistwise_dfs,
                dataset="msd",
            )
        else:
            print("Skipping lastfm/msd (pass --include-large to test them).")

        print("All smoke tests passed.")
    finally:
        if args.keep_temp:
            print(f"Kept temp dir: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
