import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import h5py

import numpy as np
from scipy.sparse import csr_matrix

from dataclasses import dataclass

def load_movielens_1m(ml1m_dir: str):
    """
    ml1m_dir should be the folder that contains:
      - ratings.dat
      - users.dat
      - movies.dat
    (Often it's .../ml-1m/)
    """
    ml1m_dir = Path(ml1m_dir)

    # ratings.dat: UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(
        ml1m_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": int, "movie_id": int, "rating": int, "timestamp": int},
    )

    # users.dat: UserID::Gender::Age::Occupation::Zip-code
    users = pd.read_csv(
        ml1m_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip": str},
    )

    # movies.dat: MovieID::Title::Genres
    # latin-1 avoids occasional encoding issues in titles
    movies = pd.read_csv(
        ml1m_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": int, "title": str, "genres": str},
        encoding="latin-1",
    )

    return ratings, users, movies


def load_myket(root: str):
    """
    root/
      myket.csv
      app_info_sample.csv
    """
    root = Path(root)

    # interactions (ratings equivalent)
    df = pd.read_csv(root / "myket.csv")

    ratings = (
        df.reset_index()
        .rename(columns={
            "index": "user_id",     # real user id
            "user_id": "item_id"    # app/package name
        })
        [["user_id", "item_id"]]
    )

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(str)

    # users
    users = (
        ratings[["user_id"]]
        .drop_duplicates()
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    # items
    items = pd.read_csv(root / "app_info_sample.csv") \
        .rename(columns={"app_name": "item_id"})

    items["item_id"] = items["item_id"].astype(str)

    return ratings, users, items


def load_artistwise_dfs(hdf5_path: str, *, min_plays: float = 1.0):
    """
    Generic artist-wise loader for LastFM or MSD.
    No dataset flag, no CSR output, no extra assumptions.

    Returns:
      ratings: user_id, item_id (artist), rating (total plays)
      users:   user_id
      items:   item_id (artist)
    """

    with h5py.File(hdf5_path, "r") as f:
        if "artist_user_plays" in f:
            # LastFM
            g = f["artist_user_plays"]
            users = np.array(f["user"].asstr()[:])
            artists = np.array(f["artist"].asstr()[:])

        elif "track_user_plays" in f:
            # MSD
            g = f["track_user_plays"]
            users = np.array(f["user"].asstr()[:])
            track = np.array(f["track"].asstr()[:])
            artists = track[:, 1] if track.ndim == 2 else track

        else:
            raise ValueError("Unknown HDF5 format")

        X = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]))

    # orient to users × artists
    if X.shape[0] != len(users):
        X = X.T

    # build triplets
    coo = X.tocoo(copy=False)
    ratings = pd.DataFrame({
        "user_id": users[coo.row],
        "item_id": artists[coo.col],
        "rating":  coo.data.astype(np.float32),
    })

    # aggregate user × artist
    ratings = (
        ratings
        .groupby(["user_id", "item_id"], as_index=False)["rating"]
        .sum()
    )

    # filter min plays
    if min_plays > 1:
        ratings = ratings[ratings["rating"] >= float(min_plays)]

    ratings = ratings.reset_index(drop=True)

    users_df = pd.DataFrame({"user_id": ratings["user_id"].unique()})
    items_df = pd.DataFrame({"item_id": ratings["item_id"].unique()})

    return ratings, users_df, items_df



def load_anime_dfs(root: str, min_rating: float = 7.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expects:
      root/Anime.csv
      root/Rating.csv

    Returns:
      ratings: user_id, item_id, rating
      users:   user_id
      items:   item_id + metadata
    """
    root = Path(root)

    items = pd.read_csv(root / "anime.csv")
    ratings = pd.read_csv(root / "rating.csv")

    # Canonical naming
    items = items.rename(columns={
        "anime_id": "item_id",
        "name": "title",
        "genre": "genres",
    })

    ratings = ratings.rename(columns={
        "anime_id": "item_id",
    })

    # Keep canonical columns (plus rating)
    ratings = ratings[["user_id", "item_id", "rating"]].copy()
    ratings = ratings[(ratings["rating"] >= min_rating) | (ratings["rating"] == -1)].reset_index(drop=True)

    users = (
        ratings[["user_id"]]
        .drop_duplicates()
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    # Keep all metadata columns in items (good for sanity checks / neighbors)
    # Ensure item_id is present and unique-ish
    items["item_id"] = items["item_id"].astype(int)
    items = items.drop_duplicates("item_id").reset_index(drop=True)

    return ratings, users, items


@dataclass
class InteractionData:
    X: csr_matrix
    user2idx: Dict[Any, int]
    idx2user: np.ndarray
    item2idx: Dict[Any, int]
    idx2item: np.ndarray
    item_info: pd.DataFrame



def build_csr_from_interactions(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    value_col: Optional[str] = None,
    item_info: Optional[pd.DataFrame] = None,
    assume_users_are_indices: bool = False,   # <- for Myket
) -> InteractionData:
    """
    Works with:
      - integer user ids (MovieLens)
      - integer user indices already 0..n-1 (Myket option)
      - string item ids (Myket package names)
      - integer item ids (MovieLens movie_id)

    If assume_users_are_indices=True, the user ids are treated as row indices directly.
    """

    df = interactions[[user_col, item_col] + ([value_col] if value_col else [])].copy()

    # ---- items: allow strings or ints ----
    item_codes, idx2item = pd.factorize(df[item_col], sort=True)
    item2idx = {idx2item[k]: np.int32(k) for k in range(len(idx2item))}

    # ---- users: either factorize or treat as already indices ----
    if assume_users_are_indices:
        u_idx = df[user_col].astype(np.int32).to_numpy()
        idx2user = np.arange(u_idx.max() + 1, dtype=np.int32)
        user2idx = {np.int32(k): np.int32(k) for k in idx2user}  # identity map
    else:
        user_codes, idx2user = pd.factorize(df[user_col], sort=True)
        u_idx = user_codes.astype(np.int32)
        user2idx = {idx2user[k]: np.int32(k) for k in range(len(idx2user))}

    i_idx = item_codes.astype(np.int32)

    if value_col is None:
        data = np.ones(len(df), dtype=np.float32)
    else:
        data = df[value_col].astype(np.float32).to_numpy()

    X = csr_matrix((data, (u_idx, i_idx)), shape=(len(idx2user), len(idx2item)))
    X.sum_duplicates()

    # ---- align item_info to column order ----
    if item_info is None:
        item_info_aligned = pd.DataFrame({"item_id": idx2item})
    else:
        info = item_info.copy()
        # normalize key name
        if item_col != "item_id" and "item_id" not in info.columns:
            info = info.rename(columns={item_col: "item_id"})
        # make same dtype as idx2item
        info["item_id"] = info["item_id"].astype(type(idx2item[0]))
        info = info.drop_duplicates("item_id").set_index("item_id")
        item_info_aligned = info.reindex(idx2item).reset_index()

    return InteractionData(
        X=X,
        user2idx=user2idx,
        idx2user=idx2user,
        item2idx=item2idx,
        idx2item=idx2item,
        item_info=item_info_aligned,
    )