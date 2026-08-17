import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import h5py

import numpy as np
from scipy.sparse import csr_matrix

from dataclasses import dataclass

from BPR.dataset_download import (
    ensure_anime,
    ensure_kuairand_pure,
    ensure_kuairec,
    ensure_lastfm,
    ensure_movielens_1m,
    ensure_msd,
    ensure_myket,
)


# Dataset-specific metadata schema. Each dataset can expose different metadata
# columns/dimensions; we intentionally do not enforce a shared global size.
DATASET_METADATA_CONFIG = {
    "ml": {
        "item_id_col": "item_id",
        "item_numeric_cols": [],
        "item_categorical_cols": ["genres"],
        "item_multivalue_sep": {"genres": "|"},
        "user_id_col": "user_id",
        "user_numeric_cols": ["age", "occupation"],
        "user_categorical_cols": ["gender"],
    },
    "myket": {
        "item_id_col": "item_id",
        "item_numeric_cols": [],
        "item_categorical_cols": ["category"],
        "item_multivalue_sep": {},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [],
    },
    "anime": {
        "item_id_col": "item_id",
        "item_numeric_cols": ["members", "episodes", "rating"],
        "item_categorical_cols": ["type", "source", "genres"],
        "item_multivalue_sep": {"genres": ","},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [],
    },
    "lastfm": {
        "item_id_col": "item_id",
        "item_numeric_cols": [],
        "item_categorical_cols": [],
        "item_multivalue_sep": {},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [],
    },
    "msd": {
        "item_id_col": "item_id",
        "item_numeric_cols": [],
        "item_categorical_cols": [],
        "item_multivalue_sep": {},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [],
    },
    "kuairec": {
        "item_id_col": "item_id",
        "item_numeric_cols": [],
        "item_categorical_cols": ["feat"],
        "item_multivalue_sep": {"feat": "|"},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [
            "onehot_feat0",
            "onehot_feat1",
            "onehot_feat2",
            "onehot_feat7",
            "onehot_feat8",
            "onehot_feat9",
            "onehot_feat10",
            "onehot_feat11",
        ],
    },
    "kuairand": {
        "item_id_col": "item_id",
        "item_numeric_cols": ["video_duration"],
        "item_categorical_cols": ["video_type", "upload_type", "tag"],
        "item_multivalue_sep": {"tag": ","},
        "user_id_col": "user_id",
        "user_numeric_cols": [],
        "user_categorical_cols": [
            "onehot_feat0",
            "onehot_feat1",
            "onehot_feat2",
            "onehot_feat7",
            "onehot_feat8",
            "onehot_feat9",
            "onehot_feat10",
            "onehot_feat11",
        ],
    },
}

def load_movielens_1m(ml1m_dir: str, *, download: bool = True):
    """
    ml1m_dir should be the folder that contains:
      - ratings.dat
      - users.dat
      - movies.dat
    (Often it's .../ml-1m/)

    If files are missing and ``download=True``, MovieLens 1M is fetched automatically.
    """
    ml1m_dir = ensure_movielens_1m(ml1m_dir, download=download)

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


def load_myket(root: str, *, download: bool = True):
    """
    root/
      myket.csv
      app_info_sample.csv

    If files are missing and ``download=True``, Myket files are fetched from Hugging Face.
    """
    root = ensure_myket(root, download=download)

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


def load_artistwise_dfs(
    hdf5_path: str,
    *,
    min_plays: float = 1.0,
    download: bool = True,
    dataset: str | None = None,
):
    """
    Generic artist-wise loader for LastFM or MSD.
    No dataset flag, no CSR output, no extra assumptions.

    ``hdf5_path`` may be a file path or a directory (default file name is inferred
    from ``dataset`` or the path name). Missing files are downloaded when
    ``download=True``.

    Returns:
      ratings: user_id, item_id (artist), rating (total plays)
      users:   user_id
      items:   item_id (artist)
    """
    path = Path(hdf5_path)
    name = (dataset or path.name).lower()
    if "msd" in name:
        hdf5_path = ensure_msd(path, download=download)
    elif "lastfm" in name:
        hdf5_path = ensure_lastfm(path, download=download)
    elif path.is_dir():
        lastfm_candidate = path / "lastfm_360k.hdf5"
        msd_candidate = path / "msd_taste_profile.hdf5"
        if lastfm_candidate.exists():
            hdf5_path = lastfm_candidate
        elif msd_candidate.exists():
            hdf5_path = msd_candidate
        elif download:
            raise ValueError(
                f"Cannot infer HDF5 dataset under {path}. "
                "Pass a .hdf5 file path or set dataset='lastfm'/'msd'."
            )
        else:
            raise FileNotFoundError(f"No HDF5 dataset found under {path}")
    else:
        hdf5_path = path
        if not hdf5_path.exists():
            if "msd" in hdf5_path.name.lower():
                hdf5_path = ensure_msd(hdf5_path, download=download)
            else:
                hdf5_path = ensure_lastfm(hdf5_path, download=download)

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



def load_anime_dfs(
    root: str,
    min_rating: float = 7.0,
    *,
    download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expects:
      root/anime.csv
      root/rating.csv

    If files are missing and ``download=True``, anime files are fetched from Hugging Face.

    Returns:
      ratings: user_id, item_id, rating
      users:   user_id
      items:   item_id + metadata
    """
    root = ensure_anime(root, download=download)

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


def _feat_list_to_pipe(value) -> str:
    """Convert KuaiRec feat cell ([27, 9] or '[27, 9]') to pipe-separated tags."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(int(x)) for x in value)
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return "|".join(str(int(x)) for x in parsed)
        except (SyntaxError, ValueError, TypeError):
            pass
    return text.replace(",", "|")


def load_kuairec(
    root: str,
    *,
    watch_ratio_min: float = 2.0,
    matrix: str = "big",
    download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load KuaiRec interactions + side info.

    Positive interactions: watch_ratio >= watch_ratio_min (paper default 2.0).
    ``matrix`` is ``big`` or ``small``.
    """
    data_dir = ensure_kuairec(root, download=download)
    matrix_name = "big_matrix.csv" if matrix != "small" else "small_matrix.csv"
    matrix_path = data_dir / matrix_name
    if not matrix_path.exists():
        raise FileNotFoundError(f"KuaiRec matrix missing: {matrix_path}")

    interactions = pd.read_csv(matrix_path)
    interactions = interactions.rename(columns={"video_id": "item_id"})
    interactions = interactions[
        interactions["watch_ratio"] >= float(watch_ratio_min)
    ].copy()
    ratings = interactions[["user_id", "item_id"]].copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)

    items_path = data_dir / "item_categories.csv"
    if items_path.exists():
        items = pd.read_csv(items_path).rename(columns={"video_id": "item_id"})
        if "feat" in items.columns:
            items["feat"] = items["feat"].map(_feat_list_to_pipe)
    else:
        items = ratings[["item_id"]].drop_duplicates()

    users_path = data_dir / "user_features.csv"
    if users_path.exists():
        users = pd.read_csv(users_path)
    else:
        users = (
            ratings[["user_id"]]
            .drop_duplicates()
            .sort_values("user_id")
            .reset_index(drop=True)
        )

    items["item_id"] = items["item_id"].astype(int)
    items = items.drop_duplicates("item_id").reset_index(drop=True)
    return ratings, users, items


def load_kuairand(
    root: str,
    *,
    positive_col: str = "is_click",
    download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load KuaiRand-Pure logs (standard + random) as implicit positives.

    Positives: ``positive_col == 1`` (default ``is_click``; fallback ``long_view``).
    """
    data_dir = ensure_kuairand_pure(root, download=download)
    log_files = sorted(data_dir.glob("log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No KuaiRand log_*.csv under {data_dir}")

    frames = [pd.read_csv(path) for path in log_files]
    logs = pd.concat(frames, ignore_index=True)
    logs = logs.rename(columns={"video_id": "item_id"})

    col = positive_col
    if col not in logs.columns:
        if "long_view" in logs.columns:
            col = "long_view"
        elif "is_click" in logs.columns:
            col = "is_click"
        else:
            raise ValueError(
                f"KuaiRand logs missing positive column '{positive_col}' "
                f"(columns={list(logs.columns)[:20]})"
            )

    positives = logs[logs[col].astype(float) >= 1.0][["user_id", "item_id"]].copy()
    positives["user_id"] = positives["user_id"].astype(int)
    positives["item_id"] = positives["item_id"].astype(int)
    ratings = positives.drop_duplicates().reset_index(drop=True)

    items_path = data_dir / "video_features_basic_pure.csv"
    if items_path.exists():
        items = pd.read_csv(items_path).rename(columns={"video_id": "item_id"})
    else:
        items = ratings[["item_id"]].drop_duplicates()

    users_path = data_dir / "user_features_pure.csv"
    if users_path.exists():
        users = pd.read_csv(users_path)
    else:
        users = (
            ratings[["user_id"]]
            .drop_duplicates()
            .sort_values("user_id")
            .reset_index(drop=True)
        )

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
        item_info_aligned = info.reindex(idx2item).rename_axis("item_id").reset_index()

    return InteractionData(
        X=X,
        user2idx=user2idx,
        idx2user=idx2user,
        item2idx=item2idx,
        idx2item=idx2item,
        item_info=item_info_aligned,
    )


def save_user_interaction_counts(
    save_path: str,
    ratings_df: pd.DataFrame,
    user2idx: dict,
    user_col: str = "user_id",
):
    """
    Saves per-user interaction counts aligned to user2idx.

    Output:
      counts[i] = number of rows in ratings_df with user_id whose index is i

    Saved as:
      save_path (np.ndarray, int64)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # count interactions per raw user_id
    counts_by_user = ratings_df[user_col].value_counts(dropna=False)

    n_users = len(user2idx)
    counts = np.zeros(n_users, dtype=np.int64)

    for uid, c in counts_by_user.items():
        if uid in user2idx:
            counts[user2idx[uid]] = int(c)

    np.save(save_path, counts)
    return counts


def _normalize_numeric_column(s: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
    if vals.size == 0:
        return vals.reshape(-1, 1)
    std = float(vals.std())
    if std < 1e-8:
        return vals.reshape(-1, 1)
    return ((vals - vals.mean()) / std).astype(np.float32).reshape(-1, 1)


def _encode_metadata_frame(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    multivalue_sep: Optional[dict[str, str]] = None,
) -> np.ndarray:
    multivalue_sep = multivalue_sep or {}
    blocks = []

    for c in numeric_cols:
        if c in df.columns:
            blocks.append(_normalize_numeric_column(df[c]))

    for c in categorical_cols:
        if c not in df.columns:
            continue
        source = df[c].fillna("").astype(str)
        if c in multivalue_sep:
            sep = multivalue_sep[c]
            tokens = source.str.split(sep)
            one_hot = tokens.str.join("|").str.get_dummies(sep="|")
            if "" in one_hot.columns:
                one_hot = one_hot.drop(columns=[""])
            blocks.append(one_hot.astype(np.float32).to_numpy())
        else:
            one_hot = pd.get_dummies(source, prefix=c, dtype=np.float32)
            blocks.append(one_hot.to_numpy())

    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def build_metadata_matrices(
    dataset_name: str,
    *,
    item_info: pd.DataFrame,
    idx2item: np.ndarray,
    users_df: Optional[pd.DataFrame] = None,
    idx2user: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    cfg = DATASET_METADATA_CONFIG.get(dataset_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'.")

    item_id_col = cfg.get("item_id_col", "item_id")
    info = item_info.copy()
    if item_id_col not in info.columns:
        # tolerate common aliases in upstream dataframes
        for candidate in ("item_id", "movie_id"):
            if candidate in info.columns:
                item_id_col = candidate
                break
    if item_id_col not in info.columns:
        raise ValueError(f"item_info must contain '{item_id_col}' for dataset '{dataset_name}'.")

    info[item_id_col] = info[item_id_col].astype(type(idx2item[0]))
    info = info.drop_duplicates(item_id_col).set_index(item_id_col)
    item_aligned = info.reindex(idx2item).reset_index(drop=True)

    item_meta = _encode_metadata_frame(
        item_aligned,
        numeric_cols=cfg.get("item_numeric_cols", []),
        categorical_cols=cfg.get("item_categorical_cols", []),
        multivalue_sep=cfg.get("item_multivalue_sep", {}),
    )

    user_meta = None
    if users_df is not None and idx2user is not None:
        user_numeric_cols = cfg.get("user_numeric_cols", [])
        user_categorical_cols = cfg.get("user_categorical_cols", [])
        if user_numeric_cols or user_categorical_cols:
            user_id_col = cfg.get("user_id_col", "user_id")
            if user_id_col not in users_df.columns:
                user_id_col = "user_id"
            if user_id_col not in users_df.columns:
                raise ValueError(f"users_df must contain '{user_id_col}' for dataset '{dataset_name}'.")

            users = users_df.copy()
            users[user_id_col] = users[user_id_col].astype(type(idx2user[0]))
            users = users.drop_duplicates(user_id_col).set_index(user_id_col)
            users_aligned = users.reindex(idx2user).reset_index(drop=True)
            user_meta = _encode_metadata_frame(
                users_aligned,
                numeric_cols=user_numeric_cols,
                categorical_cols=user_categorical_cols,
                multivalue_sep={},
            )

    return item_meta, user_meta


def save_metadata_artifacts(
    output_dir: str,
    dataset_name: str,
    *,
    item_metadata: np.ndarray,
    user_metadata: Optional[np.ndarray],
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    item_path = output / f"{dataset_name}_item_metadata.npy"
    np.save(item_path, item_metadata.astype(np.float32, copy=False))

    out_paths = {"item_metadata": item_path}
    if user_metadata is not None:
        user_path = output / f"{dataset_name}_user_metadata.npy"
        np.save(user_path, user_metadata.astype(np.float32, copy=False))
        out_paths["user_metadata"] = user_path

    return out_paths


def build_and_save_metadata_artifacts(
    dataset_name: str,
    *,
    output_dir: str,
    interaction_data: InteractionData,
    users_df: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Path]]:
    """
    Build index-aligned metadata matrices and save .npy artifacts.

    Returns:
      item_metadata, user_metadata_or_none, path_dict
    """
    item_metadata, user_metadata = build_metadata_matrices(
        dataset_name,
        item_info=interaction_data.item_info,
        idx2item=interaction_data.idx2item,
        users_df=users_df,
        idx2user=interaction_data.idx2user,
    )
    paths = save_metadata_artifacts(
        output_dir,
        dataset_name,
        item_metadata=item_metadata,
        user_metadata=user_metadata,
    )
    return item_metadata, user_metadata, paths