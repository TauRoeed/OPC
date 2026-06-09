"""Download public dataset files used by BPR loaders."""

from __future__ import annotations

import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATASETS_ROOT = Path("datasets")

URLS = {
    "ml_zip": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "anime_csv": "https://huggingface.co/datasets/jason1966/CooperUnion_anime-recommendations-database/resolve/main/anime.csv",
    "rating_csv": "https://huggingface.co/datasets/jason1966/CooperUnion_anime-recommendations-database/resolve/main/rating.csv",
    "myket_csv": "https://huggingface.co/datasets/erfanloghmani/myket-android-application-recommendation-dataset/resolve/main/myket.csv",
    "myket_app_info_csv": "https://huggingface.co/datasets/erfanloghmani/myket-android-application-recommendation-dataset/resolve/main/app_info.csv",
    "lastfm_hdf5": "https://github.com/benfred/recommender_data/releases/download/v1.0/lastfm_360k.hdf5",
    "msd_hdf5": "https://github.com/benfred/recommender_data/releases/download/v1.0/msd_taste_profile.hdf5",
}

DEFAULT_PATHS = {
    "ml": DEFAULT_DATASETS_ROOT / "ml-1m",
    "anime": DEFAULT_DATASETS_ROOT / "anime",
    "myket": DEFAULT_DATASETS_ROOT / "myket",
    "lastfm": DEFAULT_DATASETS_ROOT / "lastfm" / "lastfm_360k.hdf5",
    "msd": DEFAULT_DATASETS_ROOT / "msd" / "msd_taste_profile.hdf5",
}


def _missing_files(root: Path, names: list[str]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "OPC-dataset-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise FileNotFoundError(f"Failed to download {url}: {exc}") from exc
    tmp.replace(dest)


def ensure_movielens_1m(root: str | Path, *, download: bool = True) -> Path:
    root = Path(root)
    required = ["ratings.dat", "users.dat", "movies.dat"]
    if not _missing_files(root, required):
        return root
    if not download:
        missing = _missing_files(root, required)
        raise FileNotFoundError(f"MovieLens 1M missing at {root}: {missing}")

    archive = root.parent / "ml-1m.zip"
    if not archive.exists():
        print(f"Downloading MovieLens 1M to {archive} ...")
        _download_file(URLS["ml_zip"], archive)

    print(f"Extracting {archive} ...")
    with zipfile.ZipFile(archive, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        top_levels = {m.split("/")[0] for m in members if "/" in m}
        if len(top_levels) == 1:
            extract_root = root.parent / next(iter(top_levels))
        else:
            extract_root = root
        zf.extractall(root.parent)
        if extract_root != root and extract_root.exists():
            root.mkdir(parents=True, exist_ok=True)
            for item in extract_root.iterdir():
                target = root / item.name
                if target.exists():
                    continue
                shutil.move(str(item), str(target))
            if extract_root.is_dir() and not any(extract_root.iterdir()):
                extract_root.rmdir()

    if _missing_files(root, required):
        missing = _missing_files(root, required)
        raise FileNotFoundError(f"MovieLens 1M extract incomplete at {root}: {missing}")
    return root


def ensure_anime(root: str | Path, *, download: bool = True) -> Path:
    root = Path(root)
    required = ["anime.csv", "rating.csv"]
    missing = _missing_files(root, required)
    if not missing:
        return root
    if not download:
        raise FileNotFoundError(f"Anime dataset missing at {root}: {missing}")

    root.mkdir(parents=True, exist_ok=True)
    url_map = {"anime.csv": URLS["anime_csv"], "rating.csv": URLS["rating_csv"]}
    for name in missing:
        dest = root / name
        print(f"Downloading {name} to {dest} ...")
        _download_file(url_map[name], dest)
    return root


def ensure_myket(root: str | Path, *, download: bool = True) -> Path:
    root = Path(root)
    required = ["myket.csv", "app_info_sample.csv"]
    missing = _missing_files(root, required)
    if not missing:
        return root
    if not download:
        raise FileNotFoundError(f"Myket dataset missing at {root}: {missing}")

    root.mkdir(parents=True, exist_ok=True)
    if "myket.csv" in missing:
        dest = root / "myket.csv"
        print(f"Downloading myket.csv to {dest} ...")
        _download_file(URLS["myket_csv"], dest)
    if "app_info_sample.csv" in missing:
        dest = root / "app_info_sample.csv"
        print(f"Downloading app_info_sample.csv to {dest} ...")
        _download_file(URLS["myket_app_info_csv"], dest)
    return root


def ensure_hdf5_dataset(
    path: str | Path,
    *,
    default_name: str,
    url: str,
    download: bool = True,
) -> Path:
    path = Path(path)
    if path.is_dir():
        hdf5_path = path / default_name
    else:
        hdf5_path = path

    if hdf5_path.exists():
        return hdf5_path
    if not download:
        raise FileNotFoundError(f"HDF5 dataset missing: {hdf5_path}")

    print(f"Downloading {default_name} to {hdf5_path} ...")
    _download_file(url, hdf5_path)
    return hdf5_path


def ensure_lastfm(path: str | Path, *, download: bool = True) -> Path:
    return ensure_hdf5_dataset(
        path,
        default_name="lastfm_360k.hdf5",
        url=URLS["lastfm_hdf5"],
        download=download,
    )


def ensure_msd(path: str | Path, *, download: bool = True) -> Path:
    return ensure_hdf5_dataset(
        path,
        default_name="msd_taste_profile.hdf5",
        url=URLS["msd_hdf5"],
        download=download,
    )
