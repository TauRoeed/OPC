"""Per-dataset BPR + data-prep settings, and the BPR meta file written next to the embeddings."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from BPR.bpr_minibatch import BPRConfig, data_fingerprint

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bpr_dataset_config.json"
META_SUFFIX = "_bpr_meta.json"
# Default --datasets of the study runners and launch scripts. msd and lastfm (also in
# bpr_dataset_config.json) are opt-in: a study condition costs ~12x and ~115x ml's (README).
DEFAULT_DATASETS = ("ml", "myket", "kuairec", "kuairand", "anime")


def load_bpr_dataset_config(
    dataset: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        all_cfg = json.load(f)
    if dataset not in all_cfg:
        known = ", ".join(sorted(all_cfg))
        raise KeyError(f"Unknown dataset '{dataset}' in {path}. Known: {known}")
    cfg = all_cfg[dataset]
    if "bpr" not in cfg or "data" not in cfg:
        raise ValueError(f"Config for '{dataset}' must contain 'bpr' and 'data' sections.")
    return cfg


def resolve_bpr_params(
    dataset: str,
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every BPR setting for a dataset: config file, then CLI overrides, then BPRConfig defaults."""
    cfg = load_bpr_dataset_config(dataset, config_path=config_path)
    params = dict(cfg["bpr"])
    for key, val in (overrides or {}).items():
        if val is not None:
            params[key] = val
    return asdict(BPRConfig.from_dict(params))


# --------------------------------------------------------------------------- meta file
def bpr_meta_path(emb_dir: str | Path, dataset: str) -> Path:
    return Path(emb_dir) / f"{dataset}{META_SUFFIX}"


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def build_bpr_meta(dataset: str, model, data_cfg: dict[str, Any], X) -> dict[str, Any]:
    """JSON-able record of how a dataset's embeddings were trained."""
    meta = {
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": data_cfg,
        "data_fingerprint": data_fingerprint(X),
        "n_users": int(X.shape[0]),
        "n_items": int(X.shape[1]),
        "n_interactions": int(X.nnz),
        "numpy_version": np.__version__,
        **_git_state(),
        **model.summary(),
    }
    meta["item_bias_file"] = bool(model.item_bias is not None)
    return meta


def bpr_artifact_status(emb_dir: str | Path, dataset: str, *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Compare the BPR meta next to a dataset's embeddings with the current config.

    status: "ok", "missing" (no meta file, e.g. embeddings from before BPR v2) or "stale"
    (trained with other settings, or the dataset is not in the config).
    """
    path = bpr_meta_path(emb_dir, dataset)
    regen = f"python -m BPR.generate_artifacts --dataset {dataset} --root <data root> --emb-dir {emb_dir}"
    if not path.exists():
        return {"status": "missing", "meta_file": None,
                "message": f"no {path.name}: these embeddings predate BPR v2; regenerate with {regen}"}
    meta = json.loads(path.read_text(encoding="utf-8"))
    info = {k: meta.get(k) for k in ("created_at", "git_commit", "data_fingerprint", "epochs_trained", "validation", "item_bias_file")}
    try:
        current = resolve_bpr_params(dataset, config_path=config_path)
        current_data = load_bpr_dataset_config(dataset, config_path=config_path)["data"]
    except KeyError:
        return {"status": "stale", "meta_file": str(path), "differences": {"dataset": [dataset, None]}, **info,
                "message": f"dataset {dataset!r} is not in the BPR config"}
    saved = meta.get("settings", {})
    diffs = {k: [saved.get(k), v] for k, v in current.items() if saved.get(k) != v}
    if meta.get("data") != current_data:
        diffs["data"] = [meta.get("data"), current_data]
    if diffs:
        return {"status": "stale", "meta_file": str(path), "differences": diffs, **info,
                "message": f"embeddings were trained with settings that differ from the config ({', '.join(diffs)}); regenerate with {regen}"}
    return {"status": "ok", "meta_file": str(path), "differences": {}, **info, "message": "embeddings match the BPR config"}
