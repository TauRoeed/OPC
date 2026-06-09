"""Per-dataset BPR + data-prep settings (from BPR/datasets.ipynb)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bpr_dataset_config.json"

_BPR_KEYS = (
    "factors",
    "learning_rate",
    "regularization",
    "epochs",
    "mode",
    "samples_per_epoch",
    "random_state",
)


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
    """Merge notebook defaults with optional CLI overrides."""
    cfg = load_bpr_dataset_config(dataset, config_path=config_path)
    params = dict(cfg["bpr"])
    if overrides:
        for key in _BPR_KEYS:
            val = overrides.get(key)
            if val is not None:
                params[key] = val
    return params
