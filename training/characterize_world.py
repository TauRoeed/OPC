"""Characterize the simulated world per dataset: bias calibration, logging temperature,
click model and the resulting logging CTR / signal kept for a set of bias configurations.

Example:
    python -m training.characterize_world --datasets ml myket --bias-configs all
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.noise_snr import dataset_snr_report
from utils.representation_bias import (
    BIAS_LEVELS,
    BIAS_TYPES,
    WorldConfig,
    add_world_arguments,
    build_world,
    resolve_bias_configs,
    world_options_from_args,
)

DEFAULT_CONFIGS = ("none", "low", "medium", "high", "high/none/none", "none/high/none", "none/none/high")


def _load_optional(path: Path):
    return np.load(path) if path.exists() else None


def characterize(dataset_name: str, emb_dir: Path, seed: int, ctr: float, options: dict, bias_configs):
    emb_x = np.load(emb_dir / f"{dataset_name}_user_factors.npy")
    emb_a = np.load(emb_dir / f"{dataset_name}_item_factors.npy")
    meta_x = meta_a = None
    if options.get("group_source") == "metadata":
        meta_x = _load_optional(emb_dir / f"{dataset_name}_user_metadata.npy")
        meta_a = _load_optional(emb_dir / f"{dataset_name}_item_metadata.npy")
    config = WorldConfig(target_ctr=float(ctr), **options)
    rows, calibration = [], None
    for bias in bias_configs:
        ds = build_world(emb_x, emb_a, bias, seed=seed, config=config, metadata_x=meta_x, metadata_a=meta_a)
        w = ds["world"]
        snr = dataset_snr_report(ds)
        if calibration is None:
            calibration = {k: v for k, v in w.items() if k not in (
                "bias", "bias_label", "eps", "signal_kept", "logging_ctr", "vector_rms", "cosine_to_clean")}
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "bias": w["bias_label"],
            **{f"bias_{k}": w["bias"][k] for k in BIAS_TYPES},
            **{f"eps_{k}": w["eps"][k] for k in BIAS_TYPES},
            "signal_kept": w["signal_kept"],
            "logging_ctr": w["logging_ctr"],
            "best_item_ctr": w["best_item_ctr"],
            "best_over_logging": w["best_item_ctr"] / w["logging_ctr"],
            "uniform_ctr": w["uniform_ctr"],
            "cosine_users": w["cosine_to_clean"]["users"],
            "cosine_items": w["cosine_to_clean"]["items"],
            "snr_db_users": snr["context"]["snr_db"],
            "snr_db_items": snr["action"]["snr_db"],
        })
    return rows, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["ml", "myket", "kuairec", "kuairand"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ctr", type=float, default=0.05, help="Target CTR of the reference policy.")
    parser.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/world"))
    add_world_arguments(parser, bias_default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    if [c.lower() for c in args.bias_configs] == ["all"]:
        bias_configs = ["/".join(c) for c in itertools.product(BIAS_LEVELS, repeat=len(BIAS_TYPES))]
    else:
        bias_configs = args.bias_configs
    bias_configs = resolve_bias_configs(bias_configs)
    options = world_options_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, calibrations = [], {}
    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"== {dataset_name} seed={seed}", flush=True)
            r, cal = characterize(dataset_name, args.emb_dir, int(seed), args.ctr, options, bias_configs)
            rows += r
            calibrations[f"{dataset_name}__seed={seed}"] = cal
            print(
                f"   T={cal['logging_temperature']:.4g} (clean logger over "
                f"{cal['clean_logger_effective_items']:.0f} items), alpha={cal['alpha']:.3f}, "
                f"reference CTR {cal['reference_ctr']:.2%}, uniform {cal['uniform_ctr']:.2%}, "
                f"best item {cal['best_item_ctr']:.1%}"
            )
            for k in BIAS_TYPES:
                print(f"   eps {k:6s} " + "  ".join(f"{lvl} {cal['eps_table'][k][lvl]:.3f}" for lvl in BIAS_LEVELS[1:]))
            for row in r:
                print(f"   {row['bias']:24s} kept {row['signal_kept']:.2f}  logging CTR {row['logging_ctr']:.2%}  "
                      f"best/log {row['best_over_logging']:.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "calibration.json").write_text(json.dumps(calibrations, indent=2))
    print(f"Wrote {args.out_dir / 'summary.csv'} ({len(df)} rows) and calibration.json")


if __name__ == "__main__":
    main()
