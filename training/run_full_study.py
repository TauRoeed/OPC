import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.trainer_trials import (
    regression_trainer_trial,
    no_propensity_trainer_trial,
)
from utils.simulation_utils import generate_dataset


_NOISE_LEVEL_COMBINED = {
    "low": (0.05, 0.05, 0.0),
    "medium": (0.10, 0.15, 0.05),
    "high": (0.20, 0.25, 0.10),
}

# Per-axis magnitudes (matches the corresponding eps in the combined table).
_NOISE_LEVEL_PER_AXIS = {
    "context": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "action": {"low": 0.05, "medium": 0.15, "high": 0.25},
    "metadata": {"low": 0.0, "medium": 0.05, "high": 0.10},
}

VALID_NOISE_AXES = ("combined", "context", "action", "metadata")


def _noise_level_to_eps(level: str):
    if level not in _NOISE_LEVEL_COMBINED:
        raise ValueError(f"Unsupported noise level '{level}'")
    return _NOISE_LEVEL_COMBINED[level]


def _noise_eps(level: str, axis: str):
    """Return (eps1, eps2, eps_meta) for the requested axis at a given level.

    - axis="combined" matches the legacy bundled mapping.
    - axis in {"context", "action", "metadata"} perturbs only that axis.
    """
    if axis not in VALID_NOISE_AXES:
        raise ValueError(f"Unsupported noise axis '{axis}'")
    if axis == "combined":
        return _noise_level_to_eps(level)
    table = _NOISE_LEVEL_PER_AXIS[axis]
    if level not in table:
        raise ValueError(f"Unsupported noise level '{level}' for axis '{axis}'")
    mag = float(table[level])
    if axis == "context":
        return (mag, 0.0, 0.0)
    if axis == "action":
        return (0.0, mag, 0.0)
    return (0.0, 0.0, mag)


def _dataset_paths(emb_dir: Path, dataset_name: str):
    user_path = emb_dir / f"{dataset_name}_user_factors.npy"
    item_path = emb_dir / f"{dataset_name}_item_factors.npy"
    if not user_path.exists() or not item_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings for {dataset_name}: "
            f"{user_path} / {item_path}"
        )
    user_meta_path = emb_dir / f"{dataset_name}_user_metadata.npy"
    item_meta_path = emb_dir / f"{dataset_name}_item_metadata.npy"
    return user_path, item_path, user_meta_path, item_meta_path


def _load_optional_array(path: Path):
    if path.exists():
        return np.load(path)
    return None


def _collect_existing_summaries(base_dir: Path):
    rows = []
    for summary_path in sorted(base_dir.glob("dataset=*/summary_metrics.csv")):
        try:
            df = pd.read_csv(summary_path)
            if "noise_axis" not in df.columns:
                df["noise_axis"] = "combined"
            if "ctr" not in df.columns:
                df["ctr"] = float("nan")
            rows.append(df)
        except Exception:
            pass
    return rows


def _run_condition(
    dataset_name: str,
    emb_dir: Path,
    noise_mode: str,
    noise_axis: str,
    noise_level: str,
    ctr: float,
    seed: int,
    train_sizes: list[int],
    n_trials: int,
    num_runs: int,
    batch_size: int,
    val_size: int | None,
    val_frac: float,
    val_min: int,
    val_max: int | None,
    policy_reward_mode: str,
    policy_reward_mc_sim: int,
    policy_temperature: float,
    run_dir: Path,
    slim: bool = False,
):
    eps1, eps2, eps_meta = _noise_eps(noise_level, noise_axis)
    user_path, item_path, user_meta_path, item_meta_path = _dataset_paths(
        emb_dir, dataset_name
    )

    emb_x = np.load(user_path)
    emb_a = np.load(item_path)
    metadata_x = _load_optional_array(user_meta_path)
    metadata_a = _load_optional_array(item_meta_path)
    n_clusters = max(8, min(64, int(np.sqrt(emb_a.shape[0]))))

    if noise_axis == "metadata" and metadata_x is None and metadata_a is None:
        raise FileNotFoundError(
            f"noise_axis='metadata' but dataset '{dataset_name}' has no metadata "
            f"arrays (looked under {user_meta_path}, {item_meta_path})."
        )

    params = {
        "n_users": int(emb_x.shape[0]),
        "n_actions": int(emb_a.shape[0]),
        "emb_dim": int(emb_x.shape[1]),
        "n_clusters": int(n_clusters),
        "eps1": float(eps1),
        "eps2": float(eps2),
        "eps_meta": float(eps_meta),
        "sigma1": 1.0,
        "sigma2": 1.0,
        "sigma_meta": 1.0,
        "noise_mode": noise_mode,
        "noise_axis": noise_axis,
        "ctr": float(ctr),
        "policy_temperature": float(policy_temperature),
    }

    dataset = generate_dataset(
        params=params,
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        metadata_a=metadata_a,
        metadata_x=metadata_x,
        store_original=True,
    )

    opc_log_paths = {
        "trials": run_dir / "opc_trials_long.csv",
        "runs": run_dir / "opc_runs_long.csv",
    }
    noprop_log_paths = {
        "trials": run_dir / "no_prop_trials_long.csv",
        "runs": run_dir / "no_prop_runs_long.csv",
    }

    opc_df, opc_trials = regression_trainer_trial(
        num_runs=num_runs,
        num_neighbors=8,
        train_sizes=train_sizes,
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
        n_trials=n_trials,
        prev_best_params=None,
        propensity_mode="logged",
        log_paths=opc_log_paths,
        slim=slim,
        method_label="opc",
        policy_reward_mode=policy_reward_mode,
        policy_reward_mc_sim=policy_reward_mc_sim,
    )

    noprop_df, noprop_trials = no_propensity_trainer_trial(
        num_runs=num_runs,
        num_neighbors=8,
        train_sizes=train_sizes,
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
        n_trials=n_trials,
        prev_best_params=None,
        log_paths=noprop_log_paths,
        slim=slim,
        method_label="no_propensity",
        policy_reward_mode=policy_reward_mode,
        policy_reward_mc_sim=policy_reward_mc_sim,
    )

    # Unified long logs for post-hoc analysis.
    trials_frames = []
    runs_frames = []
    for p in (opc_log_paths["trials"], noprop_log_paths["trials"]):
        if p.exists():
            trials_frames.append(pd.read_csv(p))
    for p in (opc_log_paths["runs"], noprop_log_paths["runs"]):
        if p.exists():
            runs_frames.append(pd.read_csv(p))
    if trials_frames:
        pd.concat(trials_frames, ignore_index=True).to_csv(
            run_dir / "trials_long.csv", index=False
        )
    if runs_frames:
        pd.concat(runs_frames, ignore_index=True).to_csv(
            run_dir / "runs_long.csv", index=False
        )

    meta = {
        "dataset": dataset_name,
        "noise_mode": noise_mode,
        "noise_axis": noise_axis,
        "noise_level": noise_level,
        "seed": int(seed),
        "params": params,
        "ctr": float(params["ctr"]),
        "eps1": float(eps1),
        "eps2": float(eps2),
        "eps_meta": float(eps_meta),
        "train_sizes": [int(x) for x in train_sizes],
        "n_trials": int(n_trials),
        "num_runs": int(num_runs),
        "batch_size": int(batch_size),
        "val_size_fixed": val_size,
        "val_frac": val_frac,
        "val_min": val_min,
        "val_max": val_max,
        "policy_reward_mode": policy_reward_mode,
        "policy_reward_mc_sim": int(policy_reward_mc_sim),
        "slim": bool(slim),
    }
    return opc_df, noprop_df, opc_trials, noprop_trials, meta


def main():
    parser = argparse.ArgumentParser(
        description="Run full OPC vs no-propensity sweeps and export structured outputs."
    )
    parser.add_argument("--datasets", nargs="+", default=["ml", "anime"])
    parser.add_argument(
        "--noise-modes",
        nargs="+",
        default=["kmeans_templates"],
        help="Available: kmeans_templates, random_centroids.",
    )
    parser.add_argument(
        "--noise-axes",
        nargs="+",
        default=["combined"],
        choices=list(VALID_NOISE_AXES),
        help="Which noise axes to sweep. Default: combined only (context+action+metadata bundled). "
        "Pass context/action/metadata only if you want single-axis ablations.",
    )
    parser.add_argument("--noise-levels", nargs="+", default=["low", "medium", "high"])
    parser.add_argument(
        "--ctr-levels",
        nargs="+",
        type=float,
        default=[0.05],
        help="CTR levels to sweep in the simulator.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[5000, 25000, 50000, 100000],
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--policy-reward-mode",
        choices=["exact", "mc"],
        default="exact",
        help="Policy value in eval: exact is slower, mc is faster approximate.",
    )
    parser.add_argument(
        "--policy-reward-mc-sim",
        type=int,
        default=8,
        help="MC draws when --policy-reward-mode mc.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for dot-product policies (logging/eval). Default 1.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="If set, fixed validation logged trajectories for every train_size. "
        "Otherwise val_size = clamp(round(val_frac * train_size), val_min, val_max).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation count as a fraction of train_size when --val-size is not set.",
    )
    parser.add_argument(
        "--val-min",
        type=int,
        default=5000,
        help="Minimum validation trajectories when using --val-frac.",
    )
    parser.add_argument(
        "--val-max",
        type=int,
        default=None,
        help="Optional cap on validation trajectories when using --val-frac.",
    )
    parser.add_argument("--emb-dir", default="BPR/embeddings")
    parser.add_argument("--out-dir", default="artifacts/full_study")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--slim",
        action="store_true",
        default=False,
        help="Still log all trial hyperparameters to trials_long; skip only heavy "
        "post-hoc get_trial_results (full-catalog reward + val DM/DR/IPW/SNDR).",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip conditions whose summary_metrics.csv already exists (default: true).",
    )
    parser.add_argument("--fail-fast", action="store_true", default=False)
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    all_summary_rows = []
    failures = []

    for dataset_name in args.datasets:
        for noise_mode in args.noise_modes:
            for noise_axis in args.noise_axes:
                for noise_level in args.noise_levels:
                    for ctr in args.ctr_levels:
                        for seed in args.seeds:

                            run_key = (
                                f"dataset={dataset_name}__noise={noise_mode}"
                                f"__axis={noise_axis}__level={noise_level}"
                                f"__ctr={ctr:g}__seed={seed}"
                            )

                            print(f"\n=== Running {run_key} ===")
                            run_dir = out_dir / run_key
                            run_dir.mkdir(parents=True, exist_ok=True)
                            summary_path = run_dir / "summary_metrics.csv"
                            if args.skip_completed and summary_path.exists():
                                print(f"Skipping completed: {run_key}")
                                try:
                                    all_summary_rows.append(pd.read_csv(summary_path))
                                except Exception:
                                    pass
                                continue

                            try:
                                opc_df, noprop_df, opc_trials, noprop_trials, meta = _run_condition(
                                    dataset_name=dataset_name,
                                    emb_dir=emb_dir,
                                    noise_mode=noise_mode,
                                    noise_axis=noise_axis,
                                    noise_level=noise_level,
                                    ctr=ctr,
                                    seed=seed,
                                    train_sizes=args.train_sizes,
                                    n_trials=args.n_trials,
                                    num_runs=args.num_runs,
                                    batch_size=args.batch_size,
                                    val_size=args.val_size,
                                    val_frac=args.val_frac,
                                    val_min=args.val_min,
                                    val_max=args.val_max,
                                    policy_reward_mode=args.policy_reward_mode,
                                    policy_reward_mc_sim=args.policy_reward_mc_sim,
                                    policy_temperature=args.policy_temperature,
                                    run_dir=run_dir,
                                    slim=bool(args.slim),
                                )
                            except Exception as e:
                                failures.append({"run_key": run_key, "error": repr(e)})
                                print(f"FAILED {run_key}: {e}")
                                if args.fail_fast:
                                    raise
                                continue

                            opc_df = opc_df.reset_index().rename(columns={"index": "train_size"})
                            noprop_df = noprop_df.reset_index().rename(columns={"index": "train_size"})
                            opc_df["method"] = "opc"
                            noprop_df["method"] = "no_propensity"
                            summary_df = pd.concat([opc_df, noprop_df], ignore_index=True)
                            summary_df["dataset"] = dataset_name
                            summary_df["noise_mode"] = noise_mode
                            summary_df["noise_axis"] = noise_axis
                            summary_df["noise_level"] = noise_level
                            summary_df["seed"] = seed
                            summary_df["ctr"] = float(meta["ctr"])

                            summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
                            # Legacy filename, kept for compatibility.
                            opc_trials.to_csv(run_dir / "opc_trials.csv", index=False)
                            noprop_trials.to_csv(run_dir / "no_prop_trials.csv", index=False)

                            with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                                json.dump(meta, f, indent=2)

                            all_summary_rows.append(summary_df)

    collected_rows = _collect_existing_summaries(out_dir)
    if collected_rows:
        merged = pd.concat(collected_rows, ignore_index=True)
        merged.to_csv(out_dir / "all_summary_metrics.csv", index=False)
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_tag": run_tag,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "datasets": args.datasets,
                    "noise_modes": args.noise_modes,
                    "noise_axes": args.noise_axes,
                    "noise_levels": args.noise_levels,
                    "ctr_levels": args.ctr_levels,
                    "seeds": args.seeds,
                    "train_sizes": args.train_sizes,
                    "val_size_fixed": args.val_size,
                    "val_frac": args.val_frac,
                    "val_min": args.val_min,
                    "val_max": args.val_max,
                    "policy_reward_mode": args.policy_reward_mode,
                    "policy_reward_mc_sim": args.policy_reward_mc_sim,
                    "policy_temperature": args.policy_temperature,
                },
                f,
                indent=2,
            )
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()
