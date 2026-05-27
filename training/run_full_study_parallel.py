import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from training.run_full_study import (
    VALID_NOISE_AXES,
    _collect_existing_summaries,
    _run_condition,
)


def _iter_run_configs(args):
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
                            yield {
                                "dataset_name": dataset_name,
                                "noise_mode": noise_mode,
                                "noise_axis": noise_axis,
                                "noise_level": noise_level,
                                "ctr": float(ctr),
                                "seed": int(seed),
                                "run_key": run_key,
                            }


def _execute_run(config: dict):
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    opc_df, noprop_df, opc_trials, noprop_trials, meta = _run_condition(
        dataset_name=config["dataset_name"],
        emb_dir=Path(config["emb_dir"]),
        noise_mode=config["noise_mode"],
        noise_axis=config["noise_axis"],
        noise_level=config["noise_level"],
        ctr=config["ctr"],
        seed=config["seed"],
        train_sizes=config["train_sizes"],
        n_trials=config["n_trials"],
        num_runs=config["num_runs"],
        batch_size=config["batch_size"],
        val_size=config["val_size"],
        val_frac=config["val_frac"],
        val_min=config["val_min"],
        val_max=config["val_max"],
        policy_reward_mode=config["policy_reward_mode"],
        policy_reward_mc_sim=config["policy_reward_mc_sim"],
        policy_temperature=config["policy_temperature"],
        slim=bool(config.get("slim", False)),
        run_dir=run_dir,
    )

    opc_df = opc_df.reset_index().rename(columns={"index": "train_size"})
    noprop_df = noprop_df.reset_index().rename(columns={"index": "train_size"})
    opc_df["method"] = "opc"
    noprop_df["method"] = "no_propensity"
    summary_df = pd.concat([opc_df, noprop_df], ignore_index=True)
    summary_df["dataset"] = config["dataset_name"]
    summary_df["noise_mode"] = config["noise_mode"]
    summary_df["noise_axis"] = config["noise_axis"]
    summary_df["noise_level"] = config["noise_level"]
    summary_df["seed"] = config["seed"]
    summary_df["ctr"] = float(meta["ctr"])

    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    opc_trials.to_csv(run_dir / "opc_trials.csv", index=False)
    noprop_trials.to_csv(run_dir / "no_prop_trials.csv", index=False)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"run_key": config["run_key"]}


def main():
    parser = argparse.ArgumentParser(
        description="Run full OPC vs no-propensity sweeps in parallel."
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
        help="Which noise axes to sweep. Default: combined only (bundled context+action+metadata).",
    )
    parser.add_argument("--noise-levels", nargs="+", default=["low", "high"])# ["low", "medium", "high"])
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

    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=1)
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
        help="Softmax temperature for dot-product policies. Default 1.",
    )
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--val-min", type=int, default=5000)
    parser.add_argument("--val-max", type=int, default=None)
    parser.add_argument("--emb-dir", default="BPR/embeddings")
    parser.add_argument("--out-dir", default="artifacts/full_study")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel process workers.",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        default=False,
        help="Slim mode: still log all Optuna hyperparameters to trials_long; skip only "
        "the heavy post-hoc get_trial_results pass (full-catalog calc_reward + val "
        "DM/DR/IPW/SNDR). Keeps per-trial actual_reward/r_hat from the Optuna objective.",
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

    failures = []
    run_configs = []
    for base_cfg in _iter_run_configs(args):
        run_dir = out_dir / base_cfg["run_key"]
        summary_path = run_dir / "summary_metrics.csv"
        if args.skip_completed and summary_path.exists():
            print(f"Skipping completed: {base_cfg['run_key']}")
            continue
        cfg = {
            **base_cfg,
            "run_dir": str(run_dir),
            "emb_dir": str(emb_dir),
            "train_sizes": list(args.train_sizes),
            "n_trials": int(args.n_trials),
            "num_runs": int(args.num_runs),
            "batch_size": int(args.batch_size),
            "val_size": args.val_size,
            "val_frac": float(args.val_frac),
            "val_min": int(args.val_min),
            "val_max": args.val_max,
            "policy_reward_mode": args.policy_reward_mode,
            "policy_reward_mc_sim": int(args.policy_reward_mc_sim),
            "policy_temperature": float(args.policy_temperature),
            "slim": bool(args.slim),
        }
        run_configs.append(cfg)

    print(f"Running {len(run_configs)} conditions with max_workers={args.max_workers}")
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.max_workers)),
        mp_context=mp_ctx,
    ) as pool:
        future_to_cfg = {pool.submit(_execute_run, cfg): cfg for cfg in run_configs}
        for fut in as_completed(future_to_cfg):
            cfg = future_to_cfg[fut]
            try:
                fut.result()
                print(f"Done: {cfg['run_key']}")
            except Exception as e:
                failures.append({"run_key": cfg["run_key"], "error": repr(e)})
                print(f"FAILED {cfg['run_key']}: {e}")
                if args.fail_fast:
                    for other in future_to_cfg:
                        other.cancel()
                    raise

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
                    "max_workers": args.max_workers,
                },
                f,
                indent=2,
            )
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()
