"""
H1 experiment: bad q_hat + strong logging + large n → naive can beat OPC.

Parallel: ProcessPoolExecutor, round-robin over --num-gpus (default 2).

See docs/h1_experiment.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.run_full_study import (
    _finalize_summary_df,
    _run_condition,
)
from training.run_full_study_parallel import (
    _resolve_num_gpus,
    _run_configs_with_oom_backoff,
)
from utils.noise_levels import VALID_NOISE_LEVELS
from utils.rand_ctr import calibrate_ctr_from_rand, estimate_rand_ctr
from utils.rand_ctr_sample_size import n_for_rand_ctr


def _dataset_paths(emb_dir: Path, dataset_name: str):
    return (
        emb_dir / f"{dataset_name}_user_factors.npy",
        emb_dir / f"{dataset_name}_item_factors.npy",
    )


def _resolve_ctr(
    *,
    dataset_name: str,
    emb_dir: Path,
    ctr_arg: float | None,
    target_rand_ctr: float | None,
    seed: int,
) -> tuple[float, dict]:
    user_path, item_path = _dataset_paths(emb_dir, dataset_name)
    emb_x = np.load(user_path)
    emb_a = np.load(item_path)

    meta: dict = {}
    if target_rand_ctr is not None:
        cal = calibrate_ctr_from_rand(
            emb_x,
            emb_a,
            target_rand_ctr=float(target_rand_ctr),
            seed=int(seed),
        )
        ctr = float(cal["ctr_calibrated"])
        meta["ctr_mode"] = "calibrated_to_target"
        meta["target_rand_ctr"] = float(target_rand_ctr)
        meta.update(cal)
    elif ctr_arg is not None:
        ctr = float(ctr_arg)
        meta["ctr_mode"] = "fixed"
    else:
        ctr = 0.05
        meta["ctr_mode"] = "default"

    meta["ctr_used"] = float(ctr)
    return ctr, meta


def _execute_h1_cell(config: dict) -> None:
    """Worker entry (spawn-safe). GPU via OPC_WORKER_GPU from pool init."""
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(config["emb_dir"])
    dataset_name = config["dataset_name"]
    seed = int(config["seed"])
    target = float(config["target_rand_ctr"])
    q_err = float(config["q_error"])
    log_mix = float(config["logging_uniform_mix"])
    n_rand = int(config["n_rand_samples"])

    ctr, ctr_meta = _resolve_ctr(
        dataset_name=dataset_name,
        emb_dir=emb_dir,
        ctr_arg=config.get("ctr_fixed"),
        target_rand_ctr=target if config.get("ctr_fixed") is None else None,
        seed=seed,
    )
    q_bad = float(target if config.get("ctr_fixed") is None else ctr)

    opc_df, noprop_df, _, _, meta = _run_condition(
        dataset_name=dataset_name,
        emb_dir=emb_dir,
        noise_mode=config["noise_mode"],
        noise_axis=config["noise_axis"],
        noise_level=config["noise_level"],
        ctr=ctr,
        seed=seed,
        train_sizes=config["train_sizes"],
        n_trials=int(config["n_trials"]),
        batch_size=int(config["batch_size"]),
        val_size=config["val_size"],
        val_frac=0.15,
        val_min=5000,
        val_max=None,
        policy_reward_mode="exact",
        policy_reward_mc_sim=8,
        policy_temperature=float(config["policy_temperature"]),
        run_dir=run_dir,
        slim=bool(config.get("slim", False)),
        policy_loss_types=tuple(config["policy_loss_types"]),
        logging_uniform_mix=log_mix,
        reward_model="oracle",
        q_error=q_err,
        q_bad_value=q_bad,
        rand_ctr_meta={},
        qhat_user_chunk=int(config.get("qhat_user_chunk", 10_000)),
        qhat_action_chunk=int(config.get("qhat_action_chunk", 10_000)),
        require_cuda=bool(config.get("require_cuda", False)),
    )

    from utils.simulation_utils import generate_dataset

    user_path, item_path = _dataset_paths(emb_dir, dataset_name)
    emb_x = np.load(user_path)
    emb_a = np.load(item_path)
    ds = generate_dataset(
        params=meta["params"],
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        store_original=True,
    )
    rand_meta = estimate_rand_ctr(ds, n_samples=n_rand, seed=seed + 99)
    rand_meta.update(ctr_meta)
    rand_meta["q_error"] = q_err
    rand_meta["logging_uniform_mix"] = log_mix
    rand_meta["n_users"] = int(emb_x.shape[0])
    rand_meta["n_rand_recommended"] = n_rand

    (run_dir / "run_meta.json").write_text(
        json.dumps({**meta, "rand_ctr": rand_meta}, indent=2)
    )

    summary_df = _finalize_summary_df(
        opc_df,
        noprop_df,
        meta,
        dataset=dataset_name,
        noise_mode=config["noise_mode"],
        noise_axis=config["noise_axis"],
        noise_level=config["noise_level"],
        seed=seed,
        q_error=q_err,
        logging_uniform_mix=log_mix,
        target_rand_ctr=target,
        measured_rand_ctr=float(rand_meta["rand_ctr"]),
        density_regime=str(rand_meta["density_regime"]),
        val_size=int(config["val_size"]),
    )
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)


def _iter_h1_configs(args, out_root: Path, n_rand_by_dataset: dict[str, int]):
    ctr_targets = [args.ctr] if args.ctr is not None else list(args.target_rand_ctrs)
    val_sizes = list(args.val_sizes) if args.val_sizes else [int(args.val_size)]
    noise_levels = list(args.noise_levels)
    for dataset_name in args.datasets:
        n_rand = int(n_rand_by_dataset[dataset_name])
        for noise_level in noise_levels:
            for target in ctr_targets:
                for q_err in args.q_errors:
                    for log_mix in args.logging_mixes:
                        for val_size in val_sizes:
                            for seed in args.seeds:
                                run_key = (
                                    f"dataset={dataset_name}__noise={noise_level}"
                                    f"__target_rho={target:g}__qerr={q_err:g}"
                                    f"__logmix={log_mix:g}__val={int(val_size)}"
                                    f"__seed={seed}"
                                )
                                yield {
                                    "run_key": run_key,
                                    "run_dir": str(out_root / run_key),
                                    "dataset_name": dataset_name,
                                    "emb_dir": str(args.emb_dir),
                                    "noise_mode": args.noise_mode,
                                    "noise_axis": args.noise_axis,
                                    "noise_level": noise_level,
                                    "target_rand_ctr": float(target),
                                    "ctr_fixed": args.ctr,
                                    "q_error": float(q_err),
                                    "logging_uniform_mix": float(log_mix),
                                    "seed": int(seed),
                                    "train_sizes": list(args.train_sizes),
                                    "n_trials": int(args.n_trials),
                                    "batch_size": int(args.batch_size),
                                    "val_size": int(val_size),
                                    "policy_temperature": float(args.policy_temperature)
                                    if float(log_mix) > 0
                                    else 1.0,
                                    "slim": bool(args.slim),
                                    "policy_loss_types": list(args.policy_losses),
                                    "n_rand_samples": n_rand,
                                    "require_cuda": bool(args.require_cuda),
                                    "qhat_user_chunk": int(args.qhat_user_chunk),
                                    "qhat_action_chunk": int(args.qhat_action_chunk),
                                }


def main():
    p = argparse.ArgumentParser(description="Run H1 OPC vs naive experiment grid.")
    p.add_argument("--datasets", nargs="+", default=["ml", "anime", "myket", "kuairec"])
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/h1_study"))
    p.add_argument("--run-tag", default="h1_v1")

    p.add_argument("--noise-mode", default="kmeans_templates")
    p.add_argument("--noise-axis", default="combined")
    p.add_argument(
        "--noise-levels",
        nargs="+",
        default=["low", "medium", "high", "extreme", "brutal"],
        choices=list(VALID_NOISE_LEVELS),
        help="Embedding noise: lower (low/medium) and higher (extreme/brutal) vs high.",
    )

    p.add_argument(
        "--target-rand-ctrs",
        nargs="+",
        type=float,
        default=[0.02, 0.08, 0.18],
    )
    p.add_argument("--ctr", type=float, default=None)
    p.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[5000, 25000, 100000, 250000, 400000],
    )
    p.add_argument(
        "--q-errors",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    p.add_argument("--logging-mixes", nargs="+", type=float, default=[0.0, 0.3])
    p.add_argument("--policy-temperature", type=float, default=2.0)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(15)))

    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument(
        "--val-sizes",
        nargs="+",
        type=int,
        default=[50000, 100000, 200000],
        help="Validation sizes to sweep (overrides --val-size when set).",
    )
    p.add_argument("--qhat-user-chunk", type=int, default=10_000)
    p.add_argument("--qhat-action-chunk", type=int, default=10_000)
    p.add_argument("--shared-regression-size", type=int, default=50_000)
    p.add_argument("--n-rand-ctr-samples", type=int, default=10_000)
    p.add_argument("--policy-losses", nargs="+", default=["sndr"])
    p.add_argument("--slim", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="GPUs to round-robin workers across (default 2).",
    )
    p.add_argument(
        "--workers-per-gpu",
        type=int,
        default=16,
        help="Process workers per GPU (default 16).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Process-pool size. Default = num_gpus * workers_per_gpu.",
    )
    p.add_argument("--min-workers", type=int, default=1)
    p.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()

    out_root = args.out_dir / f"run_{args.run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    n_rand_by_dataset: dict[str, int] = {}
    for dataset_name in args.datasets:
        user_path, _ = _dataset_paths(args.emb_dir, dataset_name)
        n_users = int(np.load(user_path).shape[0])
        sample_info = n_for_rand_ctr(n_users=n_users, eps=0.01, alpha=0.05)
        n_rand_by_dataset[dataset_name] = max(
            int(args.n_rand_ctr_samples), int(sample_info["n_recommended"])
        )

    all_cfgs = list(_iter_h1_configs(args, out_root, n_rand_by_dataset))
    run_configs = []
    skipped = 0
    for cfg in all_cfgs:
        summary_path = Path(cfg["run_dir"]) / "summary_metrics.csv"
        if args.skip_completed and summary_path.exists():
            skipped += 1
            continue
        Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
        run_configs.append(cfg)

    num_gpus = _resolve_num_gpus(args.num_gpus)
    if args.max_workers is None:
        workers = max(1, int(args.workers_per_gpu) * max(1, num_gpus))
    else:
        workers = max(1, int(args.max_workers))
    print(
        f"H1: {len(run_configs)} cells to run ({skipped} skipped), "
        f"max_workers={workers} over {num_gpus} GPU(s)",
        flush=True,
    )

    failures: list[dict] = []
    if run_configs:
        failures = _run_configs_with_oom_backoff(
            run_configs,
            max_workers=workers,
            min_workers=max(1, int(args.min_workers)),
            num_gpus=num_gpus,
            fail_fast=bool(args.fail_fast),
            oom_backoff=bool(args.oom_backoff),
            execute_fn=_execute_h1_cell,
        )

    rows = []
    for cfg in all_cfgs:
        path = Path(cfg["run_dir"]) / "summary_metrics.csv"
        if path.exists():
            try:
                rows.append(pd.read_csv(path))
            except Exception:
                pass
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            out_root / "all_summary_metrics.csv", index=False
        )
    if failures:
        pd.DataFrame(failures).to_csv(out_root / "failures.csv", index=False)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hypothesis": "H1",
        "num_gpus": num_gpus,
        "max_workers": workers,
        "n_scheduled": len(all_cfgs),
        "n_ran": len(run_configs),
        "n_skipped": skipped,
        "n_fail": len(failures),
        "axes": {
            "datasets": list(args.datasets),
            "noise_levels": list(args.noise_levels),
            "train_sizes": args.train_sizes,
            "target_rand_ctrs": list(args.target_rand_ctrs)
            if args.ctr is None
            else [args.ctr],
            "q_errors": args.q_errors,
            "logging_mixes": args.logging_mixes,
            "val_sizes": list(args.val_sizes) if args.val_sizes else [int(args.val_size)],
            "seeds": args.seeds,
            "qhat_user_chunk": int(args.qhat_user_chunk),
            "qhat_action_chunk": int(args.qhat_action_chunk),
            "workers_per_gpu": int(args.workers_per_gpu),
        },
    }
    (out_root / "h1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done → {out_root}")


if __name__ == "__main__":
    main()
