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

import pandas as pd

from BPR.bpr_config import DEFAULT_DATASETS
from training.trainer_trials import DEFAULT_SELECT_WEIGHTS, DEFAULT_TRAIN_WEIGHTS
from utils.importance_weights import weight_spec_label
from utils.seeding import DEFAULT_CPU_THREADS, pin_cpu_threads
from training.run_full_study import (
    _finalize_summary_df,
    _run_condition,
)
from training.run_full_study_parallel import (
    _resolve_num_gpus,
    _run_with_memory_cap,
)
from utils.rand_ctr_sample_size import density_regime
from utils.representation_bias import (
    add_world_arguments,
    resolve_bias_configs,
    world_options_from_args,
    world_run_key_suffix,
)


def _execute_h1_cell(config: dict) -> None:
    """Worker entry (spawn-safe). GPU via OPC_WORKER_GPU from pool init.

    The world is calibrated so the uniform random policy has CTR ``target_rand_ctr``
    (``ctr_reference='uniform'``); the measured value is its exact full-catalog CTR.
    """
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = config["dataset_name"]
    seed = int(config["seed"])
    target = float(config["target_rand_ctr"])
    q_err = float(config["q_error"])
    log_mix = float(config["logging_uniform_mix"])

    opc_df, noprop_df, _, _, meta = _run_condition(
        dataset_name=dataset_name,
        emb_dir=Path(config["emb_dir"]),
        bias=config["bias"],
        ctr=target,
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
        run_dir=run_dir,
        slim=bool(config.get("slim", False)),
        deterministic=bool(config.get("deterministic", True)),
        cpu_threads=int(config.get("cpu_threads", DEFAULT_CPU_THREADS)),
        policy_loss_types=tuple(config["policy_loss_types"]),
        logging_uniform_mix=log_mix,
        reward_model="oracle",
        q_error=q_err,
        train_weights=config.get("train_weights"),
        select_weights=config.get("select_weights"),
        q_bad_value=target,
        world_options={**(config.get("world_options") or {}), "ctr_reference": "uniform"},
        record_uniform_value=True,
        qhat_user_chunk=int(config.get("qhat_user_chunk", 10_000)),
        qhat_action_chunk=int(config.get("qhat_action_chunk", 10_000)),
        require_cuda=bool(config.get("require_cuda", False)),
    )

    world = meta["world"]
    rand_ctr = float(world["uniform_value"])
    meta["rand_ctr"] = {
        "ctr_mode": "calibrated_to_target",
        "target_rand_ctr": target,
        "rand_ctr": rand_ctr,  # exact: prior-weighted users x all items
        "rand_ctr_calibration_sample": float(world["uniform_ctr"]),
        "density_regime": density_regime(rand_ctr),
        "q_error": q_err,
        "logging_uniform_mix": log_mix,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    summary_df = _finalize_summary_df(
        opc_df,
        noprop_df,
        meta,
        dataset=dataset_name,
        seed=seed,
        q_error=q_err,
        logging_uniform_mix=log_mix,
        target_rand_ctr=target,
        measured_rand_ctr=rand_ctr,
        density_regime=density_regime(rand_ctr),
        val_size=int(config["val_size"]),
    )
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)


def _iter_h1_configs(args, out_root: Path):
    val_sizes = list(args.val_sizes) if args.val_sizes else [int(args.val_size)]
    bias_configs = resolve_bias_configs(args.bias_configs)
    world_options = world_options_from_args(args)
    world_suffix = world_run_key_suffix(world_options)  # before the forced uniform CTR reference
    for dataset_name in args.datasets:
        for seed in args.seeds:
            for bias in bias_configs:
                for target in args.target_rand_ctrs:
                    for q_err in args.q_errors:
                        for log_mix in args.logging_mixes:
                            for val_size in val_sizes:
                                run_key = (
                                    f"dataset={dataset_name}__bias={bias}"
                                    f"__target_rho={target:g}__qerr={q_err:g}"
                                    f"__logmix={log_mix:g}__val={int(val_size)}"
                                    f"__seed={seed}{world_suffix}"
                                )
                                yield {
                                    "run_key": run_key,
                                    "run_dir": str(out_root / run_key),
                                    "dataset_name": dataset_name,
                                    "emb_dir": str(args.emb_dir),
                                    "bias": bias,
                                    "world_options": world_options,
                                    "target_rand_ctr": float(target),
                                    "q_error": float(q_err),
                                    "logging_uniform_mix": float(log_mix),
                                    "seed": int(seed),
                                    "train_sizes": list(args.train_sizes),
                                    "n_trials": int(args.n_trials),
                                    "batch_size": int(args.batch_size),
                                    "val_size": int(val_size),
                                    "slim": bool(args.slim),
                                    "deterministic": bool(args.deterministic),
                                    "cpu_threads": int(args.cpu_threads),
                                    "policy_loss_types": list(args.policy_losses),
                                    "train_weights": args.train_weights,
                                    "select_weights": args.select_weights,
                                    "require_cuda": bool(args.require_cuda),
                                    "qhat_user_chunk": int(args.qhat_user_chunk),
                                    "qhat_action_chunk": int(args.qhat_action_chunk),
                                }


def main():
    p = argparse.ArgumentParser(description="Run H1 OPC vs naive experiment grid.")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Default: " + " ".join(DEFAULT_DATASETS) + ".")
    p.add_argument("--emb-dir", type=Path, default=Path("BPR/embeddings"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/h1_study"))
    p.add_argument("--run-tag", default="h1_v1")

    add_world_arguments(p, ctr_reference=False)
    p.add_argument(
        "--target-rand-ctrs",
        nargs="+",
        type=float,
        default=[0.02, 0.08, 0.18],
        help="CTR of the uniform random policy; the world's click model is calibrated to it.",
    )
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
    p.add_argument("--policy-losses", nargs="+", default=["sndr"])
    p.add_argument("--train-weights", type=weight_spec_label, default=DEFAULT_TRAIN_WEIGHTS,
                   help="Importance-weight transform in the OPC training losses (none, clip:M, shrink:lambda).")
    p.add_argument("--select-weights", type=weight_spec_label, default=DEFAULT_SELECT_WEIGHTS,
                   help="Importance-weight transform of the DR selection score and post-hoc estimates.")
    p.add_argument("--slim", action="store_true")
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic torch/cuDNN/cuBLAS kernels so a seed reproduces results "
        "exactly (default: on). All RNGs are seeded from --seeds either way.",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="CPU threads for numpy/BLAS/torch in every process (default: 4). Fixed so the "
        "same seed reproduces exactly across serial/parallel runs and machines.",
    )
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
        "--memory-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap concurrent workers to fit each cell's estimated peak memory "
        "(default: on); see run_full_study_parallel --memory-cap.",
    )
    p.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    try:
        resolve_bias_configs(args.bias_configs)
    except ValueError as e:
        p.error(str(e))
    pin_cpu_threads(args.cpu_threads)  # env is inherited by spawned workers

    out_root = args.out_dir / f"run_{args.run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_cfgs = list(_iter_h1_configs(args, out_root))
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
        failures = _run_with_memory_cap(
            run_configs,
            max_workers=workers,
            min_workers=max(1, int(args.min_workers)),
            num_gpus=num_gpus,
            memory_cap=bool(args.memory_cap),
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
            "bias_configs": resolve_bias_configs(args.bias_configs),
            "world_options": {**world_options_from_args(args), "ctr_reference": "uniform"},
            "train_sizes": args.train_sizes,
            "target_rand_ctrs": list(args.target_rand_ctrs),
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
