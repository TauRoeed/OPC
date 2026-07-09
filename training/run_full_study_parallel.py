import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


def _resolve_num_gpus(explicit: int | None) -> int:
    """GPUs visible to this job (Slurm CUDA_VISIBLE_DEVICES or torch probe)."""
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return max(1, len([x for x in visible.split(",") if x.strip() != ""]))
    try:
        import torch

        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def _apply_parallel_thread_limits(workers: int) -> int:
    """Avoid CPU oversubscription when many process workers run (spawn inherits env)."""
    if workers <= 1:
        return 1
    cpus = os.cpu_count() or 8
    per_worker = max(1, min(4, cpus // workers))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, str(per_worker))
    print(
        f"Parallel pool: {workers} workers; per-worker CPU threads={per_worker} "
        f"(override via OMP_NUM_THREADS etc.)",
        flush=True,
    )
    return per_worker


def _parallel_worker_init(worker_slot, num_gpus: int) -> None:
    """Round-robin workers across cuda:0..N-1 via OPC_WORKER_GPU."""
    os.environ["OPC_IN_PARALLEL"] = "1"
    with worker_slot.get_lock():
        idx = int(worker_slot.value)
        worker_slot.value = idx + 1
    n = max(1, int(num_gpus))
    gpu_slot = idx % n
    os.environ["OPC_WORKER_GPU"] = str(gpu_slot)
    print(
        f"[worker init] pid={os.getpid()} worker={idx} "
        f"OPC_WORKER_GPU={gpu_slot} (pool over {n} GPUs)",
        flush=True,
    )

from training.run_full_study import (
    VALID_NOISE_AXES,
    VALID_STUDY_METHODS,
    _collect_existing_summaries,
    _finalize_summary_df,
    _normalize_study_methods,
    _no_prop_policy_loss_types,
    _resolve_val_size_configs,
    _run_condition,
)
from training.trainer_trials import (
    DEFAULT_QHAT_ACTION_CHUNK,
    DEFAULT_QHAT_USER_CHUNK,
)


def _iter_run_configs(args, val_size_cfg, val_label, val_root: Path):
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
                            if val_label != "frac":
                                run_key = f"{run_key}__val={val_label}"
                            yield {
                                "dataset_name": dataset_name,
                                "noise_mode": noise_mode,
                                "noise_axis": noise_axis,
                                "noise_level": noise_level,
                                "ctr": float(ctr),
                                "seed": int(seed),
                                "run_key": run_key,
                                "run_dir": str(val_root / run_key),
                                "val_size": val_size_cfg,
                            }


def _execute_run(config: dict):
    os.environ["OPC_IN_PARALLEL"] = "1"
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
        policy_loss_types=tuple(config["policy_loss_types"]),
        search_use_log_trick=bool(config.get("search_use_log_trick", True)),
        shared_regression_size=int(config.get("shared_regression_size", 50_000)),
        qhat_user_chunk=int(config.get("qhat_user_chunk", DEFAULT_QHAT_USER_CHUNK)),
        qhat_action_chunk=int(
            config.get("qhat_action_chunk", DEFAULT_QHAT_ACTION_CHUNK)
        ),
        require_cuda=bool(config.get("require_cuda", False)),
        optuna_batch_sizes=config.get("optuna_batch_sizes"),
        methods=tuple(config.get("study_methods", VALID_STUDY_METHODS)),
    )

    summary_df = _finalize_summary_df(
        opc_df,
        noprop_df,
        meta,
        dataset=config["dataset_name"],
        noise_mode=config["noise_mode"],
        noise_axis=config["noise_axis"],
        noise_level=config["noise_level"],
        seed=config["seed"],
    )

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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--optuna-batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes for Optuna to search (default: 256 512 1024 2048 4096). "
        "Not a sweep axis; only tunes inside each condition.",
    )
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
    parser.add_argument(
        "--val-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Sweep fixed validation sizes; each gets val_<n>/ subfolder.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--val-min", type=int, default=5000)
    parser.add_argument("--val-max", type=int, default=None)
    parser.add_argument(
        "--policy-losses",
        nargs="+",
        default=["kl"],
        help="Policy losses: kl, ipw, sndr, crm (multiple = Optuna categorical).",
    )
    parser.add_argument(
        "--no-log-trick",
        action="store_true",
        help="Disable log-trick for KL/IPW/SNDR/CRM; skip Optuna tuning of use_log_trick.",
    )
    parser.add_argument(
        "--shared-regression-size",
        type=int,
        default=50_000,
        help="Reg slice in each reg+train+val sim; fit once on first setup, reuse.",
    )
    parser.add_argument(
        "--qhat-user-chunk",
        type=int,
        default=DEFAULT_QHAT_USER_CHUNK,
        help="User/context block for lazy q_hat (default 5000).",
    )
    parser.add_argument(
        "--qhat-action-chunk",
        type=int,
        default=DEFAULT_QHAT_ACTION_CHUNK,
        help="Action block for lazy q_hat (default 5000).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available in worker.",
    )
    parser.add_argument("--emb-dir", default="BPR/embeddings")
    parser.add_argument("--out-dir", default="artifacts/full_study")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Round-robin pool workers across this many GPUs (default: count from "
        "CUDA_VISIBLE_DEVICES or torch.cuda.device_count()).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel process workers. Each worker loads embeddings, simulates "
        "logged splits on demand, and runs OPC + no-propensity; large --val-sizes "
        "or many train sizes increase RAM. Use --max-workers 1 if workers die "
        "(OOM / 'terminated abruptly').",
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
        "--methods",
        nargs="+",
        default=list(VALID_STUDY_METHODS),
        choices=list(VALID_STUDY_METHODS),
        help="Which arms to run. Use no_propensity alone to rerun baseline after OPC finished.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip conditions whose summary_metrics.csv already exists (default: true).",
    )
    parser.add_argument("--fail-fast", action="store_true", default=False)
    args = parser.parse_args()
    methods = _normalize_study_methods(args.methods)
    policy_loss_types = tuple(str(x).lower() for x in args.policy_losses)
    search_use_log_trick = not bool(args.no_log_trick)
    val_size_configs = _resolve_val_size_configs(args)

    emb_dir = Path(args.emb_dir)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    failures = []
    run_configs = []
    for val_size_cfg, val_label in val_size_configs:
        val_root = out_dir if len(val_size_configs) == 1 else out_dir / f"val_{val_label}"
        val_root.mkdir(parents=True, exist_ok=True)
        for base_cfg in _iter_run_configs(args, val_size_cfg, val_label, val_root):
            summary_path = Path(base_cfg["run_dir"]) / "summary_metrics.csv"
            if args.skip_completed and summary_path.exists():
                if methods == VALID_STUDY_METHODS:
                    print(f"Skipping completed: {base_cfg['run_key']}")
                    continue
                if methods == ("no_propensity",):
                    summary = pd.read_csv(summary_path)
                    if "method" in summary.columns and (
                        summary["method"] == "no_propensity"
                    ).any():
                        print(f"Skipping completed no-prop: {base_cfg['run_key']}")
                        continue
            cfg = {
                **base_cfg,
                "emb_dir": str(emb_dir),
                "train_sizes": list(args.train_sizes),
                "n_trials": int(args.n_trials),
                "batch_size": int(args.batch_size),
                "optuna_batch_sizes": args.optuna_batch_sizes,
                "val_frac": float(args.val_frac),
                "val_min": int(args.val_min),
                "val_max": args.val_max,
                "policy_reward_mode": args.policy_reward_mode,
                "policy_reward_mc_sim": int(args.policy_reward_mc_sim),
                "policy_temperature": float(args.policy_temperature),
                "slim": bool(args.slim),
                "policy_loss_types": list(policy_loss_types),
                "study_methods": list(methods),
                "search_use_log_trick": search_use_log_trick,
                "shared_regression_size": int(args.shared_regression_size),
                "qhat_user_chunk": int(args.qhat_user_chunk),
                "qhat_action_chunk": int(args.qhat_action_chunk),
                "require_cuda": bool(args.require_cuda),
            }
            run_configs.append(cfg)

    max_train = max(args.train_sizes) if args.train_sizes else 0
    max_val = max(args.val_sizes) if args.val_sizes else int(args.val_min)
    logged_rows = int(args.shared_regression_size) + int(max_train) + int(max_val)
    workers = max(1, int(args.max_workers))
    num_gpus = _resolve_num_gpus(args.num_gpus)
    _apply_parallel_thread_limits(workers)
    per_gpu = workers / max(1, num_gpus)
    if logged_rows >= 40_000 and workers > 1:
        print(
            f"WARNING: ~{logged_rows} logged rows per setup + reg fit; "
            f"--max-workers {workers} (~{per_gpu:.1f}/GPU over {num_gpus} GPUs) "
            f"may OOM RAM/VRAM on large catalogs.",
            flush=True,
        )
    print(
        f"Running {len(run_configs)} conditions with max_workers={workers} "
        f"over {num_gpus} GPU(s)",
        flush=True,
    )
    mp_ctx = mp.get_context("spawn")
    worker_slot = mp_ctx.Value("i", 0, lock=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_parallel_worker_init,
        initargs=(worker_slot, num_gpus),
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
                    "val_sizes": args.val_sizes,
                    "val_frac": args.val_frac,
                    "policy_loss_types": list(policy_loss_types),
                    "no_log_trick": bool(args.no_log_trick),
                    "shared_regression_size": int(args.shared_regression_size),
                    "qhat_user_chunk": int(args.qhat_user_chunk),
                    "qhat_action_chunk": int(args.qhat_action_chunk),
                    "require_cuda": bool(args.require_cuda),
                    "val_min": args.val_min,
                    "val_max": args.val_max,
                    "policy_reward_mode": args.policy_reward_mode,
                    "policy_reward_mc_sim": args.policy_reward_mc_sim,
                    "optuna_batch_sizes": args.optuna_batch_sizes,
                    "policy_temperature": args.policy_temperature,
                    "max_workers": args.max_workers,
                    "num_gpus": num_gpus,
                },
                f,
                indent=2,
            )
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()
