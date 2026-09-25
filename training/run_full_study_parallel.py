import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
    # --cpu-threads (pinned in main) wins; this only fills unset vars.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, str(per_worker))
    print(
        f"Parallel pool: {workers} workers; per-worker CPU threads="
        f"{os.environ.get('OMP_NUM_THREADS', per_worker)} (set via --cpu-threads)",
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

from BPR.bpr_config import DEFAULT_DATASETS
from models.models import REWARD_FEATURES
from utils.seeding import DEFAULT_CPU_THREADS, pin_cpu_threads
from training.memory_budget import describe_plan, device_capacities, plan_worker_groups
from training.run_full_study import (
    VALID_STUDY_METHODS,
    _collect_existing_summaries,
    _condition_run_key,
    _finalize_summary_df,
    _normalize_study_methods,
    _no_prop_policy_loss_types,
    _resolve_val_size_configs,
    _run_condition,
)
from training.trainer_trials import (
    DEFAULT_QHAT_ACTION_CHUNK,
    DEFAULT_QHAT_USER_CHUNK,
    VALID_OPTUNA_SELECTION,
    VALID_REWARD_MODELS,
)
from utils.representation_bias import add_world_arguments, resolve_bias_configs, world_options_from_args


def _iter_run_configs(args, out_dir: Path, val_size_configs: list):
    """Yield run configs in order: seed → dataset → ctr → [val] → bias.

    ``val`` nest/key suffix only when ``--val-size`` / ``--val-sizes`` is set
    (label != ``frac``).
    """
    multi_val = len(val_size_configs) > 1
    bias_configs = resolve_bias_configs(args.bias_configs)
    world_options = world_options_from_args(args)
    for seed in args.seeds:
        for dataset_name in args.datasets:
            for ctr in args.ctr_levels:
                for val_size_cfg, val_label in val_size_configs:
                    val_root = (
                        out_dir / f"val_{val_label}"
                        if multi_val
                        else out_dir
                    )
                    for bias in bias_configs:
                        # Tag folder with val only for real fixed/swept vals.
                        run_key = _condition_run_key(dataset_name, bias, ctr, seed, world_options, val_label)
                        yield {
                            "dataset_name": dataset_name,
                            "bias": bias,
                            "ctr": float(ctr),
                            "seed": int(seed),
                            "run_key": run_key,
                            "run_dir": str(val_root / run_key),
                            "val_size": val_size_cfg,
                        }


def _is_oom_like(exc: BaseException) -> bool:
    """True for CUDA/host OOM or worker death that usually means OOM."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, BrokenProcessPool):
        return True
    # torch.cuda.OutOfMemoryError subclasses RuntimeError
    name = type(exc).__name__.lower()
    if "outofmemory" in name or name == "memoryerror":
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cudaerror",
        "cublas",
        "cudnn_status_alloc_failed",
        "failed to allocate",
        "cannot allocate memory",
        "killed",
        "terminated abruptly",
        "sigkill",
        "brokenprocesspool",
        "oom",
    )
    return any(n in msg for n in needles)


def _default_executor_factory(workers, mp_ctx, worker_slot, num_gpus):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_parallel_worker_init,
        initargs=(worker_slot, num_gpus),
    )


def _run_configs_with_oom_backoff(
    run_configs: list[dict],
    *,
    max_workers: int,
    min_workers: int,
    num_gpus: int,
    fail_fast: bool,
    oom_backoff: bool,
    execute_fn=None,
    executor_factory=None,
) -> list[dict]:
    """
    Run configs in a process pool. On OOM / abrupt worker death:
      - re-queue unfinished + OOM'd configs
      - shrink max_workers by 1 (floor at min_workers)
      - restart the pool and continue
    Non-OOM errors are recorded as permanent failures (unless fail_fast).

    execute_fn / executor_factory are injectable for tests (defaults: process pool).
    """
    if execute_fn is None:
        execute_fn = _execute_run
    if executor_factory is None:
        executor_factory = _default_executor_factory

    pending = list(run_configs)
    failures: list[dict] = []
    workers = max(1, int(max_workers))
    min_workers = max(1, min(int(min_workers), workers))
    mp_ctx = mp.get_context("spawn")
    wave = 0

    while pending:
        wave += 1
        _apply_parallel_thread_limits(workers)
        print(
            f"[wave {wave}] {len(pending)} remaining, max_workers={workers} "
            f"over {num_gpus} GPU(s)",
            flush=True,
        )
        worker_slot = mp_ctx.Value("i", 0, lock=True)
        oom_hit = False
        oom_exc: BaseException | None = None
        still_pending: list[dict] = []
        batch = list(pending)
        pending = []

        with executor_factory(workers, mp_ctx, worker_slot, num_gpus) as pool:
            future_to_cfg = {pool.submit(execute_fn, cfg): cfg for cfg in batch}
            try:
                for fut in as_completed(future_to_cfg):
                    cfg = future_to_cfg[fut]
                    try:
                        fut.result()
                        print(f"Done: {cfg['run_key']}", flush=True)
                    except Exception as e:
                        if oom_backoff and _is_oom_like(e):
                            oom_hit = True
                            oom_exc = e
                            still_pending.append(cfg)
                            print(
                                f"OOM-like on {cfg['run_key']}: {e!r} "
                                f"— will retry after wave (workers may shrink)",
                                flush=True,
                            )
                            continue

                        failures.append(
                            {"run_key": cfg["run_key"], "error": repr(e)}
                        )
                        print(f"FAILED {cfg['run_key']}: {e}", flush=True)
                        if fail_fast:
                            for other in future_to_cfg:
                                other.cancel()
                            raise
            except BrokenProcessPool as e:
                oom_hit = True
                oom_exc = e
                print(
                    f"Broken process pool ({e!r}) — re-queue unfinished and shrink",
                    flush=True,
                )
                for fut, cfg in future_to_cfg.items():
                    if fut.done():
                        try:
                            fut.result()
                            print(f"Done: {cfg['run_key']}", flush=True)
                        except Exception as e2:
                            if _is_oom_like(e2) or isinstance(e2, BrokenProcessPool):
                                still_pending.append(cfg)
                            else:
                                # Already recorded above, or new:
                                if not any(
                                    f["run_key"] == cfg["run_key"] for f in failures
                                ):
                                    failures.append(
                                        {
                                            "run_key": cfg["run_key"],
                                            "error": repr(e2),
                                        }
                                    )
                    else:
                        still_pending.append(cfg)

        if not oom_hit:
            break

        seen: set[str] = set()
        pending = []
        for cfg in still_pending:
            key = cfg["run_key"]
            if key in seen:
                continue
            seen.add(key)
            pending.append(cfg)

        if not pending:
            break

        if not oom_backoff:
            for cfg in pending:
                failures.append(
                    {
                        "run_key": cfg["run_key"],
                        "error": repr(oom_exc)
                        if oom_exc is not None
                        else "unfinished after error",
                    }
                )
            break

        if workers > min_workers:
            workers -= 1
            print(
                f"Reducing max_workers -> {workers} after OOM "
                f"({type(oom_exc).__name__ if oom_exc else 'unknown'}); "
                f"retrying {len(pending)} job(s)",
                flush=True,
            )
            continue

        print(
            f"Still OOM at max_workers={workers} (min floor); "
            f"recording {len(pending)} remaining as failures.",
            flush=True,
        )
        for cfg in pending:
            failures.append(
                {
                    "run_key": cfg["run_key"],
                    "error": repr(oom_exc)
                    if oom_exc is not None
                    else "OOM after min_workers",
                }
            )
        break

    return failures


def _run_with_memory_cap(run_configs, *, max_workers, min_workers, num_gpus, memory_cap, **kwargs):
    """Run config groups, each with as many workers as fit in memory (see memory_budget)."""
    if not memory_cap or not run_configs:
        return _run_configs_with_oom_backoff(
            run_configs, max_workers=max_workers, min_workers=min_workers, num_gpus=num_gpus, **kwargs
        )
    kind, capacities = device_capacities(num_gpus)
    plan = plan_worker_groups(
        run_configs, max_workers=max_workers, capacities=capacities, n_slots=num_gpus
    )
    print(describe_plan(plan, kind, capacities, max_workers), flush=True)
    failures = []
    for workers, cfgs in plan:
        failures += _run_configs_with_oom_backoff(
            cfgs,
            max_workers=workers,
            min_workers=min(min_workers, workers),
            num_gpus=num_gpus,
            **kwargs,
        )
    return failures


def _execute_run(config: dict):
    os.environ["OPC_IN_PARALLEL"] = "1"
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    opc_df, noprop_df, opc_trials, noprop_trials, meta = _run_condition(
        dataset_name=config["dataset_name"],
        emb_dir=Path(config["emb_dir"]),
        bias=config["bias"],
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
        slim=bool(config.get("slim", False)),
        deterministic=bool(config.get("deterministic", True)),
        cpu_threads=int(config.get("cpu_threads", DEFAULT_CPU_THREADS)),
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
        logging_uniform_mix=float(config.get("logging_uniform_mix", 0.0)),
        optuna_selection=str(config.get("optuna_selection", "ci_low")),
        reward_model=str(config.get("reward_model", "regression")),
        reward_features=str(config.get("reward_features", "interaction")),
        world_options=config.get("world_options"),
    )

    summary_df = _finalize_summary_df(
        opc_df,
        noprop_df,
        meta,
        dataset=config["dataset_name"],
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
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Default: " + " ".join(DEFAULT_DATASETS) + ".")
    add_world_arguments(parser)
    parser.add_argument(
        "--logging-uniform-mix",
        type=float,
        default=0.0,
        help="Mix logging with uniform: π_b=(1-α)·π_biased + α/|A|. Try 0.2–0.5.",
    )
    parser.add_argument(
        "--optuna-selection",
        choices=list(VALID_OPTUNA_SELECTION),
        default="ci_low",
        help="What Optuna maximizes: ci_low (default), r_hat, or actual_reward.",
    )
    parser.add_argument(
        "--reward-model",
        choices=list(VALID_REWARD_MODELS),
        default="regression",
        help="Shared q_hat: regression (default), logging_score, or oracle (sim-only).",
    )
    parser.add_argument(
        "--reward-features",
        choices=list(REWARD_FEATURES),
        default="interaction",
        help="Features of the regression reward model: interaction = [x, a, x*a] (default; item "
        "rankings can differ between users) or concat = [x, a] (the previous model: the same item "
        "ranking for every user).",
    )
    parser.add_argument(
        "--ctr-levels",
        nargs="+",
        type=float,
        default=[0.05],
        help="Target CTR of the reference policy (see --ctr-reference) to sweep.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[5000, 25000, 50000, 100000],
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fallback batch size when best Optuna trial lacks batch_size. "
        "Default: from train_size schedule.",
    )
    parser.add_argument(
        "--optuna-batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes for Optuna to search. Default: schedule neighborhood "
        "from train_size. Not a sweep axis; only tunes inside each condition.",
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
        default=["sndr"],
        help="OPC policy loss (default sndr). Fixed DR score clip. "
        "No-prop stays naive.",
    )
    parser.add_argument(
        "--no-log-trick",
        action="store_true",
        help="Disable log-trick for policy losses; skip Optuna tuning of use_log_trick.",
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
        "--memory-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap concurrent workers so each condition's estimated peak memory fits in "
        "free GPU memory (or RAM without a GPU); conditions are grouped by size "
        "(default: on). Results are unchanged; --max-workers stays the upper bound.",
    )
    parser.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On CUDA/host OOM or abrupt worker death: re-queue unfinished jobs "
        "and reduce --max-workers by 1 (default: true). Use --no-oom-backoff to disable.",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Floor for OOM backoff worker count (default: 1).",
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
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic torch/cuDNN/cuBLAS kernels so a seed reproduces results "
        "exactly (default: on). All RNGs are seeded from --seeds either way.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="CPU threads for numpy/BLAS/torch in every process (default: 4). Fixed so the "
        "same seed reproduces exactly across serial/parallel runs and machines.",
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
    pin_cpu_threads(args.cpu_threads)  # env is inherited by spawned workers
    methods = _normalize_study_methods(args.methods)
    policy_loss_types = tuple(str(x).lower() for x in args.policy_losses)
    search_use_log_trick = not bool(args.no_log_trick)
    val_size_configs = _resolve_val_size_configs(args)
    try:
        bias_configs = resolve_bias_configs(args.bias_configs)
    except ValueError as e:
        parser.error(str(e))
    world_options = world_options_from_args(args)

    emb_dir = Path(args.emb_dir)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    run_configs = []
    # Ensure val_* roots exist before workers start.
    for _, val_label in val_size_configs:
        val_root = (
            out_dir
            if len(val_size_configs) == 1
            else out_dir / f"val_{val_label}"
        )
        val_root.mkdir(parents=True, exist_ok=True)

    print(
        "Run order: seed → dataset → ctr → "
        + ("val → " if any(lbl != "frac" for _, lbl in val_size_configs) else "")
        + f"bias {bias_configs}; world options: {world_options}",
        flush=True,
    )
    for base_cfg in _iter_run_configs(args, out_dir, val_size_configs):
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
            "batch_size": int(args.batch_size) if args.batch_size is not None else None,
            "optuna_batch_sizes": args.optuna_batch_sizes,
            "val_frac": float(args.val_frac),
            "val_min": int(args.val_min),
            "val_max": args.val_max,
            "policy_reward_mode": args.policy_reward_mode,
            "policy_reward_mc_sim": int(args.policy_reward_mc_sim),
            "world_options": world_options,
            "slim": bool(args.slim),
            "deterministic": bool(args.deterministic),
            "cpu_threads": int(args.cpu_threads),
            "policy_loss_types": list(policy_loss_types),
            "study_methods": list(methods),
            "search_use_log_trick": search_use_log_trick,
            "shared_regression_size": int(args.shared_regression_size),
            "qhat_user_chunk": int(args.qhat_user_chunk),
            "qhat_action_chunk": int(args.qhat_action_chunk),
            "require_cuda": bool(args.require_cuda),
            "logging_uniform_mix": float(args.logging_uniform_mix),
            "optuna_selection": str(args.optuna_selection),
            "reward_model": str(args.reward_model),
            "reward_features": str(args.reward_features),
        }
        run_configs.append(cfg)

    max_train = max(args.train_sizes) if args.train_sizes else 0
    max_val = max(args.val_sizes) if args.val_sizes else int(args.val_min)
    logged_rows = int(args.shared_regression_size) + int(max_train) + int(max_val)
    workers = max(1, int(args.max_workers))
    min_workers = max(1, min(int(args.min_workers), workers))
    num_gpus = _resolve_num_gpus(args.num_gpus)
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
        f"(oom_backoff={args.oom_backoff}, min_workers={min_workers}) "
        f"over {num_gpus} GPU(s)",
        flush=True,
    )
    failures = _run_with_memory_cap(
        run_configs,
        max_workers=workers,
        min_workers=min_workers,
        num_gpus=num_gpus,
        memory_cap=bool(args.memory_cap),
        fail_fast=bool(args.fail_fast),
        oom_backoff=bool(args.oom_backoff),
    )

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
                    "bias_configs": bias_configs,
                    "world_options": world_options,
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
                    "logging_uniform_mix": float(args.logging_uniform_mix),
                    "optuna_selection": str(args.optuna_selection),
                    "reward_model": str(args.reward_model),
                    "reward_features": str(args.reward_features),
                    "val_min": args.val_min,
                    "val_max": args.val_max,
                    "policy_reward_mode": args.policy_reward_mode,
                    "policy_reward_mc_sim": args.policy_reward_mc_sim,
                    "optuna_batch_sizes": args.optuna_batch_sizes,
                    "max_workers": args.max_workers,
                    "min_workers": min_workers,
                    "oom_backoff": bool(args.oom_backoff),
                    "num_gpus": num_gpus,
                },
                f,
                indent=2,
            )
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()
