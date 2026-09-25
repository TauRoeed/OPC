import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.metrics_utils import add_paired_method_pct_columns
from training.trainer_trials import (
    VALID_OPTUNA_SELECTION,
    VALID_POLICY_LOSSES,
    VALID_REWARD_MODELS,
    DEFAULT_DR_SCORE_CLIP_M,
    LazyRegressionSplitCache,
    DEFAULT_QHAT_ACTION_CHUNK,
    DEFAULT_QHAT_USER_CHUNK,
    estimate_condition_runtime_s,
    fit_shared_regression_bundle,
    format_runtime_estimate,
    no_propensity_trainer_trial,
    regression_trainer_trial,
)

VALID_STUDY_METHODS = ("opc", "no_propensity")


def _normalize_study_methods(methods: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if methods is None:
        return VALID_STUDY_METHODS
    out = []
    for name in methods:
        key = str(name).lower()
        if key not in VALID_STUDY_METHODS:
            raise ValueError(f"Unknown method {name!r}; expected one of {VALID_STUDY_METHODS}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one method required")
    return tuple(out)


def _no_prop_policy_loss_types(policy_loss_types: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """No-propensity baseline: pure naive reward (no DM/SNDR/IW/KL/CRM)."""
    _ = policy_loss_types
    return ("naive",)


def _load_cached_method_df(run_dir: Path, method: str) -> pd.DataFrame:
    """Reuse finished method rows when rerunning only the other arm."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary_metrics.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "method" in summary.columns:
            part = summary[summary["method"] == method].copy()
            if not part.empty and "train_size" in part.columns:
                return part.set_index("train_size")

    runs_name = "opc_runs_long.csv" if method == "opc" else "no_prop_runs_long.csv"
    runs_path = run_dir / runs_name
    if not runs_path.exists():
        raise FileNotFoundError(
            f"Cannot rerun only one method without cached {method} results in {run_dir}"
        )
    runs = pd.read_csv(runs_path)
    if runs.empty:
        raise FileNotFoundError(f"Cached runs file is empty: {runs_path}")
    if "is_winning_run" in runs.columns:
        runs = runs[runs["is_winning_run"].astype(bool)]
    if "train_size" not in runs.columns:
        raise ValueError(f"Missing train_size in cached runs: {runs_path}")
    rows = {}
    for train_size, grp in runs.groupby("train_size"):
        row = grp.iloc[0].to_dict()
        row.pop("train_size", None)
        row.pop("method", None)
        rows[int(train_size)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _load_cached_method_trials(run_dir: Path, method: str) -> pd.DataFrame:
    trials_name = "opc_trials_long.csv" if method == "opc" else "no_prop_trials_long.csv"
    fallback = "opc_trials.csv" if method == "opc" else "no_prop_trials.csv"
    for name in (trials_name, fallback):
        path = run_dir / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


from BPR.bpr_config import bpr_artifact_status
from utils.noise_snr import dataset_snr_report
from utils.representation_bias import (
    BIAS_TYPES,
    add_world_arguments,
    bias_label,
    describe_world,
    parse_bias,
    resolve_bias_configs,
    world_options_from_args,
    world_run_key_suffix,
)
from utils.seeding import (
    DEFAULT_CPU_THREADS,
    enable_determinism,
    pin_cpu_threads,
    seed_everything,
)
from utils.simulation_utils import generate_dataset


def _resolve_val_size_configs(args):
    """Return list of (val_size_or_none, label) for directory naming."""
    if getattr(args, "val_sizes", None):
        return [(int(v), f"{int(v):g}") for v in args.val_sizes]
    if args.val_size is not None:
        return [(int(args.val_size), f"{int(args.val_size):g}")]
    return [(None, "frac")]


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


def _condition_run_key(dataset_name: str, bias: str, ctr: float, seed: int, world_options: dict | None,
                       val_label: str = "frac") -> str:
    """Folder name of one condition: dataset, bias, CTR, seed, the world options that differ from
    the defaults (``world_run_key_suffix``) and the validation size when fixed."""
    key = f"dataset={dataset_name}__bias={bias}__ctr={ctr:g}__seed={seed}" + world_run_key_suffix(world_options)
    if val_label != "frac":
        key = f"{key}__val={val_label}"
    return key


def _wants_popularity(world_options: dict) -> bool:
    """True when the world weighs BPR's item bias (truth or logger)."""
    logger = world_options.get("logger_pop_strength")
    return float(world_options.get("pop_strength", 0.0)) > 0.0 or (logger is not None and float(logger) > 0.0)


def _load_item_bias(emb_dir: Path, dataset_name: str, world_options: dict):
    """BPR's item bias b when the world uses it (None otherwise)."""
    if not _wants_popularity(world_options):
        return None
    path = emb_dir / f"{dataset_name}_item_bias.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"--pop-strength / --logger-pop-strength need BPR's item bias {path}; "
            f"regenerate the embeddings: python -m BPR.generate_artifacts --dataset {dataset_name}"
        )
    return np.load(path)


def _collect_existing_summaries(base_dir: Path):
    rows = []
    for summary_path in sorted(base_dir.glob("**/dataset=*/summary_metrics.csv")):
        try:
            df = pd.read_csv(summary_path)
            if "noise_axis" not in df.columns:
                df["noise_axis"] = "combined"
            if "noise_component" not in df.columns:
                df["noise_component"] = "combined"
            if "ctr" not in df.columns:
                df["ctr"] = float("nan")
            rows.append(df)
        except Exception:
            pass
    return rows


def _run_condition(
    dataset_name: str,
    emb_dir: Path,
    bias: str,
    ctr: float,
    seed: int,
    train_sizes: list[int],
    n_trials: int,
    batch_size: int | None,
    val_size: int | None,
    val_frac: float,
    val_min: int,
    val_max: int | None,
    policy_reward_mode: str,
    policy_reward_mc_sim: int,
    run_dir: Path,
    slim: bool = False,
    policy_loss_types: tuple[str, ...] = ("sndr",),
    search_use_log_trick: bool = True,
    shared_regression_size: int = 50_000,
    qhat_user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    qhat_action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    require_cuda: bool = False,
    optuna_batch_sizes: list[int] | None = None,
    methods: tuple[str, ...] = VALID_STUDY_METHODS,
    logging_uniform_mix: float = 0.0,
    optuna_selection: str = "ci_low",
    reward_model: str = "regression",
    q_error: float = 0.0,
    q_bad_value: float | None = None,
    rand_ctr_meta: dict | None = None,
    world_options: dict | None = None,
    record_uniform_value: bool = False,
    dr_score_clip_m: float | None = None,
    deterministic: bool = True,
    cpu_threads: int = DEFAULT_CPU_THREADS,
):
    methods = _normalize_study_methods(methods)
    run_opc = "opc" in methods
    run_no_prop = "no_propensity" in methods
    noprop_policy_loss_types = _no_prop_policy_loss_types(policy_loss_types)
    n_methods = int(run_opc) + int(run_no_prop)
    for ts in train_sizes:
        est = estimate_condition_runtime_s(
            int(ts),
            int(n_trials),
            val_size=val_size,
            shared_regression_size=int(shared_regression_size),
            n_methods=max(1, n_methods),
            slim=bool(slim),
        )
        print(
            f"[runtime_est] train_size={int(ts)} n_trials={int(n_trials)} "
            f"{format_runtime_estimate(est)}",
            flush=True,
        )
    # One experiment seed drives every RNG; re-seed per condition so results do not
    # depend on run order or worker process.
    enable_determinism(deterministic)
    cpu_threads = pin_cpu_threads(cpu_threads)
    seed_everything(seed)
    levels = parse_bias(bias)
    label = bias_label(levels)
    world_options = dict(world_options or {})
    user_path, item_path, user_meta_path, item_meta_path = _dataset_paths(
        emb_dir, dataset_name
    )

    emb_x = np.load(user_path)
    emb_a = np.load(item_path)
    item_bias = _load_item_bias(emb_dir, dataset_name, world_options)
    bpr_status = bpr_artifact_status(emb_dir, dataset_name)
    if bpr_status["status"] != "ok":
        print(f"WARNING {dataset_name}: {bpr_status['message']}", flush=True)
    metadata_x = metadata_a = None
    if world_options.get("group_source") == "metadata":
        metadata_x = _load_optional_array(user_meta_path)
        metadata_a = _load_optional_array(item_meta_path)

    params = {
        "bias": label,
        "ctr": float(ctr),
        "logging_uniform_mix": float(np.clip(logging_uniform_mix, 0.0, 1.0)),
        **world_options,
    }

    dataset = generate_dataset(
        params=params,
        seed=seed,
        emb_a=emb_a,
        emb_x=emb_x,
        metadata_a=metadata_a,
        metadata_x=metadata_x,
        item_bias=item_bias,
    )
    world = dataset["world"]
    if record_uniform_value:
        from utils.simulation_utils import calc_uniform_reward

        world["uniform_value"] = float(calc_uniform_reward(dataset))
    print(describe_world(world), flush=True)
    snr_report = dataset_snr_report(dataset)

    reg_size = int(shared_regression_size)
    split_cache = LazyRegressionSplitCache(
        dataset,
        train_sizes,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
        condition_seed=int(seed),
        regression_size=reg_size,
    )
    first_train = int(min(train_sizes)) if len(train_sizes) > 0 else 10_000
    first_split = split_cache[(first_train, 0)]
    shared_regression_bundle = fit_shared_regression_bundle(
        dataset,
        first_split["reg_data"],
        reward_model=str(reward_model),
        user_chunk=int(qhat_user_chunk),
        action_chunk=int(qhat_action_chunk),
        q_error=float(q_error),
        q_bad_value=q_bad_value,
    )

    opc_log_paths = {
        "trials": run_dir / "opc_trials_long.csv",
        "runs": run_dir / "opc_runs_long.csv",
    }
    noprop_log_paths = {
        "trials": run_dir / "no_prop_trials_long.csv",
        "runs": run_dir / "no_prop_runs_long.csv",
    }

    if run_opc:
        opc_df, opc_trials = regression_trainer_trial(
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
            split_cache=split_cache,
            policy_loss_types=policy_loss_types,
            dataset_name=dataset_name,
            search_use_log_trick=search_use_log_trick,
            use_log_trick_fixed=True,
            shared_regression_bundle=shared_regression_bundle,
            shared_regression_size=shared_regression_size,
            qhat_user_chunk=qhat_user_chunk,
            qhat_action_chunk=qhat_action_chunk,
            require_cuda=require_cuda,
            optuna_batch_sizes=optuna_batch_sizes,
            optuna_selection=optuna_selection,
            reward_model=str(reward_model),
            dr_score_clip_m=dr_score_clip_m,
            seed=int(seed),
        )
    else:
        try:
            opc_df = _load_cached_method_df(run_dir, "opc")
            opc_trials = _load_cached_method_trials(run_dir, "opc")
        except FileNotFoundError:
            opc_df = pd.DataFrame()
            opc_trials = pd.DataFrame()

    if run_no_prop:
        noprop_df, noprop_trials = no_propensity_trainer_trial(
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
            split_cache=split_cache,
            policy_loss_types=noprop_policy_loss_types,
            dataset_name=dataset_name,
            search_use_log_trick=False,
            use_log_trick_fixed=False,
            shared_regression_bundle=shared_regression_bundle,
            shared_regression_size=shared_regression_size,
            qhat_user_chunk=qhat_user_chunk,
            qhat_action_chunk=qhat_action_chunk,
            require_cuda=require_cuda,
            optuna_batch_sizes=optuna_batch_sizes,
            optuna_selection=optuna_selection,
            reward_model=str(reward_model),
            seed=int(seed),
        )
    else:
        try:
            noprop_df = _load_cached_method_df(run_dir, "no_propensity")
            noprop_trials = _load_cached_method_trials(run_dir, "no_propensity")
        except FileNotFoundError:
            noprop_df = pd.DataFrame()
            noprop_trials = pd.DataFrame()

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
        "bias": levels,
        "bias_label": label,
        # analysis scripts group by these columns
        "noise_mode": "representation_bias",
        "noise_axis": "both",
        "noise_component": "combined",
        "noise_level": label,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "cpu_threads": int(cpu_threads),
        "params": params,
        "ctr": float(params["ctr"]),
        "world": world,
        "bpr": bpr_status,
        "snr": snr_report,
        "train_sizes": [int(x) for x in train_sizes],
        "n_trials": int(n_trials),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "val_size_fixed": val_size,
        "val_frac": val_frac,
        "val_min": val_min,
        "val_max": val_max,
        "policy_reward_mode": policy_reward_mode,
        "policy_reward_mc_sim": int(policy_reward_mc_sim),
        "optuna_batch_sizes": list(optuna_batch_sizes or []),
        "slim": bool(slim),
        "policy_loss_types": list(policy_loss_types),
        "search_use_log_trick": bool(search_use_log_trick),
        "opc_use_log_trick_fixed": True,
        "no_prop_use_log_trick_fixed": False,
        "study_methods": list(methods),
        "opc_policy_loss_types": list(policy_loss_types),
        "no_prop_policy_loss_types": list(noprop_policy_loss_types),
        "dr_score_clip_m": float(
            DEFAULT_DR_SCORE_CLIP_M if dr_score_clip_m is None else dr_score_clip_m
        ),
        "shared_regression_size": int(
            shared_regression_bundle.get("sample_size", reg_size)
        ),
        "require_cuda": bool(require_cuda),
        "shared_train_val_splits": True,
        "combined_reg_train_val_sim": True,
        "random_logged_partition": True,
        "qhat_user_chunk": int(qhat_user_chunk),
        "qhat_action_chunk": int(qhat_action_chunk),
        "logging_uniform_mix": float(params.get("logging_uniform_mix", 0.0)),
        "optuna_selection": str(optuna_selection),
        "reward_model": str(
            shared_regression_bundle.get("reward_model", reward_model)
        ),
        "q_error": float(shared_regression_bundle.get("q_error", q_error)),
        "q_bad_value": shared_regression_bundle.get("q_bad_value", q_bad_value),
        "rand_ctr": rand_ctr_meta or {},
    }
    return opc_df, noprop_df, opc_trials, noprop_trials, meta


def _finalize_summary_df(opc_df, noprop_df, meta: dict, **tags) -> pd.DataFrame:
    frames = []
    if opc_df is not None and not getattr(opc_df, "empty", True):
        part = opc_df.reset_index().rename(columns={"index": "train_size"})
        part["method"] = "opc"
        frames.append(part)
    if noprop_df is not None and not getattr(noprop_df, "empty", True):
        part = noprop_df.reset_index().rename(columns={"index": "train_size"})
        part["method"] = "no_propensity"
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    summary_df = pd.concat(frames, ignore_index=True)
    tags = {"dataset": meta.get("dataset"), **tags}
    for k in ("noise_mode", "noise_axis", "noise_component", "noise_level"):
        if k in meta:
            tags.setdefault(k, meta[k])
    for k, v in tags.items():
        summary_df[k] = v
    summary_df["ctr"] = float(meta["ctr"])
    world = meta.get("world")
    if world:
        for k in BIAS_TYPES:
            summary_df[f"bias_{k}"] = world["bias"][k]
        summary_df["signal_kept"] = float(world["signal_kept"])
        summary_df["logging_temperature"] = float(world["logging_temperature"])
        summary_df["pop_strength"] = float(world.get("pop_strength", 0.0))
        summary_df["logger_pop_strength"] = float(world.get("logger_pop_strength", 0.0))
    if "val_size" in summary_df.columns:
        summary_df["val_size_config"] = summary_df["val_size"]
    if {"opc", "no_propensity"}.issubset(set(summary_df.get("method", pd.Series(dtype=str)))):
        return add_paired_method_pct_columns(summary_df)
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Run full OPC vs no-propensity sweeps and export structured outputs."
    )
    parser.add_argument("--datasets", nargs="+", default=["ml", "myket", "kuairec", "kuairand"], help="Default: the four datasets with personalized clean worlds (see docs/representation_bias.md).")
    add_world_arguments(parser)
    parser.add_argument(
        "--logging-uniform-mix",
        type=float,
        default=0.0,
        help="Mix logging policy with uniform: π_b=(1-α)·π_biased + α/|A|. "
        "0=off. Try 0.2–0.5 to hurt coverage / make logging worse.",
    )
    parser.add_argument(
        "--optuna-selection",
        choices=list(VALID_OPTUNA_SELECTION),
        default="ci_low",
        help="What Optuna maximizes: ci_low (default), r_hat, or actual_reward "
        "(oracle; debug only).",
    )
    parser.add_argument(
        "--reward-model",
        choices=list(VALID_REWARD_MODELS),
        default="regression",
        help="Shared q_hat for DM/DR: regression (default LR fit on biased vectors), "
        "logging_score (env click model on our_x/our_a), oracle (clean env; sim-only).",
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

    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fallback batch size when best Optuna trial lacks batch_size. "
        "Default: from train_size schedule (see batch_schedule).",
    )
    parser.add_argument(
        "--optuna-batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes for Optuna to search. Default: schedule neighborhood "
        "from train_size (2× prior table, e.g. 2048/4096/8192 at 100k; "
        "163840/81920/327680 above 2M ≈10× 1M default). Not a sweep axis; only tunes inside each condition.",
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
        "--val-size",
        type=int,
        default=None,
        help="If set (and --val-sizes omitted), fixed validation logged trajectories "
        "for every train_size. Otherwise val_size = clamp(round(val_frac * train_size), val_min, val_max).",
    )
    parser.add_argument(
        "--val-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Sweep fixed validation sizes (e.g. 10000 50000). Overrides --val-size. "
        "Each size gets its own subfolder under the run tag.",
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
        "--policy-losses",
        nargs="+",
        default=["sndr"],
        choices=list(VALID_POLICY_LOSSES),
        help="OPC policy-gradient loss (default sndr = pure SNDR train). "
        "DR selection uses fixed IW clip (DEFAULT_DR_SCORE_CLIP_M). "
        "Multiple values = Optuna categorical over losses. "
        "No-propensity stays naive (no IW/clip).",
    )
    parser.add_argument(
        "--no-log-trick",
        action="store_true",
        help="Disable log-trick policy surrogate (direct probs). "
        "Skips tuning use_log_trick in Optuna.",
    )
    parser.add_argument(
        "--shared-regression-size",
        type=int,
        default=50_000,
        help="Reg slice size in each combined sim (reg+train+val). Reward model fits once "
        "on the first setup's reg slice; reused for OPC and no-prop.",
    )
    parser.add_argument(
        "--qhat-user-chunk",
        type=int,
        default=DEFAULT_QHAT_USER_CHUNK,
        help="User/context block size for lazy q_hat / softmax (default 5000).",
    )
    parser.add_argument(
        "--qhat-action-chunk",
        type=int,
        default=DEFAULT_QHAT_ACTION_CHUNK,
        help="Action block size for lazy q_hat / softmax (default 5000).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available.",
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
    print(f"Validation configs: {val_size_configs}")
    print(f"Bias configs: {bias_configs}; world options: {world_options}")
    print(f"Policy losses: {policy_loss_types}")
    print(f"Methods: {methods}")
    if "no_propensity" in methods:
        print(f"No-prop losses: {_no_prop_policy_loss_types(policy_loss_types)}")
    print(f"DR score clip M (OPC fixed): {DEFAULT_DR_SCORE_CLIP_M}")
    print(f"Search use_log_trick: {search_use_log_trick}")

    all_summary_rows = []
    failures = []

    print(
        "Run order: seed → dataset → ctr → "
        + ("val → " if any(lbl != "frac" for _, lbl in val_size_configs) else "")
        + "bias"
    )

    for seed in args.seeds:
        for dataset_name in args.datasets:
            for ctr in args.ctr_levels:
                for val_size_cfg, val_label in val_size_configs:
                    val_root = (
                        out_dir
                        if len(val_size_configs) == 1
                        else out_dir / f"val_{val_label}"
                    )
                    val_root.mkdir(parents=True, exist_ok=True)

                    for bias in bias_configs:
                        run_key = _condition_run_key(dataset_name, bias, ctr, seed, world_options, val_label)

                        print(f"\n=== Running {run_key} ===")
                        run_dir = val_root / run_key
                        run_dir.mkdir(parents=True, exist_ok=True)
                        summary_path = run_dir / "summary_metrics.csv"
                        if args.skip_completed and summary_path.exists():
                            if methods == VALID_STUDY_METHODS:
                                print(f"Skipping completed: {run_key}")
                                try:
                                    all_summary_rows.append(pd.read_csv(summary_path))
                                except Exception:
                                    pass
                                continue
                            if methods == ("no_propensity",):
                                summary = pd.read_csv(summary_path)
                                if (
                                    "method" in summary.columns
                                    and (summary["method"] == "no_propensity").any()
                                ):
                                    print(f"Skipping completed no-prop: {run_key}")
                                    try:
                                        all_summary_rows.append(summary)
                                    except Exception:
                                        pass
                                    continue

                        try:
                            opc_df, noprop_df, opc_trials, noprop_trials, meta = _run_condition(
                                dataset_name=dataset_name,
                                emb_dir=emb_dir,
                                bias=bias,
                                ctr=ctr,
                                seed=seed,
                                train_sizes=args.train_sizes,
                                n_trials=args.n_trials,
                                batch_size=args.batch_size,
                                val_size=val_size_cfg,
                                val_frac=args.val_frac,
                                val_min=args.val_min,
                                val_max=args.val_max,
                                policy_reward_mode=args.policy_reward_mode,
                                policy_reward_mc_sim=args.policy_reward_mc_sim,
                                run_dir=run_dir,
                                slim=bool(args.slim),
                                deterministic=bool(args.deterministic),
                                cpu_threads=int(args.cpu_threads),
                                policy_loss_types=policy_loss_types,
                                search_use_log_trick=search_use_log_trick,
                                shared_regression_size=args.shared_regression_size,
                                qhat_user_chunk=args.qhat_user_chunk,
                                qhat_action_chunk=args.qhat_action_chunk,
                                require_cuda=bool(args.require_cuda),
                                optuna_batch_sizes=args.optuna_batch_sizes,
                                methods=methods,
                                logging_uniform_mix=float(args.logging_uniform_mix),
                                optuna_selection=str(args.optuna_selection),
                                reward_model=str(args.reward_model),
                                world_options=world_options,
                            )
                        except Exception as e:
                            failures.append({"run_key": run_key, "error": repr(e)})
                            print(f"FAILED {run_key}: {e}")
                            if args.fail_fast:
                                raise
                            continue

                        summary_df = _finalize_summary_df(
                            opc_df,
                            noprop_df,
                            meta,
                            dataset=dataset_name,
                            seed=seed,
                        )

                        summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
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
                    "bias_configs": bias_configs,
                    "world_options": world_options,
                    "ctr_levels": args.ctr_levels,
                    "seeds": args.seeds,
                    "train_sizes": args.train_sizes,
                    "val_size_fixed": args.val_size,
                    "val_sizes": args.val_sizes,
                    "val_frac": args.val_frac,
                    "val_min": args.val_min,
                    "val_max": args.val_max,
                    "policy_reward_mode": args.policy_reward_mode,
                    "policy_reward_mc_sim": args.policy_reward_mc_sim,
                    "optuna_batch_sizes": args.optuna_batch_sizes,
                    "policy_loss_types": list(policy_loss_types),
                    "study_methods": list(methods),
                    "no_prop_policy_loss_types": list(
                        _no_prop_policy_loss_types(policy_loss_types)
                    ),
                    "no_log_trick": bool(args.no_log_trick),
                    "shared_regression_size": int(args.shared_regression_size),
                    "qhat_user_chunk": int(args.qhat_user_chunk),
                    "qhat_action_chunk": int(args.qhat_action_chunk),
                    "require_cuda": bool(args.require_cuda),
                    "val_size_configs": [
                        {"val_size": v, "label": lbl} for v, lbl in val_size_configs
                    ],
                    "logging_uniform_mix": float(args.logging_uniform_mix),
                    "optuna_selection": str(args.optuna_selection),
                    "reward_model": str(args.reward_model),
                },
                f,
                indent=2,
            )
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)


if __name__ == "__main__":
    main()
