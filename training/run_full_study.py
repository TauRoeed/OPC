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
    LazyRegressionSplitCache,
    DEFAULT_QHAT_ACTION_CHUNK,
    DEFAULT_QHAT_USER_CHUNK,
    fit_shared_regression_bundle,
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


from utils.noise_levels import VALID_NOISE_AXES, noise_eps as _noise_eps
from utils.noise_snr import dataset_snr_report
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


def _collect_existing_summaries(base_dir: Path):
    rows = []
    for summary_path in sorted(base_dir.glob("**/dataset=*/summary_metrics.csv")):
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
    policy_loss_types: tuple[str, ...] = ("kl_crm",),
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
):
    methods = _normalize_study_methods(methods)
    run_opc = "opc" in methods
    run_no_prop = "no_propensity" in methods
    noprop_policy_loss_types = _no_prop_policy_loss_types(policy_loss_types)
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
            f"arrays (looked under {user_meta_path}, {item_meta_path}). "
            f"Run: python BPR/generate_artifacts.py --dataset {dataset_name}"
        )

    if (
        float(eps_meta) > 0
        and metadata_x is None
        and metadata_a is None
        and noise_axis == "combined"
    ):
        print(
            f"WARNING: {dataset_name}: no metadata under {emb_dir}; "
            f"combined noise uses context+action only (eps_meta {eps_meta:g} -> 0). "
            f"For full combined noise: python BPR/generate_artifacts.py --dataset {dataset_name}",
            flush=True,
        )
        eps_meta = 0.0

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
        "logging_uniform_mix": float(np.clip(logging_uniform_mix, 0.0, 1.0)),
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
    snr_report = dataset_snr_report(
        dataset, eps1=float(eps1), eps2=float(eps2), eps_meta=float(eps_meta)
    )

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
        )
    else:
        opc_df = _load_cached_method_df(run_dir, "opc")
        opc_trials = _load_cached_method_trials(run_dir, "opc")

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
        )
    else:
        noprop_df = _load_cached_method_df(run_dir, "no_propensity")
        noprop_trials = _load_cached_method_trials(run_dir, "no_propensity")

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
        "snr": snr_report,
        "train_sizes": [int(x) for x in train_sizes],
        "n_trials": int(n_trials),
        "batch_size": int(batch_size),
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
    opc_df = opc_df.reset_index().rename(columns={"index": "train_size"})
    noprop_df = noprop_df.reset_index().rename(columns={"index": "train_size"})
    opc_df["method"] = "opc"
    noprop_df["method"] = "no_propensity"
    summary_df = pd.concat([opc_df, noprop_df], ignore_index=True)
    for k, v in tags.items():
        summary_df[k] = v
    summary_df["ctr"] = float(meta["ctr"])
    if "val_size" in summary_df.columns:
        summary_df["val_size_config"] = summary_df["val_size"]
    return add_paired_method_pct_columns(summary_df)


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
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        default=["low", "medium", "high"],
        help="Noise levels: low/medium/high/extreme/brutal "
        "(extreme≈0.35+0.40+0.20, brutal≈0.50+0.50+0.30 eps mix).",
    )
    parser.add_argument(
        "--logging-uniform-mix",
        type=float,
        default=0.0,
        help="Mix logging policy with uniform: π_b=(1-α)·π_noisy + α/|A|. "
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
        help="Shared q_hat for DM/DR: regression (default LR fit on noisy emb), "
        "logging_score (CTR link on our_x/our_a), oracle (clean env; sim-only).",
    )
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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--optuna-batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes for Optuna to search (default: 4096 8192 16384). "
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
        help="Softmax temperature for dot-product policies (logging/eval). Default 1.",
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
        "--policy-losses",
        nargs="+",
        default=["kl_crm"],
        choices=list(VALID_POLICY_LOSSES),
        help="Policy-gradient losses for Optuna (default kl_crm = SNDR+KL+CRM, no split). "
        "Multiple values = categorical over legacy losses.",
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

    emb_dir = Path(args.emb_dir)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")
    print(f"Validation configs: {val_size_configs}")
    print(f"Policy losses: {policy_loss_types}")
    print(f"Methods: {methods}")
    if "no_propensity" in methods:
        print(f"No-prop losses: {_no_prop_policy_loss_types(policy_loss_types)}")
    print(f"Search use_log_trick: {search_use_log_trick}")

    all_summary_rows = []
    failures = []

    for val_size_cfg, val_label in val_size_configs:
        val_root = out_dir if len(val_size_configs) == 1 else out_dir / f"val_{val_label}"
        val_root.mkdir(parents=True, exist_ok=True)

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
                                if len(val_size_configs) > 1:
                                    run_key = f"{run_key}__val={val_label}"

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
                                        noise_mode=noise_mode,
                                        noise_axis=noise_axis,
                                        noise_level=noise_level,
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
                                        policy_temperature=args.policy_temperature,
                                        run_dir=run_dir,
                                        slim=bool(args.slim),
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
                                    noise_mode=noise_mode,
                                    noise_axis=noise_axis,
                                    noise_level=noise_level,
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
                    "noise_modes": args.noise_modes,
                    "noise_axes": args.noise_axes,
                    "noise_levels": args.noise_levels,
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
                    "policy_temperature": args.policy_temperature,
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
