import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from training.metrics_utils import add_paired_method_pct_columns
from training.run_state import (
    CONFIG_KEY_COLUMN,
    ROW_KEY,
    _read_json,
    arm_config,
    arm_labels,
    arm_log_paths,
    atomic_write_csv,
    atomic_write_json,
    condition_lock,
    condition_plan,
    config_key,
    dedupe_runs,
    merge_run_meta,
    merge_summary,
    read_csv_or_none,
    read_summary,
    rebuild_long_logs,
    record_invocation,
    replace_rows,
    reset_arm_logs,
    utc_now,
)
from training.trainer_trials import (
    DEFAULT_SEARCH_SPACE,
    VALID_OPTUNA_SELECTION,
    VALID_POLICY_LOSSES,
    VALID_REWARD_MODELS,
    DEFAULT_SELECT_WEIGHTS,
    LazyRegressionSplitCache,
    DEFAULT_QHAT_ACTION_CHUNK,
    DEFAULT_QHAT_USER_CHUNK,
    LOGGED_RUN_IDX,
    estimate_condition_runtime_s,
    SN_SCOPES,
    fit_shared_regression_bundle,
    format_runtime_estimate,
    no_propensity_trainer_trial,
    regression_trainer_trial,
    resolve_search_space,
    _dataset_log_constants,
    _policy_greedy_reward_from_embeddings,
    _training_device,
)

from models.shared_objectives import LAMBDA_GRID
from training.blob_trials import add_blob_arguments, blob_options_from_args
from training.cause_trials import add_cause_arguments, cause_options_from_args

VALID_STUDY_METHODS = ("opc", "no_propensity")  # the default arms
# Opt-in baselines (--methods): dm = policy trained and selected on q_hat alone (no propensities);
# tempered_logger = no training, the logger's logits x s with s chosen by the DR selection score.
BASELINE_METHODS = ("dm", "tempered_logger")
# Opt-in prior-work baselines: cause = native CausE on the fixed-budget warm/uniform data
# (training/cause_trials.py; one summary method per prediction and rho, cause_<prediction>_r<rho per mille>).
PRIOR_WORK_METHODS = ("cause", "blob")
# The shared-objective arms (docs/shared_objective_study.md): one global correction model trained with penalized
# likelihood, uniform-reference weighted likelihood (raw, clipped at 10) or OPC's value objective, each plus the same
# source anchor; one paired random search (seed label "shared"). Arm -> trainer arguments.
SHARED_OBJECTIVE_ARMS = {
    "shared_likelihood": {"shared_objective": "likelihood"},
    "shared_iw_likelihood": {"shared_objective": "iw_likelihood"},
    "shared_iw_likelihood_clip10": {"shared_objective": "iw_likelihood", "iw_clip": 10.0},
    "shared_opc": {"shared_objective": "opc"},
    # docs/opc_gradient_regime_study.md §6-§8: raw-DR OPC, oracle-q training (diagnostic only) and the matched-steps
    # replays with larger batches; every other setting as shared_opc
    "shared_opc_raw": {"shared_objective": "opc", "train_weights": "none"},
    "shared_opc_oq": {"shared_objective": "opc", "train_reward": "oracle"},
    "shared_opc_raw_oq": {"shared_objective": "opc", "train_weights": "none", "train_reward": "oracle"},
    "shared_opc_b8192": {"shared_objective": "opc", "train_batch": 8192},
    "shared_opc_raw_b8192": {"shared_objective": "opc", "train_weights": "none", "train_batch": 8192},
    "shared_opc_bfull": {"shared_objective": "opc", "train_batch": "full"},
    "shared_opc_raw_bfull": {"shared_objective": "opc", "train_weights": "none", "train_batch": "full"},
    # docs/structured_scenario_shift_study.md §4-§5, §7: the matched rank-4 adapter (lr) and its user-gated form (lrg),
    # the ordinary and the calibration-aware likelihood, harmonic / raw DR OPC, harmonic OPC with the simulator's q
    **{f"shared_{fam}_{name}": {**spec, "shared_adapter": adapter}
       for fam, adapter in (("lr", "lowrank"), ("lrg", "lowrank_gated"))
       for name, spec in (("likelihood", {"shared_objective": "likelihood"}),
                          ("likelihood_calib", {"shared_objective": "likelihood", "shared_head_kind": "calib"}),
                          ("opc", {"shared_objective": "opc"}),
                          ("opc_raw", {"shared_objective": "opc", "train_weights": "none"}),
                          ("opc_oq", {"shared_objective": "opc", "train_reward": "oracle"}))},
}
SHARED_OBJECTIVE_METHODS = tuple(SHARED_OBJECTIVE_ARMS)
SHARED_DEFAULTS = {"lambdas": list(LAMBDA_GRID)}
SHARED_SEED_TAG = "shared"  # the paired search's seed label; another tag draws independent trials (a supplementary round)
ALL_STUDY_METHODS = VALID_STUDY_METHODS + BASELINE_METHODS + PRIOR_WORK_METHODS + SHARED_OBJECTIVE_METHODS
# Where the regression reward model's data come from: 'external' = a separate reg slice
# (--shared-regression-size, the same at every train size); 'train' = each train size's own
# training rows, so every arm uses only the n logged rows it is given.
REWARD_DATA_MODES = ("external", "train")
# What the results are for: development runs design the method (objective, weights, defaults);
# confirmatory runs evaluate the frozen method on fresh seeds and conditions.
RUN_STAGES = ("development", "confirmatory")
# How OPC's training loss is differentiated (docs/training_losses.md 3.4): 'log-trick' (the transformed
# weight g(w) as a detached coefficient on grad log pi: the exact gradient of DM + H(w)(r - q_hat) with
# H(w) = int_0^w g(t)/t dt) or 'direct' (pathwise through g(w): the exact gradient of the transformed
# estimate DM + g(w)(r - q_hat) itself). The two coincide for raw weights.
OPC_GRADIENTS = ("log-trick", "direct")
# Working development defaults since f5cade9 (2026-09-27) (docs/decision_record_opc_objective_weighting.md): OPC trains DR,
# differentiated directly, with Metelli et al.'s harmonic weights at lambda = 0.1; selection keeps clip:10. This is
# the working method for development runs, not the final paper choice. Standard alternatives: shrink:100 (Su et al.
# 2020, the prespecified smooth-weight comparison) and none (raw DR, the unregularized reference). The previous
# defaults are reproduced by --policy-losses sndr --sn-scope batch --opc-gradient log-trick --train-weights shrink:100.
STUDY_POLICY_LOSSES = ("dr",)
STUDY_OPC_GRADIENT = "direct"
STUDY_TRAIN_WEIGHTS = "harmonic:0.1"
# Study defaults (2026-09-26): the reward model shares the policy's budget (fit on each train size's
# own training rows, cross-fitted by user in 5 folds), and the validation split is fixed at 20,000
# logged rows (DR standard error ~0.5-0.6 CTR points for OPC's selected policy, vs ~1.1 at 5,000).
DEFAULT_REWARD_DATA = "train"
DEFAULT_CROSSFIT_FOLDS = 5
DEFAULT_VAL_SIZE = 20_000
METHODS_HELP = ("Which arms to run (default: opc no_propensity). Opt-in baselines: dm (policy trained and selected on "
                "q_hat alone) and tempered_logger (the logger's logits x s, s chosen by the DR selection score); "
                "prior-work baselines cause and blob. Arms run into an existing run tag add their rows next to the "
                "folder's other arms.")
SKIP_COMPLETED_HELP = ("Run only the work a condition folder does not hold yet (default: on): a requested arm is done when "
                       "every label it writes has a summary row for every requested train size made with the same "
                       "settings; only the other arms run. Rows of a requested arm made with other settings stop the "
                       "invocation before anything runs. --no-skip-completed reruns every requested arm and replaces "
                       "its rows; other arms' rows are always kept (training/run_state.py).")


def _study_budget_from_args(args) -> tuple[str, int]:
    """(reward_data, crossfit_folds) from the CLI: cross-fitting defaults to 5 folds with the
    train-mode reward model and to off with the external one."""
    reward_data = str(getattr(args, "reward_data", DEFAULT_REWARD_DATA))
    folds = getattr(args, "crossfit_folds", None)
    if folds is None:
        folds = DEFAULT_CROSSFIT_FOLDS if reward_data == "train" else 0
    return reward_data, int(folds)


def _normalize_study_methods(methods: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if methods is None:
        return VALID_STUDY_METHODS
    out = []
    for name in methods:
        key = str(name).lower()
        if key not in ALL_STUDY_METHODS:
            raise ValueError(f"Unknown method {name!r}; expected one of {ALL_STUDY_METHODS}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one method required")
    return tuple(out)


def _no_prop_policy_loss_types(policy_loss_types: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """No-propensity baseline: pure naive reward (no DM/SNDR/IW/KL/CRM)."""
    _ = policy_loss_types
    return ("naive",)


def _subset_rows(data: dict, mask) -> dict:
    """The logged rows of ``data`` (a ``get_train_data`` dict) where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    out = {k: (np.asarray(v)[mask] if k in ("x", "a", "r", "x_idx", "pscore") else v) for k, v in data.items()}
    out["num_data"] = int(mask.sum())
    return out


def _crossfit_bundles(dataset, train_data, user_fold, folds, **fit_kw) -> list:
    """One reward model per fold k, fit on the training rows whose user is not in fold k."""
    row_fold = np.asarray(user_fold)[np.asarray(train_data["x_idx"], dtype=np.int64)]
    bundles = []
    for k in range(int(folds)):
        rows = row_fold != k
        if not rows.any() or rows.all():
            raise ValueError(f"cross-fitting fold {k}: {int(rows.sum())} of {len(rows)} rows left to fit on")
        bundles.append(fit_shared_regression_bundle(dataset, _subset_rows(train_data, rows), materialize_qhat="never", **fit_kw))
    return bundles


def _summary_has_methods(summary_path: Path, methods, *, cause_options: dict | None = None,
                         blob_options: dict | None = None, train_sizes=None) -> bool:
    """True when ``summary_metrics.csv`` holds every label the requested arms write (training/run_state.py
    ``arm_labels``: the CausE family's predictions at each rho, each BLOB family and prior variant, as given by the
    options), for every train size in ``train_sizes`` (any size when None). The runners' skip-completed check also
    compares each row's configuration key (``run_state.condition_plan``)."""
    try:
        summary = read_summary(summary_path)
    except Exception:
        return False
    if summary is None or summary.empty or "method" not in summary.columns:
        return False
    have = summary["method"].astype(str)
    for m in methods:
        labels = arm_labels(m, cause_options, blob_options) if m in ALL_STUDY_METHODS else (str(m),)
        for label in labels:
            rows = summary[have == label]
            if rows.empty:
                return False
            if train_sizes is not None:
                sizes = set(pd.to_numeric(rows["train_size"], errors="coerce").dropna().astype(int))
                if not {int(n) for n in train_sizes} <= sizes:
                    return False
    return True


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
    runs = dedupe_runs(pd.read_csv(runs_path))  # a rerun's rows, not an interrupted attempt's
    if runs.empty:
        raise FileNotFoundError(f"Cached runs file is empty: {runs_path}")
    if "is_winning_run" in runs.columns:
        runs = runs[runs["is_winning_run"].astype(bool)]
    if "train_size" not in runs.columns:
        raise ValueError(f"Missing train_size in cached runs: {runs_path}")
    rows = {}
    for train_size, grp in runs.groupby("train_size"):
        row = grp.iloc[-1].to_dict()
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


from BPR.bpr_config import DEFAULT_DATASETS, bpr_artifact_status
from models.models import POLICY_TRANSFORMS, REWARD_FEATURES
from utils.importance_weights import parse_weight_spec, weight_spec_label
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
from utils.provenance import code_commit
from utils.seeding import (
    DEFAULT_CPU_THREADS,
    OPTUNA_SAMPLERS,
    derive_seed,
    enable_determinism,
    pin_cpu_threads,
    seed_everything,
)
from utils.simulation_utils import generate_dataset


def _resolve_val_size_configs(args):
    """Return list of (val_size_or_none, label) for directory naming; ``--val-size 0`` = the older
    fraction rule (label 'frac')."""
    if getattr(args, "val_sizes", None):
        return [(int(v), f"{int(v):g}") for v in args.val_sizes]
    if args.val_size is not None and int(args.val_size) > 0:
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
                       val_label: str = "frac", reward_data: str = "external", crossfit_folds: int = 0,
                       logging_uniform_mix: float = 0.0) -> str:
    """Folder name of one condition: dataset, bias, CTR, seed, the world options that differ from
    the defaults (``world_run_key_suffix``), the logger's uniform mix when on (another logger: its
    conditions never share a folder with the unmixed one's), the reward-model data when not external
    (and its cross-fitting folds), and the validation size when fixed."""
    key = f"dataset={dataset_name}__bias={bias}__ctr={ctr:g}__seed={seed}" + world_run_key_suffix(world_options)
    mix = float(np.clip(float(logging_uniform_mix or 0.0), 0.0, 1.0))
    if mix > 0.0:
        key = f"{key}__mix={mix:g}"
    if str(reward_data) != "external":
        key = f"{key}__qhat={reward_data}"
    if int(crossfit_folds or 0) > 0:
        key = f"{key}__cf={int(crossfit_folds)}"
    if val_label != "frac":
        key = f"{key}__val={val_label}"
    return key


def add_search_space_arguments(parser) -> None:
    """``--lr-range``, ``--epochs-range``, ``--lr-decay-range``, ``--weight-decay-range`` (both runners)."""
    d = DEFAULT_SEARCH_SPACE
    g = parser.add_argument_group("policy search space (shared by OPC, no-propensity and DM-only)")
    g.add_argument("--lr-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
                   help=f"Log-uniform learning-rate range (default {d['lr'][0]:g} {d['lr'][1]:g}).")
    g.add_argument("--epochs-range", nargs=2, type=int, metavar=("LOW", "HIGH"), default=None,
                   help=f"Epoch range, uniform over the integers (default {d['num_epochs'][0]} {d['num_epochs'][1]}).")
    g.add_argument("--lr-decay-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
                   help=f"Per-epoch learning-rate decay range (default {d['lr_decay'][0]:g} {d['lr_decay'][1]:g}).")
    g.add_argument("--weight-decay-range", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
                   help="AdamW weight decay toward the logger, log-uniform, drawn per trial from its own seeded "
                        "stream (needs --sampler random). Default: none (Adam).")


def add_shared_arguments(parser) -> None:
    """``--shared-lambdas`` (both runners): the source anchor's candidate strengths for the shared-objective arms;
    ``--shared-arm-space`` / ``--shared-arm-lambdas``: one arm's own range or grid, when the tuning protocol's edge
    rule extends it (docs/shared_objective_study.md §8; every other arm keeps the common space)."""
    g = parser.add_argument_group("shared-objective arms (opt-in: --methods " + " ".join(SHARED_OBJECTIVE_METHODS) + ")")
    g.add_argument("--shared-lambdas", type=float, nargs="+", default=list(LAMBDA_GRID),
                   help="Candidate strengths of the source anchor λ R(θ), searched per trial and the same for every "
                        "shared arm (default %(default)s; docs/shared_objective_study.md §2).")
    g.add_argument("--shared-arm-space", nargs="+", default=[], metavar="ARM:DIM=LO,HI",
                   help="One shared arm's own search range, DIM one of lr, num_epochs, lr_decay (e.g. "
                        "shared_opc:lr=1e-4,6.32e-3); its draws map the same random numbers onto its range.")
    g.add_argument("--shared-arm-lambdas", nargs="+", default=[], metavar="ARM=V1,V2,...",
                   help="One shared arm's own λ grid (e.g. shared_likelihood=0,0.001,0.01,0.1,1,10).")
    g.add_argument("--shared-seed-tag", default=SHARED_SEED_TAG,
                   help="Seed label of the paired search (default %(default)s). Another tag draws new, independent "
                        "configurations, seeds and batch orders, still paired across the shared arms: a supplementary "
                        "tuning round (docs/shared_objective_study.md §8).")


SHARED_SPACE_DIMENSIONS = ("lr", "num_epochs", "lr_decay")


def parse_shared_arm_options(spaces=(), lambdas=()) -> dict:
    """{"spaces": {arm: {dim: (lo, hi)}}, "arm_lambdas": {arm: [...]}} from the --shared-arm-* entries."""
    out = {"spaces": {}, "arm_lambdas": {}}
    for entry in spaces:
        try:
            arm, rest = str(entry).split(":", 1)
            dim, rng = rest.split("=", 1)
            lo, hi = (float(v) for v in rng.split(","))
        except ValueError:
            raise ValueError(f"--shared-arm-space entries are ARM:DIM=LO,HI, got {entry!r}") from None
        if arm not in SHARED_OBJECTIVE_ARMS or dim not in SHARED_SPACE_DIMENSIONS:
            raise ValueError(f"--shared-arm-space: unknown arm or dimension in {entry!r}")
        out["spaces"].setdefault(arm, {})[dim] = (int(lo), int(hi)) if dim == "num_epochs" else (lo, hi)
    for entry in lambdas:
        try:
            arm, values = str(entry).split("=", 1)
            grid = [float(v) for v in values.split(",")]
        except ValueError:
            raise ValueError(f"--shared-arm-lambdas entries are ARM=V1,V2,..., got {entry!r}") from None
        if arm not in SHARED_OBJECTIVE_ARMS:
            raise ValueError(f"--shared-arm-lambdas: unknown arm in {entry!r}")
        out["arm_lambdas"][arm] = grid
    return {k: v for k, v in out.items() if v}


def shared_options_from_args(args) -> dict:
    opts = {"lambdas": [float(x) for x in getattr(args, "shared_lambdas", LAMBDA_GRID)]}
    opts.update(parse_shared_arm_options(getattr(args, "shared_arm_space", ()), getattr(args, "shared_arm_lambdas", ())))
    tag = str(getattr(args, "shared_seed_tag", SHARED_SEED_TAG))
    if tag != SHARED_SEED_TAG:  # recorded only when it differs, so the default's configuration keys stay as they were
        opts["seed_tag"] = tag
    return opts


def shared_arm_settings(method: str, shared_options: dict | None, search_space: dict | None) -> tuple[dict, list]:
    """One shared arm's search space (the common one with the arm's own ranges on top) and λ grid."""
    opts = {**SHARED_DEFAULTS, **(shared_options or {})}
    space = {**(search_space or {}), **{k: tuple(v) for k, v in (opts.get("spaces") or {}).get(method, {}).items()}}
    lambdas = [float(x) for x in (opts.get("arm_lambdas") or {}).get(method, opts["lambdas"])]
    return space, lambdas


def search_space_from_args(args) -> dict | None:
    """The ranges given on the command line (None when every range is the default)."""
    space = {}
    for key, attr in (("lr", "lr_range"), ("num_epochs", "epochs_range"), ("lr_decay", "lr_decay_range"),
                      ("weight_decay", "weight_decay_range")):
        value = getattr(args, attr, None)
        if value is not None:
            space[key] = tuple(value)
    if not space:
        return None
    resolve_search_space(space)  # validate early
    return space


def build_condition_world(dataset_name: str, emb_dir: Path, bias: str, ctr: float, seed: int, *,
                          world_options: dict | None = None, logging_uniform_mix: float = 0.0):
    """The simulated world of one condition, exactly as ``_run_condition`` builds it (call after
    ``seed_everything(seed)``, as it does). Returns ``(dataset, params, levels, label)``; the oracle
    repair bound (``training.oracle_repair``) uses the same worlds."""
    world_options = dict(world_options or {})
    if world_options.get("world_family", "representation_bias") == "structured_shift":
        return _build_structured_condition_world(dataset_name, emb_dir, bias, ctr, seed, world_options=world_options,
                                                 logging_uniform_mix=logging_uniform_mix)
    levels = parse_bias(bias)
    label = bias_label(levels)
    user_path, item_path, user_meta_path, item_meta_path = _dataset_paths(emb_dir, dataset_name)
    emb_x = np.load(user_path)
    emb_a = np.load(item_path)
    item_bias = _load_item_bias(emb_dir, dataset_name, world_options)
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
    return dataset, params, levels, label


def _build_structured_condition_world(dataset_name: str, emb_dir: Path, label: str, ctr: float, seed: int, *,
                                      world_options: dict, logging_uniform_mix: float = 0.0):
    """``build_condition_world`` for the structured-shift family (docs/structured_scenario_shift_study.md §1-§3):
    the BPR vectors as source, the label s-<shift>.r-<response>[.gated]."""
    from utils.representation_bias import DEFAULT_LOGGER_GREEDY_SHARE, WorldConfig
    from utils.structured_shift import build_structured_world, parse_structured

    levels = parse_structured(label)
    label = str(label).strip()
    if float(world_options.get("pop_strength", 0.0)) > 0.0 or world_options.get("logger_pop_strength"):
        raise ValueError("the structured-shift worlds have no popularity term")
    if world_options.get("ctr_reference", "logger") != "logger" or world_options.get("group_source", "cluster") != "cluster":
        raise ValueError("the structured-shift worlds use the logger reference and the default groups")
    user_path, item_path, _, _ = _dataset_paths(emb_dir, dataset_name)
    d = WorldConfig()
    config = WorldConfig(centering=float(world_options.get("centering", d.centering)),
                         logging_spread=float(world_options.get("logging_spread", d.logging_spread)),
                         target_ctr=float(ctr), best_ctr=float(world_options.get("best_ctr", d.best_ctr)))
    params = {"bias": label, "ctr": float(ctr), "logging_uniform_mix": float(np.clip(logging_uniform_mix, 0.0, 1.0)),
              **world_options}
    dataset = build_structured_world(np.load(user_path), np.load(item_path), label, seed=int(seed), config=config,
                                     logging_uniform_mix=logging_uniform_mix,
                                     logger_greedy_share=world_options.get("logger_greedy_share",
                                                                           DEFAULT_LOGGER_GREEDY_SHARE))
    return dataset, params, levels, label


def _weight_spec_fields(role: str, label: str) -> dict:
    """``{role}_weight_mode`` and ``{role}_weight_param`` of a weight spec (the param is None for none / dm)."""
    mode, param = parse_weight_spec(label)
    return {f"{role}_weight_mode": mode, f"{role}_weight_param": None if mode in ("none", "dm") else float(param)}


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
    policy_loss_types: tuple[str, ...] = STUDY_POLICY_LOSSES,
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
    reward_features: str = "interaction",
    q_error: float = 0.0,
    q_bad_value: float | None = None,
    rand_ctr_meta: dict | None = None,
    world_options: dict | None = None,
    record_uniform_value: bool = False,
    dr_score_clip_m: float | None = None,
    train_weights: str | None = None,
    select_weights: str | None = None,
    log_select_weights=(),
    policy_transform: str = "linear",
    deterministic: bool = True,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    learn_logit_scale: bool = False,
    return_extra: bool = False,
    reward_data: str = "external",
    crossfit_folds: int = 0,
    post_temper: bool = False,
    sn_scope: str = "batch",
    sampler: str = "tpe",
    stage: str = "development",
    opc_gradient: str = STUDY_OPC_GRADIENT,
    search_space: dict | None = None,
    cause_options: dict | None = None,
    blob_options: dict | None = None,
    save_policies: bool = False,
    rebuild_logs: bool = True,
    shared_options: dict | None = None,
):
    """One condition. ``methods`` may add the opt-in baselines (``BASELINE_METHODS``); their
    summaries and trials come back as a 6th item ``{method: (summary_df, trials_df)}`` when
    ``return_extra`` (the 5-item return is unchanged otherwise). ``learn_logit_scale``: every
    trained policy (OPC, no-prop, DM) also learns a logit scale. ``reward_data``: ``external``
    (default: q_hat from the separate reg slice) or ``train`` (q_hat fit per train size on that
    size's training rows, shared by every arm; the splits are the same in both modes).
    ``crossfit_folds`` K >= 2 (train mode only): users are split into K folds; the training losses
    take each user's q_hat from the model fit on the other folds' rows. ``post_temper``: every
    trained policy (OPC, no-prop, DM) gets its sharpness chosen after training on validation.
    ``sn_scope``: normalizer of the sndr / kl training correction (per minibatch or full-data).
    ``sampler``: Optuna's ``tpe`` (default) or ``random`` (seeded random search without warm starts:
    the same trial configurations and seeds in every run of the grid, the paired comparison of
    objectives). ``stage``: ``development`` (default) or ``confirmatory``, recorded with the results.
    ``opc_gradient``: ``direct`` (default) or ``log-trick``, how OPC's loss is differentiated
    (``OPC_GRADIENTS``); the other arms are fixed (no-propensity and DM direct, tempered untrained).
    Defaults are the working development method (``STUDY_POLICY_LOSSES``, ``STUDY_OPC_GRADIENT``,
    ``STUDY_TRAIN_WEIGHTS``); ``train_weights=None`` means ``STUDY_TRAIN_WEIGHTS``. ``rebuild_logs``: rebuild
    ``trials_long.csv`` / ``runs_long.csv`` from the arms' logs here (``execute_condition`` does it itself, after the
    summary, so an interrupted arm never shows up in them)."""
    if str(stage) not in RUN_STAGES:
        raise ValueError(f"stage must be one of {RUN_STAGES}, got {stage!r}")
    if str(opc_gradient) not in OPC_GRADIENTS:
        raise ValueError(f"opc_gradient must be one of {OPC_GRADIENTS}, got {opc_gradient!r}")
    # random search pairs the arms that share OPC's search space (OPC, no-propensity, DM-only): the same
    # configurations and trial seeds; TPE keeps each arm's own stream (unchanged)
    shared_seed_label = "opc" if str(sampler) == "random" else None
    opc_log_trick = str(opc_gradient) == "log-trick"
    train_weights = STUDY_TRAIN_WEIGHTS if train_weights is None else train_weights  # explicit for every trainer
    train_mode = parse_weight_spec(train_weights)[0]
    if train_mode == "harmonic" and opc_log_trick and "opc" in _normalize_study_methods(methods):
        raise ValueError("harmonic training weights are optimized by their direct gradient (Metelli et al. 2021): "
                         "use opc_gradient='direct' (--opc-gradient direct)")
    if str(sn_scope) == "exact" and opc_log_trick and "opc" in _normalize_study_methods(methods):
        raise ValueError("sn_scope='exact' (the SNDR ratio's gradient) needs opc_gradient='direct'")
    reward_data = str(reward_data).lower()
    if reward_data not in REWARD_DATA_MODES:
        raise ValueError(f"reward_data must be one of {REWARD_DATA_MODES}, got {reward_data!r}")
    if reward_data == "train" and str(reward_model).lower() != "regression":
        raise ValueError("reward_data='train' fits the regression reward model; other reward models use no data")
    crossfit_folds = int(crossfit_folds or 0)
    if crossfit_folds and (crossfit_folds < 2 or reward_data != "train"):
        raise ValueError("crossfit_folds needs K >= 2 and reward_data='train' (the model is fit on the training rows)")
    methods = _normalize_study_methods(methods)
    run_opc = "opc" in methods
    run_no_prop = "no_propensity" in methods
    noprop_policy_loss_types = _no_prop_policy_loss_types(policy_loss_types)
    n_methods = int(run_opc) + int(run_no_prop) + int("dm" in methods)
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
    world_options = dict(world_options or {})
    train_label = weight_spec_label(train_weights)
    select_label = weight_spec_label(
        ("clip", dr_score_clip_m) if dr_score_clip_m is not None
        else (DEFAULT_SELECT_WEIGHTS if select_weights is None else select_weights)
    )
    bpr_status = bpr_artifact_status(emb_dir, dataset_name)
    if bpr_status["status"] != "ok":
        print(f"WARNING {dataset_name}: {bpr_status['message']}", flush=True)
    dataset, params, levels, label = build_condition_world(
        dataset_name, emb_dir, bias, ctr, seed, world_options=world_options, logging_uniform_mix=logging_uniform_mix
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
    # CausE uses no reward model: a CausE-only run skips the q_hat fits (the splits are built the same way)
    needs_qhat = bool(set(methods) - set(PRIOR_WORK_METHODS))
    shared_regression_bundle = fit_shared_regression_bundle(
        dataset,
        first_split["reg_data"],
        reward_model=str(reward_model),
        reward_features=str(reward_features),
        user_chunk=int(qhat_user_chunk),
        action_chunk=int(qhat_action_chunk),
        q_error=float(q_error),
        q_bad_value=q_bad_value,
    ) if needs_qhat else {}
    size_bundles = None
    if reward_data == "train" and needs_qhat:  # each size's own training rows fit its q_hat; every arm shares it
        size_bundles = {
            int(n): fit_shared_regression_bundle(
                dataset,
                split_cache[(int(n), LOGGED_RUN_IDX)]["train_data"],
                reward_model="regression",
                reward_features=str(reward_features),
                user_chunk=int(qhat_user_chunk),
                action_chunk=int(qhat_action_chunk),
                q_error=float(q_error),
                q_bad_value=q_bad_value,
            )
            for n in train_sizes
        }
    size_crossfit = None
    if crossfit_folds and needs_qhat:
        user_fold = np.random.default_rng(derive_seed(seed, "crossfit", "user_fold")).integers(
            0, crossfit_folds, int(dataset["n_users"])
        )
        fit_kw = dict(reward_model="regression", reward_features=str(reward_features), user_chunk=int(qhat_user_chunk),
                      action_chunk=int(qhat_action_chunk), q_error=float(q_error), q_bad_value=q_bad_value)
        size_crossfit = {
            int(n): (_crossfit_bundles(dataset, split_cache[(int(n), LOGGED_RUN_IDX)]["train_data"], user_fold,
                                       crossfit_folds, **fit_kw), user_fold)
            for n in train_sizes
        }

    # Per-arm trial and run logs, appended one train size at a time. Each arm first drops its own rows for the sizes it
    # is about to run (an earlier or interrupted attempt's), so a rerun never repeats them (training/run_state.py).
    opc_log_paths = arm_log_paths(run_dir, "opc")
    noprop_log_paths = arm_log_paths(run_dir, "no_propensity")

    if run_opc:
        reset_arm_logs(opc_log_paths, "opc", train_sizes)
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
            use_log_trick_fixed=opc_log_trick,
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
            train_weights=train_weights,
            select_weights=select_weights,
            log_select_weights=tuple(log_select_weights or ()),
            policy_transform=policy_transform,
            learn_logit_scale=bool(learn_logit_scale),
            size_regression_bundles=size_bundles,
            size_crossfit=size_crossfit,
            post_temper=bool(post_temper),
            sn_scope=str(sn_scope),
            sampler=str(sampler),
            seed_label=shared_seed_label,
            search_space=search_space,
            policy_dir=str(run_dir) if save_policies else None,
        )
    else:
        try:
            opc_df = _load_cached_method_df(run_dir, "opc")
            opc_trials = _load_cached_method_trials(run_dir, "opc")
        except FileNotFoundError:
            opc_df = pd.DataFrame()
            opc_trials = pd.DataFrame()

    if run_no_prop:
        reset_arm_logs(noprop_log_paths, "no_propensity", train_sizes)
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
            select_weights=select_weights,
            policy_transform=policy_transform,
            learn_logit_scale=bool(learn_logit_scale),
            size_regression_bundles=size_bundles,
            log_select_weights=tuple(log_select_weights or ()),
            size_crossfit=size_crossfit,
            post_temper=bool(post_temper),
            sampler=str(sampler),
            seed_label=shared_seed_label,
            search_space=search_space,
        )
    else:
        try:
            noprop_df = _load_cached_method_df(run_dir, "no_propensity")
            noprop_trials = _load_cached_method_trials(run_dir, "no_propensity")
        except FileNotFoundError:
            noprop_df = pd.DataFrame()
            noprop_trials = pd.DataFrame()

    # Opt-in baselines: the same splits, reward model, selection weights and search budget.
    extra = {}
    extra_log_paths = {}
    # (the loop variable is not `label`: that name holds the bias label written to run_meta.json below)
    for method in (m for m in BASELINE_METHODS if m in methods):
        extra_log_paths[method] = arm_log_paths(run_dir, method)
        reset_arm_logs(extra_log_paths[method], method, train_sizes)
        arm = {"dm": dict(policy_loss_types=("dm",), select_estimator="dm", learn_logit_scale=bool(learn_logit_scale),
                          post_temper=bool(post_temper)),
               "tempered_logger": dict(policy_loss_types=("sndr",), temper_only=True)}[method]
        extra[method] = regression_trainer_trial(
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
            log_paths=extra_log_paths[method],
            slim=slim,
            method_label=method,
            policy_reward_mode=policy_reward_mode,
            policy_reward_mc_sim=policy_reward_mc_sim,
            split_cache=split_cache,
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
            dr_score_clip_m=dr_score_clip_m,
            seed=int(seed),
            train_weights=train_weights,
            select_weights=select_weights,
            log_select_weights=tuple(log_select_weights or ()),
            policy_transform=policy_transform,
            size_regression_bundles=size_bundles,
            size_crossfit=size_crossfit,
            sampler=str(sampler),
            seed_label=shared_seed_label if method == "dm" else None,  # the tempered logger searches its own space
            search_space=search_space,
            policy_dir=str(run_dir) if save_policies else None,
            **arm,
        )

    # The shared-objective arms: the same splits, reward model and selection weights, one paired search over the study's
    # space plus the source anchor's strength (docs/shared_objective_study.md §3-§5)
    shared_meta = None
    if post_temper and set(methods) & set(SHARED_OBJECTIVE_METHODS):
        raise ValueError("the shared-objective arms are not post-tempered (docs/shared_objective_study.md §3)")
    for method in (m for m in SHARED_OBJECTIVE_METHODS if m in methods):
        shared_meta = {**SHARED_DEFAULTS, **(shared_options or {})}
        extra_log_paths[method] = arm_log_paths(run_dir, method)
        reset_arm_logs(extra_log_paths[method], method, train_sizes)
        arm = dict(SHARED_OBJECTIVE_ARMS[method])
        arm.pop("train_weights", None)  # passed as train_weights below: an arm's own transform replaces the study's
        likelihood_arm = arm["shared_objective"] != "opc"
        arm_space, arm_lambdas = shared_arm_settings(method, shared_options, search_space)
        extra[method] = regression_trainer_trial(
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
            log_paths=extra_log_paths[method],
            slim=slim,
            method_label=method,
            policy_reward_mode=policy_reward_mode,
            policy_reward_mc_sim=policy_reward_mc_sim,
            split_cache=split_cache,
            policy_loss_types=tuple(policy_loss_types),
            dataset_name=dataset_name,
            search_use_log_trick=False,
            use_log_trick_fixed=False if likelihood_arm else opc_log_trick,
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
            train_weights=SHARED_OBJECTIVE_ARMS[method].get("train_weights", train_weights),
            select_weights=select_weights,
            log_select_weights=tuple(log_select_weights or ()),
            policy_transform="linear",
            learn_logit_scale=True,  # the shared model always learns its scale (OPC's study configuration)
            size_regression_bundles=size_bundles,
            size_crossfit=size_crossfit,
            sn_scope=str(sn_scope),
            sampler=str(sampler),
            # trial k: the same configuration, seed and batch order in all four arms
            seed_label=str((shared_options or {}).get("seed_tag", SHARED_SEED_TAG)),
            search_space=arm_space or None,
            policy_dir=str(run_dir) if save_policies else None,
            anchor_lambdas=tuple(arm_lambdas),
            **arm,
        )

    cause_meta = None
    if "cause" in methods:  # prior-work baseline on the fixed-budget warm/uniform data (training/cause_trials.py)
        from training.cause_trials import CAUSE_DEFAULTS, cause_trainer_trial

        our_x, our_a = dataset["our_x"], dataset["our_a"]
        cause_constants = {"initial_reward": _dataset_log_constants(dataset, our_x, our_a)["initial_reward"],
                           "logger_greedy": _policy_greedy_reward_from_embeddings(dataset, our_x, our_a)}
        cause_meta = {**CAUSE_DEFAULTS, **(cause_options or {})}
        cause_meta["n_trials"] = int(cause_meta.get("n_trials") or n_trials)
        if save_policies:  # the selected policies, for the cross-arm pick diagnostics (training/policy_diagnostics.py)
            cause_options = {**(cause_options or {}), "policy_dir": str(run_dir)}
        extra.update(cause_trainer_trial(
            train_sizes=train_sizes, dataset=dataset, split_cache=split_cache, condition_seed=int(seed), seed=int(seed),
            n_trials=int(n_trials), log_constants=cause_constants, options=cause_options, sampler=str(sampler),
            stage=str(stage), run_idx=LOGGED_RUN_IDX,
            device=torch.device("cpu") if cause_meta.get("device") == "cpu" else _training_device(require_cuda=require_cuda),
        ))

    blob_meta = None
    if "blob" in methods:  # BLOB-supplied-source on the same N warm rows (training/blob_trials.py)
        from training.blob_trials import BLOB_DEFAULTS, blob_trainer_trial

        our_x, our_a = dataset["our_x"], dataset["our_a"]
        blob_constants = {"initial_reward": _dataset_log_constants(dataset, our_x, our_a)["initial_reward"],
                          "logger_greedy": _policy_greedy_reward_from_embeddings(dataset, our_x, our_a)}
        blob_meta = {**BLOB_DEFAULTS, **(blob_options or {})}
        blob_meta["n_trials"] = int(blob_meta.get("n_trials") or n_trials)
        if save_policies:
            blob_options = {**(blob_options or {}), "policy_dir": str(run_dir)}
        extra.update(blob_trainer_trial(
            train_sizes=train_sizes, dataset=dataset, split_cache=split_cache, seed=int(seed), n_trials=int(n_trials),
            log_constants=blob_constants, options=blob_options, sampler=str(sampler), stage=str(stage),
            run_idx=LOGGED_RUN_IDX,
            device=torch.device("cpu") if blob_meta.get("device") == "cpu" else _training_device(require_cuda=require_cuda),
        ))

    # Unified long logs for post-hoc analysis: every arm's logs in the folder (also arms run by earlier invocations),
    # one row per trial / run.
    if rebuild_logs:
        rebuild_long_logs(run_dir)

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
        "opc_use_log_trick_fixed": opc_log_trick,
        "opc_gradient": str(opc_gradient),
        "no_prop_use_log_trick_fixed": False,
        "study_methods": list(methods),
        "opc_policy_loss_types": list(policy_loss_types),
        "no_prop_policy_loss_types": list(noprop_policy_loss_types),
        "train_weights": train_label,
        "select_weights": select_label,
        **_weight_spec_fields("train", train_label),
        **_weight_spec_fields("select", select_label),
        "log_select_weights": [weight_spec_label(w) for w in (log_select_weights or ())],
        "policy_transform": str(policy_transform),
        "learn_logit_scale": bool(learn_logit_scale),
        "reward_data": reward_data,
        "crossfit_folds": crossfit_folds,
        "post_temper": bool(post_temper),
        "sn_scope": str(sn_scope),
        "sampler": str(sampler),
        "paired_arms": shared_seed_label is not None,
        "stage": str(stage),
        "code_commit": code_commit(),
        "search_space": {k: (list(v) if v is not None else None) for k, v in resolve_search_space(search_space).items()},
        "dr_score_clip_m": parse_weight_spec(select_label)[1] if select_label.startswith("clip") else None,
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
        "reward_features": shared_regression_bundle.get("reward_features"),
        "q_error": float(shared_regression_bundle.get("q_error", q_error)),
        "q_bad_value": shared_regression_bundle.get("q_bad_value", q_bad_value),
        "rand_ctr": rand_ctr_meta or {},
        "cause": cause_meta,
        "blob": blob_meta,
        "shared": shared_meta,
    }
    if return_extra:
        return opc_df, noprop_df, opc_trials, noprop_trials, meta, extra
    return opc_df, noprop_df, opc_trials, noprop_trials, meta


def _finalize_summary_df(opc_df, noprop_df, meta: dict, *, extra: dict | None = None, **tags) -> pd.DataFrame:
    """One row per (method, train size); ``extra``: ``{method: (summary_df, trials_df)}`` baselines."""
    frames = []
    arms = [("opc", opc_df), ("no_propensity", noprop_df)] + [(k, v[0]) for k, v in (extra or {}).items()]
    for label, df in arms:
        if df is not None and not getattr(df, "empty", True):
            part = df.reset_index().rename(columns={"index": "train_size"})
            part["method"] = label
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
        summary_df["logger_greedy_share"] = float(world.get("logger_greedy_share", 0.0))
        summary_df["logger_sharpness"] = float(world.get("logger_sharpness", 1.0))
        summary_df["logger_greedy_ctr"] = world.get("logger_greedy_ctr")
    summary_df["reward_features"] = meta.get("reward_features")  # None unless reward_model=regression
    for k in ("train_weights", "select_weights", "policy_transform", "learn_logit_scale"):
        summary_df[k] = meta.get(k)
    for role in ("train", "select"):
        fields = _weight_spec_fields(role, meta[f"{role}_weights"]) if meta.get(f"{role}_weights") else {}
        for k, v in fields.items():
            summary_df[k] = v
    summary_df["reward_data"] = meta.get("reward_data", "external")
    summary_df["crossfit_folds"] = int(meta.get("crossfit_folds", 0) or 0)
    summary_df["post_temper"] = bool(meta.get("post_temper", False))
    summary_df["policy_loss_types"] = "+".join(meta.get("policy_loss_types", []) or [])
    summary_df["sn_scope"] = meta.get("sn_scope", "batch")
    summary_df["sampler"] = meta.get("sampler", "tpe")
    summary_df["stage"] = meta.get("stage", "development")
    summary_df["opc_gradient"] = meta.get("opc_gradient", "log-trick")
    if "val_size" in summary_df.columns:
        summary_df["val_size_config"] = summary_df["val_size"]
    if {"opc", "no_propensity"}.issubset(set(summary_df.get("method", pd.Series(dtype=str)))):
        return add_paired_method_pct_columns(summary_df)
    return summary_df


# ------------------------------------------------------------------------------- one invocation, both runners
def condition_configs(args, out_dir: Path, *, methods, bias_configs, world_options, val_size_configs,
                      policy_loss_types, search_use_log_trick) -> list[dict]:
    """One config per condition, in run order seed → dataset → CTR → [validation size] → bias: everything
    ``execute_condition`` needs. ``study_methods`` is the requested arms until ``plan_conditions`` narrows it to the
    arms a condition still needs."""
    multi_val = len(val_size_configs) > 1
    common = {
        "emb_dir": str(args.emb_dir),
        "train_sizes": [int(n) for n in args.train_sizes],
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
        "search_use_log_trick": bool(search_use_log_trick),
        "shared_regression_size": int(args.shared_regression_size),
        "qhat_user_chunk": int(args.qhat_user_chunk),
        "qhat_action_chunk": int(args.qhat_action_chunk),
        "require_cuda": bool(args.require_cuda),
        "logging_uniform_mix": float(args.logging_uniform_mix),
        "optuna_selection": str(args.optuna_selection),
        "reward_model": str(args.reward_model),
        "reward_features": str(args.reward_features),
        "train_weights": args.train_weights,
        "select_weights": args.select_weights,
        "log_select_weights": list(args.log_select_weights),
        "policy_transform": args.policy_transform,
        "learn_logit_scale": bool(args.learn_logit_scale),
        "reward_data": args.reward_data,
        "crossfit_folds": int(args.crossfit_folds),
        "post_temper": bool(args.post_temper),
        "sn_scope": str(args.sn_scope),
        "sampler": str(args.sampler),
        "stage": str(args.stage),
        "opc_gradient": str(args.opc_gradient),
        "search_space": search_space_from_args(args),
        "cause_options": cause_options_from_args(args),
        "blob_options": blob_options_from_args(args),
        "shared_options": shared_options_from_args(args),
        "save_policies": bool(args.save_policies),
        "skip_completed": bool(args.skip_completed),
    }
    configs = []
    for seed in args.seeds:
        for dataset_name in args.datasets:
            for ctr in args.ctr_levels:
                for val_size_cfg, val_label in val_size_configs:
                    val_root = out_dir / f"val_{val_label}" if multi_val else out_dir
                    for bias in bias_configs:
                        run_key = _condition_run_key(dataset_name, bias, ctr, seed, world_options, val_label,
                                                     reward_data=args.reward_data,
                                                     crossfit_folds=int(args.crossfit_folds or 0),
                                                     logging_uniform_mix=float(args.logging_uniform_mix))
                        configs.append({"dataset_name": dataset_name, "bias": bias, "ctr": float(ctr),
                                        "seed": int(seed), "run_key": run_key, "run_dir": str(val_root / run_key),
                                        "val_size": val_size_cfg, **common})
    return configs


def plan_conditions(configs: list[dict], methods, *, skip_completed: bool) -> tuple[list[dict], list[str]]:
    """The configs to run, each with ``study_methods`` set to the arms its folder still needs (all requested arms
    with ``skip_completed`` off), and the run keys skipped as complete (training/run_state.py). A requested label and
    train size whose row was made with other settings stops the invocation before anything runs, listing the
    conflicts, unless ``skip_completed`` is off: then that arm reruns and its rows are replaced."""
    todo, skipped, conflicts = [], [], []
    for cfg in configs:
        plan = condition_plan(cfg, methods, skip_completed=skip_completed)
        if plan["conflicts"]:
            if skip_completed:
                conflicts += plan["detail"]
                continue
            for line in plan["detail"]:
                print(f"Replacing rows made with other settings (--no-skip-completed): {line}", flush=True)
        if not plan["pending"]:
            print(f"Skipping completed: {cfg['run_key']} ({', '.join(methods)})", flush=True)
            skipped.append(cfg["run_key"])
            continue
        if plan["complete"] or plan["options"]:
            print(f"Partly complete: {cfg['run_key']}: running {', '.join(plan['pending'])}"
                  + "".join(f"; {k} {v}" for k, v in plan["options"].items())
                  + (f"; already done: {', '.join(plan['complete'])}" if plan["complete"] else ""), flush=True)
        todo.append({**cfg, **plan["options"], "requested_methods": list(methods),
                     "study_methods": list(plan["pending"])})
    if conflicts:
        raise SystemExit("These condition folders already hold rows of the requested arms made with other settings:\n  "
                         + "\n  ".join(conflicts)
                         + "\nNothing was run. Use a new --run-tag, or --no-skip-completed to rerun the requested arms "
                           "and replace their rows.")
    return todo, skipped


def _condition_kwargs(cfg: dict) -> dict:
    """``_run_condition``'s keyword arguments from a condition config."""
    return dict(
        dataset_name=cfg["dataset_name"],
        emb_dir=Path(cfg["emb_dir"]),
        bias=cfg["bias"],
        ctr=cfg["ctr"],
        seed=cfg["seed"],
        train_sizes=cfg["train_sizes"],
        n_trials=cfg["n_trials"],
        batch_size=cfg["batch_size"],
        val_size=cfg["val_size"],
        val_frac=cfg["val_frac"],
        val_min=cfg["val_min"],
        val_max=cfg["val_max"],
        policy_reward_mode=cfg["policy_reward_mode"],
        policy_reward_mc_sim=cfg["policy_reward_mc_sim"],
        slim=bool(cfg.get("slim", False)),
        deterministic=bool(cfg.get("deterministic", True)),
        cpu_threads=int(cfg.get("cpu_threads", DEFAULT_CPU_THREADS)),
        policy_loss_types=tuple(cfg["policy_loss_types"]),
        search_use_log_trick=bool(cfg.get("search_use_log_trick", True)),
        shared_regression_size=int(cfg.get("shared_regression_size", 50_000)),
        qhat_user_chunk=int(cfg.get("qhat_user_chunk", DEFAULT_QHAT_USER_CHUNK)),
        qhat_action_chunk=int(cfg.get("qhat_action_chunk", DEFAULT_QHAT_ACTION_CHUNK)),
        require_cuda=bool(cfg.get("require_cuda", False)),
        optuna_batch_sizes=cfg.get("optuna_batch_sizes"),
        logging_uniform_mix=float(cfg.get("logging_uniform_mix", 0.0)),
        optuna_selection=str(cfg.get("optuna_selection", "ci_low")),
        reward_model=str(cfg.get("reward_model", "regression")),
        reward_features=str(cfg.get("reward_features", "interaction")),
        train_weights=cfg.get("train_weights"),
        select_weights=cfg.get("select_weights"),
        log_select_weights=cfg.get("log_select_weights") or (),
        policy_transform=str(cfg.get("policy_transform", "linear")),
        world_options=cfg.get("world_options"),
        learn_logit_scale=bool(cfg.get("learn_logit_scale", False)),
        reward_data=str(cfg.get("reward_data", "external")),
        crossfit_folds=int(cfg.get("crossfit_folds", 0) or 0),
        post_temper=bool(cfg.get("post_temper", False)),
        sn_scope=str(cfg.get("sn_scope", "batch")),
        sampler=str(cfg.get("sampler", "tpe")),
        stage=str(cfg.get("stage", "development")),
        opc_gradient=str(cfg.get("opc_gradient", STUDY_OPC_GRADIENT)),
        search_space=cfg.get("search_space"),
        cause_options=cfg.get("cause_options"),
        blob_options=cfg.get("blob_options"),
        save_policies=bool(cfg.get("save_policies", False)),
        shared_options=cfg.get("shared_options"),
    )


def execute_condition(cfg: dict, *, run_condition=None) -> dict:
    """Run a condition config's arms (``study_methods``) and merge their rows into its folder (training/run_state.py):
    ``summary_metrics.csv`` and the per-label ``<label>_trials.csv`` keep every row this run does not produce,
    ``trials_long.csv`` / ``runs_long.csv`` are rebuilt from every arm's logs, and ``run_meta.json`` records each
    label's configuration. Holds the folder's lock; with ``skip_completed`` it re-checks which arms are still needed
    (another invocation may have finished some meanwhile). ``run_condition`` defaults to ``_run_condition``."""
    run_condition = run_condition or _run_condition
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    methods = list(cfg["study_methods"])
    with condition_lock(run_dir):
        if cfg.get("skip_completed"):
            plan = condition_plan(cfg, methods, skip_completed=True)
            if plan["conflicts"]:
                raise RuntimeError("rows made with other settings appeared meanwhile: " + "; ".join(plan["detail"]))
            methods = plan["pending"]
            if not methods:
                print(f"Completed meanwhile by another invocation: {cfg['run_key']}", flush=True)
                return {"run_key": cfg["run_key"], "ran": []}
            cfg = {**cfg, **plan["options"]}
        opc_df, noprop_df, opc_trials, noprop_trials, meta, extra = run_condition(
            **_condition_kwargs(cfg), run_dir=run_dir, methods=tuple(methods), return_extra=True, rebuild_logs=False)
        ran = set(methods)
        summary_new = _finalize_summary_df(opc_df if "opc" in ran else None,
                                           noprop_df if "no_propensity" in ran else None,
                                           meta, extra=extra, dataset=cfg["dataset_name"], seed=cfg["seed"])
        label_arm = {label: m for m in methods
                     for label in arm_labels(m, cfg.get("cause_options"), cfg.get("blob_options"))}
        produced = set(summary_new["method"].astype(str)) if "method" in summary_new.columns else set()
        if produced != set(label_arm):
            raise RuntimeError(f"{cfg['run_key']}: the arms wrote {sorted(produced)}, expected {sorted(label_arm)}")
        configs = {m: arm_config(m, cfg) for m in methods}
        keys = {m: config_key(c) for m, c in configs.items()}
        summary_new = summary_new.assign(**{CONFIG_KEY_COLUMN: [keys[label_arm[str(x)]] for x in summary_new["method"]]})
        summary = merge_summary(read_summary(run_dir / "summary_metrics.csv"), summary_new)
        atomic_write_csv(summary, run_dir / "summary_metrics.csv")
        if "opc" in ran:  # the last train size's trials, as before (trials_long.csv holds every size)
            atomic_write_csv(opc_trials, run_dir / "opc_trials.csv")
        if "no_propensity" in ran:
            atomic_write_csv(noprop_trials, run_dir / "no_prop_trials.csv")
        for label, (_, trials) in extra.items():
            path = run_dir / f"{label}_trials.csv"
            atomic_write_csv(replace_rows(read_csv_or_none(path), trials, ROW_KEY), path)
        rebuild_long_logs(run_dir, summary)
        commit, now = code_commit(), utc_now()
        labels = {label: {"arm": m, "config_key": keys[m], "code_commit": commit.get("commit"),
                          "code_dirty": commit.get("dirty"), "written_at": now} for label, m in label_arm.items()}
        atomic_write_json(merge_run_meta(_read_json(run_dir / "run_meta.json"), meta, labels,
                                         {keys[m]: configs[m] for m in methods}), run_dir / "run_meta.json")
    return {"run_key": cfg["run_key"], "ran": methods}


def study_manifest(args, *, runner: str, run_tag: str, methods, bias_configs, world_options, policy_loss_types,
                   val_size_configs, extra: dict | None = None) -> dict:
    """``run_manifest.json`` of an invocation (both runners): every setting that shapes the results, the arms, and the
    CausE / BLOB options when those arms were requested."""
    return {
        "run_tag": run_tag,
        "runner": runner,
        "created_at": utc_now(),
        "datasets": args.datasets,
        "bias_configs": bias_configs,
        "world_options": world_options,
        "ctr_levels": args.ctr_levels,
        "seeds": args.seeds,
        "train_sizes": args.train_sizes,
        "n_trials": int(args.n_trials),
        "batch_size": args.batch_size,
        "val_size_fixed": args.val_size,
        "val_sizes": args.val_sizes,
        "val_frac": args.val_frac,
        "val_min": args.val_min,
        "val_max": args.val_max,
        "val_size_configs": [{"val_size": v, "label": lbl} for v, lbl in val_size_configs],
        "policy_reward_mode": args.policy_reward_mode,
        "policy_reward_mc_sim": args.policy_reward_mc_sim,
        "optuna_batch_sizes": args.optuna_batch_sizes,
        "study_methods": list(methods),
        "policy_loss_types": list(policy_loss_types),
        "no_prop_policy_loss_types": list(_no_prop_policy_loss_types(policy_loss_types)),
        "no_log_trick": bool(args.no_log_trick),
        "shared_regression_size": int(args.shared_regression_size),
        "qhat_user_chunk": int(args.qhat_user_chunk),
        "qhat_action_chunk": int(args.qhat_action_chunk),
        "require_cuda": bool(args.require_cuda),
        "logging_uniform_mix": float(args.logging_uniform_mix),
        "optuna_selection": str(args.optuna_selection),
        "reward_model": str(args.reward_model),
        "reward_features": str(args.reward_features),
        "train_weights": args.train_weights,
        "select_weights": args.select_weights,
        "log_select_weights": list(args.log_select_weights),
        "policy_transform": args.policy_transform,
        "learn_logit_scale": bool(args.learn_logit_scale),
        "reward_data": args.reward_data,
        "crossfit_folds": int(args.crossfit_folds),
        "post_temper": bool(args.post_temper),
        "sn_scope": str(args.sn_scope),
        "sampler": str(args.sampler),
        "stage": str(args.stage),
        "opc_gradient": str(args.opc_gradient),
        "search_space": {k: (list(v) if v is not None else None)
                         for k, v in resolve_search_space(search_space_from_args(args)).items()},
        "cause_options": cause_options_from_args(args) if "cause" in methods else None,
        "blob_options": blob_options_from_args(args) if "blob" in methods else None,
        "shared_options": (shared_options_from_args(args) if set(methods) & set(SHARED_OBJECTIVE_METHODS) else None),
        "save_policies": bool(args.save_policies),
        "slim": bool(args.slim),
        "deterministic": bool(args.deterministic),
        "cpu_threads": int(args.cpu_threads),
        "skip_completed": bool(args.skip_completed),
        "code_commit": code_commit(),
        **(extra or {}),
    }


def finish_invocation(out_dir: Path, manifest: dict, *, started: str, ran: list, skipped: list,
                      failures: list) -> None:
    """After an invocation: ``all_summary_metrics.csv`` from every condition's summary, ``run_manifest.json`` (this
    invocation), ``failures.csv`` (this invocation's failures; removed when there are none) and one line in
    ``run_invocations.jsonl`` (the history of the run tag)."""
    out_dir = Path(out_dir)
    collected = _collect_existing_summaries(out_dir)
    if collected:
        atomic_write_csv(pd.concat(collected, ignore_index=True), out_dir / "all_summary_metrics.csv")
        atomic_write_json(manifest, out_dir / "run_manifest.json")
    failures_path = out_dir / "failures.csv"
    if failures:
        atomic_write_csv(pd.DataFrame(failures), failures_path)
    elif failures_path.exists():
        failures_path.unlink()
    record_invocation(out_dir, {"started_at": started, "finished_at": utc_now(), "argv": list(sys.argv),
                                "ran": ran, "skipped": skipped, "failures": failures, "manifest": manifest})


def main():
    parser = argparse.ArgumentParser(
        description="Run full OPC vs no-propensity sweeps and export structured outputs."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Default: " + " ".join(DEFAULT_DATASETS) + ".")
    add_world_arguments(parser)
    add_search_space_arguments(parser)
    add_cause_arguments(parser)
    add_blob_arguments(parser)
    add_shared_arguments(parser)
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
        "--train-weights",
        type=weight_spec_label,
        default=STUDY_TRAIN_WEIGHTS,
        help="Importance-weight transform in the OPC training losses sndr / dr / ipw / kl: none, clip:M, "
        "shrink:lambda (Su et al. 2020) or harmonic:lambda (Metelli et al. 2021, w / (1 - lambda + lambda w), "
        "lambda in [0, 1], at most 1 / lambda; needs --opc-gradient direct). Default %(default)s: the working "
        "development default, not the final paper choice; shrink:100 is the standard smooth-weight comparison "
        "and none (raw DR) the unregularized reference (crm / kl_crm keep their own searched clip). Recorded "
        "as train_weights, train_weight_mode and train_weight_param.",
    )
    parser.add_argument(
        "--select-weights",
        type=weight_spec_label,
        default=DEFAULT_SELECT_WEIGHTS,
        help="Importance-weight transform of the DR selection score and the post-hoc DR / SNIPW / "
        "SNDR estimates: none, clip:M, shrink:lambda or harmonic:lambda (default %(default)s; clip:1 = the "
        "older selection).",
    )
    parser.add_argument(
        "--policy-transform",
        choices=list(POLICY_TRANSFORMS),
        default="linear",
        help="How the learned policy corrects the biased vectors: linear = (I + D) x + b per side, "
        "starting at the logger (default); mlp = x + MLP(LN(x)) (the older transform); linear+mlp = "
        "(I + D) x + b + MLP(x), also starting at the logger.",
    )
    parser.add_argument(
        "--log-select-weights",
        nargs="*",
        type=weight_spec_label,
        default=[],
        help="Also log each OPC trial's selection score under these weight specs (trials_long "
        "columns sel_r_hat[spec], sel_ci_low[spec]; for tuning --select-weights).",
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
        default=DEFAULT_VAL_SIZE,
        help="Fixed validation logged trajectories for every train_size (default %(default)s; --val-sizes "
        "overrides). 0 = the older rule val_size = clamp(round(val_frac * train_size), val_min, val_max).",
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
        "--save-policies",
        action="store_true",
        default=False,
        help="Also save each arm's selected policy vectors in the condition folder (training/policy_diagnostics.py).",
    )
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
        default=list(STUDY_POLICY_LOSSES),
        choices=list(VALID_POLICY_LOSSES),
        help="OPC training loss. Default dr (DM + weighted correction, no self-normalization): the working "
        "development default since f5cade9 (2026-09-27), not the final paper choice. sndr with --sn-scope batch (legacy "
        "minibatch-normalized SNDR, the default before f5cade9) and --sn-scope global remain for "
        "reproducibility and diagnostics (docs/training_losses.md 3.4, 9). DR selection uses a fixed weight "
        "transform (--select-weights). Multiple values = Optuna categorical over losses. No-propensity stays naive.",
    )
    parser.add_argument(
        "--sn-scope",
        choices=list(SN_SCOPES),
        default="batch",
        help="Normalizer of the sndr / kl correction: batch (default, legacy: the minibatch mean weight, so "
        "the objective depends on the Optuna-searched batch size) or global (the full-data mean weight, "
        "computed at the start of every epoch and held fixed: no gradient through it, stale after the "
        "epoch's first step; a stop-gradient SNDR surrogate, not exact SNDR, docs/training_losses.md 3.4), or exact "
        "(sndr with --opc-gradient direct only: the gradient of the full-data SNDR ratio, whose two means are "
        "refreshed at the start of every epoch).",
    )
    parser.add_argument(
        "--sampler",
        choices=list(OPTUNA_SAMPLERS),
        default="tpe",
        help="Optuna sampler: tpe (default; each train size starts from the previous size's best) or "
        "random (seeded random search, no warm start). With random, trial k has the same configuration "
        "and the same trial seed in every run with the same grid and seeds, whatever the objective: runs "
        "that differ only in --policy-losses / --sn-scope / --train-weights are then a paired (replayed) "
        "comparison of the objectives.",
    )
    parser.add_argument(
        "--stage",
        choices=list(RUN_STAGES),
        default="development",
        help="Recorded with the results (run_meta.json, summaries, manifest): development (default; runs "
        "used to design the method) or confirmatory (the frozen method on fresh seeds and conditions).",
    )
    parser.add_argument(
        "--opc-gradient",
        choices=list(OPC_GRADIENTS),
        default=STUDY_OPC_GRADIENT,
        help="How OPC's training loss is differentiated: direct (default: pathwise through the transformed "
        "weight g(w), the exact gradient of the named estimate DM + g(w)(r - q_hat)) or log-trick (the "
        "transformed weight as a detached coefficient on grad log pi: the exact gradient of DM + H(w)(r - q_hat) "
        "with H(w) = int_0^w g(t)/t dt; the default before f5cade9, kept for reproducibility). The two "
        "coincide for --train-weights none. docs/training_losses.md 3.4.",
    )
    parser.add_argument(
        "--no-log-trick",
        action="store_true",
        help="Only for trainers that search use_log_trick; every arm of this study has it fixed (OPC: "
        "--opc-gradient; no-propensity and DM: direct), so this flag does not change training here.",
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
        help="User/context block size for lazy q_hat / softmax (default %(default)s).",
    )
    parser.add_argument(
        "--qhat-action-chunk",
        type=int,
        default=DEFAULT_QHAT_ACTION_CHUNK,
        help="Action block size for lazy q_hat / softmax (default %(default)s).",
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
        choices=list(ALL_STUDY_METHODS),
        help=METHODS_HELP,
    )
    parser.add_argument(
        "--learn-logit-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trained policies (OPC, no-prop, DM) also learn a logit scale s, softmax(s·u·a/T), "
        "starting at 1: sharpen or flatten without re-ranking (default: off).",
    )
    parser.add_argument(
        "--reward-data",
        choices=list(REWARD_DATA_MODES),
        default=DEFAULT_REWARD_DATA,
        help="Data of the regression reward model: train (default: each train size's own training rows, "
        "so every arm uses only its n rows; folders get __qhat=train) or external (a separate slice of "
        "--shared-regression-size logged rows, the same at every train size: the runs before 2026-09-26).",
    )
    parser.add_argument(
        "--crossfit-folds",
        type=int,
        default=None,
        help="With --reward-data train: split users into K folds; the training losses take each user's "
        f"q_hat from the model fit on the other folds' rows (default {DEFAULT_CROSSFIT_FOLDS} with train, "
        "off with external; 0 = off; folders get __cf=K).",
    )
    parser.add_argument(
        "--post-temper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After training, scale each trained policy's logits by the factor (0.25-16) with the best "
        "selection score on validation: sharpness chosen after training (default: off).",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=SKIP_COMPLETED_HELP,
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
    args.reward_data, args.crossfit_folds = _study_budget_from_args(args)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")
    print(f"Reward model data: {args.reward_data} (cross-fitting folds: {args.crossfit_folds or 'off'})")
    print(f"Validation configs: {val_size_configs}")
    print(f"Bias configs: {bias_configs}; world options: {world_options}")
    print(f"Policy losses: {policy_loss_types}")
    print(f"Methods: {methods}")
    if "no_propensity" in methods:
        print(f"No-prop losses: {_no_prop_policy_loss_types(policy_loss_types)}")
    print(f"Importance weights: training {args.train_weights}, selection and post-hoc {args.select_weights}")
    print(f"Search use_log_trick: {search_use_log_trick}")

    started = utc_now()
    print(
        "Run order: seed → dataset → ctr → "
        + ("val → " if any(lbl != "frac" for _, lbl in val_size_configs) else "")
        + "bias"
    )
    configs = condition_configs(args, out_dir, methods=methods, bias_configs=bias_configs, world_options=world_options,
                                val_size_configs=val_size_configs, policy_loss_types=policy_loss_types,
                                search_use_log_trick=search_use_log_trick)
    todo, skipped = plan_conditions(configs, methods, skip_completed=bool(args.skip_completed))
    manifest = study_manifest(args, runner="serial", run_tag=run_tag, methods=methods, bias_configs=bias_configs,
                              world_options=world_options, policy_loss_types=policy_loss_types,
                              val_size_configs=val_size_configs)
    ran, failures = [], []
    try:
        for cfg in todo:
            print(f"\n=== Running {cfg['run_key']} ({', '.join(cfg['study_methods'])}) ===", flush=True)
            try:
                execute_condition(cfg)
                ran.append(cfg["run_key"])
            except Exception as e:
                failures.append({"run_key": cfg["run_key"], "error": repr(e)})
                print(f"FAILED {cfg['run_key']}: {e}", flush=True)
                if args.fail_fast:
                    raise
    finally:
        finish_invocation(out_dir, manifest, started=started, ran=ran, skipped=skipped, failures=failures)


if __name__ == "__main__":
    main()
