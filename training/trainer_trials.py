import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import os
import numpy as np
import pandas as pd
import sys
import time

sys.path.append("/code")

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

torch.backends.cudnn.benchmark = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _training_device(*, require_cuda: bool = False) -> torch.device:
    """Pick training device; honors OPC_WORKER_GPU from parallel pool workers."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = int(os.environ.get("OPC_WORKER_GPU", "0"))
        if gpu < 0 or gpu >= torch.cuda.device_count():
            gpu = 0
        dev = torch.device(f"cuda:{gpu}")
    else:
        dev = torch.device("cpu")
    if require_cuda and dev.type != "cuda":
        raise RuntimeError(
            "CUDA is required (--require-cuda), but torch.cuda.is_available() is False. "
            "Use a CUDA PyTorch build (e.g. docker build --build-arg "
            "TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t opc:gpu) and "
            "run with --gpus all."
        )
    return dev


def _dataloader_num_workers() -> int:
    """DataLoader worker processes.

  - Default: 4 when CUDA is available, else 0 (restores pre-parallel-runner behavior).
  - Under ``run_full_study_parallel`` workers, default 0 to avoid EMFILE / fd exhaustion
    (set automatically via ``OPC_IN_PARALLEL=1``).
  - Override anytime with ``OPC_DATALOADER_WORKERS``.
    """
    raw = os.environ.get("OPC_DATALOADER_WORKERS", "").strip()
    if raw != "":
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    if os.environ.get("OPC_IN_PARALLEL", "").strip() in ("1", "true", "yes"):
        return 0
    return 4 if torch.cuda.is_available() else 0


def _log_training_device(method_label: str, device: torch.device) -> None:
    """One-line device probe at trainer entry (helps debug CPU-only runs)."""
    cuda_ok = torch.cuda.is_available()
    parts = [
        f"[{method_label}] device={device}",
        f"cuda_available={cuda_ok}",
        f"dataloader_workers={_dataloader_num_workers()}",
    ]
    if cuda_ok:
        try:
            parts.append(f"gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(" | ".join(parts), flush=True)


from sklearn.base import is_classifier
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import t as student_t
import optuna

from utils.policies import Policy, generate_policies
from utils.plots_and_stats import compute_statistics_and_plots

from models.model_scoring import score_model_modular_large
from models.estimators import (
    DirectMethod as DM,
)

from utils.chunk_progress import iter_action_blocks, iter_user_action_blocks
from utils.simulation_utils import (
    eval_policy,
    generate_dataset,
    create_simulation_data_from_pi,
    get_train_data,
    get_opl_results_dict,
    CustomCFDataset,
    CustomCFDatasetPS,
    calc_reward,
    calc_reward_mc,
    get_weights_info,
    create_simulation_data_from_policy,
)

from models.models import (
    LinearCFModel,
    CFModel,
    MLPRewardModel,
    SingleMLPTransform,
    NeighborhoodModel,
    RegressionModel,
)

from training.training_utils import (
    train,
)

from models.custom_losses import (
    CRMPolicyLoss,
    IPWPolicyLoss,
    KLCRMPolicyLoss,
    KLPolicyLoss,
    NaiveRewardPolicyLoss,
    SNDRPolicyLoss,
    uses_importance_weighting,
)
from training.metrics_utils import (
    enrich_summary_pct_fields,
    enrich_trial_pct_fields,
)

VALID_POLICY_LOSSES = ("kl_crm", "kl", "ipw", "sndr", "crm", "naive")


def _policy_loss_needs_kl(policy_loss_types: tuple[str, ...] | list[str]) -> bool:
    return any(str(x).lower() in ("kl", "kl_crm") for x in policy_loss_types)


def _policy_loss_needs_crm(policy_loss_types: tuple[str, ...] | list[str]) -> bool:
    return any(str(x).lower() in ("crm", "kl_crm") for x in policy_loss_types)

# Max working blocks for q_hat / softmax (user_chunk, action_chunk); no full n_users x n_actions.
DEFAULT_QHAT_USER_CHUNK = 5000
DEFAULT_QHAT_ACTION_CHUNK = 5000
DEFAULT_OPTUNA_BATCH_SIZES = (256, 512, 1024, 2048, 4096)
DEFAULT_NEIGHBORHOOD_OPTUNA_BATCH_SIZES = (64, 128, 256, 512)
LOGGED_RUN_IDX = 0


def _normalize_optuna_batch_sizes(
    sizes: list[int] | tuple[int, ...] | None,
    *,
    default: tuple[int, ...] = DEFAULT_OPTUNA_BATCH_SIZES,
) -> list[int]:
    if sizes is None:
        return list(default)
    out = sorted({int(x) for x in sizes if int(x) > 0})
    if not out:
        raise ValueError("optuna_batch_sizes must contain at least one positive int")
    return out


def _kl_policy_loss(
    gamma: float,
    use_log_trick: bool = True,
    propensity_mode: str = "logged",
) -> KLPolicyLoss:
    return KLPolicyLoss(
        gamma=float(gamma),
        use_log_trick=use_log_trick,
        propensity_mode=propensity_mode,
    )


def _crm_policy_loss(
    clip_m: float,
    crm_lambda: float,
    use_log_trick: bool = True,
    propensity_mode: str = "logged",
) -> CRMPolicyLoss:
    return CRMPolicyLoss(
        clip_m=float(clip_m),
        crm_lambda=float(crm_lambda),
        use_log_trick=use_log_trick,
        propensity_mode=propensity_mode,
    )


def _kl_crm_policy_loss(
    gamma: float,
    clip_m: float,
    crm_lambda: float,
    use_log_trick: bool = True,
    propensity_mode: str = "logged",
) -> KLCRMPolicyLoss:
    return KLCRMPolicyLoss(
        gamma=float(gamma),
        clip_m=float(clip_m),
        crm_lambda=float(crm_lambda),
        use_log_trick=use_log_trick,
        propensity_mode=propensity_mode,
    )


def _policy_loss_from_name(
    loss_name: str,
    *,
    kl_gamma: float = 0.05,
    clip_m: float = 10.0,
    crm_lambda: float = 1.0,
    use_log_trick: bool = True,
    propensity_mode: str = "logged",
):
    name = str(loss_name).lower()
    if name == "kl_crm":
        return _kl_crm_policy_loss(
            kl_gamma,
            clip_m,
            crm_lambda,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    if name == "kl":
        return _kl_policy_loss(
            kl_gamma,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    if name == "ipw":
        return IPWPolicyLoss(
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    if name == "naive":
        return NaiveRewardPolicyLoss(
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    if name == "sndr":
        return SNDRPolicyLoss(
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    if name == "crm":
        return _crm_policy_loss(
            clip_m,
            crm_lambda,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
    raise ValueError(f"Unknown policy loss '{loss_name}'; expected one of {VALID_POLICY_LOSSES}")


def _resolve_trial_use_log_trick(
    trial,
    search_use_log_trick: bool,
    use_log_trick_fixed: bool | None = None,
) -> bool:
    """Optuna toggle for log-trick vs direct-prob policy surrogate (all losses)."""
    if use_log_trick_fixed is not None:
        return bool(use_log_trick_fixed)
    if not search_use_log_trick:
        return False
    return trial.suggest_categorical("use_log_trick", [True, False])


def _fix_best_params_use_log_trick(
    best_params: dict,
    *,
    search_use_log_trick: bool,
    use_log_trick_fixed: bool | None,
) -> dict:
    out = dict(best_params)
    if use_log_trick_fixed is not None:
        out["use_log_trick"] = bool(use_log_trick_fixed)
    elif not search_use_log_trick:
        out["use_log_trick"] = False
    return out


def _split_seed_for_condition(base_seed: int, train_size: int, run_idx: int) -> int:
    """Deterministic RNG seed for logged train/val simulation (shared across methods)."""
    return int(base_seed) + int(train_size) * 1009 + int(run_idx) * 17 + 7


def _partition_seed(split_seed: int) -> int:
    """RNG seed for random reg/train/val index draws (distinct from simulation seed)."""
    return int(split_seed) + 2_000_003


def _random_partition_indices(
    total: int,
    sizes: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Disjoint random index sets into a simulated pool (no sequential prefix/suffix bias)."""
    sizes = tuple(int(s) for s in sizes)
    need = sum(sizes)
    total = int(total)
    if need > total:
        raise ValueError(f"partition needs {need} indices but simulated pool has {total}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(total)
    out: list[np.ndarray] = []
    i = 0
    for s in sizes:
        out.append(perm[i : i + s].copy())
        i += s
    return tuple(out)


def _build_baseline_logged_split(
    dataset,
    our_x_orig,
    our_a_orig,
    val_size: int,
    split_seed: int,
):
    """Logging-policy baseline: train/val slices from one simulation (no CF training)."""
    n_actions = int(dataset["n_actions"])
    v = int(val_size)
    simulation_data = _simulate_from_embedding_policy(
        dataset,
        our_x_orig,
        our_a_orig,
        v + v,
        random_state=int(split_seed),
    )
    train_idx, val_idx = _random_partition_indices(
        v + v, (v, v), _partition_seed(split_seed)
    )
    train_data = get_train_data(
        n_actions, v, simulation_data, train_idx, our_x_orig
    )
    val_data = get_train_data(
        n_actions, v, simulation_data, val_idx, our_x_orig
    )
    return {
        "reg_data": None,
        "train_data": train_data,
        "val_data": val_data,
        "val_size": v,
        "split_seed": int(split_seed),
        "random_partition": True,
    }


def _build_regression_logged_split(
    dataset,
    our_x_orig,
    our_a_orig,
    train_size: int,
    val_size: int,
    run_idx: int,
    split_seed: int | None = None,
    regression_size: int = 50_000,
):
    """One logged simulation of reg+train+val -> disjoint slices (OPC vs no-prop)."""
    n_actions = int(dataset["n_actions"])
    train_size = int(train_size)
    v = int(val_size)
    reg_size = int(regression_size)
    if split_seed is None:
        split_seed = (int(run_idx) + 1) * (train_size + 17)
    total = reg_size + train_size + v
    simulation_data = _simulate_from_embedding_policy(
        dataset,
        our_x_orig,
        our_a_orig,
        total,
        random_state=int(split_seed),
    )
    reg_idx, train_idx, val_idx = _random_partition_indices(
        total,
        (reg_size, train_size, v),
        _partition_seed(split_seed),
    )
    reg_data = get_train_data(
        n_actions, reg_size, simulation_data, reg_idx, our_x_orig
    )
    train_data = get_train_data(
        n_actions, train_size, simulation_data, train_idx, our_x_orig
    )
    val_data = get_train_data(
        n_actions, v, simulation_data, val_idx, our_x_orig
    )
    return {
        "reg_data": reg_data,
        "train_data": train_data,
        "val_data": val_data,
        "val_size": v,
        "regression_size": reg_size,
        "split_seed": int(split_seed),
        "random_partition": True,
    }


class LazyRegressionSplitCache:
    """
    Logged train/val splits built on first access per key.

    Shared by OPC and no-propensity in the same process so each simulation runs
  once per (train_size, run) without pre-allocating every split up front.
    """

    def __init__(
        self,
        dataset,
        train_sizes,
        *,
        val_size=None,
        val_frac=0.15,
        val_min=5000,
        val_max=None,
        condition_seed: int = 0,
        regression_size: int = 50_000,
    ):
        self._dataset = dataset
        self._our_x_orig = dataset["our_x"]
        self._our_a_orig = dataset["our_a"]
        self._train_sizes = [int(x) for x in train_sizes]
        self._val_size = val_size
        self._val_frac = val_frac
        self._val_min = val_min
        self._val_max = val_max
        self._condition_seed = int(condition_seed)
        self._regression_size = int(regression_size)
        self._built: dict = {}

    def _all_keys(self):
        keys = {(0, 0)}
        for train_size in self._train_sizes:
            keys.add((int(train_size), LOGGED_RUN_IDX))
        return keys

    def keys(self):
        return self._all_keys()

    def __contains__(self, key):
        k = (int(key[0]), int(key[1]))
        return k in self._all_keys()

    def __getitem__(self, key):
        k = (int(key[0]), int(key[1]))
        if k not in self:
            raise KeyError(key)
        if k not in self._built:
            self._built[k] = self._build_one(k)
        return self._built[k]

    def _build_one(self, key):
        train_size, run = key
        if key == (0, 0):
            base_train = min(self._train_sizes) if self._train_sizes else 10_000
            v_baseline = resolve_validation_size(
                base_train,
                val_size=self._val_size,
                val_frac=self._val_frac,
                val_min=self._val_min,
                val_max=self._val_max,
            )
            return _build_baseline_logged_split(
                self._dataset,
                self._our_x_orig,
                self._our_a_orig,
                v_baseline,
                _split_seed_for_condition(self._condition_seed, 0, 0),
            )
        v = resolve_validation_size(
            train_size,
            val_size=self._val_size,
            val_frac=self._val_frac,
            val_min=self._val_min,
            val_max=self._val_max,
        )
        seed = _split_seed_for_condition(self._condition_seed, train_size, run)
        return _build_regression_logged_split(
            self._dataset,
            self._our_x_orig,
            self._our_a_orig,
            train_size,
            v,
            run,
            split_seed=seed,
            regression_size=self._regression_size,
        )


def build_regression_split_cache(
    dataset,
    train_sizes,
    *,
    val_size=None,
    val_frac=0.15,
    val_min=5000,
    val_max=None,
    condition_seed: int = 0,
    regression_size: int = 50_000,
):
    """
    Eagerly pre-draw all logged splits (high memory). Prefer ``LazyRegressionSplitCache``.
    """
    lazy = LazyRegressionSplitCache(
        dataset,
        train_sizes,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
        condition_seed=condition_seed,
        regression_size=regression_size,
    )
    return {k: lazy[k] for k in lazy.keys()}


def fit_shared_regression_bundle(
    dataset: dict,
    reg_data: dict,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
):
    """Fit one reward model on the reg slice; q_hat is computed on demand (not materialized)."""
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    n_actions = int(np.asarray(our_a).shape[0])
    if n_actions != int(dataset["n_actions"]):
        raise ValueError(
            f"dataset n_actions={dataset['n_actions']} != emb_a rows={n_actions}"
        )
    n = int(len(reg_data["r"]))
    model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a,
        base_model=LogisticRegression(random_state=12345),
    )
    t0 = time.time()
    model.fit(reg_data["x"], reg_data["a"], reg_data["r"])
    print(
        f"[Regression] shared fit n={n} time={time.time() - t0:.2f}s "
        f"(lazy q_hat user_chunk={user_chunk} action_chunk={action_chunk})",
        flush=True,
    )
    return {
        "regression_model": model,
        "user_context": our_x,
        "user_chunk": int(user_chunk),
        "action_chunk": int(action_chunk),
        "catalog_n_actions": int(n_actions),
        "sample_size": int(n),
    }

from models.model_scoring import score_model_modular, score_model_modular_large
random_state = 12345
random_ = check_random_state(random_state)


# --------------------------------------------------------------------
# Small helper: wrap a RegressionModel so eval_policy (which expects
# model.predict(x_idx)) can still work by mapping indices -> contexts.
# --------------------------------------------------------------------
class IndexToContextModelWrapper:
    def __init__(self, base_model, user_context):
        """
        base_model: RegressionModel (from models.py)
        user_context: np.ndarray of shape (n_users, d_x)
        """
        self.base_model = base_model
        self.user_context = user_context

    def predict(self, context):
        """
        x_idx: indices of users (np.array, torch.tensor, list, etc.)
        Returns q_hat(context, a) with the same shape as base_model.predict(context).
        """
        # x_idx = np.asarray(x_idx)
        # context = np.asarray(context)
        # context = self.user_context[x_idx]
        return self.base_model.predict(context)


# --------------------------------------------------------------------
# Validation sizing + numeric precision helpers
# --------------------------------------------------------------------
def resolve_validation_size(
    train_size: int,
    *,
    val_size: int | None = None,
    val_frac: float | None = None,
    val_min: int = 5000,
    val_max: int | None = None,
) -> int:
    """
    Number of logged validation trajectories.

    - If ``val_size`` is set: fixed count (legacy / explicit).
    - Else: ``round(val_frac * train_size)``, clamped to ``[val_min, val_max]``.
    """
    train_size = int(train_size)
    if val_size is not None:
        v = int(val_size)
    else:
        frac = 0.15 if val_frac is None else float(val_frac)
        v = int(round(frac * max(train_size, 1)))
        v = max(int(val_min), v)
        if val_max is not None:
            v = min(int(val_max), v)
    return max(v, 1)


# --------------------------------------------------------------------
# Shared utility: robust mean over dict list
# --------------------------------------------------------------------
def _mean_dict(dicts):
    """Robust mean over a list of dicts with numeric/array values (skips strings)."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = []
        for d in dicts:
            if k not in d:
                continue
            try:
                arr = np.asarray(d[k])
            except Exception:
                continue
            if arr.dtype.kind not in "biufc":
                continue
            vals.append(arr)
        if not vals:
            continue
        stacked = np.stack(vals, axis=0)
        out[k] = np.mean(stacked, axis=0)
    return out


def _scalar_for_run_log(value):
    """Coerce numeric scalars for CSV rows; keep strings and multi-element arrays."""
    if isinstance(value, (str, bytes, bool)):
        return value
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.dtype.kind in ("U", "S", "O"):
        if arr.size == 1:
            x = arr.reshape(-1)[0]
            return x.item() if hasattr(x, "item") else x
        return value
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return value


def _aggregate_runs_by_validation_score(dicts, score_key: str = "selection_val_score"):
    """
    Pick metrics from the run whose Optuna validation objective was best, and add
    *_runs_mean for every metric averaged across runs (excluding score_key).

    Primary columns (policy_rewards, conv_dr, ...) are from the winning run — the
    regime you get after selecting hyperparameters by validation score per run, then
    choosing the best run by that same score.
    """
    if not dicts:
        return {}, -1
    stripped = [{k: v for k, v in d.items() if k != score_key} for d in dicts]
    runs_mean = _mean_dict(stripped)

    scores = []
    for d in dicts:
        s = d.get(score_key, float("-inf"))
        try:
            sf = float(np.asarray(s).reshape(-1)[0])
            if not np.isfinite(sf):
                sf = float("-inf")
        except Exception:
            sf = float("-inf")
        scores.append(sf)
    idx = int(np.argmax(scores))
    best = dict(dicts[idx])
    win_score = best.pop(score_key, float("nan"))

    out = {}
    for k, v in best.items():
        out[k] = v
    out[score_key] = win_score
    for k, v in runs_mean.items():
        out[f"{k}_runs_mean"] = v
    return out, idx


def _append_csv(path, df: pd.DataFrame):
    """Append rows to a CSV, writing header only if the file doesn't exist yet."""
    if path is None or df is None or df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def _study_trials_long(
    study,
    method_label: str,
    train_size: int,
    run_idx: int,
    best_trial_number: int | None = None,
    ctr: float | None = None,
    initial_reward: float | None = None,
    dataset_name: str | None = None,
):
    """Flatten an Optuna study into a long-format DataFrame for replayable logs."""
    rows = []
    if study is None:
        return pd.DataFrame()
    ctr_v = float(ctr) if ctr is not None else float("nan")
    ir_v = float(initial_reward) if initial_reward is not None else float("nan")
    for t in study.trials:
        attrs = t.user_attrs or {}
        params = t.params or {}
        rows.append(
            {
                "dataset": str(dataset_name) if dataset_name is not None else "",
                "method": method_label,
                "train_size": int(train_size),
                "run": int(run_idx),
                "trial_number": int(t.number),
                "ctr": ctr_v,
                "initial_reward": ir_v,
                "value": (
                    float(t.value)
                    if t.value is not None and np.isfinite(t.value)
                    else float("nan")
                ),
                "r_hat": float(attrs.get("r_hat", float("nan"))),
                "r_hat_train": float(attrs.get("r_hat_train", float("nan"))),
                "q_error": float(attrs.get("q_error", float("nan"))),
                "ess": float(attrs.get("ess", float("nan"))),
                "ess_train": float(attrs.get("ess_train", float("nan"))),
                "actual_reward": float(
                    np.asarray(attrs.get("actual_reward", float("nan"))).reshape(-1)[0]
                )
                if attrs.get("actual_reward") is not None
                else float("nan"),
                "param_lr": float(params.get("lr", float("nan"))),
                "param_num_epochs": int(params.get("num_epochs", -1)),
                "param_batch_size": int(params.get("batch_size", -1)),
                "param_lr_decay": float(params.get("lr_decay", float("nan"))),
                "param_kl_gamma": float(params.get("kl_gamma", float("nan"))),
                "param_crm_M": float(params.get("crm_M", float("nan"))),
                "param_crm_lambda": float(params.get("crm_lambda", float("nan"))),
                "param_use_log_trick": int(bool(params.get("use_log_trick", True))),
                "param_policy_loss": str(params.get("policy_loss", "kl_crm")),
                "param_num_neighbors": int(params.get("num_neighbors", -1)),
                "is_best_in_run": bool(
                    best_trial_number is not None and t.number == best_trial_number
                ),
            }
        )
        act = rows[-1]["actual_reward"]
        val = rows[-1]["value"]
        if np.isfinite(ir_v):
            rows[-1].update(enrich_trial_pct_fields(act, val, ir_v))
    return pd.DataFrame(rows)


def _nan_posthoc_eval_metrics() -> dict:
    """Placeholder metrics when ``slim`` skips ``get_trial_results``."""
    keys = (
        "policy_rewards",
        "ipw",
        "reg_dm",
        "conv_dm",
        "conv_dr",
        "conv_sndr",
        "ipw_var",
        "reg_dm_var",
        "conv_dm_var",
        "conv_dr_var",
        "conv_sndr_var",
        "action_diff_to_real",
        "action_delta",
        "context_diff_to_real",
        "context_delta",
    )
    return {k: float("nan") for k in keys}


def _slim_learned_policy_metrics(
    dataset: dict,
    learned_x: np.ndarray,
    learned_a: np.ndarray,
) -> dict:
    """True policy value + embedding refs for slim runs (skip post-hoc eval)."""
    reward = float(_policy_reward_from_embeddings(dataset, learned_x, learned_a))
    return {
        **_nan_posthoc_eval_metrics(),
        "policy_rewards": reward,
        "actual_reward_selected": reward,
        "_learned_user_emb": learned_x,
        "_learned_item_emb": learned_a,
    }


def _learned_embeddings_from_trial_dict(d: dict, our_x_orig, our_a_orig):
    return (
        d.get("_learned_user_emb", our_x_orig),
        d.get("_learned_item_emb", our_a_orig),
    )


def _append_slim_winning_run_extras(
    row: dict,
    d: dict,
    *,
    dataset: dict,
    our_x_orig,
    our_a_orig,
    train_users,
    train_actions,
    pscore_tr,
) -> None:
    """Attach IW diagnostics and selected-policy reward for slim run logs."""
    ux, ua = _learned_embeddings_from_trial_dict(d, our_x_orig, our_a_orig)
    pi_e_tr = _batched_pi_at_logged_actions(
        ux,
        ua,
        train_users,
        train_actions,
        policy_temperature=_policy_temperature(dataset),
    )
    wi = get_weights_info(pi_e_tr, pscore_tr)
    for k2, v2 in wi.items():
        row[f"weights_{k2}"] = float(v2)
    selected = d.get("actual_reward_selected")
    if selected is not None and np.isfinite(float(np.asarray(selected).reshape(-1)[0])):
        row["actual_reward_selected"] = float(np.asarray(selected).reshape(-1)[0])
    else:
        row["actual_reward_selected"] = float(
            _policy_reward_from_embeddings(dataset, ux, ua)
        )


def _enqueue_with_kl_gamma(
    last_best: dict | None,
    kl_default: float = 0.05,
    crm_m_default: float = 10.0,
    crm_lambda_default: float = 1.0,
    policy_loss_types: tuple[str, ...] = ("kl_crm",),
    search_use_log_trick: bool = True,
    use_log_trick_fixed: bool | None = None,
) -> dict | None:
    """Optuna enqueue compatibility when new search dims were added."""
    if last_best is None:
        return None
    merged = dict(last_best)
    if _policy_loss_needs_kl(policy_loss_types):
        merged.setdefault("kl_gamma", float(kl_default))
    else:
        merged.pop("kl_gamma", None)
    if _policy_loss_needs_crm(policy_loss_types):
        merged.setdefault("crm_M", float(crm_m_default))
        merged.setdefault("crm_lambda", float(crm_lambda_default))
    else:
        merged.pop("crm_M", None)
        merged.pop("crm_lambda", None)
    if use_log_trick_fixed is not None:
        merged["use_log_trick"] = bool(use_log_trick_fixed)
    elif not search_use_log_trick:
        merged["use_log_trick"] = False
    else:
        merged.setdefault("use_log_trick", True)
    if len(policy_loss_types) == 1:
        merged["policy_loss"] = policy_loss_types[0]
    return merged


def _resolve_logged_pscore(train_data, original_policy_prob, mode="logged"):
    """
    Behavior propensity pi_b(a|x) at the logged action.

    ``mode`` is accepted for API compatibility; always returns true logged
    propensities. Importance-weight bypass (iw=1) is handled in the loss via
    ``propensity_mode="uniform"``.
    """
    _ = mode
    if "pscore" in train_data and train_data["pscore"] is not None:
        return np.asarray(train_data["pscore"], dtype=np.float32)

    return np.asarray(
        original_policy_prob[train_data["x_idx"], train_data["a"]].squeeze(),
        dtype=np.float32,
    )


def _build_cf_dataset(train_data, original_policy_prob, propensity_mode="logged"):
    _ = propensity_mode
    pscore = _resolve_logged_pscore(
        train_data=train_data,
        original_policy_prob=original_policy_prob,
        mode="logged",
    )
    return CustomCFDatasetPS(
        train_data["x_idx"],
        train_data["a"],
        train_data["r"],
        pscore,
    )


def _policy_temperature(dataset: dict) -> float:
    """Softmax temperature for Policy logits (dot-product policies). Default 1.0."""
    return float(dataset.get("policy_temperature", 1.0))


def _simulate_from_embedding_policy(dataset, our_x, our_a, n_samples, random_state):
    """Sample logged bandit data without materializing dense pi (n_users x n_actions)."""
    rng = int(random_state) % (2**31 - 1)
    logging_policy = Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=int(our_x.shape[1]),
        temperature=_policy_temperature(dataset),
        user_chunk=2048,
        rng=np.random.default_rng(rng),
    )
    return create_simulation_data_from_policy(
        dataset=dataset,
        policy=logging_policy,
        n_samples=int(n_samples),
        random_state=int(random_state),
    )


def _softmax_action_probs_stable(logits, axis=1, min_floor: float = 1e-15):
    """Softmax in float64, floor, renormalize — avoids float32 underflow on large |A|."""
    x = np.asarray(logits, dtype=np.float64)
    p = softmax(x, axis=axis)
    p = np.maximum(p, min_floor)
    p /= np.sum(p, axis=axis, keepdims=True)
    return p.astype(np.float32)


def _catalog_n_actions(regression_model) -> int:
    """Item count for q_hat / softmax; prefer action_context rows over n_actions field."""
    ac = getattr(regression_model, "action_context", None)
    if ac is not None:
        return int(np.asarray(ac).shape[0])
    return int(regression_model.n_actions)


def _predict_regression_qhat_user_action_block(
    regression_model,
    context: np.ndarray,
    action_start: int,
    action_end: int,
) -> np.ndarray:
    """q_hat for users x actions[action_start:action_end]; shape (n_users, n_actions_block, len_list)."""
    n = int(context.shape[0])
    n_a = int(action_end - action_start)
    n_list = int(regression_model.len_list)
    q_hat = np.zeros((n, n_a, n_list), dtype=np.float32)
    for local_a, action_ in enumerate(range(int(action_start), int(action_end))):
        for pos_ in range(n_list):
            X = regression_model._pre_process_for_reg_model(
                context=context,
                action=action_ * np.ones(n, dtype=int),
                action_context=regression_model.action_context,
            )
            model = regression_model.base_model_list[pos_]
            q_hat_ = (
                model.predict_proba(X)[:, 1]
                if is_classifier(model)
                else model.predict(X)
            )
            q_hat[:, local_a, pos_] = np.asarray(q_hat_, dtype=np.float32)
    return q_hat


def predict_regression_qhat_users(
    regression_model,
    user_context: np.ndarray,
    user_ids: np.ndarray,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    show_progress: bool = False,
) -> np.ndarray:
    """q_hat for selected users; max working block (user_chunk, action_chunk)."""
    user_ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    n = len(user_ids)
    n_actions = _catalog_n_actions(regression_model)
    n_list = int(regression_model.len_list)
    out = np.zeros((n, n_actions, n_list), dtype=np.float32)
    for us, ue, a0, a1 in iter_user_action_blocks(
        n,
        n_actions,
        user_chunk,
        action_chunk,
        desc="q_hat",
        show_progress=show_progress,
    ):
        ctx = np.asarray(user_context[user_ids[us:ue]], dtype=np.float32)
        block = _predict_regression_qhat_user_action_block(
            regression_model, ctx, a0, a1
        )
        out[us:ue, a0:a1, :] = block
    return out


class RegressionScoresLookup:
    """On-demand q_hat rows for training batches (no full n_users x n_actions tensor)."""

    def __init__(
        self,
        regression_model,
        user_context: np.ndarray,
        device,
        *,
        user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
        action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    ):
        self.regression_model = regression_model
        self.user_context = np.asarray(user_context)
        self.device = device
        self.user_chunk = int(user_chunk)
        self.action_chunk = int(action_chunk)
        self.n_actions = _catalog_n_actions(regression_model)
        self.show_progress = False

    def __getitem__(self, user_idx):
        if isinstance(user_idx, torch.Tensor):
            user_idx = user_idx.detach().cpu().numpy()
        user_idx = np.asarray(user_idx, dtype=np.int64).reshape(-1)
        q = predict_regression_qhat_users(
            self.regression_model,
            self.user_context,
            user_idx,
            user_chunk=self.user_chunk,
            action_chunk=self.action_chunk,
            show_progress=self.show_progress,
        )
        if q.ndim == 3 and q.shape[2] == 1:
            q = q[:, :, 0]
        return torch.as_tensor(q, device=self.device, dtype=torch.float32)


def _scores_lookup_from_bundle(bundle: dict, device) -> RegressionScoresLookup:
    if "q_hat_all" in bundle:
        arr = np.asarray(bundle["q_hat_all"], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        class _DenseScores:
            def __init__(self, tensor):
                self._t = torch.as_tensor(tensor, device=device, dtype=torch.float32)

            def __getitem__(self, user_idx):
                return self._t[user_idx.long()]

        return _DenseScores(arr)
    return RegressionScoresLookup(
        bundle["regression_model"],
        bundle["user_context"],
        device,
        user_chunk=int(bundle.get("user_chunk", DEFAULT_QHAT_USER_CHUNK)),
        action_chunk=int(bundle.get("action_chunk", DEFAULT_QHAT_ACTION_CHUNK)),
    )


def _logsumexp_action_chunks(
    user_emb_block: np.ndarray,
    item_emb: np.ndarray,
    policy_temperature: float,
    action_chunk: int,
) -> np.ndarray:
    """Per-row logsumexp over all actions; peak array (n_rows, action_chunk)."""
    pt = max(float(policy_temperature), 1e-8)
    n_rows = int(user_emb_block.shape[0])
    n_actions = int(item_emb.shape[0])
    acc = np.full(n_rows, -np.inf, dtype=np.float64)
    for a0, a1 in iter_action_blocks(
        n_actions,
        action_chunk,
        desc="logsumexp pi",
        n_rows=n_rows,
    ):
        logits = (user_emb_block @ item_emb[a0:a1].T) / pt
        m = logits.max(axis=1)
        s = np.exp(logits - m[:, None]).sum(axis=1)
        acc = np.logaddexp(acc, m + np.log(s + 1e-300))
    return acc


def _iter_full_softmax_action_blocks(
    user_emb_block: np.ndarray,
    item_emb: np.ndarray,
    policy_temperature: float,
    action_chunk: int,
    *,
    desc: str = "policy probs",
    n_rows: int | None = None,
):
    """Yield (a0, a1, pi_block) with pi normalized over the full action catalog."""
    pt = max(float(policy_temperature), 1e-8)
    user_emb_block = np.asarray(user_emb_block, dtype=np.float32)
    item_emb = np.asarray(item_emb, dtype=np.float32)
    log_denom = _logsumexp_action_chunks(
        user_emb_block, item_emb, pt, action_chunk
    )
    n_actions = int(item_emb.shape[0])
    if n_rows is None:
        n_rows = int(user_emb_block.shape[0])
    for a0, a1 in iter_action_blocks(
        n_actions,
        action_chunk,
        desc=desc,
        n_rows=n_rows,
    ):
        logits = (user_emb_block @ item_emb[a0:a1].T).astype(np.float64) / pt
        pi = np.exp(logits - log_denom[:, None]).astype(np.float32)
        yield int(a0), int(a1), pi


def _batched_pi_at_logged_actions(
    user_emb,
    item_emb,
    user_ids,
    action_ids,
    chunk_size: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    policy_temperature: float = 1.0,
):
    """Per-row pi(a_i|x_i) without materializing (n_rows, n_actions)."""
    user_ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    action_ids = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    out = np.empty(len(user_ids), dtype=np.float32)
    xw = np.asarray(user_emb, dtype=np.float32)
    aw = np.asarray(item_emb, dtype=np.float32)
    pt = max(float(policy_temperature), 1e-8)
    n_actions = aw.shape[0]
    for s in range(0, len(user_ids), chunk_size):
        e = min(len(user_ids), s + chunk_size)
        u_emb = xw[user_ids[s:e]]
        log_denom = _logsumexp_action_chunks(u_emb, aw, pt, action_chunk)
        logit_a = np.empty(e - s, dtype=np.float64)
        for a0 in range(0, n_actions, action_chunk):
            a1 = min(n_actions, a0 + action_chunk)
            mask = (action_ids[s:e] >= a0) & (action_ids[s:e] < a1)
            if not np.any(mask):
                continue
            logits = (u_emb[mask] @ aw[a0:a1].T) / pt
            loc = (action_ids[s:e][mask] - a0).astype(np.int64)
            logit_a[np.where(mask)[0]] = logits[np.arange(mask.sum()), loc]
        out[s:e] = np.exp(logit_a - log_denom).astype(np.float32)
    return out


def _dm_reward_rows_chunked(
    users: np.ndarray,
    trial_x: np.ndarray,
    trial_a: np.ndarray,
    score_lookup: RegressionScoresLookup,
    dataset: dict,
) -> np.ndarray:
    """Per-row DM term sum_a q(x,a) pi(a|x) with (row_chunk, action_chunk) blocks."""
    users = np.asarray(users, dtype=np.int64).reshape(-1)
    n_rows = len(users)
    dm = np.zeros(n_rows, dtype=np.float64)
    pt = max(_policy_temperature(dataset), 1e-8)
    xw = np.asarray(trial_x, dtype=np.float32)
    aw = np.asarray(trial_a, dtype=np.float32)
    n_actions = aw.shape[0]
    uc = score_lookup.user_chunk
    ac = score_lookup.action_chunk
    for rs in range(0, n_rows, uc):
        re = min(n_rows, rs + uc)
        u_block = users[rs:re]
        ctx = score_lookup.user_context[u_block]
        u_emb = xw[u_block]
        for a0, a1, pi in _iter_full_softmax_action_blocks(
            u_emb,
            aw,
            pt,
            ac,
            desc="DR DM",
            n_rows=re - rs,
        ):
            q = _predict_regression_qhat_user_action_block(
                score_lookup.regression_model, ctx, a0, a1
            )[:, :, 0]
            dm[rs:re] += (q * pi).sum(axis=1)
    return dm


def cv_score_model(
    val_data,
    score_lookup,
    user_emb,
    item_emb,
    policy_temperature: float = 1.0,
):
    """Conservative validation score (chunked; no dense n_val x n_actions matrices)."""
    if isinstance(score_lookup, torch.Tensor):
        return _cv_score_model_dense(
            val_data,
            score_lookup,
            user_emb,
            item_emb,
            policy_temperature=policy_temperature,
        )
    pscore = np.asarray(val_data["pscore"], dtype=np.float32)
    users = np.asarray(val_data["x_idx"], dtype=np.int64)
    reward = np.asarray(val_data["r"], dtype=np.float32)
    actions = np.asarray(val_data["a"], dtype=np.int64)

    pi_e_at_position = _batched_pi_at_logged_actions(
        user_emb,
        item_emb,
        users,
        actions,
        chunk_size=score_lookup.user_chunk,
        action_chunk=score_lookup.action_chunk,
        policy_temperature=policy_temperature,
    )
    ctx = score_lookup.user_context[users]
    q_hat_factual = np.asarray(
        score_lookup.regression_model.predict_pairs(ctx, actions),
        dtype=np.float32,
    )
    dm_reward = _dm_reward_rows_chunked(
        users, user_emb, item_emb, score_lookup, dataset
    ).astype(np.float32)
    iw = pi_e_at_position / (pscore + 1e-12)
    dr_vec = dm_reward + iw * (reward - q_hat_factual)
    n = max(len(dr_vec), 2)
    r_hat = float(dr_vec.mean())
    se = float(dr_vec.std(ddof=1) / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, n - 1))
    return r_hat - tcrit * se


def _cv_score_model_dense(
    val_data,
    scores_all: torch.Tensor,
    user_emb,
    item_emb,
    policy_temperature: float = 1.0,
):
    """Validation score when scores are a precomputed tensor (neighborhood path)."""
    pscore = np.asarray(val_data["pscore"], dtype=np.float32)
    users = np.asarray(val_data["x_idx"], dtype=np.int64)
    reward = np.asarray(val_data["r"], dtype=np.float32)
    actions = np.asarray(val_data["a"], dtype=np.int64)
    ac = DEFAULT_QHAT_ACTION_CHUNK
    pi_e_at_position = _batched_pi_at_logged_actions(
        user_emb,
        item_emb,
        users,
        actions,
        action_chunk=ac,
        policy_temperature=policy_temperature,
    )
    scores_val = np.asarray(
        scores_all[users].detach().cpu().numpy(), dtype=np.float32
    ).squeeze()
    loc = np.arange(len(users), dtype=np.int64)
    q_hat_factual = np.asarray(scores_val[loc, actions].squeeze(), dtype=np.float32)
    n_actions = int(scores_val.shape[1])
    xw = np.asarray(user_emb, dtype=np.float32)
    aw = np.asarray(item_emb, dtype=np.float32)
    pt = max(float(policy_temperature), 1e-8)
    u_emb = xw[users]
    dm_reward = np.zeros(len(users), dtype=np.float32)
    for a0, a1, pi in _iter_full_softmax_action_blocks(
        u_emb,
        aw,
        pt,
        ac,
        desc="cv DM (dense scores)",
        n_rows=len(users),
    ):
        dm_reward += (scores_val[:, a0:a1] * pi).sum(axis=1)
    iw = pi_e_at_position / (pscore + 1e-12)
    dr_vec = dm_reward + iw * (reward - q_hat_factual)
    n = max(len(dr_vec), 2)
    r_hat = float(dr_vec.mean())
    se = float(dr_vec.std(ddof=1) / np.sqrt(n))
    tcrit = float(student_t.ppf(0.975, n - 1))
    return r_hat - tcrit * se


def _split_dr_vec_and_ess(
    split_data: dict,
    trial_x: np.ndarray,
    trial_a: np.ndarray,
    score_lookup,
    dataset: dict,
    *,
    propensity_mode: str = "logged",
) -> tuple[np.ndarray, float]:
    """Per-row value vector and ESS on a logged split (train or val), chunked.

    Off-policy (``logged``): DR_i = DM_i + (pi_e/pi_b) * (r - q).
    No-propensity (``uniform``): pure naive R_i = r_i * pi_e(a_i|x_i)
    (no DM, no SNDR correction, no propensity weights).
    """
    pscore = np.asarray(split_data["pscore"], dtype=np.float32)
    users = np.asarray(split_data["x_idx"], dtype=np.int64)
    reward = np.asarray(split_data["r"], dtype=np.float32)
    actions = np.asarray(split_data["a"], dtype=np.int64)
    pt = _policy_temperature(dataset)
    pi_e_at_position = _batched_pi_at_logged_actions(
        trial_x,
        trial_a,
        users,
        actions,
        chunk_size=score_lookup.user_chunk,
        action_chunk=score_lookup.action_chunk,
        policy_temperature=pt,
    )
    if not uses_importance_weighting(propensity_mode):
        value_vec = (reward * pi_e_at_position).astype(np.float32)
        ess = float(len(value_vec))
        return value_vec, ess

    ctx = score_lookup.user_context[users]
    q_hat_factual = np.asarray(
        score_lookup.regression_model.predict_pairs(ctx, actions),
        dtype=np.float32,
    )
    dm_reward = _dm_reward_rows_chunked(
        users, trial_x, trial_a, score_lookup, dataset
    ).astype(np.float32)
    iw = pi_e_at_position / (pscore + 1e-12)
    dr_vec = dm_reward + iw * (reward - q_hat_factual)
    ess = float((iw.sum() ** 2) / ((iw**2).sum() + 1e-12))
    return dr_vec, ess


def _policy_reward_from_embeddings(
    dataset,
    user_emb,
    item_emb,
    seed=12345,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
):
    pi_obj = Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=user_emb,
        item_emb=item_emb,
        emb_dim=int(dataset["emb_dim"]),
        temperature=_policy_temperature(dataset),
        user_chunk=user_chunk,
        action_chunk=action_chunk,
        rng=np.random.default_rng(seed),
    )
    return calc_reward(dataset, pi_obj, chunk_size=user_chunk)


def _dataset_log_constants(dataset, our_x, our_a):
    """CTR from env (oracle setting in sim) and exact true reward for logging embeddings."""
    env = dataset.get("env")
    ctr = float(getattr(env, "ctr", np.nan)) if env is not None else float("nan")
    initial_reward = float(_policy_reward_from_embeddings(dataset, our_x, our_a))
    return {"ctr": ctr, "initial_reward": initial_reward}


# --------------------------------------------------------------------
# Data preparation utility: generate train/val split
# (uses your signature of create_simulation_data_from_pi)
# --------------------------------------------------------------------
def generate_train_val_split(dataset, pi_0, train_size, val_size, run, user_context):
    """
    Generates synthetic bandit data and splits into train/validation sets.

    dataset: dict
    pi_0: (n_users, n_actions) behavior policy
    user_context: np.ndarray, shape (n_users, d)
    """
    n_actions = dataset["n_actions"]

    simulation_data = create_simulation_data_from_pi(
        dataset,
        pi_0,
        train_size + val_size,
        random_state=(run + 1) * (train_size + 17),
    )

    idx_train = np.arange(train_size)
    idx_val = np.arange(val_size) + train_size

    train_data = get_train_data(
        n_actions, train_size, simulation_data, idx_train, user_context
    )

    val_data = get_train_data(
        n_actions, val_size, simulation_data, idx_val, user_context
    )

    return train_data, val_data


# --------------------------------------------------------------------
# Evaluation wrapper: your original get_trial_results, unchanged
# except that for regression trainer we will pass a wrapper as
# `neighberhoodmodel` so eval_policy works correctly.
# --------------------------------------------------------------------
def _policy_probs_for_contexts(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    contexts: np.ndarray,
    policy_temperature: float,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    show_progress: bool = False,
) -> np.ndarray:
    """Softmax pi(a|x) for val rows only; peak (user_chunk, action_chunk)."""
    contexts = np.asarray(contexts, dtype=np.float32)
    item_emb = np.asarray(item_emb, dtype=np.float32)
    n = int(contexts.shape[0])
    n_actions = int(item_emb.shape[0])
    pt = max(float(policy_temperature), 1e-8)
    out = np.zeros((n, n_actions), dtype=np.float32)
    for us in range(0, n, user_chunk):
        ue = min(n, us + user_chunk)
        u_emb = contexts[us:ue]
        for a0, a1, pi in _iter_full_softmax_action_blocks(
            u_emb,
            item_emb,
            pt,
            action_chunk,
            desc="policy probs",
            n_rows=ue - us,
        ):
            out[us:ue, a0:a1] = pi
    return np.expand_dims(out, -1)


def get_trial_results(
    our_x,
    our_a,
    emb_x,
    emb_a,
    original_x,
    original_a,
    dataset,
    val_data,
    original_policy_prob,
    neighberhoodmodel,
    regression_model,
    dm,
    policy_reward_mode: str = "exact",
    policy_reward_mc_sim: int = 8,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
):
    t0 = time.time()
    uids = np.asarray(val_data["x_idx"], dtype=np.int64)
    xw = np.asarray(our_x[uids], dtype=np.float32)
    pt = max(_policy_temperature(dataset), 1e-8)
    policy_val = _policy_probs_for_contexts(
        our_x, our_a, xw, pt, user_chunk=user_chunk, action_chunk=action_chunk
    )
    policy_object = Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=int(dataset["emb_dim"]),
        temperature=_policy_temperature(dataset),
        user_chunk=user_chunk,
        action_chunk=action_chunk,
        rng=np.random.default_rng(12345),
    )
    if policy_reward_mode == "mc":
        policy_reward = calc_reward_mc(
            dataset,
            policy_object,
            n_sim=max(1, int(policy_reward_mc_sim)),
        )
    else:
        policy_reward = calc_reward(dataset, policy_object)
    print(f"Policy reward time: {time.time() - t0} seconds")
    # eval_policy expects model.predict(x_idx)
    eval_metrics = eval_policy(neighberhoodmodel, val_data, original_policy_prob, policy_val)

    action_diff_to_real = np.sqrt(np.mean((emb_a - our_a) ** 2))
    action_delta = np.sqrt(np.mean((original_a - our_a) ** 2))
    context_diff_to_real = np.sqrt(np.mean((emb_x - our_x) ** 2))
    context_delta = np.sqrt(np.mean((original_x - our_x) ** 2))

    t0 = time.time()
    row = np.concatenate(
        [
            np.atleast_1d(policy_reward),
            np.atleast_1d(eval_metrics),
            np.atleast_1d(action_diff_to_real),
            np.atleast_1d(action_delta),
            np.atleast_1d(context_diff_to_real),
            np.atleast_1d(context_delta),
        ]
    )

    q_hat_val = predict_regression_qhat_users(
        regression_model,
        our_x,
        uids,
        user_chunk=user_chunk,
        action_chunk=action_chunk,
    )
    reg_dm = dm.estimate_policy_value(policy_val, q_hat_val)
    print(f"Reg DM time: {time.time() - t0} seconds")
    reg_results = np.array([reg_dm])
    conv_results = np.array([row])

    print(f"Evaluation total results time: {time.time() - t0:.2f} seconds")
    return get_opl_results_dict(reg_results, conv_results)


# --------------------------------------------------------------------
#  NEIGHBORHOOD MODEL TRAINER (MODULAR)
# --------------------------------------------------------------------
def neighberhoodmodel_trainer_trial(
    num_neighbors,
    train_sizes,
    dataset,
    batch_size,
    val_size=None,
    val_frac=0.15,
    val_min=5000,
    val_max=None,
    n_trials=20,
    prev_best_params=None,
    propensity_mode="logged",
    log_paths: dict | None = None,
    method_label: str = "neighborhood",
    slim: bool = False,
    search_use_log_trick: bool = True,
    use_log_trick_fixed: bool | None = None,
    optuna_batch_sizes: list[int] | None = None,
):

    device = _training_device()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    trial_batch_choices = _normalize_optuna_batch_sizes(
        optuna_batch_sizes, default=DEFAULT_NEIGHBORHOOD_OPTUNA_BATCH_SIZES
    )

    dm = DM()
    results = {}
    best_hyperparams_by_size = {}
    last_best_params = prev_best_params if prev_best_params is not None else None

    # ===== Unpack dataset =====
    our_x_orig = dataset["our_x"]
    our_a_orig = dataset["our_a"]
    emb_x = dataset["emb_x"]
    emb_a = dataset["emb_a"]
    original_x = dataset["original_x"]
    original_a = dataset["original_a"]
    n_users = dataset["n_users"]
    n_actions = dataset["n_actions"]
    emb_dim = dataset["emb_dim"]

    _log_constants = _dataset_log_constants(dataset, our_x_orig, our_a_orig)

    all_user_indices = np.arange(n_users, dtype=np.int64)
    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

    train_sizes_list = [int(x) for x in train_sizes]
    base_train = min(train_sizes_list) if train_sizes_list else 10_000
    v_baseline = resolve_validation_size(
        base_train,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
    )

    # ===== baseline (sample size = 0) using get_trial_results =====
    simulation_data = _simulate_from_embedding_policy(
        dataset, our_x_orig, our_a_orig, v_baseline * 2, random_state=0
    )
    original_policy_prob = None

    # use same data for train/val just to generate the baseline row
    train_data = get_train_data(
        n_actions, v_baseline, simulation_data, np.arange(v_baseline), our_x_orig
    )
    val_data = get_train_data(
        n_actions, v_baseline, simulation_data, np.arange(v_baseline), our_x_orig
    )

    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a_orig,  # IMPORTANT: action embeddings, not user embeddings
        base_model=LogisticRegression(random_state=12345),
    )
    regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    print(f"Baseline regression model fit time: {time.time() - t0:.2f} seconds")

    t0 = time.time()
    neighberhoodmodel = NeighborhoodModel(
        train_data["x_idx"],
        train_data["a"],
        our_a_orig,
        our_x_orig,
        train_data["r"],
        num_neighbors=num_neighbors,
    )
    print(f"Baseline neighborhood model fit time: {time.time() - t0:.2f} seconds")

    # If you want baseline row:
    # results[0] = get_trial_results(
    #     our_x_orig, our_a_orig, emb_x, emb_a, original_x, original_a,
    #     dataset, val_data, original_policy_prob,
    #     neighberhoodmodel, regression_model, dm
    # )

    # ===== main loop over training sizes =====
    for train_size in train_sizes:
        trial_dicts_this_size = []
        best_hyperparams_by_size[train_size] = {}
        v = resolve_validation_size(
            int(train_size),
            val_size=val_size,
            val_frac=val_frac,
            val_min=val_min,
            val_max=val_max,
        )

        run = LOGGED_RUN_IDX
        print(f"\n=== [Neighborhood] Train size {train_size}, run {run} (val_size={v}) ===")

        # --- resample for this run ---
        simulation_data = _simulate_from_embedding_policy(
            dataset,
            our_x_orig,
            our_a_orig,
            int(train_size) + v,
            random_state=(run + 1) * (train_size + 17),
        )
        original_policy_prob = None

        part_seed = (run + 1) * (train_size + 17)
        train_idx, val_idx = _random_partition_indices(
            int(train_size) + v,
            (int(train_size), v),
            _partition_seed(part_seed),
        )
        train_data = get_train_data(
            n_actions, train_size, simulation_data, train_idx, our_x_orig
        )
        val_data = get_train_data(
            n_actions, v, simulation_data, val_idx, our_x_orig
        )

        num_workers = _dataloader_num_workers()

        cf_dataset = _build_cf_dataset(
            train_data=train_data,
            original_policy_prob=original_policy_prob,
            propensity_mode=propensity_mode,
        )

        # --- Optuna objective bound to this run's data ---
        def objective(trial):
            print()
            print(f"[Neighborhood] Trial {trial.number} started")
            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
            epochs = trial.suggest_int("num_epochs", 1, 10)
            trial_batch_size = trial.suggest_categorical(
                "batch_size", trial_batch_choices
            )
            trial_num_neighbors = trial.suggest_int("num_neighbors", 3, 15)
            lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)
            kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
            trial_use_log_trick = _resolve_trial_use_log_trick(
                trial, search_use_log_trick, use_log_trick_fixed
            )

            trial_neigh_model = NeighborhoodModel(
                train_data["x_idx"],
                train_data["a"],
                our_a_orig,
                our_x_orig,
                train_data["r"],
                num_neighbors=trial_num_neighbors,
            )

            trial_scores_all = torch.as_tensor(
                trial_neigh_model.predict(all_user_indices),
                device=device,
                dtype=torch.float32,
            )

            trial_model = LinearCFModel(
                n_users,
                n_actions,
                emb_dim,
                initial_user_embeddings=T(our_x_orig),
                initial_actions_embeddings=T(our_a_orig),
            ).to(device)

            assert (not torch.cuda.is_available()) or next(
                trial_model.parameters()
            ).is_cuda

            final_train_loader = DataLoader(
                cf_dataset,
                batch_size=trial_batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
                persistent_workers=bool(num_workers),
            )

            current_lr = lr
            for epoch in range(epochs):
                if epoch > 0:
                    current_lr *= lr_decay

                train(
                    trial_model,
                    final_train_loader,
                    trial_scores_all,
                    criterion=_kl_policy_loss(
                        kl_gamma,
                        use_log_trick=trial_use_log_trick,
                        propensity_mode=propensity_mode,
                    ),
                    num_epochs=1,
                    lr=current_lr,
                    device=str(device),
                )

            trial_x, trial_a = trial_model.get_params()
            trial_x = trial_x.detach().cpu().numpy()
            trial_a = trial_a.detach().cpu().numpy()

            train_actions = train_data["a"]
            train_users = train_data["x_idx"]
            pi_e_tr = _batched_pi_at_logged_actions(
                trial_x,
                trial_a,
                train_users,
                train_actions,
                action_chunk=DEFAULT_QHAT_ACTION_CHUNK,
                policy_temperature=_policy_temperature(dataset),
            )
            pscore_tr = np.asarray(train_data["pscore"], dtype=np.float32)

            print(
                "Train wi info: {}".format(
                    get_weights_info(pi_e_tr, pscore_tr)
                )
            )
            print(f"actual reward: {_policy_reward_from_embeddings(dataset, trial_x, trial_a)}")

            # validation reward for selection (you had cv_score_model)
            return cv_score_model(
                val_data,
                trial_scores_all,
                trial_x,
                trial_a,
                policy_temperature=_policy_temperature(dataset),
            )

        # --- run Optuna for this run ---
        study = optuna.create_study(direction="maximize")

        if last_best_params is not None:
            study.enqueue_trial(
                _enqueue_with_kl_gamma(
                    last_best_params,
                    search_use_log_trick=search_use_log_trick,
                    use_log_trick_fixed=use_log_trick_fixed,
                )
            )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params = _fix_best_params_use_log_trick(
            best_params,
            search_use_log_trick=search_use_log_trick,
            use_log_trick_fixed=use_log_trick_fixed,
        )
        last_best_params = best_params
        best_trial_number = (
            int(study.best_trial.number) if study.best_trial is not None else None
        )
        best_hyperparams_by_size[train_size][run] = {
            "params": best_params,
            "reward": study.best_value,
        }

        if log_paths is not None and log_paths.get("trials") is not None:
            _append_csv(
                log_paths["trials"],
                _study_trials_long(
                    study,
                    method_label=method_label,
                    train_size=int(train_size),
                    run_idx=int(run),
                    best_trial_number=best_trial_number,
                    ctr=_log_constants["ctr"],
                    initial_reward=_log_constants["initial_reward"],
                ),
            )

        # --- final training with best params on this run’s data ---
        regression_model = RegressionModel(
            n_actions=n_actions,
            action_context=our_a_orig,
            base_model=LogisticRegression(random_state=12345),
        )
        regression_model.fit(
            train_data["x"],
            train_data["a"],
            train_data["r"],
            np.asarray(train_data["pscore"], dtype=np.float32),
        )

        neighberhoodmodel = NeighborhoodModel(
            train_data["x_idx"],
            train_data["a"],
            our_a_orig,
            our_x_orig,
            train_data["r"],
            num_neighbors=best_params["num_neighbors"],
        )

        scores_all = torch.as_tensor(
            neighberhoodmodel.predict(all_user_indices),
            device=device,
            dtype=torch.float32,
        )

        model = LinearCFModel(
            n_users,
            n_actions,
            emb_dim,
            initial_user_embeddings=T(our_x_orig),
            initial_actions_embeddings=T(our_a_orig),
        ).to(device)
        assert (not torch.cuda.is_available()) or next(
            model.parameters()
        ).is_cuda

        train_loader = DataLoader(
            cf_dataset,
            batch_size=int(best_params.get("batch_size", batch_size)),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
        )

        current_lr = best_params["lr"]
        for epoch in range(best_params["num_epochs"]):
            if epoch > 0:
                current_lr *= best_params["lr_decay"]
            train(
                model,
                train_loader,
                scores_all,
                criterion=_kl_policy_loss(
                    best_params.get("kl_gamma", 0.05),
                    use_log_trick=bool(best_params.get("use_log_trick", True)),
                    propensity_mode=propensity_mode,
                ),
                num_epochs=1,
                lr=current_lr,
                device=str(device),
            )

        # learned embeddings (do NOT overwrite originals)
        learned_x_t, learned_a_t = model.get_params()
        learned_x = learned_x_t.detach().cpu().numpy()
        learned_a = learned_a_t.detach().cpu().numpy()

        if slim:
            trial_res = _slim_learned_policy_metrics(dataset, learned_x, learned_a)
        else:
            # --- produce the per-run result via get_trial_results ---
            trial_res = get_trial_results(
                learned_x,
                learned_a,  # learned (policy) embeddings
                emb_x,
                emb_a,  # ground-truth embedding refs
                original_x,
                original_a,  # original clean refs
                dataset,
                val_data,  # this run's val split
                None,
                neighberhoodmodel,
                regression_model,
                dm,
            )
        trial_res = {
            **trial_res,
            "val_size": float(v),
            "selection_val_score": float(study.best_value),
            **_log_constants,
        }

        trial_dicts_this_size.append(trial_res)

        # memory hygiene
        torch.cuda.empty_cache()

        # Primary metrics = run with best validation objective; *_runs_mean = mean across runs.
        agg, win_idx = _aggregate_runs_by_validation_score(trial_dicts_this_size)
        results[train_size] = agg

        if log_paths is not None and log_paths.get("runs") is not None:
            rows = []
            for ridx, d in enumerate(trial_dicts_this_size):
                row = {
                    "method": method_label,
                    "train_size": int(train_size),
                    "run": ridx,
                    "is_winning_run": bool(ridx == win_idx),
                }
                for k, v_ in d.items():
                    if str(k).startswith("_"):
                        continue
                    row[k] = _scalar_for_run_log(v_)
                # For slim mode, also attach weights info and actual reward for the winning run only.
                if slim and ridx == win_idx:
                    train_size_int = int(train_size)
                    v = resolve_validation_size(
                        train_size_int,
                        val_size=val_size,
                        val_frac=val_frac,
                        val_min=val_min,
                        val_max=val_max,
                    )
                    sim_seed = (ridx + 1) * (train_size_int + 17)
                    simulation_data = _simulate_from_embedding_policy(
                        dataset,
                        our_x_orig,
                        our_a_orig,
                        train_size_int + v,
                        random_state=sim_seed,
                    )
                    train_idx, _ = _random_partition_indices(
                        train_size_int + v,
                        (train_size_int, v),
                        _partition_seed(sim_seed),
                    )
                    train_data = get_train_data(
                        n_actions, train_size_int, simulation_data, train_idx, our_x_orig
                    )
                    train_actions = train_data["a"]
                    train_users = train_data["x_idx"]
                    pscore_tr = np.asarray(train_data.get("pscore"), dtype=np.float32)
                    _append_slim_winning_run_extras(
                        row,
                        d,
                        dataset=dataset,
                        our_x_orig=our_x_orig,
                        our_a_orig=our_a_orig,
                        train_users=train_users,
                        train_actions=train_actions,
                        pscore_tr=pscore_tr,
                    )
                rows.append(row)
            _append_csv(log_paths["runs"], pd.DataFrame(rows))

    return pd.DataFrame.from_dict(results, orient="index"), best_hyperparams_by_size


# --------------------------------------------------------------------
#  REGRESSION-BASED TRAINER (MODULAR)
# --------------------------------------------------------------------
def regression_trainer_trial(
    train_sizes,
    dataset,
    batch_size,
    val_size=None,
    val_frac=0.15,
    val_min=5000,
    val_max=None,
    n_trials=20,
    prev_best_params=None,
    propensity_mode="logged",
    log_paths: dict | None = None,
    method_label: str = "opc",
    slim: bool = False,
    policy_reward_mode: str = "exact",
    policy_reward_mc_sim: int = 8,
    split_cache: dict | None = None,
    policy_loss_types: tuple[str, ...] = ("kl_crm",),
    dataset_name: str | None = None,
    search_use_log_trick: bool = True,
    use_log_trick_fixed: bool | None = None,
    shared_regression_bundle: dict | None = None,
    shared_regression_size: int = 50_000,
    qhat_user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    qhat_action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    require_cuda: bool = False,
    optuna_batch_sizes: list[int] | None = None,
):
    """
    OPC / no-propensity trainer with Optuna over CF hyperparameters.

    ``slim``: always writes ``trials_long`` (all hyperparameters per trial). Skips only
    the heavy post-hoc ``get_trial_results`` pass (full-catalog reward + val DM/DR/IPW/SNDR).

    ``split_cache``: optional logged-split cache (``LazyRegressionSplitCache`` or a
    pre-built dict) so OPC and no-propensity see identical train/val data.

    ``policy_loss_types``: policy-gradient losses to try (``kl_crm``, ``kl``, ``ipw``,
    ``sndr``, ``crm``, ``naive``); if more than one, Optuna picks per trial. Default
    ``kl_crm`` is the unified SNDR log-trick + KL + CRM variance objective.

    ``search_use_log_trick``: if False, always use direct-prob surrogate (no log trick)
    for applicable losses and do not tune ``use_log_trick`` in Optuna.
    """
    policy_loss_types = tuple(str(x).lower() for x in policy_loss_types)
    for name in policy_loss_types:
        if name not in VALID_POLICY_LOSSES:
            raise ValueError(f"Unknown policy loss '{name}'")

    device = _training_device(require_cuda=require_cuda)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    _log_training_device(method_label, device)
    trial_batch_choices = _normalize_optuna_batch_sizes(optuna_batch_sizes)

    dm = DM()
    results = {}
    best_hyperparams_by_size = {}
    last_trials_export = pd.DataFrame()
    last_best_params = prev_best_params if prev_best_params is not None else None
    last_optuna_study = None

    # ===== Unpack dataset =====
    our_x_orig = dataset["our_x"]
    our_a_orig = dataset["our_a"]
    emb_x = dataset["emb_x"]
    emb_a = dataset["emb_a"]
    original_x = dataset["original_x"]
    original_a = dataset["original_a"]
    n_users = dataset["n_users"]
    n_actions = dataset["n_actions"]
    emb_dim = dataset["emb_dim"]

    _log_constants = _dataset_log_constants(
        dataset, our_x_orig, our_a_orig
    )

    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

    train_sizes_list = [int(x) for x in train_sizes]
    base_train = min(train_sizes_list) if train_sizes_list else 10_000
    v_baseline = resolve_validation_size(
        base_train,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
    )

    if shared_regression_bundle is None:
        if split_cache is not None and train_sizes_list:
            _warm = split_cache[(int(min(train_sizes_list)), 0)]
            shared_regression_bundle = fit_shared_regression_bundle(
                dataset,
                _warm["reg_data"],
                user_chunk=qhat_user_chunk,
                action_chunk=qhat_action_chunk,
            )
        else:
            _v0 = resolve_validation_size(
                int(base_train),
                val_size=val_size,
                val_frac=val_frac,
                val_min=val_min,
                val_max=val_max,
            )
            _split0 = _build_regression_logged_split(
                dataset,
                our_x_orig,
                our_a_orig,
                int(base_train),
                _v0,
                0,
                split_seed=12345,
                regression_size=int(shared_regression_size),
            )
            shared_regression_bundle = fit_shared_regression_bundle(
                dataset,
                _split0["reg_data"],
                user_chunk=qhat_user_chunk,
                action_chunk=qhat_action_chunk,
            )
    shared_regression_model = shared_regression_bundle["regression_model"]
    shared_scores_all_t = _scores_lookup_from_bundle(shared_regression_bundle, device)

    # ===== Baseline row =====
    if split_cache is not None and (0, 0) in split_cache:
        _base = split_cache[(0, 0)]
        train_data = _base["train_data"]
        val_data = _base["val_data"]
        v_baseline = int(_base["val_size"])
    else:
        simulation_data = _simulate_from_embedding_policy(
            dataset, our_x_orig, our_a_orig, v_baseline + v_baseline, random_state=0
        )
        train_idx, val_idx = _random_partition_indices(
            v_baseline + v_baseline,
            (v_baseline, v_baseline),
            _partition_seed(0),
        )
        train_data = get_train_data(
            n_actions, v_baseline, simulation_data, train_idx, our_x_orig
        )
        val_data = get_train_data(
            n_actions, v_baseline, simulation_data, val_idx, our_x_orig
        )
    original_policy_prob = None

    regression_model = shared_regression_model

    # wrap for eval_policy
    wrapped_reg_model = IndexToContextModelWrapper(regression_model, our_x_orig)

    if slim:
        results[0] = _nan_posthoc_eval_metrics()
        results[0]["policy_rewards"] = float(_log_constants["initial_reward"])
        results[0]["actual_reward_selected"] = float(_log_constants["initial_reward"])
    else:
        results[0] = get_trial_results(
            our_x_orig,
            our_a_orig,
            emb_x,
            emb_a,
            original_x,
            original_a,
            dataset,
            val_data,
            None,
            wrapped_reg_model,  # for eval_policy
            regression_model,  # for reg_dm
            dm,
            policy_reward_mode=policy_reward_mode,
            policy_reward_mc_sim=policy_reward_mc_sim,
        )
    results[0]["val_size"] = float(v_baseline)
    results[0].update(_log_constants)
    results[0].update(
        enrich_summary_pct_fields({**results[0], **_log_constants})
    )

    # ===== Main loop over training sizes =====
    for train_size in train_sizes:
        trial_dicts_this_size = []
        best_hyperparams_by_size[train_size] = {}
        v = resolve_validation_size(
            int(train_size),
            val_size=val_size,
            val_frac=val_frac,
            val_min=val_min,
            val_max=val_max,
        )

        run = LOGGED_RUN_IDX
        print(f"\n=== [Regression] Training size {train_size}, run {run} (val_size={v}) ===")

        if split_cache is not None and (int(train_size), int(run)) in split_cache:
            _bundle = split_cache[(int(train_size), int(run))]
            reg_data = _bundle["reg_data"]
            train_data = _bundle["train_data"]
            val_data = _bundle["val_data"]
        else:
            _split = _build_regression_logged_split(
                dataset,
                our_x_orig,
                our_a_orig,
                int(train_size),
                v,
                run,
                regression_size=int(shared_regression_size),
            )
            reg_data = _split["reg_data"]
            train_data = _split["train_data"]
            val_data = _split["val_data"]

        original_policy_prob = None
        cf_dataset = _build_cf_dataset(
            train_data=train_data,
            original_policy_prob=None,
            propensity_mode=propensity_mode,
        )

        num_workers = _dataloader_num_workers()

        # --- Define Optuna objective ---
        def objective(trial):
            print(f"\n[Regression] Optuna Trial {trial.number}")
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            epochs = trial.suggest_int("num_epochs", 5, 25)
            trial_batch_size = trial.suggest_categorical(
                "batch_size", trial_batch_choices
            )
            lr_decay = trial.suggest_float("lr_decay", 1e-5, 1e-3, log=True)
            if _policy_loss_needs_kl(policy_loss_types):
                kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
            else:
                kl_gamma = 0.05
            if _policy_loss_needs_crm(policy_loss_types):
                crm_M = trial.suggest_float("crm_M", 1.0, 100.0, log=True)
                crm_lambda = trial.suggest_float("crm_lambda", 1e-4, 10.0, log=True)
            else:
                crm_M = 10.0
                crm_lambda = 1.0
            trial_use_log_trick = _resolve_trial_use_log_trick(
                trial, search_use_log_trick, use_log_trick_fixed
            )
            if len(policy_loss_types) > 1:
                trial_policy_loss = trial.suggest_categorical(
                    "policy_loss", list(policy_loss_types)
                )
            else:
                trial_policy_loss = policy_loss_types[0]

            trial_scores_all = shared_scores_all_t

            # Initialize CF model
            trial_model = CFModel(
                n_users,
                n_actions,
                emb_dim,
                initial_user_embeddings=T(our_x_orig),
                initial_actions_embeddings=T(our_a_orig),
                user_transform=SingleMLPTransform(emb_dim),
                action_transform=SingleMLPTransform(emb_dim),
            ).to(device)

            final_train_loader = DataLoader(
                cf_dataset,
                batch_size=trial_batch_size,
                shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
                persistent_workers=bool(num_workers),
            )

            current_lr = lr
            for epoch in range(epochs):
                if epoch > 0:
                    current_lr *= lr_decay
                train(
                    trial_model,
                    final_train_loader,
                    trial_scores_all,
                    criterion=_policy_loss_from_name(
                        trial_policy_loss,
                        kl_gamma=kl_gamma,
                        clip_m=crm_M,
                        crm_lambda=crm_lambda,
                        use_log_trick=trial_use_log_trick,
                        propensity_mode=propensity_mode,
                    ),
                    num_epochs=1,
                    lr=current_lr,
                    device=str(device),
                )

            # Evaluate validation score
            trial_x, trial_a = trial_model.get_params()
            trial_x, trial_a = (
                trial_x.detach().cpu().numpy(),
                trial_a.detach().cpu().numpy(),
            )
            r = _policy_reward_from_embeddings(dataset, trial_x, trial_a)
            print(
                f"actual reward: {r}"
            )
            dr_vec, ess_val = _split_dr_vec_and_ess(
                val_data,
                trial_x,
                trial_a,
                trial_scores_all,
                dataset,
                propensity_mode=propensity_mode,
            )
            dr_vec_tr, ess_train = _split_dr_vec_and_ess(
                train_data,
                trial_x,
                trial_a,
                trial_scores_all,
                dataset,
                propensity_mode=propensity_mode,
            )
            n = max(len(dr_vec), 2)
            r_hat = float(dr_vec.mean())
            r_hat_train = float(dr_vec_tr.mean())
            err = float(dr_vec.std(ddof=1) / np.sqrt(n))
            tcrit = float(student_t.ppf(0.975, n - 1))
            value = r_hat - tcrit * err

            trial.set_user_attr("all_values", [r_hat, err, value])
            trial.set_user_attr(
                "scores_dict",
                {"r_hat": r_hat, "r_hat_train": r_hat_train, "se": err, "ci_low": value},
            )
            trial.set_user_attr("r_hat", r_hat)
            trial.set_user_attr("r_hat_train", r_hat_train)
            trial.set_user_attr("q_error", err)
            trial.set_user_attr("actual_reward", r)
            trial.set_user_attr("ess", ess_val)
            trial.set_user_attr("ess_train", ess_train)

            return value

        # --- Run Optuna search ---
        study = optuna.create_study(direction="maximize")
        if last_best_params is not None:
            study.enqueue_trial(
                _enqueue_with_kl_gamma(
                    last_best_params,
                    policy_loss_types=policy_loss_types,
                    search_use_log_trick=search_use_log_trick,
                    use_log_trick_fixed=use_log_trick_fixed,
                )
            )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        last_optuna_study = study

        best_params = study.best_params
        if "policy_loss" not in best_params and len(policy_loss_types) == 1:
            best_params = {**best_params, "policy_loss": policy_loss_types[0]}
        best_params = _fix_best_params_use_log_trick(
            best_params,
            search_use_log_trick=search_use_log_trick,
            use_log_trick_fixed=use_log_trick_fixed,
        )
        last_best_params = best_params
        best_trial_number = (
            int(study.best_trial.number) if study.best_trial is not None else None
        )
        best_hyperparams_by_size[train_size][run] = {
            "params": best_params,
            "reward": study.best_value,
        }

        if log_paths is not None and log_paths.get("trials") is not None:
            last_trials_export = _study_trials_long(
                study,
                method_label=method_label,
                train_size=int(train_size),
                run_idx=int(run),
                best_trial_number=best_trial_number,
                ctr=_log_constants["ctr"],
                initial_reward=_log_constants["initial_reward"],
                dataset_name=dataset_name,
            )
            _append_csv(log_paths["trials"], last_trials_export)

        # --- Final training with best params ---
        regression_model = shared_regression_model
        scores_all = shared_scores_all_t

        model = CFModel(
            n_users,
            n_actions,
            emb_dim,
            initial_user_embeddings=T(our_x_orig),
            initial_actions_embeddings=T(our_a_orig),
            user_transform=SingleMLPTransform(emb_dim),
            action_transform=SingleMLPTransform(emb_dim),
        ).to(device)

        train_loader = DataLoader(
            cf_dataset,
            batch_size=int(best_params.get("batch_size", batch_size)),
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
        )

        current_lr = best_params["lr"]
        for epoch in range(best_params["num_epochs"]):
            if epoch > 0:
                current_lr *= best_params["lr_decay"]
            train(
                model,
                train_loader,
                scores_all,
                criterion=_policy_loss_from_name(
                    best_params.get("policy_loss", policy_loss_types[0]),
                    kl_gamma=best_params.get("kl_gamma", 0.05),
                    clip_m=best_params.get("crm_M", 10.0),
                    crm_lambda=best_params.get("crm_lambda", 1.0),
                    use_log_trick=bool(best_params.get("use_log_trick", True)),
                    propensity_mode=propensity_mode,
                ),
                num_epochs=1,
                lr=current_lr,
                device=str(device),
            )

        learned_x_t, learned_a_t = model.get_params()
        learned_x = learned_x_t.detach().cpu().numpy()
        learned_a = learned_a_t.detach().cpu().numpy()

        wrapped_reg_model = IndexToContextModelWrapper(
            regression_model, our_x_orig
        )

        if slim:
            trial_res = _slim_learned_policy_metrics(dataset, learned_x, learned_a)
        else:
            trial_res = get_trial_results(
                learned_x,
                learned_a,
                emb_x,
                emb_a,
                original_x,
                original_a,
                dataset,
                val_data,
                None,
                wrapped_reg_model,
                regression_model,
                dm,
                policy_reward_mode=policy_reward_mode,
                policy_reward_mc_sim=policy_reward_mc_sim,
            )
        trial_res = {
            **trial_res,
            "val_size": float(v),
            "selection_val_score": float(study.best_value),
            "policy_loss": str(
                best_params.get("policy_loss", policy_loss_types[0])
            ),
            **_log_constants,
        }
        trial_res.update(enrich_summary_pct_fields(trial_res))

        trial_dicts_this_size.append(trial_res)
        torch.cuda.empty_cache()

        # Primary metrics = run with best validation objective; *_runs_mean = mean across runs.
        agg, win_idx = _aggregate_runs_by_validation_score(trial_dicts_this_size)
        results[train_size] = agg

        if log_paths is not None and log_paths.get("runs") is not None:
            rows = []
            for ridx, d in enumerate(trial_dicts_this_size):
                row = {
                    "dataset": str(dataset_name) if dataset_name is not None else "",
                    "method": method_label,
                    "train_size": int(train_size),
                    "run": ridx,
                    "is_winning_run": bool(ridx == win_idx),
                }
                for k, v_ in d.items():
                    if str(k).startswith("_"):
                        continue
                    row[k] = _scalar_for_run_log(v_)
                if slim and ridx == win_idx:
                    train_size_int = int(train_size)
                    if (
                        split_cache is not None
                        and (train_size_int, ridx) in split_cache
                    ):
                        train_data = split_cache[(train_size_int, ridx)]["train_data"]
                    else:
                        v = resolve_validation_size(
                            train_size_int,
                            val_size=val_size,
                            val_frac=val_frac,
                            val_min=val_min,
                            val_max=val_max,
                        )
                        sim_seed = (ridx + 1) * (train_size_int + 17)
                        simulation_data = _simulate_from_embedding_policy(
                            dataset,
                            our_x_orig,
                            our_a_orig,
                            train_size_int + v,
                            random_state=sim_seed,
                        )
                        train_idx, _ = _random_partition_indices(
                            train_size_int + v,
                            (train_size_int, v),
                            _partition_seed(sim_seed),
                        )
                        train_data = get_train_data(
                            n_actions,
                            train_size_int,
                            simulation_data,
                            train_idx,
                            our_x_orig,
                        )
                    train_actions = train_data["a"]
                    train_users = train_data["x_idx"]
                    pscore_tr = np.asarray(train_data.get("pscore"), dtype=np.float32)
                    _append_slim_winning_run_extras(
                        row,
                        d,
                        dataset=dataset,
                        our_x_orig=our_x_orig,
                        our_a_orig=our_a_orig,
                        train_users=train_users,
                        train_actions=train_actions,
                        pscore_tr=pscore_tr,
                    )
                rows.append(row)
            _append_csv(log_paths["runs"], pd.DataFrame(rows))

    # opc_trials.csv: same hyperparameter columns as trials_long (last run).
    trial_df = (
        last_trials_export
        if not last_trials_export.empty
        else pd.DataFrame()
    )

    return pd.DataFrame.from_dict(results, orient="index"), trial_df


def no_propensity_trainer_trial(
    train_sizes,
    dataset,
    batch_size,
    val_size=None,
    val_frac=0.15,
    val_min=5000,
    val_max=None,
    n_trials=20,
    prev_best_params=None,
    log_paths: dict | None = None,
    method_label: str = "no_propensity",
    policy_reward_mode: str = "exact",
    policy_reward_mc_sim: int = 8,
    slim: bool = False,
    split_cache: dict | None = None,
    policy_loss_types: tuple[str, ...] = ("naive",),
    dataset_name: str | None = None,
    search_use_log_trick: bool = False,
    use_log_trick_fixed: bool | None = False,
    shared_regression_bundle: dict | None = None,
    shared_regression_size: int = 50_000,
    qhat_user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    qhat_action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
    require_cuda: bool = False,
    optuna_batch_sizes: list[int] | None = None,
):
    """
    Explicit no-propensity baseline with parity to regression trainer:
    same model family, same search budget, same train/val splits.

    Uses pure naive reward ``mean(r * pi)`` (no DM/SNDR/IW/KL/CRM, no log-trick).
    """
    return regression_trainer_trial(
        train_sizes=train_sizes,
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        val_frac=val_frac,
        val_min=val_min,
        val_max=val_max,
        n_trials=n_trials,
        prev_best_params=prev_best_params,
        propensity_mode="uniform",
        log_paths=log_paths,
        method_label=method_label,
        policy_reward_mode=policy_reward_mode,
        policy_reward_mc_sim=policy_reward_mc_sim,
        slim=slim,
        split_cache=split_cache,
        policy_loss_types=policy_loss_types,
        dataset_name=dataset_name,
        search_use_log_trick=search_use_log_trick,
        use_log_trick_fixed=use_log_trick_fixed,
        shared_regression_bundle=shared_regression_bundle,
        shared_regression_size=shared_regression_size,
        qhat_user_chunk=qhat_user_chunk,
        qhat_action_chunk=qhat_action_chunk,
        require_cuda=require_cuda,
        optuna_batch_sizes=optuna_batch_sizes,
    )


class MLPScoresLookup:
    """On-demand q_hat from MLPRewardModel (user batches; model chunks actions internally)."""

    def __init__(
        self,
        reward_model,
        user_context: np.ndarray,
        device,
        *,
        user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    ):
        self.reward_model = reward_model
        self.user_context = np.asarray(user_context)
        self.device = device
        self.user_chunk = int(user_chunk)
        self.action_chunk = DEFAULT_QHAT_ACTION_CHUNK

    def __getitem__(self, user_idx):
        if isinstance(user_idx, torch.Tensor):
            user_idx = user_idx.detach().cpu().numpy()
        user_idx = np.asarray(user_idx, dtype=np.int64).reshape(-1)
        n = len(user_idx)
        n_actions = int(self.reward_model.n_actions)
        out = np.zeros((n, n_actions), dtype=np.float32)
        for us in range(0, n, self.user_chunk):
            ue = min(n, us + self.user_chunk)
            q = self.reward_model.predict(self.user_context[user_idx[us:ue]])
            q = np.asarray(q, dtype=np.float32)
            if q.ndim == 3:
                q = q[:, :, 0]
            out[us:ue] = q
        return torch.as_tensor(out, device=self.device, dtype=torch.float32)


# --------------------------------------------------------------------
# Random / oracle / baseline mixture policies live in policies.py.
# This file only consumes policy objects.
# --------------------------------------------------------------------

def random_policy_trainer_trial(
    train_size: int,
    dataset: dict,
    n_policies: int = 50,
    val_size: int = 2000,
    use_random: bool = True,
    use_oracle: bool = True,
    jaws: bool = False,
    n_bootstrap: int = 500,
    n_dm_mc: int = 32,
    chunk_size=2048,
    seed: int = 12345,
):
    """
    Evaluate a set of mixture policies at scale (exact full-softmax over all actions).

    Requirements:
      - dataset contains:
          our_x, our_a, emb_x, emb_a, n_actions, n_users, env
        optionally:
          user_prior (length n_users) to sample users in simulation
      - create_simulation_data_from_policy(policy=...) logs (x_idx, a, r, pscore, x)
      - score_model_modular_large consumes policy objects with:
          sample_actions(users), prob_actions(users, actions)
      - RegressionModel has predict_pairs fixed to use base_model_list like predict()
    """

    # ----- unpack -----
    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    emb_x = dataset["emb_x"]
    emb_a = dataset["emb_a"]
    n_actions = int(dataset["n_actions"])
    n_users = int(dataset["n_users"])

    policy_temp = float(dataset.get("policy_temperature", 1.0))

    # ----- policies (exact full softmax over all items) -----
    pi0_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=our_x.shape[1],
        temperature=policy_temp,
        user_chunk=chunk_size,
        rng=np.random.default_rng(seed + 1),
    )

    oracle_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=emb_x,
        item_emb=emb_a,
        emb_dim=emb_x.shape[1],
        temperature=policy_temp,
        user_chunk=chunk_size,
        rng=np.random.default_rng(seed + 2),
    )

    # noise policy: random per run, fixed within run; dim=1 (valid as “random latent factor”)
    noise_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=None,
        item_emb=None,
        emb_dim=1,
        temperature=policy_temp,
        user_chunk=chunk_size,
        rng=np.random.default_rng(seed + 3),
    )

    # ----- simulate logged data using logging policy -----
    simulation_data = create_simulation_data_from_policy(
        dataset=dataset,
        policy=pi0_policy,
        n_samples=int(train_size + val_size),
        random_state=int(seed + train_size + 17),
    )

    idx_train = np.arange(train_size, dtype=np.int64)
    train_data = get_train_data(n_actions, train_size, simulation_data, idx_train, our_x)

    idx_val = np.arange(val_size, dtype=np.int64) + train_size
    val_data = get_train_data(n_actions, val_size, simulation_data, idx_val, our_x)

    # ----- fit Q model (pairwise usage in scorer) -----
    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a,
        base_model=LogisticRegression(random_state=seed),
    )
    regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    print(f"[Regression] fit time: {time.time() - t0:.2f}s")
    # t0 = time.time()
    # regression_model = MLPRewardModel(
    #     n_actions=n_actions,
    #     action_context=our_a,      # (n_actions, d_action)
    #     hidden_dims=[64, 16],            # one hidden layer
    #     dropout=0.2,
    #     epochs=15,
    #     lr=1e-3,
    #     batch_size=8192,
    #     device="cuda",             # or "cpu"
    # )

    # regression_model.fit(
    #     context=train_data["x"],   # (n_rounds, d_context)
    #     action=train_data["a"],    # (n_rounds,)
    #     reward=train_data["r"],    # (n_rounds,)
    # )
    # print(f"[Regression-MLP] fit time: {time.time() - t0:.2f}s")

    # ----- generate mixture policies (same alpha/beta/jaws logic) -----
    policies = generate_policies(
        num_policies=n_policies,
        base_policy=pi0_policy,
        oracle_policy=oracle_policy,
        noise_policy=noise_policy,
        use_random=use_random,
        use_oracle=use_oracle,
        jaws=jaws,
        seed=seed + 999,
    )

    # ----- results df -----
    df = pd.DataFrame(
        columns=[
            "value",
            "user_attrs_actual_reward",
            "user_attrs_q_error",
            "user_attrs_r_hat",
            "user_attrs_ess",
            "user_attrs_scores_dict",
            "user_attrs_all_values",
            "ipw",
            "sign_uni",
            "sign_exp",
        ]
    )

    tq = tqdm(policies)
    for pi_i in tq:
        scores_dict, scores_array, weight_info = score_model_modular_large(
            val_dataset=val_data,
            regression_model=regression_model,
            policy=pi_i,
            lam_dr=3.0,
            n_bootstrap=n_bootstrap,
            n_dm_mc=n_dm_mc,
            random_state=seed + 123,
        )

        value = float(scores_dict["dr_naive_ci_low"])
        r_hat = float(scores_dict["dr_naive_mean"])
        err = float(scores_dict["dr_naive_se"])

        # "actual" (MC) on the validation users
        r_actual = calc_reward(dataset, pi_i, chunk_size=chunk_size)

        df.loc[len(df)] = {
            "value": value,
            "user_attrs_actual_reward": r_actual,
            "user_attrs_q_error": err,
            "user_attrs_r_hat": r_hat,
            "user_attrs_ess": float(weight_info["ess"]),
            "user_attrs_scores_dict": scores_dict,
            "user_attrs_all_values": scores_array,
            "ipw": float(scores_dict["ipw_uni_mean"]),
            "sign_uni": float(scores_dict["cv_signed_rmse_uniform"]),
            "sign_exp": float(scores_dict["cv_signed_rmse_exp"]),
        }

        tq.set_description(
            f"ESS={weight_info['ess']:.2f} "
            f"(max_wi, min_wi)=({weight_info['max_wi']:.2f}, {weight_info['min_wi']:.2f}) "
            f"value={value:.4f}±{err:.4f} r_actual={r_actual:.4f}"
        )
        # compute_statistics_and_plots(df, full_plot=False)

    df = df[df["value"] > 0]
    return df, df


def predict_qhat_all_chunked(
    reward_model,
    user_contexts,
    *,
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    action_chunk: int = DEFAULT_QHAT_ACTION_CHUNK,
):
    """Deprecated: materializes (n_users, n_actions). Use RegressionScoresLookup instead."""
    if hasattr(reward_model, "base_model_list"):
        return predict_regression_qhat_users(
            reward_model,
            user_contexts,
            np.arange(user_contexts.shape[0], dtype=np.int64),
            user_chunk=user_chunk,
            action_chunk=action_chunk,
        )
    n_users = int(user_contexts.shape[0])
    n_actions = int(reward_model.n_actions)
    out = np.zeros((n_users, n_actions, 1), dtype=np.float32)
    for s in range(0, n_users, user_chunk):
        e = min(n_users, s + user_chunk)
        q = reward_model.predict(user_contexts[s:e])
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 2:
            q = q[..., None]
        out[s:e] = q
    return out


def mlp_trial_reward_fit_once(
    train_size: int,
    dataset: dict,
    val_size: int = 2000,
    n_trials: int = 20,
    reg_frac: float = 0.5,         # part of train used to fit reward model once
    user_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    qhat_chunk: int = DEFAULT_QHAT_USER_CHUNK,
    seed: int = 12345,
    # keep eval same as random_trial:
    lam_dr: float = 3.0,
    n_bootstrap: int = 1,
    n_dm_mc: int = 1,
    search_use_log_trick: bool = True,
):
    """
    Fit reward model ONCE, then Optuna tunes CF training only.
    Evaluation matches random_policy_trainer_trial (score_model_modular_large).
    """

    our_x = dataset["our_x"]
    our_a = dataset["our_a"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    emb_dim = int(dataset["emb_dim"])

    device = _training_device()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # ---------------------------
    # Logging policy (Policy object)
    # ---------------------------
    logging_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=our_x.shape[1],
        temperature=_policy_temperature(dataset),
        user_chunk=user_chunk,
        rng=np.random.default_rng(seed + 1),
    )

    initial_reward = calc_reward(dataset, logging_policy, chunk_size=user_chunk)
    print(f"Initial reward: {initial_reward:.6f}")
    # ---------------------------
    # Simulate logged data once
    # ---------------------------
    sim = create_simulation_data_from_policy(
        dataset=dataset,
        policy=logging_policy,
        n_samples=int(train_size + val_size),
        random_state=int(seed + train_size + 17),
    )

    idx_train = np.arange(train_size, dtype=np.int64)
    idx_val = np.arange(val_size, dtype=np.int64) + train_size

    # train_full = get_train_data(n_actions, train_size, sim, idx_train, our_x)
    val_data   = get_train_data(n_actions, val_size, sim, idx_val, our_x)

    # ---------------------------
    # Split train: reward-model fit vs CF fit
    # ---------------------------
    reg_size = int(reg_frac * train_size)
    reg_idx = np.arange(reg_size, dtype=np.int64)
    cf_idx  = np.arange(train_size - reg_size, dtype=np.int64) + reg_size

    reg_data = get_train_data(n_actions, reg_size, sim, reg_idx, our_x)
    cf_data  = get_train_data(n_actions, train_size - reg_size, sim, cf_idx, our_x)

    # ---------------------------
    # Fit reward model ONCE
    # ---------------------------
    t0 = time.time()
    reward_model = MLPRewardModel(
        n_actions=n_actions,
        action_context=our_a,
        device=str(device),
    )

    reward_model.fit(
        context=reg_data["x"],
        action=reg_data["a"],
        reward=reg_data["r"],
    )
    print(f"[MLPRewardModel] fit time: {time.time() - t0:.2f}s")
    # t0 = time.time()
    # reward_model = RegressionModel(
    #     n_actions=n_actions,
    #     action_context=our_a,  # IMPORTANT: action embeddings
    #     base_model=LogisticRegression(random_state=12345),
    # )
    # reward_model.fit(reg_data["x"], reg_data["a"], reg_data["r"])

    # print(f"[Regression] Baseline regression model fit time: {time.time() - t0:.2f}s")

    scores_all = MLPScoresLookup(
        reward_model, our_x, device, user_chunk=qhat_chunk
    )

    # ---------------------------
    # CF dataset + loader settings
    # ---------------------------
    cf_dataset = CustomCFDatasetPS(
        cf_data["x_idx"],
        cf_data["a"],
        cf_data["r"],
        cf_data["pscore"],   # <-- scalar propensity per logged sample
    )
    num_workers = _dataloader_num_workers()

    # ---------------------------
    # Optuna objective: CF only
    # ---------------------------
    def objective(trial):
        # lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        epochs = trial.suggest_int("num_epochs", 1, 10)
        hidden = trial.suggest_int("hidden", 4, 32)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        batch_size = trial.suggest_categorical(
            "batch_size", list(DEFAULT_OPTUNA_BATCH_SIZES)
        )
        lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)
        kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
        trial_use_log_trick = _resolve_trial_use_log_trick(trial, search_use_log_trick)

        model = CFModel(
            n_users,
            n_actions,
            emb_dim,
            initial_user_embeddings=torch.as_tensor(our_x, device=device, dtype=torch.float32),
            initial_actions_embeddings=torch.as_tensor(our_a, device=device, dtype=torch.float32),
            user_transform=SingleMLPTransform(emb_dim, hidden=hidden, dropout=dropout),
            action_transform=SingleMLPTransform(emb_dim, hidden=hidden, dropout=dropout),
        ).to(device)

        loader = DataLoader(
            cf_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
        )

        current_lr = lr
        for ep in range(epochs):
            if ep > 0:
                current_lr *= lr_decay
            train(
                model,
                loader,
                scores_all,
                criterion=_kl_policy_loss(kl_gamma, use_log_trick=trial_use_log_trick),
                num_epochs=1,
                lr=current_lr,
                device=str(device),
            )

        learned_x_t, learned_a_t = model.get_params()

        learned_policy = Policy(
            n_users=n_users,
            n_items=n_actions,
            user_emb=learned_x_t.detach().cpu().numpy(),
            item_emb=learned_a_t.detach().cpu().numpy(),
            emb_dim=emb_dim,
            temperature=float(dataset.get("policy_temperature", 1.0)),
            user_chunk=user_chunk,
            rng=np.random.default_rng(seed + 1000 + trial.number),
        )

        # evaluate exactly like random_trial
        scores_dict, scores_array, weight_info = score_model_modular_large(
            val_dataset=val_data,
            regression_model=reward_model,
            policy=learned_policy,
            lam_dr=lam_dr,
            n_bootstrap=n_bootstrap,
            n_dm_mc=n_dm_mc,
            random_state=seed + 123 + trial.number,
        )

        value = float(scores_dict["dr_naive_ci_low"])
        r_hat = float(scores_dict["dr_naive_mean"])
        err   = float(scores_dict["dr_naive_se"])
        ess   = float(weight_info["ess"])

        r_actual = calc_reward(dataset, learned_policy, chunk_size=user_chunk)

        trial.set_user_attr("all_values", scores_array)
        trial.set_user_attr("scores_dict", scores_dict)
        trial.set_user_attr("r_hat", r_hat)
        trial.set_user_attr("q_error", err)
        trial.set_user_attr("actual_reward", r_actual)
        trial.set_user_attr("ess", ess)

        del loader
        torch.cuda.empty_cache()

        print(f"actual reward={r_actual:6f}, score={value:6f}")
        return value
        # return r_actual

    # ---------------------------
    # Run Optuna
    # ---------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trial_df = study.trials_dataframe()[[
        "value",
        "user_attrs_actual_reward",
        "user_attrs_q_error",
        "user_attrs_r_hat",
        "user_attrs_ess",
        "user_attrs_scores_dict",
        "user_attrs_all_values",
    ]].copy()
    trial_df = trial_df[trial_df["value"] > 0]

    summary = {"best_value": float(study.best_value), **study.best_params}
    
    return pd.DataFrame([summary]), trial_df, initial_reward