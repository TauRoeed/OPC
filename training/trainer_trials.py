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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _dataloader_num_workers() -> int:
    """DataLoader worker processes. Default 0 avoids EMFILE when nested under
    ProcessPoolExecutor / many Optuna trials (each trial used num_workers>0 before).
    Set OPC_DATALOADER_WORKERS to a small integer (e.g. 2) to re-enable prefetch."""
    raw = os.environ.get("OPC_DATALOADER_WORKERS", "").strip()
    if raw != "":
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 0


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
    KLPolicyLoss,
)


def _kl_policy_loss(gamma: float, use_log_trick: bool = True) -> KLPolicyLoss:
    return KLPolicyLoss(gamma=float(gamma), use_log_trick=use_log_trick)

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
    """Robust mean over a list of dicts with numeric/array values."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [np.asarray(d[k]) for d in dicts if k in d]
        stacked = np.stack(vals, axis=0)
        out[k] = np.mean(stacked, axis=0)
    return out


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
                "param_use_log_trick": int(bool(params.get("use_log_trick", True))),
                "param_num_neighbors": int(params.get("num_neighbors", -1)),
                "is_best_in_run": bool(
                    best_trial_number is not None and t.number == best_trial_number
                ),
            }
        )
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


def _enqueue_with_kl_gamma(last_best: dict | None, kl_default: float = 0.05) -> dict | None:
    """Optuna enqueue compatibility when new search dims were added."""
    if last_best is None:
        return None
    merged = dict(last_best)
    merged.setdefault("kl_gamma", float(kl_default))
    merged.setdefault("use_log_trick", True)
    return merged


def _resolve_logged_pscore(train_data, original_policy_prob, mode="logged"):
    """
    Unify propensity handling across trial trainers.
    mode:
      - "logged": use behavior propensity from logged data.
      - "uniform": force pscore=1.0 (explicit no-propensity training).
    """
    if mode == "uniform":
        return np.ones_like(train_data["r"], dtype=np.float32)

    if "pscore" in train_data and train_data["pscore"] is not None:
        return np.asarray(train_data["pscore"], dtype=np.float32)

    return np.asarray(
        original_policy_prob[train_data["x_idx"], train_data["a"]].squeeze(),
        dtype=np.float32,
    )


def _build_cf_dataset(train_data, original_policy_prob, propensity_mode="logged"):
    pscore = _resolve_logged_pscore(
        train_data=train_data,
        original_policy_prob=original_policy_prob,
        mode=propensity_mode,
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


def predict_regression_qhat_chunked(
    regression_model, user_context, chunk_size: int = 256
):
    """RegressionModel.predict over users in chunks; returns float32 (n_users, n_actions, len_list)."""
    n_users = int(user_context.shape[0])
    n_list = int(regression_model.len_list)
    out = np.zeros((n_users, int(regression_model.n_actions), n_list), dtype=np.float32)
    for s in range(0, n_users, chunk_size):
        e = min(n_users, s + chunk_size)
        q = regression_model.predict(user_context[s:e])
        out[s:e] = np.asarray(q, dtype=np.float32)
    return out


def _batched_pi_at_logged_actions(
    user_emb,
    item_emb,
    user_ids,
    action_ids,
    chunk_size=4096,
    policy_temperature: float = 1.0,
):
    """Per-row pi(a_i|x_i) for logged (user_ids, action_ids) without full softmax matrix."""
    user_ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    action_ids = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    out = np.empty(len(user_ids), dtype=np.float32)
    xw = np.asarray(user_emb, dtype=np.float32)
    aw = np.asarray(item_emb, dtype=np.float32)
    pt = max(float(policy_temperature), 1e-8)
    for s in range(0, len(user_ids), chunk_size):
        e = min(len(user_ids), s + chunk_size)
        logits = (xw[user_ids[s:e]] @ aw.T) / pt
        prob = _softmax_action_probs_stable(logits, axis=1)
        loc = np.arange(e - s, dtype=np.int64)
        out[s:e] = prob[loc, action_ids[s:e]]
    return out


def cv_score_model(
    val_data,
    trial_scores_all,
    user_emb,
    item_emb,
    policy_temperature: float = 1.0,
):
    """
    Conservative validation score:
    r_hat - tdist * se
    Uses only validation rows (no dense n_users x n_actions policy matrix).
    """
    pscore = np.asarray(val_data["pscore"], dtype=np.float32)
    users = np.asarray(val_data["x_idx"], dtype=np.int64)
    reward = np.asarray(val_data["r"], dtype=np.float32)
    actions = np.asarray(val_data["a"], dtype=np.int64)

    scores = np.asarray(trial_scores_all.detach().cpu().numpy(), dtype=np.float32).squeeze()
    xw = np.asarray(user_emb, dtype=np.float32)
    aw = np.asarray(item_emb, dtype=np.float32)
    pt = max(float(policy_temperature), 1e-8)
    logits = (xw[users] @ aw.T) / pt
    pi_val = _softmax_action_probs_stable(logits, axis=1)
    scores_val = scores[users]
    loc = np.arange(len(users), dtype=np.int64)
    pi_e_at_position = np.asarray(pi_val[loc, actions].squeeze(), dtype=np.float32)
    iw = pi_e_at_position / (pscore + 1e-12)
    q_hat_factual = np.asarray(scores_val[loc, actions].squeeze(), dtype=np.float32)
    dm_reward = np.asarray((scores_val * pi_val).sum(axis=1), dtype=np.float32)

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
    trial_scores_all: torch.Tensor,
    dataset: dict,
) -> tuple[np.ndarray, float]:
    """DR per-row vector and IW ESS on a logged split (train or val)."""
    pscore = np.asarray(split_data["pscore"], dtype=np.float32)
    users = np.asarray(split_data["x_idx"], dtype=np.int64)
    reward = np.asarray(split_data["r"], dtype=np.float32)
    actions = np.asarray(split_data["a"], dtype=np.int64)
    tx = np.asarray(trial_x, dtype=np.float32)
    ta = np.asarray(trial_a, dtype=np.float32)
    pt = max(_policy_temperature(dataset), 1e-8)
    pi_val = _softmax_action_probs_stable((tx[users] @ ta.T) / pt, axis=1)
    scores_val = np.asarray(
        trial_scores_all[users].detach().cpu().numpy(),
        dtype=np.float32,
    ).squeeze()
    loc = np.arange(len(users), dtype=np.int64)
    pi_e_at_position = np.asarray(pi_val[loc, actions].squeeze(), dtype=np.float32)
    iw = pi_e_at_position / (pscore + 1e-12)
    q_hat_factual = np.asarray(scores_val[loc, actions].squeeze(), dtype=np.float32)
    dm_reward = np.asarray((scores_val * pi_val).sum(axis=1), dtype=np.float32)
    dr_vec = dm_reward + iw * (reward - q_hat_factual)
    ess = float((iw.sum() ** 2) / ((iw**2).sum() + 1e-12))
    return dr_vec, ess


def _policy_reward_from_embeddings(dataset, user_emb, item_emb, seed=12345):
    pi_obj = Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=user_emb,
        item_emb=item_emb,
        emb_dim=int(dataset["emb_dim"]),
        temperature=_policy_temperature(dataset),
        user_chunk=2048,
        rng=np.random.default_rng(seed),
    )
    return calc_reward(dataset, pi_obj)


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
):
    t0 = time.time()
    # Val-aligned softmax only (avoid dense n_users x n_actions matrix).
    uids = np.asarray(val_data["x_idx"], dtype=np.int64)
    xw = np.asarray(our_x[uids], dtype=np.float32)
    aw = np.asarray(our_a, dtype=np.float32)
    pt = max(_policy_temperature(dataset), 1e-8)
    logits = (xw @ aw.T) / pt
    policy_val = np.expand_dims(_softmax_action_probs_stable(logits, axis=1), -1)
    policy_object = Policy(
        n_users=int(dataset["n_users"]),
        n_items=int(dataset["n_actions"]),
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=int(dataset["emb_dim"]),
        temperature=_policy_temperature(dataset),
        user_chunk=2048,
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

    # DM with regression model (needs 3D action_dist: n x n_actions x len_list)
    reg_dm = dm.estimate_policy_value(
        policy_val, regression_model.predict(val_data["x"])
    )
    print(f"Reg DM time: {time.time() - t0} seconds")
    reg_results = np.array([reg_dm])
    conv_results = np.array([row])

    print(f"Evaluation total results time: {time.time() - t0:.2f} seconds")
    return get_opl_results_dict(reg_results, conv_results)


# --------------------------------------------------------------------
#  NEIGHBORHOOD MODEL TRAINER (MODULAR)
# --------------------------------------------------------------------
def neighberhoodmodel_trainer_trial(
    num_runs,
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
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

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

        for run in range(num_runs):
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

            idx_train = np.arange(train_size)
            train_data = get_train_data(
                n_actions, train_size, simulation_data, idx_train, our_x_orig
            )
            val_idx = np.arange(v) + train_size
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
                    "batch_size", [64, 128, 256, 512]
                )
                trial_num_neighbors = trial.suggest_int("num_neighbors", 3, 15)
                lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)
                kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
                trial_use_log_trick = trial.suggest_categorical(
                    "use_log_trick", [True, False]
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
                            kl_gamma, use_log_trick=trial_use_log_trick
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
                study.enqueue_trial(_enqueue_with_kl_gamma(last_best_params))

            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
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
                batch_size=batch_size,
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
                trial_res = _nan_posthoc_eval_metrics()
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
                    arr = np.asarray(v_)
                    row[k] = float(arr.reshape(-1)[0]) if arr.size == 1 else v_
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
                    simulation_data = _simulate_from_embedding_policy(
                        dataset,
                        our_x_orig,
                        our_a_orig,
                        train_size_int + v,
                        random_state=(ridx + 1) * (train_size_int + 17),
                    )
                    idx_train = np.arange(train_size_int)
                    train_data = get_train_data(
                        n_actions, train_size_int, simulation_data, idx_train, our_x_orig
                    )
                    train_actions = train_data["a"]
                    train_users = train_data["x_idx"]
                    pi_e_tr = _batched_pi_at_logged_actions(
                        trial_dicts_this_size[ridx].get("our_x", our_x_orig),
                        trial_dicts_this_size[ridx].get("our_a", our_a_orig),
                        train_users,
                        train_actions,
                        policy_temperature=_policy_temperature(dataset),
                    )
                    pscore_tr = np.asarray(train_data.get("pscore"), dtype=np.float32)
                    wi = get_weights_info(pi_e_tr, pscore_tr)
                    for k2, v2 in wi.items():
                        row[f"weights_{k2}"] = float(v2)
                    # actual reward of the chosen policy
                    row["actual_reward_selected"] = float(
                        _policy_reward_from_embeddings(
                            dataset,
                            trial_dicts_this_size[ridx].get("our_x", our_x_orig),
                            trial_dicts_this_size[ridx].get("our_a", our_a_orig),
                        )
                    )
                rows.append(row)
            _append_csv(log_paths["runs"], pd.DataFrame(rows))

    return pd.DataFrame.from_dict(results, orient="index"), best_hyperparams_by_size


# --------------------------------------------------------------------
#  REGRESSION-BASED TRAINER (MODULAR)
# --------------------------------------------------------------------
def regression_trainer_trial(
    num_runs,
    num_neighbors,  # unused, kept for API compatibility
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
):
    """
    OPC / no-propensity trainer with Optuna over CF hyperparameters.

    ``slim``: always writes ``trials_long`` (all hyperparameters per trial). Skips only
    the heavy post-hoc ``get_trial_results`` pass (full-catalog reward + val DM/DR/IPW/SNDR).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

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

    # ===== Baseline row =====
    simulation_data = _simulate_from_embedding_policy(
        dataset, our_x_orig, our_a_orig, v_baseline + v_baseline, random_state=0
    )
    original_policy_prob = None
    
    train_data = get_train_data(
        n_actions, v_baseline, simulation_data, np.arange(v_baseline), our_x_orig
    )

    val_data = get_train_data(
        n_actions, v_baseline, simulation_data, np.arange(v_baseline) + v_baseline, our_x_orig
    )

    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a_orig,  # IMPORTANT: action embeddings
        base_model=LogisticRegression(random_state=12345),
    )
    regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    print(f"[Regression] Baseline regression model fit time: {time.time() - t0:.2f}s")

    # wrap for eval_policy
    wrapped_reg_model = IndexToContextModelWrapper(regression_model, our_x_orig)

    if slim:
        results[0] = _nan_posthoc_eval_metrics()
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

        for run in range(num_runs):
            print(f"\n=== [Regression] Training size {train_size}, run {run} (val_size={v}) ===")

            simulation_data = _simulate_from_embedding_policy(
                dataset,
                our_x_orig,
                our_a_orig,
                int(train_size) + v,
                random_state=(run + 1) * (train_size + 17),
            )
            original_policy_prob = None
            reg_size = int(0.5 * train_size)
            reg_data_idx = np.arange(reg_size)
            
            reg_data = get_train_data(
                n_actions, reg_size, simulation_data, reg_data_idx, our_x_orig
            )

            idx_train = np.arange(reg_size) + reg_size
            train_data = get_train_data(
                n_actions, train_size - reg_size, simulation_data, idx_train, our_x_orig
            )

            val_idx = np.arange(v) + train_size
            val_data = get_train_data(
                n_actions, v, simulation_data, val_idx, our_x_orig
            )

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
                    "batch_size", [1024, 2048, 4096, 8192]
                )
                lr_decay = trial.suggest_float("lr_decay", 1e-5, 1e-3, log=True)
                kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
                trial_use_log_trick = trial.suggest_categorical(
                    "use_log_trick", [True, False]
                )

                # Regression model (instead of neighborhood)
                trial_reg_model = RegressionModel(
                    n_actions=n_actions,
                    action_context=our_a_orig,
                    base_model=LogisticRegression(random_state=12345),
                )
            
                trial_reg_model.fit(
                    reg_data["x"], reg_data["a"], reg_data["r"]
                )

                # Predict q_hat for ALL users (static scores), chunked to limit RAM.
                trial_q_hat = predict_regression_qhat_chunked(
                    trial_reg_model, our_x_orig, chunk_size=256
                )
                trial_scores_all = torch.as_tensor(
                    np.asarray(trial_q_hat, dtype=np.float32),
                    device=device,
                    dtype=torch.float32,
                )

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
                        criterion=_kl_policy_loss(
                            kl_gamma, use_log_trick=trial_use_log_trick
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
                    val_data, trial_x, trial_a, trial_scores_all, dataset
                )
                dr_vec_tr, ess_train = _split_dr_vec_and_ess(
                    train_data, trial_x, trial_a, trial_scores_all, dataset
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
                study.enqueue_trial(_enqueue_with_kl_gamma(last_best_params))

            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            last_optuna_study = study

            best_params = study.best_params
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
                )
                _append_csv(log_paths["trials"], last_trials_export)

            # --- Final training with best params ---
            regression_model = RegressionModel(
                n_actions=n_actions,
                action_context=our_a_orig,
                base_model=LogisticRegression(random_state=12345),
            )
            regression_model.fit(
                train_data["x"], train_data["a"], train_data["r"]
            )

            q_hat_all = predict_regression_qhat_chunked(
                regression_model, our_x_orig, chunk_size=256
            )
            scores_all = torch.as_tensor(
                np.asarray(q_hat_all, dtype=np.float32),
                device=device,
                dtype=torch.float32,
            )

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
                batch_size=batch_size,
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
                trial_res = _nan_posthoc_eval_metrics()
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
                **_log_constants,
            }

            trial_dicts_this_size.append(trial_res)
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
                    arr = np.asarray(v_)
                    row[k] = float(arr.reshape(-1)[0]) if arr.size == 1 else v_
                if slim and ridx == win_idx:
                    train_size_int = int(train_size)
                    v = resolve_validation_size(
                        train_size_int,
                        val_size=val_size,
                        val_frac=val_frac,
                        val_min=val_min,
                        val_max=val_max,
                    )
                    simulation_data = _simulate_from_embedding_policy(
                        dataset,
                        our_x_orig,
                        our_a_orig,
                        train_size_int + v,
                        random_state=(ridx + 1) * (train_size_int + 17),
                    )
                    idx_train = np.arange(train_size_int)
                    train_data = get_train_data(
                        n_actions, train_size_int, simulation_data, idx_train, our_x_orig
                    )
                    train_actions = train_data["a"]
                    train_users = train_data["x_idx"]
                    pi_e_tr = _batched_pi_at_logged_actions(
                        trial_dicts_this_size[ridx].get("our_x", our_x_orig),
                        trial_dicts_this_size[ridx].get("our_a", our_a_orig),
                        train_users,
                        train_actions,
                        policy_temperature=_policy_temperature(dataset),
                    )
                    pscore_tr = np.asarray(train_data.get("pscore"), dtype=np.float32)
                    wi = get_weights_info(pi_e_tr, pscore_tr)
                    for k2, v2 in wi.items():
                        row[f"weights_{k2}"] = float(v2)
                    row["actual_reward_selected"] = float(
                        _policy_reward_from_embeddings(
                            dataset,
                            trial_dicts_this_size[ridx].get("our_x", our_x_orig),
                            trial_dicts_this_size[ridx].get("our_a", our_a_orig),
                        )
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
    num_runs,
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
    log_paths: dict | None = None,
    method_label: str = "no_propensity",
    policy_reward_mode: str = "exact",
    policy_reward_mc_sim: int = 8,
    slim: bool = False,
):
    """
    Explicit no-propensity baseline with parity to regression trainer:
    same model family, same search budget, same train/val splits.
    """
    return regression_trainer_trial(
        num_runs=num_runs,
        num_neighbors=num_neighbors,
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
    )


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


def predict_qhat_all_chunked(reward_model, user_contexts, chunk_size=4096):
    n_users = user_contexts.shape[0]
    outs = []
    for s in range(0, n_users, chunk_size):
        e = min(n_users, s + chunk_size)
        q = reward_model.predict(user_contexts[s:e])
        q = np.asarray(q)
        if q.ndim == 2:
            q = q[..., None]
        outs.append(q)
    return np.concatenate(outs, axis=0)


def mlp_trial_reward_fit_once(
    train_size: int,
    dataset: dict,
    val_size: int = 2000,
    n_trials: int = 20,
    reg_frac: float = 0.5,         # part of train used to fit reward model once
    user_chunk: int = 2048,
    qhat_chunk: int = 4096,
    seed: int = 12345,
    # keep eval same as random_trial:
    lam_dr: float = 3.0,
    n_bootstrap: int = 1,
    n_dm_mc: int = 1,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Precompute q_hat_all ONCE (chunked)
    q_hat_all = predict_qhat_all_chunked(reward_model, our_x, chunk_size=qhat_chunk)
    scores_all = torch.as_tensor(q_hat_all, device=device, dtype=torch.float32)

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
            "batch_size", [1024, 2048, 4096, 8192]
        )
        lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)
        kl_gamma = trial.suggest_float("kl_gamma", 1e-4, 0.5, log=True)
        trial_use_log_trick = trial.suggest_categorical("use_log_trick", [True, False])

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