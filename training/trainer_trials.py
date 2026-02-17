import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from datetime import datetime
from pathlib import Path

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

from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from scipy.special import softmax
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
    KLPolicyLoss
)

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
):
    t0 = time.time()
    policy = np.expand_dims(softmax(our_x @ our_a.T, axis=1), -1)
    policy_reward = calc_reward(dataset, policy)

    # eval_policy expects model.predict(x_idx)
    eval_metrics = eval_policy(neighberhoodmodel, val_data, original_policy_prob, policy)

    action_diff_to_real = np.sqrt(np.mean((emb_a - our_a) ** 2))
    action_delta = np.sqrt(np.mean((original_a - our_a) ** 2))
    context_diff_to_real = np.sqrt(np.mean((emb_x - our_x) ** 2))
    context_delta = np.sqrt(np.mean((original_x - our_x) ** 2))

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

    # DM with regression model
    reg_dm = dm.estimate_policy_value(
        policy[val_data["x_idx"]], regression_model.predict(val_data["x"])
    )
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
    val_size=2000,
    n_trials=10,
    prev_best_params=None,
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

    all_user_indices = np.arange(n_users, dtype=np.int64)
    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

    # ===== baseline (sample size = 0) using get_trial_results =====
    pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
    original_policy_prob = np.expand_dims(pi_0, -1)

    simulation_data = create_simulation_data_from_pi(
        dataset, pi_0, val_size, random_state=0
    )

    # use same data for train/val just to generate the baseline row
    train_data = get_train_data(
        n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig
    )
    val_data = get_train_data(
        n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig
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

        for run in range(num_runs):
            print(f"\n=== [Neighborhood] Train size {train_size}, run {run} ===")

            # --- resample for this run ---
            pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
            original_policy_prob = np.expand_dims(pi_0, -1)

            simulation_data = create_simulation_data_from_pi(
                dataset,
                pi_0,
                train_size + val_size,
                random_state=(run + 1) * (train_size + 17),
            )

            idx_train = np.arange(train_size)
            train_data = get_train_data(
                n_actions, train_size, simulation_data, idx_train, our_x_orig
            )
            val_idx = np.arange(val_size) + train_size
            val_data = get_train_data(
                n_actions, val_size, simulation_data, val_idx, our_x_orig
            )

            num_workers = 4 if torch.cuda.is_available() else 0

            cf_dataset = CustomCFDataset(
                train_data["x_idx"],
                train_data["a"],
                train_data["r"],
                original_policy_prob,
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
                        criterion=KLPolicyLoss(),
                        num_epochs=1,
                        lr=current_lr,
                        device=str(device),
                    )

                trial_x, trial_a = trial_model.get_params()
                trial_x = trial_x.detach().cpu().numpy()
                trial_a = trial_a.detach().cpu().numpy()

                pi_i = softmax(trial_x @ trial_a.T, axis=1)
                train_actions = train_data["a"]
                train_users = train_data["x_idx"]

                print(
                    "Train wi info: {}".format(
                        get_weights_info(
                            pi_i[train_users, train_actions],
                            original_policy_prob[train_users, train_actions],
                        )
                    )
                )
                print(
                    f"actual reward: {calc_reward(dataset, np.expand_dims(pi_i, -1))}"
                )

                # validation reward for selection (you had cv_score_model)
                return cv_score_model(val_data, trial_scores_all, pi_i)

            # --- run Optuna for this run ---
            study = optuna.create_study(direction="maximize")

            if last_best_params is not None:
                study.enqueue_trial(last_best_params)

            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            last_best_params = best_params
            best_hyperparams_by_size[train_size][run] = {
                "params": best_params,
                "reward": study.best_value,
            }

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
                original_policy_prob[train_data["x_idx"], train_data["a"]].squeeze(),
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
                    criterion=KLPolicyLoss(),
                    num_epochs=1,
                    lr=current_lr,
                    device=str(device),
                )

            # learned embeddings (do NOT overwrite originals)
            learned_x_t, learned_a_t = model.get_params()
            learned_x = learned_x_t.detach().cpu().numpy()
            learned_a = learned_a_t.detach().cpu().numpy()

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
                original_policy_prob,
                neighberhoodmodel,
                regression_model,
                dm,
            )

            trial_dicts_this_size.append(trial_res)

            # memory hygiene
            torch.cuda.empty_cache()

        # === aggregate per-run results (mean) and store under this train_size ===
        results[train_size] = _mean_dict(trial_dicts_this_size)

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
    val_size=2000,
    n_trials=10,
    prev_best_params=None,
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

    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

    # ===== Baseline row =====
    pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
    original_policy_prob = np.expand_dims(pi_0, -1)

    simulation_data = create_simulation_data_from_pi(
        dataset, pi_0, val_size + val_size, random_state=0
    )
    
    train_data = get_train_data(
        n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig
    )

    val_data = get_train_data(
        n_actions, val_size, simulation_data, np.arange(val_size) + val_size, our_x_orig
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

    results[0] = get_trial_results(
        our_x_orig,
        our_a_orig,
        emb_x,
        emb_a,
        original_x,
        original_a,
        dataset,
        val_data,
        original_policy_prob,
        wrapped_reg_model,  # for eval_policy
        regression_model,  # for reg_dm
        dm,
    )

    # ===== Main loop over training sizes =====
    for train_size in train_sizes:
        trial_dicts_this_size = []
        best_hyperparams_by_size[train_size] = {}

        for run in range(num_runs):
            print(f"\n=== [Regression] Training size {train_size}, run {run} ===")

            pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
            original_policy_prob = np.expand_dims(pi_0, -1)

            simulation_data = create_simulation_data_from_pi(
                dataset,
                pi_0,
                train_size + val_size,
                random_state=(run + 1) * (train_size + 17),
            )
            reg_size = int(0.5 * train_size)
            reg_data_idx = np.arange(reg_size)
            
            reg_data = get_train_data(
                n_actions, reg_size, simulation_data, reg_data_idx, our_x_orig
            )

            idx_train = np.arange(reg_size) + reg_size
            train_data = get_train_data(
                n_actions, train_size - reg_size, simulation_data, idx_train, our_x_orig
            )

            val_idx = np.arange(val_size) + train_size
            val_data = get_train_data(
                n_actions, val_size, simulation_data, val_idx, our_x_orig
            )

            cf_dataset = CustomCFDataset(
                train_data["x_idx"],
                train_data["a"],
                train_data["r"],
                original_policy_prob,
            )

            num_workers = 4 if torch.cuda.is_available() else 0

            # --- Define Optuna objective ---
            def objective(trial):
                print(f"\n[Regression] Optuna Trial {trial.number}")
                lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                epochs = trial.suggest_int("num_epochs", 1, 10)
                trial_batch_size = trial.suggest_categorical(
                    "batch_size", [64, 128, 256, 512]
                )
                lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)

                # Regression model (instead of neighborhood)
                trial_reg_model = RegressionModel(
                    n_actions=n_actions,
                    action_context=our_a_orig,
                    base_model=LogisticRegression(random_state=12345),
                )
            
                trial_reg_model.fit(
                    reg_data["x"], reg_data["a"], reg_data["r"]
                )

                # Predict q_hat for ALL users (static scores)
                trial_q_hat = trial_reg_model.predict(our_x_orig)  # (n_users, n_actions, 1)
                trial_scores_all = torch.as_tensor(
                    trial_q_hat, device=device, dtype=torch.float32
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
                        criterion=KLPolicyLoss(),
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
                pi_i = softmax(trial_x @ trial_a.T, axis=1)
                r = calc_reward(dataset, np.expand_dims(pi_i, -1))
                print(
                    f"actual reward: {r}"
                )
                scores_dict, scores_array, weight_info = score_model_modular(val_data, trial_scores_all, pi_i)
                r_hat = scores_dict['dr_naive_mean']
                err = scores_dict['dr_naive_se']

                value = scores_dict['dr_naive_ci_low']  # conservative estimate

                trial.set_user_attr("all_values", scores_array)
                trial.set_user_attr("scores_dict", scores_dict)
                trial.set_user_attr("r_hat", r_hat)
                trial.set_user_attr("q_error", err)
                trial.set_user_attr("actual_reward", r)
                trial.set_user_attr("ess", weight_info["ess"])

                return value

            # --- Run Optuna search ---
            study = optuna.create_study(direction="maximize")
            if last_best_params is not None:
                study.enqueue_trial(last_best_params)

            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            last_best_params = best_params
            best_hyperparams_by_size[train_size][run] = {
                "params": best_params,
                "reward": study.best_value,
            }

            # --- Final training with best params ---
            regression_model = RegressionModel(
                n_actions=n_actions,
                action_context=our_a_orig,
                base_model=LogisticRegression(random_state=12345),
            )
            regression_model.fit(
                train_data["x"], train_data["a"], train_data["r"]
            )

            q_hat_all = regression_model.predict(our_x_orig)
            scores_all = torch.as_tensor(q_hat_all, device=device, dtype=torch.float32)

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
                    criterion=KLPolicyLoss(),
                    num_epochs=1,
                    lr=current_lr,
                    device=str(device),
                )

            # Extract learned embeddings
            learned_x_t, learned_a_t = model.get_params()
            learned_x = learned_x_t.detach().cpu().numpy()
            learned_a = learned_a_t.detach().cpu().numpy()

            # Wrap regression model so eval_policy uses context instead of idx
            wrapped_reg_model = IndexToContextModelWrapper(
                regression_model, our_x_orig
            )

            # Evaluate results
            trial_res = get_trial_results(
                learned_x,
                learned_a,
                emb_x,
                emb_a,
                original_x,
                original_a,
                dataset,
                val_data,
                original_policy_prob,
                wrapped_reg_model,
                regression_model,
                dm,
            )

            trial_dicts_this_size.append(trial_res)
            torch.cuda.empty_cache()

        # Aggregate across runs
        results[train_size] = _mean_dict(trial_dicts_this_size)
        
    trial_df = study.trials_dataframe()[["value", 
                                         "user_attrs_actual_reward", 
                                         "user_attrs_q_error", 
                                         "user_attrs_r_hat", 
                                         "user_attrs_ess",                                          
                                         "user_attrs_scores_dict", 
                                         "user_attrs_all_values"
                                         ]]

    trial_df['user_attrs_actual_reward'] = trial_df['user_attrs_actual_reward'].apply(lambda x:x[0])
    trial_df = trial_df[trial_df['value'] > 0]

    return pd.DataFrame.from_dict(results, orient="index"), trial_df


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

    # ----- policies (exact full softmax over all items) -----
    pi0_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=our_x,
        item_emb=our_a,
        emb_dim=our_x.shape[1],
        temperature=1.0,
        user_chunk=chunk_size,
        rng=np.random.default_rng(seed + 1),
    )

    oracle_policy = Policy(
        n_users=n_users,
        n_items=n_actions,
        user_emb=emb_x,
        item_emb=emb_a,
        emb_dim=emb_x.shape[1],
        temperature=1.0,
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
        temperature=1.0,
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
    # t0 = time.time()
    # regression_model = RegressionModel(
    #     n_actions=n_actions,
    #     action_context=our_a,
    #     base_model=LogisticRegression(random_state=seed),
    # )
    # regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    # print(f"[Regression] fit time: {time.time() - t0:.2f}s")
    t0 = time.time()
    regression_model = MLPRewardModel(
        n_actions=n_actions,
        action_context=our_a,      # (n_actions, d_action)
        hidden_dims=[64, 16],            # one hidden layer
        dropout=0.2,
        epochs=15,
        lr=1e-3,
        batch_size=8192,
        device="cuda",             # or "cpu"
    )

    regression_model.fit(
        context=train_data["x"],   # (n_rounds, d_context)
        action=train_data["a"],    # (n_rounds,)
        reward=train_data["r"],    # (n_rounds,)
    )
    print(f"[Regression-MLP] fit time: {time.time() - t0:.2f}s")

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
        users_val = np.asarray(val_data["x_idx"], dtype=np.int64)
        a_mc, _ = pi_i.sample_actions(users_val)
        r_actual = float(dataset["env"].reward_prob(users_val, a_mc).mean())

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
    n_bootstrap: int = 500,
    n_dm_mc: int = 32,
    # reward model fixed hyperparams (since fitted once):
    rm_hidden_dims=(64, 16),
    rm_dropout=0.2,
    rm_epochs=15,
    rm_lr=1e-3,
    rm_batch_size=8192,
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
        temperature=1.0,
        user_chunk=user_chunk,
        rng=np.random.default_rng(seed + 1),
    )

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

    train_full = get_train_data(n_actions, train_size, sim, idx_train, our_x)
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
    reward_model = MLPRewardModel(
        n_actions=n_actions,
        action_context=our_a,
        hidden_dims=list(rm_hidden_dims),
        dropout=rm_dropout,
        epochs=rm_epochs,
        lr=rm_lr,
        batch_size=rm_batch_size,
        device=str(device),
    )
    reward_model.fit(
        context=reg_data["x"],
        action=reg_data["a"],
        reward=reg_data["r"],
    )

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
    num_workers = 4 if torch.cuda.is_available() else 0

    # ---------------------------
    # Optuna objective: CF only
    # ---------------------------
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        epochs = trial.suggest_int("num_epochs", 1, 10)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)

        model = CFModel(
            n_users,
            n_actions,
            emb_dim,
            initial_user_embeddings=torch.as_tensor(our_x, device=device, dtype=torch.float32),
            initial_actions_embeddings=torch.as_tensor(our_a, device=device, dtype=torch.float32),
            user_transform=SingleMLPTransform(emb_dim),
            action_transform=SingleMLPTransform(emb_dim),
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
                criterion=KLPolicyLoss(),
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
            temperature=1.0,
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

        # "actual" on validation users (same as random_trial)
        users_val = np.asarray(val_data["x_idx"], dtype=np.int64)
        a_mc, _ = learned_policy.sample_actions(users_val)
        r_actual = float(dataset["env"].reward_prob(users_val, a_mc).mean())

        trial.set_user_attr("all_values", scores_array)
        trial.set_user_attr("scores_dict", scores_dict)
        trial.set_user_attr("r_hat", r_hat)
        trial.set_user_attr("q_error", err)
        trial.set_user_attr("actual_reward", r_actual)
        trial.set_user_attr("ess", ess)

        return value

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
    return pd.DataFrame([summary]), trial_df