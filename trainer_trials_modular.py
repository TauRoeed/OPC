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

from estimators import (
    DirectMethod as DM,
)

from simulation_utils import (
    eval_policy,
    generate_dataset,
    create_simulation_data_from_pi,
    get_train_data,
    get_opl_results_dict,
    CustomCFDataset,
    calc_reward,
    get_weights_info,
)

from models import (
    LinearCFModel,
    CFModel,
    SingleMLPTransform,
    NeighborhoodModel,
    RegressionModel,
)

from training_utils import (
    train,
    validation_loop,
)

from custom_losses import (
    SNDRPolicyLoss,
    IPWPolicyLoss,
    KLPolicyLoss,
)

from model_scoring import score_model_modular

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



def generate_policies(num_policies, pi_0, pi_oracle, use_random=True, use_oracle=True, jaws=False):
    policies = []
    n_users, n_actions = pi_0.shape

    for i in range(num_policies):
        noise = random_.normal(size=(n_users, n_actions))
        noise_p = softmax(noise, axis=1)

        alpha = np.random.uniform(0, 1) * use_random
        beta = np.random.uniform(0, 1 - alpha) * use_oracle

        if jaws:
            p = np.random.uniform(0, 1)
            if p > 0.5:
                beta = 0.0
            else:
                alpha = 0.0
                beta = np.random.uniform(0, 1)

        # pi_i = (1 - alpha) * pi_0 + alpha * pi_oracle
        pi_i = (1 - alpha - beta) * pi_0 + alpha * noise_p + beta * pi_oracle
        # pi_i = softmax(pi_i, axis=1)
        policies.append(pi_i)

    return policies


def random_policy_trainer_trial(
    train_size,
    dataset,
    n_policies=50,
    val_size=2000,
    use_random=True,
    use_oracle=True,
    jaws=False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    results = {}

    # ===== Unpack dataset =====
    our_x_orig = dataset["our_x"]
    our_a_orig = dataset["our_a"]
    true_x = dataset["emb_x"]
    true_a = dataset["emb_a"]
    n_actions = dataset["n_actions"]

    T = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

    # ===== Baseline row =====
    pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
    pi_oracle = softmax(true_x @ true_a.T, axis=1)

    simulation_data = create_simulation_data_from_pi(
        dataset,
        pi_0,
        train_size + val_size,
        random_state=train_size + 17
    )

    idx_train = np.arange(train_size)
    train_data = get_train_data(
        n_actions, train_size, simulation_data, idx_train, our_x_orig
    )
    val_idx = np.arange(val_size) + train_size
    val_data = get_train_data(
        n_actions, val_size, simulation_data, val_idx, our_x_orig
    )
    df = pd.DataFrame(columns=[
                                "value", 
                                "user_attrs_actual_reward", 
                                "user_attrs_q_error", 
                                "user_attrs_r_hat", 
                                "user_attrs_ess",                                          
                                "user_attrs_scores_dict", 
                                "user_attrs_all_values"
                            ])
    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a_orig,  # IMPORTANT: action embeddings
        base_model=LogisticRegression(random_state=12345),
    )

    regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    print(f"[Regression] Baseline regression model fit time: {time.time() - t0:.2f}s")

    q_hat_all = regression_model.predict(our_x_orig)

    scores_all = torch.as_tensor(q_hat_all, device=device, dtype=torch.float32)
    policies = generate_policies(num_policies=n_policies, pi_0=pi_0, pi_oracle=pi_oracle, use_random=use_random, use_oracle=use_oracle, jaws=jaws)
    # ===== Main loop over training sizes =====
    tq = tqdm(policies)
    for pi_i in tq:

        scores_dict, scores_array, weight_info = score_model_modular(val_data, scores_all, pi_i)
        r_hat = scores_dict['dr_naive_mean']
        err = scores_dict['dr_naive_se']
        r = calc_reward(dataset, np.expand_dims(pi_i, -1))[0]
        value = scores_dict['dr_naive_ci_low'] # conservative estimate
        new_row = pd.DataFrame([{
            "value": float(value),
            "user_attrs_actual_reward": float(r),
            "user_attrs_q_error": float(err),
            "user_attrs_r_hat": float(r_hat),
            "user_attrs_ess": float(weight_info["ess"]),
            "user_attrs_scores_dict": scores_dict,
            "user_attrs_all_values": scores_array,
            'ipw': scores_dict['ipw_uni_mean'],
            "sign_uni": scores_dict['cv_signed_rmse_uniform'],
            "sign_exp": scores_dict['cv_signed_rmse_exp']
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        tq.set_description(f"Validation weights_info: {weight_info}")

    df = df[df['value'] > 0]

    return df, df