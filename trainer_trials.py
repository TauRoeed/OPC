import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.append("/code")

from tqdm import tqdm
import torch
import time
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# import gym
# import recogym

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

torch.backends.cudnn.benchmark = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # TF32 = big speedup on Ada


from sklearn.utils import check_random_state

# implementing OPE of the IPWLearner using synthetic bandit data
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from scipy.special import softmax
import optuna
# from memory_profiler import profile


from estimators import (
    DirectMethod as DM
)

from simulation_utils import (
    eval_policy,
    generate_dataset,
    create_simulation_data_from_pi,
    get_train_data,
    get_opl_results_dict,
    CustomCFDataset,
    calc_reward,
    get_weights_info
)

from models import (    
    LinearCFModel,
    NeighborhoodModel,
    BPRModel, 
    RegressionModel
)

from training_utils import (
    train,
    validation_loop, 
    cv_score_model
 )

from custom_losses import (
    SNDRPolicyLoss,
    IPWPolicyLoss, 
    KLPolicyLoss
    )

random_state=12345
random_ = check_random_state(random_state)

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
    dm
):
    t0 = time.time()
    policy = np.expand_dims(softmax(our_x @ our_a.T, axis=1), -1)
    policy_reward = calc_reward(dataset, policy)
    eval_metrics = eval_policy(neighberhoodmodel, val_data, original_policy_prob, policy)
    action_diff_to_real = np.sqrt(np.mean((emb_a - our_a) ** 2))
    action_delta = np.sqrt(np.mean((original_a - our_a) ** 2))
    context_diff_to_real = np.sqrt(np.mean((emb_x - our_x) ** 2))
    context_delta = np.sqrt(np.mean((original_x - our_x) ** 2))

    row = np.concatenate([
        np.atleast_1d(policy_reward),
        np.atleast_1d(eval_metrics),
        np.atleast_1d(action_diff_to_real),
        np.atleast_1d(action_delta),
        np.atleast_1d(context_diff_to_real),
        np.atleast_1d(context_delta)
    ])
    reg_dm = dm.estimate_policy_value(policy[val_data['x_idx']], regression_model.predict(val_data['x']))
    reg_results = np.array([reg_dm])
    conv_results = np.array([row])
    print(f"Evaluation total results time: {time.time() - t0} seconds")
    return get_opl_results_dict(reg_results, conv_results)


def neighberhoodmodel_trainer_trial(
    num_runs,
    num_neighbors,
    train_sizes,
    dataset,
    batch_size,
    val_size=2000,
    n_trials=10,    
    prev_best_params=None
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dm = DM()
    results = {}

    our_x, our_a = dataset["our_x"], dataset["our_a"]
    emb_x, emb_a = dataset["emb_x"], dataset["emb_a"]

    original_x, original_a = dataset["original_x"], dataset["original_a"]
    n_users, n_actions, emb_dim = dataset["n_users"], dataset["n_actions"], dataset["emb_dim"]

    all_user_indices = np.arange(n_users, dtype=np.int64)

    def T(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32)

    def _mean_dict(dicts):
        """
        Robust mean over a list of dicts with numeric/scalar/1D-array values.
        Returns a single dict with elementwise means.
        """
        if not dicts:
            return {}
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            vals = [d[k] for d in dicts if k in d]
            # try to convert each to np.array and average
            arrs = [np.asarray(v) for v in vals]
            # broadcast to same shape if scalars/1D
            stacked = np.stack(arrs, axis=0)
            out[k] = np.mean(stacked, axis=0)
        return out

    # ===== unpack dataset (keep originals safe) =====
    our_x_orig, our_a_orig = our_x, our_a
    emb_x, emb_a = emb_x, emb_a
    original_x, original_a = original_x, original_a
    n_users, n_actions, emb_dim = n_users, n_actions, emb_dim
    all_user_indices = np.arange(n_users, dtype=np.int64)

    dm = DM()
    results = {}
    best_hyperparams_by_size = {}
    last_best_params = prev_best_params if prev_best_params is not None else None

    # ===== baseline (sample size = 0) using get_trial_results =====
    pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
    original_policy_prob = np.expand_dims(pi_0, -1)

    simulation_data = create_simulation_data_from_pi(
        dataset, pi_0, val_size, random_state=0
    )

    # use same data for train/val just to generate the baseline row
    train_data = get_train_data(n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig)
    val_data   = get_train_data(n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig)
    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions, action_context=our_x_orig,
        base_model=LogisticRegression(random_state=12345)
    )

    regression_model.fit(train_data['x'], train_data['a'], train_data['r'])
    print(f"Baseline regression model fit time: {time.time() - t0} seconds")

    t0 = time.time()
    neighberhoodmodel = NeighborhoodModel(
        train_data['x_idx'], train_data['a'],
        our_a_orig, our_x_orig, train_data['r'],
        num_neighbors=num_neighbors
    )
    print(f"Baseline neighborhood model fit time: {time.time() - t0} seconds")
    
    # baseline row produced via get_trial_results
    # results[0] = get_trial_results(
    #     our_x_orig, our_a_orig, emb_x, emb_a, original_x, original_a,
    #     dataset, val_data, original_policy_prob,
    #     neighberhoodmodel, regression_model, dm
    # )

    # ===== main loop over training sizes =====
    for train_size in train_sizes:

        # we’ll collect per-run trial dicts generated by get_trial_results
        trial_dicts_this_size = []
        best_hyperparams_by_size[train_size] = {}

        # --- prepare a resampling for Optuna’s objective (shared loaders built per-run inside objective) ---
        # We’ll do Optuna per-run (fresh resample + search), then final fit with best params, then get_trial_results.

        for run in range(num_runs):

            # --- resample for this run ---
            pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
            original_policy_prob = np.expand_dims(pi_0, -1)

            simulation_data = create_simulation_data_from_pi(
                dataset, pi_0, train_size + val_size,
                random_state=(run + 1) * (train_size + 17)
            )
                        
            idx_train = np.arange(train_size)
            train_data = get_train_data(n_actions, train_size, simulation_data, idx_train, our_x_orig)
            val_idx   = np.arange(val_size) + train_size
            val_data  = get_train_data(n_actions, val_size, simulation_data, val_idx, our_x_orig)

            num_workers = 4 if torch.cuda.is_available() else 0

            cf_dataset = CustomCFDataset(
                train_data['x_idx'], train_data['a'], train_data['r'], original_policy_prob
            )

            # val_loader = DataLoader(
            #     val_dataset, batch_size=val_size, shuffle=False,
            #     pin_memory=torch.cuda.is_available(),
            #     num_workers=num_workers, persistent_workers=bool(num_workers)
            # )


            # --- Optuna objective bound to this run's data ---
            def objective(trial):
                print()
                print(f"Trial {trial.number} started")
                lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                epochs = trial.suggest_int("num_epochs", 1, 10)
                trial_batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
                trial_num_neighbors = trial.suggest_int("num_neighbors", 3, 15)
                lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)

                trial_neigh_model = NeighborhoodModel(
                    train_data['x_idx'], train_data['a'],
                    our_a_orig, our_x_orig, train_data['r'],
                    num_neighbors=trial_num_neighbors
                )

                trial_scores_all = torch.as_tensor(
                    trial_neigh_model.predict(all_user_indices),
                    device=device, dtype=torch.float32
                )

                trial_model = LinearCFModel(
                    n_users, n_actions, emb_dim,
                    initial_user_embeddings=T(our_x_orig),
                    initial_actions_embeddings=T(our_a_orig)
                ).to(device)

                assert (not torch.cuda.is_available()) or next(trial_model.parameters()).is_cuda

                final_train_loader = DataLoader(
                    cf_dataset, batch_size=trial_batch_size, shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=num_workers, persistent_workers=bool(num_workers)
                )

                current_lr = lr
                for epoch in range(epochs):
                    if epoch > 0:
                        current_lr *= lr_decay
                        
                    train(
                        trial_model, final_train_loader, trial_scores_all,
                        criterion=KLPolicyLoss(), num_epochs=1, lr=current_lr, device=str(device)
                    )

                trial_x, trial_a = trial_model.get_params()
                trial_x = trial_x.detach().cpu().numpy()
                trial_a = trial_a.detach().cpu().numpy()

                pi_i = softmax(trial_x @ trial_a.T, axis=1)
                train_actions = train_data['a']
                train_users = train_data['x_idx']

                print("Train wi info: {}".format(get_weights_info(pi_i[train_users, train_actions], original_policy_prob[train_users, train_actions])))
                print(f"actual reward: {calc_reward(dataset, np.expand_dims(pi_i, -1))}")

                # print(get_weights_info(pi_i, original_policy_prob))
                # validation reward for selection
                return cv_score_model(val_data, trial_scores_all, pi_i)


            # --- run Optuna for this run ---
            study = optuna.create_study(direction="maximize")
            
            if last_best_params is not None:
                study.enqueue_trial(last_best_params)

            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            last_best_params = best_params  # optional warm-start to next run
            best_hyperparams_by_size[train_size][run] = {
                "params": best_params,
                "reward": study.best_value
            }


            # --- final training with best params on this run’s data ---
            regression_model = RegressionModel(
                n_actions=n_actions, action_context=our_x_orig,
                base_model=LogisticRegression(random_state=12345)
            )
            regression_model.fit(
                train_data['x'], train_data['a'], train_data['r'],
                original_policy_prob[train_data['x_idx'], train_data['a']].squeeze()
            )

            neighberhoodmodel = NeighborhoodModel(
                train_data['x_idx'], train_data['a'],
                our_a_orig, our_x_orig, train_data['r'],
                num_neighbors=best_params['num_neighbors']
            )
            
            scores_all = torch.as_tensor(
                neighberhoodmodel.predict(all_user_indices),
                device=device, dtype=torch.float32
            )

            model = LinearCFModel(
                n_users, n_actions, emb_dim,
                initial_user_embeddings=T(our_x_orig),
                initial_actions_embeddings=T(our_a_orig)
            ).to(device)
            assert (not torch.cuda.is_available()) or next(model.parameters()).is_cuda

            train_loader = DataLoader(
                cf_dataset, batch_size=batch_size, shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers, persistent_workers=bool(num_workers)
            )

            current_lr = best_params['lr']
            for epoch in range(best_params['num_epochs']):
                if epoch > 0:
                    current_lr *= best_params['lr_decay']
                train(
                    model, train_loader, scores_all,
                    criterion=KLPolicyLoss(), num_epochs=1, lr=current_lr, device=str(device)
                )

            # learned embeddings (do NOT overwrite originals)
            learned_x_t, learned_a_t = model.get_params()
            learned_x = learned_x_t.detach().cpu().numpy()
            learned_a = learned_a_t.detach().cpu().numpy()

            # --- produce the per-run result via get_trial_results ---
            trial_res = get_trial_results(
                learned_x, learned_a,          # learned (policy) embeddings
                emb_x, emb_a,                  # ground-truth embedding refs
                original_x, original_a,        # original clean refs
                dataset,
                val_data,                      # use this run's val split
                original_policy_prob,
                neighberhoodmodel,
                regression_model,
                dm
            )

            trial_dicts_this_size.append(trial_res)

            # memory hygiene
            torch.cuda.empty_cache()

        # === aggregate per-run results (mean) and store under this train_size ===
        results[train_size] = _mean_dict(trial_dicts_this_size)

    return pd.DataFrame.from_dict(results, orient='index'), best_hyperparams_by_size



def regression_trainer_trial(
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

    def _mean_dict(dicts):
        if not dicts:
            return {}
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            vals = [np.asarray(d[k]) for d in dicts if k in d]
            stacked = np.stack(vals, axis=0)
            out[k] = np.mean(stacked, axis=0)
        return out

    # ===== Baseline row =====
    pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
    original_policy_prob = np.expand_dims(pi_0, -1)

    simulation_data = create_simulation_data_from_pi(dataset, pi_0, val_size, random_state=0)
    train_data = get_train_data(n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig)
    val_data = get_train_data(n_actions, val_size, simulation_data, np.arange(val_size), our_x_orig)

    t0 = time.time()
    regression_model = RegressionModel(
        n_actions=n_actions,
        action_context=our_a_orig,
        base_model=LogisticRegression(random_state=12345)
    )
    regression_model.fit(train_data["x"], train_data["a"], train_data["r"])
    print(f"Baseline regression model fit time: {time.time() - t0:.2f}s")

    results[0] = get_trial_results(
        our_x_orig, our_a_orig, emb_x, emb_a, original_x, original_a,
        dataset, val_data, original_policy_prob,
        regression_model, regression_model, dm
    )

    # ===== Main loop over training sizes =====
    for train_size in train_sizes:
        trial_dicts_this_size = []
        best_hyperparams_by_size[train_size] = {}

        for run in range(num_runs):
            print(f"\n=== Training size {train_size}, run {run} ===")

            pi_0 = softmax(our_x_orig @ our_a_orig.T, axis=1)
            original_policy_prob = np.expand_dims(pi_0, -1)

            simulation_data = create_simulation_data_from_pi(
                dataset, pi_0, train_size + val_size,
                random_state=(run + 1) * (train_size + 17)
            )

            idx_train = np.arange(train_size)
            train_data = get_train_data(n_actions, train_size, simulation_data, idx_train, our_x_orig)
            val_idx = np.arange(val_size) + train_size
            val_data = get_train_data(n_actions, val_size, simulation_data, val_idx, our_x_orig)

            cf_dataset = CustomCFDataset(
                train_data["x_idx"], train_data["a"], train_data["r"], original_policy_prob
            )
            num_workers = 4 if torch.cuda.is_available() else 0

            # --- Define Optuna objective ---
            def objective(trial):
                print(f"\n[Optuna Trial {trial.number}]")
                lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                epochs = trial.suggest_int("num_epochs", 1, 10)
                trial_batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
                lr_decay = trial.suggest_float("lr_decay", 0.8, 1.0)

                # Regression model instead of neighborhood
                trial_reg_model = RegressionModel(
                    n_actions=n_actions,
                    action_context=our_a_orig,
                    base_model=LogisticRegression(random_state=12345)
                )
                trial_reg_model.fit(train_data["x"], train_data["a"], train_data["r"])

                # Predict q_hat for all users
                trial_q_hat = trial_reg_model.predict(our_x_orig)  # (n_users, n_actions, 1)
                trial_scores_all = torch.as_tensor(trial_q_hat, device=device, dtype=torch.float32)

                # Initialize CF model
                trial_model = LinearCFModel(
                    n_users, n_actions, emb_dim,
                    initial_user_embeddings=T(our_x_orig),
                    initial_actions_embeddings=T(our_a_orig)
                ).to(device)

                final_train_loader = DataLoader(
                    cf_dataset, batch_size=trial_batch_size, shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=num_workers, persistent_workers=bool(num_workers)
                )

                current_lr = lr
                for epoch in range(epochs):
                    if epoch > 0:
                        current_lr *= lr_decay
                    train(
                        trial_model, final_train_loader, trial_scores_all,
                        criterion=KLPolicyLoss(), num_epochs=1, lr=current_lr, device=str(device)
                    )

                # Evaluate validation score
                trial_x, trial_a = trial_model.get_params()
                trial_x, trial_a = trial_x.detach().cpu().numpy(), trial_a.detach().cpu().numpy()
                pi_i = softmax(trial_x @ trial_a.T, axis=1)

                print(f"actual reward: {calc_reward(dataset, np.expand_dims(pi_i, -1))}")
                r_hat, err = cv_score_model(val_data, trial_scores_all, pi_i)
                return r_hat - 2 * err

            # --- Run Optuna search ---
            study = optuna.create_study(direction="maximize")
            if last_best_params is not None:
                study.enqueue_trial(last_best_params)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            last_best_params = best_params
            best_hyperparams_by_size[train_size][run] = {
                "params": best_params,
                "reward": study.best_value
            }

            # --- Final training with best params ---
            regression_model = RegressionModel(
                n_actions=n_actions,
                action_context=our_a_orig,
                base_model=LogisticRegression(random_state=12345)
            )
            regression_model.fit(train_data["x"], train_data["a"], train_data["r"])

            q_hat_all = regression_model.predict(our_x_orig)
            scores_all = torch.as_tensor(q_hat_all, device=device, dtype=torch.float32)

            model = LinearCFModel(
                n_users, n_actions, emb_dim,
                initial_user_embeddings=T(our_x_orig),
                initial_actions_embeddings=T(our_a_orig)
            ).to(device)

            train_loader = DataLoader(
                cf_dataset, batch_size=batch_size, shuffle=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers, persistent_workers=bool(num_workers)
            )

            current_lr = best_params["lr"]
            for epoch in range(best_params["num_epochs"]):
                if epoch > 0:
                    current_lr *= best_params["lr_decay"]
                train(
                    model, train_loader, scores_all,
                    criterion=KLPolicyLoss(), num_epochs=1, lr=current_lr, device=str(device)
                )

            # Extract learned embeddings
            learned_x_t, learned_a_t = model.get_params()
            learned_x = learned_x_t.detach().cpu().numpy()
            learned_a = learned_a_t.detach().cpu().numpy()

            # Evaluate results
            trial_res = get_trial_results(
                learned_x, learned_a,
                emb_x, emb_a, original_x, original_a,
                dataset, val_data, original_policy_prob,
                regression_model, regression_model, dm
            )

            trial_dicts_this_size.append(trial_res)
            torch.cuda.empty_cache()

        # Aggregate across runs
        results[train_size] = _mean_dict(trial_dicts_this_size)

    return pd.DataFrame.from_dict(results, orient="index"), best_hyperparams_by_size