"""Native CausE as a study arm: fixed-budget data, a seeded search, selection by validation NLL, exact values.

For each train size N and randomized share rho (docs/cause_baseline.md §4):
* data: N_c = N - round(rho·N) warm-logger rows (the first rows of the training split every other arm uses)
  plus N_t = round(rho·N) uniform-random rows from the same world (utils/budget_split.py);
* model: models/cause.py, from scratch, CausE-prod (prediction with the control rows = prod-C, or the
  treatment rows = prod-T, from one trained model) and CausE-avg;
* search: ``n_trials`` seeded Optuna trials per variant over lr, epochs, L2 and the tie strength. The
  configurations and trial seeds do not depend on rho, so with ``sampler='random'`` the rho curve is paired;
* selection: validation NLL of each prediction on the warm validation rows shared with every arm;
* outcomes: the exact true value of the greedy (argmax) policy and of the softmax over CausE's logits
  (the softmax is a convention, not part of the paper), plus collection rewards and diagnostics.
Each (prediction, rho) is reported as its own method, ``cause_<prediction>_r<rho in per mille>``.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.cause import (CAUSE_OPTIMIZERS, CAUSE_PREDICTIONS, CAUSE_TIES, CAUSE_VARIANTS, CausELayout, CausEModel,
                          fit_cause, predict_logits, prediction_metrics)
from utils.budget_split import CAUSE_RHOS, build_budget_split, simulate_uniform_pool, uniform_pool_seed
from utils.seeding import derive_seed, optuna_sampler, seed_everything
from utils.simulation_utils import calc_greedy_reward, calc_reward, calc_uniform_reward, ensure_exact_env_q_cache

CAUSE_EPOCHS = (1, 3, 10, 30, 100, 300)
CAUSE_L2 = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
CAUSE_CF_PEN = (0.0, 0.1, 1.0, 10.0, 100.0)
CAUSE_LR_RANGE = (1e-3, 1.0)
DIVERGED_NLL = 1e9
CAUSE_DEFAULTS = {"rhos": list(CAUSE_RHOS), "variants": list(CAUSE_VARIANTS), "dim": 32,
                  "optimizer": "momentum_decay", "tie": "one_way", "batch_size": 512, "n_trials": None}


def cause_method_label(prediction: str, rho: float) -> str:
    return f"cause_{prediction}_r{int(round(float(rho) * 1000)):03d}"


def add_cause_arguments(parser) -> None:
    g = parser.add_argument_group("CausE baseline (opt-in: --methods ... cause)")
    g.add_argument("--cause-rhos", type=float, nargs="+", default=list(CAUSE_RHOS),
                   help="Randomized shares of the budget N (default: %(default)s).")
    g.add_argument("--cause-variants", nargs="+", default=list(CAUSE_VARIANTS), choices=list(CAUSE_VARIANTS),
                   help="prod (gives prod-C and prod-T) and/or avg.")
    g.add_argument("--cause-dim", type=int, default=CAUSE_DEFAULTS["dim"], help="Embedding dimension (default 32).")
    g.add_argument("--cause-optimizer", default=CAUSE_DEFAULTS["optimizer"], choices=list(CAUSE_OPTIMIZERS),
                   help="momentum_decay (the paper's description) or sgd (the released code).")
    g.add_argument("--cause-tie", default=CAUSE_DEFAULTS["tie"], choices=list(CAUSE_TIES),
                   help="one_way (released) or symmetric (the paper's eq. 18).")
    g.add_argument("--cause-batch-size", type=int, default=CAUSE_DEFAULTS["batch_size"])
    g.add_argument("--cause-trials", type=int, default=None, help="Trials per variant (default: --n-trials).")


def cause_options_from_args(args) -> dict:
    return {"rhos": [float(r) for r in args.cause_rhos], "variants": list(args.cause_variants),
            "dim": int(args.cause_dim), "optimizer": str(args.cause_optimizer), "tie": str(args.cause_tie),
            "batch_size": int(args.cause_batch_size), "n_trials": args.cause_trials}


def _exact_values(dataset: dict, model: CausEModel, rows: np.ndarray) -> tuple[float, float]:
    """(stochastic, greedy) true values of the policies the logits define over the catalog."""
    ux, ia = model.policy_vectors(rows)
    policy = SimpleNamespace(user_emb=ux, item_emb=ia, temperature=1.0, action_chunk=8192)
    return float(calc_reward(dataset, policy)), float(calc_greedy_reward(dataset, ux, ia))


@torch.no_grad()
def _diagnostics(model: CausEModel, layout: CausELayout) -> dict:
    p = model.item_emb.detach()
    c_rows = torch.as_tensor(layout.prediction_rows("control"), device=p.device)
    out = {"alpha": float(model.alpha), "rms_control": float(p[c_rows].pow(2).mean().sqrt())}
    if layout.variant == "prod":
        t_rows = c_rows + layout.n_items
        out["rms_treatment"] = float(p[t_rows].pow(2).mean().sqrt())
        out["mean_l1_treatment_minus_control"] = float((p[t_rows] - p[c_rows]).abs().sum(dim=1).mean())
    else:
        out["pooled_norm"] = float(p[layout.pooled_row].norm())
    return out


def _training_arrays(split: dict, layout: CausELayout):
    c, t = split["control"], split["treatment"]
    users = np.concatenate([c["x_idx"], t["x_idx"]]).astype(np.int64)
    actions = np.concatenate([c["a"], t["a"]]).astype(np.int64)
    treat = np.concatenate([np.zeros(len(c["a"]), bool), np.ones(len(t["a"]), bool)])
    labels = np.concatenate([c["r"], t["r"]]).astype(np.float32)
    return users, layout.train_rows(actions, treat), labels


def cause_trainer_trial(
    *,
    train_sizes,
    dataset: dict,
    split_cache,
    condition_seed: int,
    seed: int,
    n_trials: int,
    log_constants: dict,
    options: dict | None = None,
    sampler: str = "tpe",
    stage: str = "development",
    device: torch.device | str = "cpu",
    run_idx: int = 0,
) -> dict:
    """``{method_label: (summary_df indexed by train size, trials_df)}`` for every (prediction, rho)."""
    import optuna

    opts = {**CAUSE_DEFAULTS, **(options or {})}
    n_trials = int(opts["n_trials"] or n_trials)
    symmetric = opts["tie"] == "symmetric"
    ensure_exact_env_q_cache(dataset)
    n_users, n_actions = int(dataset["n_users"]), int(dataset["n_actions"])
    v_logger = float(log_constants["initial_reward"])
    v_logger_greedy = float(log_constants["logger_greedy"])
    v_uniform = float(calc_uniform_reward(dataset))
    summaries: dict[str, list] = {}
    trials_by_label: dict[str, list] = {}
    for n in [int(x) for x in train_sizes]:
        split = split_cache[(n, run_idx)]
        warm, val = split["train_data"], split["val_data"]
        pool = simulate_uniform_pool(dataset, n, seed=uniform_pool_seed(condition_seed, n))
        val_users = np.asarray(val["x_idx"], dtype=np.int64)
        val_actions = np.asarray(val["a"], dtype=np.int64)
        val_labels = np.asarray(val["r"], dtype=np.float64)
        warm_reward_sum = float(np.sum(warm["r"]))
        for rho in [float(r) for r in opts["rhos"]]:
            budget = build_budget_split(dataset, warm, pool, n, rho)
            meta = budget["meta"]
            for variant in opts["variants"]:
                layout = CausELayout(variant, n_actions)
                users, rows, labels = _training_arrays(budget, layout)
                preds = [p for p, (v, _side) in CAUSE_PREDICTIONS.items() if v == variant]
                pred_rows = {p: layout.prediction_rows(CAUSE_PREDICTIONS[p][1]) for p in preds}
                labels_seed = ("cause", variant, n)  # no rho: the same configurations and seeds at every rho
                trial_rows = []

                def objective(trial):
                    t0 = time.time()
                    lr = trial.suggest_float("lr", *CAUSE_LR_RANGE, log=True)
                    epochs = trial.suggest_categorical("epochs", list(CAUSE_EPOCHS))
                    l2_pen = trial.suggest_categorical("l2_pen", list(CAUSE_L2))
                    cf_pen = trial.suggest_categorical("cf_pen", list(CAUSE_CF_PEN))
                    trial_seed = derive_seed(seed, *labels_seed, "trial", trial.number)
                    seed_everything(trial_seed)
                    gen = torch.Generator().manual_seed(derive_seed(trial_seed, "init"))
                    model = CausEModel.for_layout(layout, n_users, int(opts["dim"]), generator=gen)
                    info = fit_cause(model, users, rows, labels, epochs=int(epochs), batch_size=int(opts["batch_size"]),
                                     optimizer=str(opts["optimizer"]), lr=lr, l2_pen=float(l2_pen), cf_pen=float(cf_pen),
                                     symmetric=symmetric, seed=derive_seed(trial_seed, "order"), device=device)
                    rec = {"train_size": n, "rho": rho, "variant": variant, "trial": trial.number, "lr": lr,
                           "epochs": int(epochs), "l2_pen": float(l2_pen), "cf_pen": float(cf_pen),
                           "steps": info["steps"], "finite": info["finite"], **_diagnostics(model, layout)}
                    for p in preds:
                        if info["finite"]:
                            z = predict_logits(model, val_users, pred_rows[p][val_actions])
                            m = prediction_metrics(z, val_labels)
                            m["nll"] = m["nll"] if np.isfinite(m["nll"]) else DIVERGED_NLL
                            v_soft, v_greedy = _exact_values(dataset, model, pred_rows[p])
                        else:
                            m = {"nll": DIVERGED_NLL, "mse": np.nan, "auc": np.nan}
                            v_soft = v_greedy = np.nan
                        rec.update({f"{p}_val_nll": m["nll"], f"{p}_val_mse": m["mse"], f"{p}_val_auc": m["auc"],
                                    f"{p}_value": v_soft, f"{p}_value_greedy": v_greedy})
                    rec["seconds"] = time.time() - t0
                    trial_rows.append(rec)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Optuna follows the first prediction (prod-C for prod, avg for avg); each prediction is
                    # selected separately below from the same trials
                    return float(rec[f"{preds[0]}_val_nll"])

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction="minimize",
                                            sampler=optuna_sampler(seed, *labels_seed, kind=str(sampler)))
                t_study = time.time()
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                trials_df = pd.DataFrame(trial_rows)
                print(f"[cause] N={n} rho={rho:g} {variant}: {len(trials_df)} trials in {time.time() - t_study:.0f}s; "
                      f"finite {int(trials_df['finite'].sum())}; best {preds[0]} val NLL "
                      f"{trials_df[f'{preds[0]}_val_nll'].min():.4f}", flush=True)
                for p in preds:
                    label = cause_method_label(p, rho)
                    finite = trials_df[trials_df[f"{p}_val_nll"] < DIVERGED_NLL]
                    best = finite.loc[finite[f"{p}_val_nll"].idxmin()] if len(finite) else trials_df.iloc[0]
                    oracle = trials_df.loc[trials_df[f"{p}_value_greedy"].idxmax()] if trials_df[f"{p}_value_greedy"].notna().any() else best
                    v_soft, v_greedy = float(best[f"{p}_value"]), float(best[f"{p}_value_greedy"])
                    row = {
                        "train_size": n, "cause_prediction": p, "cause_variant": variant, "cause_rho": rho,
                        "policy_rewards": v_soft, "policy_rewards_greedy": v_greedy,
                        "initial_reward": v_logger, "logger_greedy": v_logger_greedy, "uniform_value": v_uniform,
                        "gain_over_logger": v_soft - v_logger, "gain_over_logger_greedy": v_greedy - v_logger_greedy,
                        "selection_val_score": float(best[f"{p}_val_nll"]), "selection_metric": "val_nll",
                        "val_nll": float(best[f"{p}_val_nll"]), "val_mse": float(best[f"{p}_val_mse"]),
                        "val_auc": float(best[f"{p}_val_auc"]), "val_size": int(len(val_labels)),
                        "selected_trial": int(best["trial"]), "lr": float(best["lr"]), "epochs": int(best["epochs"]),
                        "l2_pen": float(best["l2_pen"]), "cf_pen": float(best["cf_pen"]), "alpha": float(best["alpha"]),
                        "n_finite_trials": int(len(finite)), "n_trials": int(len(trials_df)),
                        "oracle_selected_value_greedy": float(oracle[f"{p}_value_greedy"]),
                        "oracle_selected_value": float(oracle[f"{p}_value"]),
                        "cause_dim": int(opts["dim"]), "cause_optimizer": str(opts["optimizer"]),
                        "cause_tie": str(opts["tie"]), "cause_batch_size": int(opts["batch_size"]),
                        "cause_sampler": str(sampler), "stage": str(stage),
                        **{k: v for k, v in meta.items() if k != "rho"},
                        "opc_collection_reward_sum": warm_reward_sum,
                        "exploration_cost_expected": rho * n * (v_logger - v_uniform),
                        "exploration_cost_realised": warm_reward_sum - meta["collection_reward_sum"],
                    }
                    summaries.setdefault(label, []).append(row)
                    trials_by_label.setdefault(label, []).append(trials_df.assign(method=label, prediction=p))
    out = {}
    for label, rows in summaries.items():
        df = pd.DataFrame(rows).set_index("train_size")
        out[label] = (df, pd.concat(trials_by_label[label], ignore_index=True))
    return out
