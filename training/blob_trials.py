"""BLOB-supplied-source as a study arm (docs/blob_controlled_integration.md §3).

For each train size N:
* data: the same N warm-logger rows every other arm trains on (the training split), no propensities, no
  randomized rows; selection on the 20,000 warm validation rows shared by every arm;
* source: the logger's own vectors take the place of BLOB's organic model. Ψ = the item vectors ``our_a`` and
  ω̂ = the user vectors ``our_x`` / RMS(our_x): the logger's softmax(⟨x, a⟩ / T) plays BLOB's organic
  softmax(Ψω + ρ), and dividing by the RMS puts ω̂ on the scale of BLOB's N(0, I) organic prior;
* model: models/blob.py, the released bandit layer (``mnq`` or ``nq``; Ψ normalization as released);
* search: ``n_trials`` seeded random trials over lr, epochs and the priors' material hyperparameters, trained
  together (``fit_blob_batch``) in groups that share an epoch count;
* selection: validation NLL of the posterior-mean click model (with its intercept w_c);
* outcomes: the exact true value of its greedy policy argmax_a ω̂ β̂_a + κ̂_a (the released recommendation) and of
  the softmax of those logits, raw (τ = 1) and with the fair tempering of the CausE arm (§2 of
  docs/cause_fair_comparison_25k.md), plus diagnostics;
* optionally (``pick_diagnostics``) every trial's pick-level diagnostics against the truth and the logger
  (training/policy_diagnostics.py), and (``policy_dir``) the selected trial's policy saved for the cross-arm pass.
Each family is reported as its own method, ``blob_<family>``.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.blob import BLOB_FAMILIES, BlobBanditBatch, BlobPriors, fit_blob_batch
from models.cause import prediction_metrics
from utils.seeding import derive_seed, optuna_sampler
from utils.simulation_utils import calc_greedy_reward, calc_reward, calc_uniform_reward, ensure_exact_env_q_cache

BLOB_LR_RANGE = (1e-4, 3e-2)
BLOB_EPOCHS = (10, 30, 100, 300)
BLOB_WA_M = (-1.0,)
BLOB_WB_M = (-6.0,)
BLOB_KAPPA_S = (0.01,)
DIVERGED_NLL = 1e9
BLOB_DEFAULTS = {"families": ["nq"], "n_trials": None, "lr_range": list(BLOB_LR_RANGE), "epochs": list(BLOB_EPOCHS),
                 "wa_m": list(BLOB_WA_M), "wb_m": list(BLOB_WB_M), "kappa_s": list(BLOB_KAPPA_S), "batch_size": 1024,
                 "temper": True, "norm": True, "alias_loc": True, "device": "auto", "source_scale": "rms",
                 "pick_diagnostics": False, "policy_dir": None}


def blob_method_label(family: str) -> str:
    return f"blob_{family}"


def add_blob_arguments(parser) -> None:
    g = parser.add_argument_group("BLOB-supplied-source baseline (opt-in: --methods ... blob)")
    g.add_argument("--blob-families", nargs="+", default=list(BLOB_DEFAULTS["families"]), choices=list(BLOB_FAMILIES),
                   help="Variational families: nq (BLOB-NQ) and/or mnq (BLOB-MNQ).")
    g.add_argument("--blob-trials", type=int, default=None, help="Trials per family (default: --n-trials).")
    g.add_argument("--blob-lr-range", type=float, nargs=2, default=list(BLOB_LR_RANGE), metavar=("LOW", "HIGH"))
    g.add_argument("--blob-epochs", type=int, nargs="+", default=list(BLOB_EPOCHS))
    g.add_argument("--blob-wa-m", type=float, nargs="+", default=list(BLOB_WA_M),
                   help="Prior mean of w_a (searched when several; released -1).")
    g.add_argument("--blob-wb-m", type=float, nargs="+", default=list(BLOB_WB_M),
                   help="Prior mean of w_b, the K x K deviation's scale (released -6).")
    g.add_argument("--blob-kappa-s", type=float, nargs="+", default=list(BLOB_KAPPA_S),
                   help="Prior std of the per-item intercepts kappa (released 0.01).")
    g.add_argument("--blob-batch-size", type=int, default=BLOB_DEFAULTS["batch_size"])
    g.add_argument("--blob-no-temper", action="store_true", help="Skip the fair tempering of the selected trial.")
    g.add_argument("--blob-device", default="auto", choices=["auto", "cpu"])
    g.add_argument("--blob-pick-diagnostics", action="store_true",
                   help="Record every trial's pick-level diagnostics (training/policy_diagnostics.py).")


def blob_options_from_args(args) -> dict:
    return {"families": list(args.blob_families), "n_trials": args.blob_trials,
            "lr_range": [float(x) for x in args.blob_lr_range], "epochs": [int(x) for x in args.blob_epochs],
            "wa_m": [float(x) for x in args.blob_wa_m], "wb_m": [float(x) for x in args.blob_wb_m],
            "kappa_s": [float(x) for x in args.blob_kappa_s], "batch_size": int(args.blob_batch_size),
            "temper": not bool(args.blob_no_temper), "device": str(args.blob_device),
            "pick_diagnostics": bool(args.blob_pick_diagnostics)}


def supplied_source(dataset: dict, scale: str = "rms") -> tuple[np.ndarray, np.ndarray]:
    """(ω̂, Ψ): the logger's user vectors divided by their RMS (``scale='rms'``; ``'none'`` keeps them) and its item
    vectors."""
    x = np.asarray(dataset["our_x"], dtype=np.float32)
    a = np.asarray(dataset["our_a"], dtype=np.float32)
    if scale == "rms":
        x = x / float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    elif scale != "none":
        raise ValueError(f"scale must be 'rms' or 'none', got {scale!r}")
    return x, a


def _distributions(lr_range, epochs, wa_m, wb_m, kappa_s):
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    d = {"lr": FloatDistribution(*lr_range, log=True), "epochs": CategoricalDistribution(list(epochs))}
    for name, values in (("wa_m", wa_m), ("wb_m", wb_m), ("kappa_s", kappa_s)):
        if len(values) > 1:  # a single value is fixed, not drawn, so its draws do not shift the others
            d[name] = CategoricalDistribution(list(values))
    return d


class _Policy:
    """The policy vectors of a trained trial: ux = [ω̂, 1], ia = [β̂, κ̂], so ux·ia = ω̂ β̂_a + κ̂_a."""

    def __init__(self, omega: np.ndarray, beta: torch.Tensor, kappa: torch.Tensor):
        self.ux = np.concatenate([omega, np.ones((omega.shape[0], 1), np.float32)], axis=1).astype(np.float32)
        self.ia = torch.cat([beta, kappa[:, None]], dim=1).float().cpu().numpy()

    def policy_vectors(self, rows=None):
        return self.ux, self.ia if rows is None else self.ia[np.asarray(rows)]


def _diagnostics(model: BlobBanditBatch, t: int) -> dict:
    sp = torch.nn.functional.softplus
    with torch.no_grad():
        zeta = model.zeta_means[t].reshape(model.K, model.K)
        return {"sp_wa": float(sp(model.wa_means[t, 0])), "sp_wb": float(sp(model.wb_means[t, 0])),
                "wc": float(model.bias_means[t, 0]), "zeta_norm": float(zeta.norm()),
                "kappa_rms": float(model.kappa_means[t, :, 0].pow(2).mean().sqrt()),
                "kappa_sd_post": float(torch.exp(model.kappa_logstd[t]).mean()),
                # the bandit term's size relative to the organic term, s+(w_b)‖L ζᵀ‖ / s+(w_a)
                "deviation_ratio": float(sp(model.wb_means[t, 0]) * (model.chol @ zeta.T).norm() /
                                         (sp(model.wa_means[t, 0]) * np.sqrt(model.K)))}


def blob_trainer_trial(
    *,
    train_sizes,
    dataset: dict,
    split_cache,
    seed: int,
    n_trials: int,
    log_constants: dict,
    options: dict | None = None,
    sampler: str = "random",
    stage: str = "development",
    device: torch.device | str = "cpu",
    run_idx: int = 0,
) -> dict:
    """``{blob_<family>: (summary_df indexed by train size, trials_df)}``."""
    import optuna

    from training.cause_trials import _greedy_dr, _tempered
    from training.policy_diagnostics import POLICY_SUFFIX, greedy_pick_diagnostics, logger_reference, save_selected_policy
    from training.trainer_trials import _scores_lookup_from_bundle, fit_shared_regression_bundle

    opts = {**BLOB_DEFAULTS, **(options or {})}
    if str(sampler) != "random":
        raise ValueError("the BLOB arm uses the paired random sampler")
    n_trials = int(opts["n_trials"] or n_trials)
    dist = _distributions(tuple(opts["lr_range"]), tuple(opts["epochs"]), tuple(opts["wa_m"]), tuple(opts["wb_m"]),
                          tuple(opts["kappa_s"]))
    ensure_exact_env_q_cache(dataset)
    omega, psi = supplied_source(dataset, str(opts["source_scale"]))
    v_logger = float(log_constants["initial_reward"])
    v_logger_greedy = float(log_constants["logger_greedy"])
    v_uniform = float(calc_uniform_reward(dataset))
    ref = logger_reference(dataset) if opts["pick_diagnostics"] else None
    out = {}
    for n in [int(x) for x in train_sizes]:
        split = split_cache[(n, run_idx)]
        warm, val = split["train_data"], split["val_data"]
        users = np.asarray(warm["x_idx"], dtype=np.int64)
        actions = np.asarray(warm["a"], dtype=np.int64)
        clicks = np.asarray(warm["r"], dtype=np.float32)
        val_users = np.asarray(val["x_idx"], dtype=np.int64)
        val_actions = np.asarray(val["a"], dtype=np.int64)
        val_labels = np.asarray(val["r"], dtype=np.float64)
        lookup = None
        if opts["temper"]:  # q_hat of the same N warm rows (= OPC's training rows)
            bundle = fit_shared_regression_bundle(dataset, {k: warm[k] for k in ("x", "a", "r", "x_idx", "pscore")})
            lookup = _scores_lookup_from_bundle(bundle, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for family in opts["families"]:
            label = blob_method_label(family)
            labels_seed = ("blob", family, n)
            study = optuna.create_study(direction="minimize", sampler=optuna_sampler(seed, *labels_seed, kind="random"))
            asked = [study.ask(dist) for _ in range(n_trials)]
            groups: dict[int, list] = {}
            for tr in asked:
                groups.setdefault(int(tr.params["epochs"]), []).append(tr)
            done, best_vectors = {}, None
            t_study = time.time()
            order_seed = derive_seed(seed, *labels_seed, "order")
            for epochs, group in sorted(groups.items()):
                t0 = time.time()
                priors = [BlobPriors(wa_m=float(tr.params.get("wa_m", opts["wa_m"][0])),
                                     wb_m=float(tr.params.get("wb_m", opts["wb_m"][0])),
                                     kappa_s=float(tr.params.get("kappa_s", opts["kappa_s"][0]))) for tr in group]
                model = BlobBanditBatch(psi, priors, family=family, norm=bool(opts["norm"]),
                                        alias_loc=bool(opts["alias_loc"])).to(device)
                info = fit_blob_batch(model, omega[users], actions, clicks, epochs=epochs,
                                      lrs=[tr.params["lr"] for tr in group], batch_size=int(opts["batch_size"]),
                                      order_seed=order_seed, noise_seed=derive_seed(seed, *labels_seed, "noise", epochs),
                                      device=device)
                for i, tr in enumerate(group):
                    rec = {"train_size": n, "trial": tr.number, "lr": float(tr.params["lr"]), "epochs": epochs,
                           "wa_m": priors[i].wa_m, "wb_m": priors[i].wb_m, "kappa_s": priors[i].kappa_s,
                           "steps": int(info["steps"]), "finite": bool(info["finite"][i])}
                    if rec["finite"]:
                        beta, kappa, wc = model.point(i)
                        pol = _Policy(omega, beta, kappa)
                        ux, ia = pol.policy_vectors()
                        z = (ux[val_users] * ia[val_actions]).sum(1) + wc  # the click model includes w_c
                        m = prediction_metrics(z, val_labels)
                        rec.update({"val_nll": m["nll"] if np.isfinite(m["nll"]) else DIVERGED_NLL,
                                    "val_mse": m["mse"], "val_auc": m["auc"],
                                    "value": float(calc_reward(dataset, SimpleNamespace(user_emb=ux, item_emb=ia,
                                                                                        temperature=1.0,
                                                                                        action_chunk=8192))),
                                    "value_greedy": float(calc_greedy_reward(dataset, ux, ia)),
                                    **_diagnostics(model, i)})
                        if lookup is not None:
                            g_hat, g_low = _greedy_dr(val, ux, ia, lookup)
                            rec.update({"val_dr_greedy": g_hat, "val_dr_greedy_low": g_low})
                        if ref is not None:
                            rec.update({f"diag_{k}": v for k, v in
                                        greedy_pick_diagnostics(dataset, ux, ia, offset=wc, ref=ref).items()})
                        key = (rec["val_nll"], tr.number)
                        if rec["val_nll"] < DIVERGED_NLL and (best_vectors is None or key < best_vectors[0]):
                            best_vectors = (key, (ux, ia), wc)
                    else:
                        rec.update({"val_nll": DIVERGED_NLL, "value": np.nan, "value_greedy": np.nan})
                    rec["seconds"] = (time.time() - t0) / len(group)
                    done[tr.number] = rec
                del model
            trials_df = pd.DataFrame([done[tr.number] for tr in asked])
            for tr in asked:
                study.tell(tr, float(done[tr.number]["val_nll"]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            finite = trials_df[trials_df["val_nll"] < DIVERGED_NLL]
            print(f"[blob:{family}] N={n}: {len(trials_df)} trials in {time.time() - t_study:.0f}s; finite {len(finite)}; "
                  f"best val NLL {finite['val_nll'].min() if len(finite) else float('nan'):.4f}", flush=True)
            best = finite.loc[finite["val_nll"].idxmin()] if len(finite) else trials_df.iloc[0]
            row = {"train_size": n, "blob_family": family, "policy_rewards": float(best["value"]),
                   "policy_rewards_greedy": float(best["value_greedy"]), "initial_reward": v_logger,
                   "logger_greedy": v_logger_greedy, "uniform_value": v_uniform,
                   "gain_over_logger": float(best["value"]) - v_logger,
                   "gain_over_logger_greedy": float(best["value_greedy"]) - v_logger_greedy,
                   "selection_val_score": float(best["val_nll"]), "selection_metric": "val_nll",
                   "val_nll": float(best["val_nll"]), "val_auc": float(best.get("val_auc", np.nan)),
                   "val_size": int(len(val_labels)), "selected_trial": int(best["trial"]),
                   "lr": float(best["lr"]), "epochs": int(best["epochs"]), "wa_m": float(best["wa_m"]),
                   "wb_m": float(best["wb_m"]), "kappa_s": float(best["kappa_s"]),
                   "n_finite_trials": int(len(finite)), "n_trials": int(len(trials_df)),
                   "oracle_selected_value_greedy": float(trials_df["value_greedy"].max()),
                   "oracle_selected_value": float(trials_df["value"].max()),
                   "blob_norm": bool(opts["norm"]), "blob_alias_loc": bool(opts["alias_loc"]),
                   "blob_source_scale": str(opts["source_scale"]), "blob_batch_size": int(opts["batch_size"]),
                   "blob_sampler": "random", "stage": str(stage), "n_total": int(n)}
            for k in ("sp_wa", "sp_wb", "wc", "zeta_norm", "kappa_rms", "kappa_sd_post", "deviation_ratio",
                      "val_dr_greedy", "val_dr_greedy_low"):
                if k in best:
                    row[k] = float(best[k])
            if lookup is not None and best_vectors is not None:
                extra = _tempered(dataset, val, *best_vectors[1], lookup)
                assert int(best["trial"]) == best_vectors[0][1], "tempered a different trial"
                row.update({"policy_rewards_tempered": float(extra["value_tempered"]),
                            "temper_scale": float(extra["temper_scale"]), "val_dr_tempered": float(extra["val_dr_tempered"]),
                            "val_dr_tempered_low": float(extra["val_dr_tempered_low"])})
            if opts.get("policy_dir") and best_vectors is not None:
                assert int(best["trial"]) == best_vectors[0][1], "saving a different trial"
                save_selected_policy(Path(opts["policy_dir"]) / f"{label}_n{n}{POLICY_SUFFIX}", *best_vectors[1],
                                     offset=best_vectors[2], arm=label, trial=int(best["trial"]),
                                     value_greedy=float(best["value_greedy"]), value=float(best["value"]))
            out[label] = (pd.DataFrame([row]).set_index("train_size"), trials_df.assign(method=label))
    return out
