"""Native CausE as a study arm: fixed-budget data, a seeded search, selection by validation NLL, exact values.

For each train size N and randomized share rho (docs/cause_baseline.md §4):
* data: N_c = N - round(rho·N) warm-logger rows (the first rows of the training split every other arm uses)
  plus N_t = round(rho·N) uniform-random rows from the same world (utils/budget_split.py);
* model: models/cause.py, from scratch, CausE-prod (prediction with the control rows = prod-C, or the
  treatment rows = prod-T, from one trained model) and CausE-avg;
* search: ``n_trials`` seeded Optuna trials per variant over lr, epochs, L2 and the tie strength. The
  configurations and trial seeds do not depend on rho, so with ``sampler='random'`` the rho curve is paired.
  Every trial of a study trains on the same seeded batch order (its own initial values). With the random
  sampler the configurations are drawn up front and trials sharing an epoch count train together
  (``fit_cause_batch``: trial k equals its own single-model run);
* selection: validation NLL of each prediction on the warm validation rows shared with every arm;
* outcomes: the exact true value of the greedy (argmax) policy and of the softmax over CausE's logits
  (the softmax is a convention, not part of the paper), plus collection rewards and diagnostics.
Each (prediction, rho) is reported as its own method, ``cause_<prediction>_r<rho in per mille>``.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.cause import (CAUSE_OPTIMIZERS, CAUSE_PREDICTIONS, CAUSE_TIES, CAUSE_VARIANTS, CausELayout, CausELinModel,
                          CausEModel, fit_cause, fit_cause_batch, predict_logits, prediction_metrics, warm_start_)
from utils.budget_split import CAUSE_RHOS, build_budget_split, simulate_uniform_pool, uniform_pool_seed
from utils.seeding import derive_seed, optuna_sampler
from utils.simulation_utils import calc_greedy_reward, calc_reward, calc_uniform_reward, ensure_exact_env_q_cache

CAUSE_EPOCHS = (1, 3, 10, 30, 100, 300)
CAUSE_L2 = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
CAUSE_CF_PEN = (0.0, 0.1, 1.0, 10.0, 100.0)
CAUSE_LR_RANGE = (1e-3, 1.0)
DIVERGED_NLL = 1e9


# CausE families (docs/cause_fair_comparison_25k.md §1): native = the published model from scratch (M5); warm = native
# capacity started at OPC's source vectors (CausE-warm); cap = OPC's correction family on the frozen source vectors
# (CausE-capacity-matched). warm and cap train the prod layout and predict with the control (c) or treatment (t) rows.
CAUSE_FAMILIES = ("native", "warm", "cap")
FAMILY_PREDICTIONS = {"c": "control", "t": "treatment"}
FAMILY_LABEL = {"native": "cause", "warm": "causewarm", "cap": "causecap"}
# where a warm-started model's intercept starts: 0 (the native initialization) or the logit of the click rate of its own
# training rows. With source vectors, whose logged dot products are large and positive, a zero intercept makes the
# scale alpha absorb the initial miscalibration (sigma(0) = 0.5 against a ~10% click rate) and can drive it negative
BIAS_INITS = ("zero", "base_rate")
# CausE's tempering grid: its logits are click log-odds, much flatter than a policy's, so the grid reaches higher than
# OPC's post-tempering grid (0.25-16)
CAUSE_TEMPER_GRID = tuple(float(2.0 ** k) for k in range(-2, 13))
TEMPER_FIELDS = ("temper_scale", "value_tempered", "val_dr_tempered", "val_dr_tempered_low")


def _distributions(lr_range=CAUSE_LR_RANGE, epochs=CAUSE_EPOCHS, l2=CAUSE_L2, cf=CAUSE_CF_PEN, ties=None, bias_inits=None):
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    # insertion order = the order of the suggest_* calls of the sequential search; the tie direction is searched only
    # when more than one is given (the native default has one, so its draws are unchanged)
    d = {"lr": FloatDistribution(*lr_range, log=True), "epochs": CategoricalDistribution(list(epochs)),
         "l2_pen": CategoricalDistribution(list(l2)), "cf_pen": CategoricalDistribution(list(cf))}
    if ties is not None and len(ties) > 1:
        d["tie"] = CategoricalDistribution(list(ties))
    if bias_inits is not None and len(bias_inits) > 1:
        d["bias_init"] = CategoricalDistribution(list(bias_inits))
    return d


CAUSE_DISTRIBUTIONS = _distributions()
CAUSE_DEFAULTS = {"rhos": list(CAUSE_RHOS), "variants": list(CAUSE_VARIANTS), "dim": 32,
                  "optimizer": "momentum_decay", "tie": "one_way", "batch_size": 512, "n_trials": None, "device": "auto",
                  "family": "native", "lr_range": list(CAUSE_LR_RANGE), "epochs": list(CAUSE_EPOCHS),
                  "l2": list(CAUSE_L2), "cf": list(CAUSE_CF_PEN), "ties": None, "bias_inits": None, "temper": False,
                  "temper_trials": "selected"}


def cause_method_label(prediction: str, rho: float, family: str = "native") -> str:
    return f"{FAMILY_LABEL[family]}_{prediction}_r{int(round(float(rho) * 1000)):03d}"


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
    g.add_argument("--cause-device", default="auto", choices=["auto", "cpu"],
                   help="Where CausE trains: auto (the GPU when there is one) or cpu (many runs in parallel on the "
                        "cores; exact evaluation still uses the GPU).")
    g.add_argument("--cause-family", default="native", choices=list(CAUSE_FAMILIES),
                   help="native (the published model from scratch), warm (CausE-warm: native capacity started at "
                        "OPC's source vectors) or cap (CausE-capacity-matched: OPC's (I + D)x + b family); "
                        "docs/cause_fair_comparison_25k.md §1.")
    g.add_argument("--cause-lr-range", type=float, nargs=2, default=list(CAUSE_LR_RANGE), metavar=("LOW", "HIGH"))
    g.add_argument("--cause-epochs", type=int, nargs="+", default=list(CAUSE_EPOCHS))
    g.add_argument("--cause-l2", type=float, nargs="+", default=list(CAUSE_L2))
    g.add_argument("--cause-cf", type=float, nargs="+", default=list(CAUSE_CF_PEN))
    g.add_argument("--cause-ties", nargs="+", default=None, choices=list(CAUSE_TIES),
                   help="Search the tie direction over these (default: --cause-tie only).")
    g.add_argument("--cause-bias-inits", nargs="+", default=None, choices=list(BIAS_INITS),
                   help="Warm families: where the intercept starts (searched when several are given; default zero).")
    g.add_argument("--cause-temper", action="store_true",
                   help="Also temper each selected model's softmax by the DR lower bound on validation (clip:10; q_hat "
                        "fit on CausE's own rows at each rho), and record DR estimates of its greedy and tempered "
                        "policies (docs/cause_fair_comparison_25k.md §2).")
    g.add_argument("--cause-temper-trials", default="selected", choices=["selected", "all"],
                   help="With --cause-temper: search the scale for each prediction's selected trial only (default; the "
                        "selection does not depend on it) or for every trial (the tuning stage's diagnostics). Every "
                        "trial gets the DR estimate of its greedy policy either way.")


def cause_options_from_args(args) -> dict:
    return {"rhos": [float(r) for r in args.cause_rhos], "variants": list(args.cause_variants),
            "dim": int(args.cause_dim), "optimizer": str(args.cause_optimizer), "tie": str(args.cause_tie),
            "batch_size": int(args.cause_batch_size), "n_trials": args.cause_trials, "device": str(args.cause_device),
            "family": str(args.cause_family), "lr_range": [float(x) for x in args.cause_lr_range],
            "epochs": [int(x) for x in args.cause_epochs], "l2": [float(x) for x in args.cause_l2],
            "cf": [float(x) for x in args.cause_cf], "ties": None if args.cause_ties is None else list(args.cause_ties),
            "bias_inits": None if args.cause_bias_inits is None else list(args.cause_bias_inits),
            "temper": bool(args.cause_temper), "temper_trials": str(args.cause_temper_trials)}


def _exact_values(dataset: dict, model: CausEModel, rows: np.ndarray) -> tuple[float, float]:
    """(stochastic, greedy) true values of the policies the logits define over the catalog."""
    ux, ia = model.policy_vectors(rows)
    policy = SimpleNamespace(user_emb=ux, item_emb=ia, temperature=1.0, action_chunk=8192)
    return float(calc_reward(dataset, policy)), float(calc_greedy_reward(dataset, ux, ia))


def click_offset(model) -> np.ndarray:
    """What the click logit adds to ``policy_vectors``' dot product: the global bias, plus the user's bias in the
    free-vector model (one value per user)."""
    with torch.no_grad():
        if isinstance(model, CausELinModel):
            return np.asarray(float(model.global_bias), dtype=np.float32)
        return (model.user_bias + model.global_bias).float().cpu().numpy()


@torch.no_grad()
def _diagnostics(model, layout: CausELayout, source: tuple | None = None) -> dict:
    """alpha and representation diagnostics; for warm starts also how far the vectors moved from the source."""
    if isinstance(model, CausELinModel):
        norm = lambda m: float(torch.cat([m.delta.reshape(-1), m.bias]).norm())
        a = model.A
        gap = (model.treatment_map(a) - model.control_map(a)).abs().sum(dim=1).mean()
        return {"alpha": float(model.alpha), "map_norm_user": norm(model.user_map),
                "map_norm_control": norm(model.control_map), "map_norm_treatment": norm(model.treatment_map),
                "mean_l1_treatment_minus_control": float(gap)}
    out = _diagnostics_native(model, layout)
    if source is not None:
        x, a = (torch.as_tensor(v, device=model.user_emb.device) for v in source)
        n = layout.n_items
        out["mean_l1_users_from_source"] = float((model.user_emb - x).abs().sum(dim=1).mean())
        out["mean_l1_control_from_source"] = float((model.item_emb[:n] - a).abs().sum(dim=1).mean())
        out["mean_l1_treatment_from_source"] = float((model.item_emb[n:2 * n] - a).abs().sum(dim=1).mean())
    return out


@torch.no_grad()
def _diagnostics_native(model: CausEModel, layout: CausELayout) -> dict:
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


def _merged_rows(budget: dict) -> dict:
    """CausE's own N training rows (warm then uniform) as one logged slice, for the q_hat behind its tempering."""
    c, t = budget["control"], budget["treatment"]
    return {k: np.concatenate([np.asarray(c[k]), np.asarray(t[k])]) for k in ("x", "a", "r", "x_idx", "pscore")}


def _greedy_dr(val: dict, ux: np.ndarray, ia: np.ndarray, lookup, *, clip: float = 10.0, chunk: int = 2048) -> tuple:
    """(point, 95% lower bound) of the DR estimate (weights clipped at ``clip``) of the greedy policy argmax_j ux·ia on
    the validation rows: DM = q_hat(u, g) plus the weighted correction 1[a = g] / pscore · (r − q_hat(u, a))."""
    from scipy.stats import t as student_t

    users = np.asarray(val["x_idx"], dtype=np.int64)
    actions = np.asarray(val["a"], dtype=np.int64)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ia_t = torch.as_tensor(ia, device=device)
    greedy = np.empty(len(users), dtype=np.int64)
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(ux[users[s:s + chunk]], device=device)
        greedy[s:s + chunk] = (u @ ia_t.T).argmax(dim=1).cpu().numpy()
    ctx = lookup.user_context[users]
    q_g = np.asarray(lookup.regression_model.predict_pairs(ctx, greedy), dtype=np.float64)
    q_a = np.asarray(lookup.regression_model.predict_pairs(ctx, actions), dtype=np.float64)
    w = np.minimum((greedy == actions) / np.asarray(val["pscore"], dtype=np.float64), clip)
    v = q_g + w * (np.asarray(val["r"], dtype=np.float64) - q_a)
    n = max(len(v), 2)
    r_hat = float(v.mean())
    return r_hat, r_hat - float(student_t.ppf(0.975, n - 1)) * float(v.std(ddof=1) / np.sqrt(n))


def _tempered(dataset: dict, val: dict, ux: np.ndarray, ia: np.ndarray, lookup) -> dict:
    """The fair sharpening of a click predictor's softmax (docs/cause_fair_comparison_25k.md §2): its logits ux·ia × s,
    s in ``CAUSE_TEMPER_GRID``, chosen by the DR lower bound (clip:10) on the validation rows, as the tempered logger's
    scale is; the exact value of the tempered softmax, and the DR estimates of it and of the greedy policy."""
    from scipy.stats import t as student_t

    from training.trainer_trials import _policy_temperature, _split_dr_vec_and_ess

    temperature = float(_policy_temperature(dataset))  # the DR score's softmax divides by the logger's T
    best = (-np.inf, 1.0, np.nan)
    for s in CAUSE_TEMPER_GRID:
        vec, _, _ = _split_dr_vec_and_ess(val, ux * np.float32(s * temperature), ia, lookup, dataset,
                                          propensity_mode="logged", weights=("clip", 10.0))
        n = max(len(vec), 2)
        r_hat = float(vec.mean())
        low = r_hat - float(student_t.ppf(0.975, n - 1)) * float(vec.std(ddof=1) / np.sqrt(n))
        if low > best[0]:
            best = (low, float(s), r_hat)
    low, s, r_hat = best
    policy = SimpleNamespace(user_emb=ux * np.float32(s), item_emb=ia, temperature=1.0, action_chunk=8192)
    g_hat, g_low = _greedy_dr(val, ux, ia, lookup)
    return {"temper_scale": s, "value_tempered": float(calc_reward(dataset, policy)), "val_dr_tempered": r_hat,
            "val_dr_tempered_low": low, "val_dr_greedy": g_hat, "val_dr_greedy_low": g_low}


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
    """``{method_label: (summary_df indexed by train size, trials_df)}`` for every (prediction, rho).

    ``options['family']``: ``native`` (the published model from scratch: predictions prod_c, prod_t, avg), ``warm``
    (CausE-warm) or ``cap`` (CausE-capacity-matched); warm and cap train the prod layout from the logger's source vectors
    and predict with its control (c) or treatment (t) rows (docs/cause_fair_comparison_25k.md §1). The search space is
    ``lr_range``, ``epochs``, ``l2``, ``cf`` and, when several are given, the tie direction ``ties``. ``temper``: also
    the fair sharpening of every trial's softmax and DR estimates of its policies (§2)."""
    import optuna

    from training.policy_diagnostics import POLICY_SUFFIX, save_selected_policy
    from training.trainer_trials import _scores_lookup_from_bundle, fit_shared_regression_bundle

    opts = {**CAUSE_DEFAULTS, **(options or {})}
    family = str(opts["family"])
    if family not in CAUSE_FAMILIES:
        raise ValueError(f"family must be one of {CAUSE_FAMILIES}, got {family!r}")
    n_trials = int(opts["n_trials"] or n_trials)
    ties = list(opts["ties"]) if opts.get("ties") else None
    searched_tie = ties is not None and len(ties) > 1
    default_tie = ties[0] if (ties and not searched_tie) else str(opts["tie"])
    bias_inits = list(opts["bias_inits"]) if opts.get("bias_inits") else None
    searched_init = bias_inits is not None and len(bias_inits) > 1
    default_init = bias_inits[0] if bias_inits else "zero"
    if family == "native" and default_init != "zero" or (family == "native" and searched_init):
        raise ValueError("the intercept initialization is an option of the warm-started families only")
    distributions = _distributions(tuple(opts["lr_range"]), tuple(opts["epochs"]), tuple(opts["l2"]), tuple(opts["cf"]),
                                   ties, bias_inits)
    if opts.get("temper_trials", "selected") not in ("selected", "all"):
        raise ValueError(f"temper_trials must be 'selected' or 'all', got {opts['temper_trials']!r}")
    temper_all = opts.get("temper_trials", "selected") == "all"
    ensure_exact_env_q_cache(dataset)
    n_users, n_actions = int(dataset["n_users"]), int(dataset["n_actions"])
    source = None
    if family != "native":
        source = (np.asarray(dataset["our_x"], dtype=np.float32), np.asarray(dataset["our_a"], dtype=np.float32))
        if source[0].shape[1] != int(opts["dim"]):
            raise ValueError(f"warm-started CausE needs dim = the source dimension {source[0].shape[1]}")
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
            lookup = None
            if opts["temper"]:  # q_hat of CausE's own N rows at this rho: its tempering stays within its budget
                bundle = fit_shared_regression_bundle(dataset, _merged_rows(budget))
                lookup = _scores_lookup_from_bundle(bundle, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            variants = list(opts["variants"]) if family == "native" else ["prod"]
            for variant in variants:
                layout = CausELayout(variant, n_actions)
                users, rows, labels = _training_arrays(budget, layout)
                if family == "native":
                    preds = [p for p, (v, _side) in CAUSE_PREDICTIONS.items() if v == variant]
                    pred_rows = {p: layout.prediction_rows(CAUSE_PREDICTIONS[p][1]) for p in preds}
                    labels_seed = ("cause", variant, n)  # no rho: the same configurations and seeds at every rho
                else:
                    preds = list(FAMILY_PREDICTIONS)
                    pred_rows = {p: layout.prediction_rows(side) for p, side in FAMILY_PREDICTIONS.items()}
                    labels_seed = ("cause", family, n)
                trial_rows = []
                order_seed = derive_seed(seed, *labels_seed, "order")  # one batch order for every trial of the study

                # per prediction: ((val NLL, trial), its policy vectors, its click model's offset over them)
                selected_vectors: dict[str, tuple] = {}

                def evaluate(model, rec):
                    """Validation metrics and exact true values of every prediction of a trained trial; with
                    tempering, the DR estimate of its greedy policy, and the scale search for every trial
                    (``temper_trials='all'``) or, after the search, for the selected trial only."""
                    for p in preds:
                        extra = {}
                        if rec["finite"]:
                            z = predict_logits(model, val_users, pred_rows[p][val_actions])
                            m = prediction_metrics(z, val_labels)
                            m["nll"] = m["nll"] if np.isfinite(m["nll"]) else DIVERGED_NLL
                            v_soft, v_greedy = _exact_values(dataset, model, pred_rows[p])
                            if lookup is not None:
                                ux, ia = model.policy_vectors(pred_rows[p])
                                if temper_all:
                                    extra = _tempered(dataset, val, ux, ia, lookup)
                                else:
                                    g_hat, g_low = _greedy_dr(val, ux, ia, lookup)
                                    extra = {k: np.nan for k in TEMPER_FIELDS} | {"val_dr_greedy": g_hat,
                                                                                  "val_dr_greedy_low": g_low}
                                    key = (float(m["nll"]), int(rec["trial"]))
                                    if m["nll"] < DIVERGED_NLL and (p not in selected_vectors or key < selected_vectors[p][0]):
                                        selected_vectors[p] = (key, (ux, ia), click_offset(model))
                        else:
                            m = {"nll": DIVERGED_NLL, "mse": np.nan, "auc": np.nan}
                            v_soft = v_greedy = np.nan
                            if lookup is not None:
                                extra = {k: np.nan for k in TEMPER_FIELDS + ("val_dr_greedy", "val_dr_greedy_low")}
                        rec.update({f"{p}_val_nll": m["nll"], f"{p}_val_mse": m["mse"], f"{p}_val_auc": m["auc"],
                                    f"{p}_value": v_soft, f"{p}_value_greedy": v_greedy,
                                    **{f"{p}_{k}": v for k, v in extra.items()}})
                    return rec

                base_rate_logit = float(np.log(labels.mean() / (1.0 - labels.mean())))  # of its own N training rows

                def new_model(number, bias_init="zero"):
                    trial_seed = derive_seed(seed, *labels_seed, "trial", number)
                    if family == "cap":
                        model = CausELinModel(*source)
                    else:
                        gen = torch.Generator().manual_seed(derive_seed(trial_seed, "init"))
                        model = CausEModel.for_layout(layout, n_users, int(opts["dim"]), generator=gen)
                        if family == "warm":
                            warm_start_(model, *source)
                    if bias_init == "base_rate":
                        with torch.no_grad():
                            model.global_bias.fill_(base_rate_logit)
                    return model

                def record(number, params, info_steps, finite, model):
                    rec = {"train_size": n, "rho": rho, "variant": variant, "trial": number, "lr": params["lr"],
                           "epochs": int(params["epochs"]), "l2_pen": float(params["l2_pen"]),
                           "cf_pen": float(params["cf_pen"]), "tie": str(params.get("tie", default_tie)),
                           "bias_init": str(params.get("bias_init", default_init)),
                           "steps": int(info_steps), "finite": bool(finite),
                           **_diagnostics(model, layout, source if family == "warm" else None)}
                    return evaluate(model, rec)

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction="minimize",
                                            sampler=optuna_sampler(seed, *labels_seed, kind=str(sampler)))
                t_study = time.time()
                if str(sampler) == "random":
                    # configurations do not depend on results: draw them all, then train the trials that share an
                    # epoch count and a tie direction together (models/cause.py fit_cause_batch; trial k equals its own
                    # fit_cause)
                    asked = [study.ask(distributions) for _ in range(n_trials)]
                    groups: dict[tuple, list] = {}
                    for tr in asked:
                        key = (int(tr.params["epochs"]), str(tr.params.get("tie", default_tie)))
                        groups.setdefault(key, []).append(tr)
                    done = {}
                    for (epochs, tie), group in sorted(groups.items()):
                        t0 = time.time()
                        models = [new_model(tr.number, str(tr.params.get("bias_init", default_init))) for tr in group]
                        info = fit_cause_batch(models, users, rows, labels, epochs=epochs,
                                               batch_size=int(opts["batch_size"]), optimizer=str(opts["optimizer"]),
                                               lrs=[tr.params["lr"] for tr in group],
                                               l2_pens=[tr.params["l2_pen"] for tr in group],
                                               cf_pens=[tr.params["cf_pen"] for tr in group],
                                               symmetric=tie == "symmetric", order_seed=order_seed, device=device)
                        for i, tr in enumerate(group):
                            rec = record(tr.number, tr.params, info["steps"], info["finite"][i], models[i])
                            rec["seconds"] = (time.time() - t0) / len(group)
                            done[tr.number] = rec
                        del models
                    for tr in asked:  # tell in trial order, as a sequential search would
                        trial_rows.append(done[tr.number])
                        study.tell(tr, float(done[tr.number][f"{preds[0]}_val_nll"]))
                else:
                    def objective(trial):
                        t0 = time.time()
                        params = {"lr": trial.suggest_float("lr", *opts["lr_range"], log=True),
                                  "epochs": trial.suggest_categorical("epochs", list(opts["epochs"])),
                                  "l2_pen": trial.suggest_categorical("l2_pen", list(opts["l2"])),
                                  "cf_pen": trial.suggest_categorical("cf_pen", list(opts["cf"]))}
                        if searched_tie:
                            params["tie"] = trial.suggest_categorical("tie", ties)
                        if searched_init:
                            params["bias_init"] = trial.suggest_categorical("bias_init", bias_inits)
                        model = new_model(trial.number, params.get("bias_init", default_init))
                        info = fit_cause(model, users, rows, labels, epochs=int(params["epochs"]),
                                         batch_size=int(opts["batch_size"]), optimizer=str(opts["optimizer"]),
                                         lr=params["lr"], l2_pen=float(params["l2_pen"]), cf_pen=float(params["cf_pen"]),
                                         symmetric=params.get("tie", default_tie) == "symmetric", seed=order_seed,
                                         device=device)
                        rec = record(trial.number, params, info["steps"], info["finite"], model)
                        rec["seconds"] = time.time() - t0
                        trial_rows.append(rec)
                        # Optuna follows the first prediction (prod-C for prod, avg for avg, c for warm / cap); each
                        # prediction is selected separately below from the same trials
                        return float(rec[f"{preds[0]}_val_nll"])

                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trials_df = pd.DataFrame(trial_rows)
                if lookup is not None and not temper_all:  # the selected trial of each prediction (selection by NLL)
                    for p, ((_nll, number), (ux, ia), _offset) in selected_vectors.items():
                        extra = _tempered(dataset, val, ux, ia, lookup)
                        at = trials_df.index[trials_df["trial"] == number][0]
                        for k in TEMPER_FIELDS:
                            trials_df.loc[at, f"{p}_{k}"] = extra[k]
                print(f"[cause:{family}] N={n} rho={rho:g} {variant}: {len(trials_df)} trials in "
                      f"{time.time() - t_study:.0f}s; finite {int(trials_df['finite'].sum())}; best {preds[0]} val NLL "
                      f"{trials_df[f'{preds[0]}_val_nll'].min():.4f}", flush=True)
                for p in preds:
                    label = cause_method_label(p, rho, family)
                    finite = trials_df[trials_df[f"{p}_val_nll"] < DIVERGED_NLL]
                    best = finite.loc[finite[f"{p}_val_nll"].idxmin()] if len(finite) else trials_df.iloc[0]
                    if lookup is not None and not temper_all and len(finite):
                        assert int(best["trial"]) == selected_vectors[p][0][1], "tempered a different trial"
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
                        "cause_train_device": str(torch.device(device).type),
                        "cause_tie": str(best.get("tie", default_tie)), "cause_batch_size": int(opts["batch_size"]),
                        "cause_sampler": str(sampler), "stage": str(stage),
                        **{k: v for k, v in meta.items() if k != "rho"},
                        "opc_collection_reward_sum": warm_reward_sum,
                        "exploration_cost_expected": rho * n * (v_logger - v_uniform),
                        "exploration_cost_realised": warm_reward_sum - meta["collection_reward_sum"],
                    }
                    if family != "native":
                        row["cause_family"] = family
                        row["cause_bias_init"] = str(best.get("bias_init", default_init))
                    if lookup is not None:
                        row.update({"policy_rewards_tempered": float(best[f"{p}_value_tempered"]),
                                    "temper_scale": float(best[f"{p}_temper_scale"]),
                                    "val_dr_greedy": float(best[f"{p}_val_dr_greedy"]),
                                    "val_dr_greedy_low": float(best[f"{p}_val_dr_greedy_low"]),
                                    "val_dr_tempered": float(best[f"{p}_val_dr_tempered"]),
                                    "val_dr_tempered_low": float(best[f"{p}_val_dr_tempered_low"]),
                                    # the best tempered value over the trials (a regret diagnostic): only when
                                    # every trial was tempered
                                    "oracle_selected_value_tempered": float(trials_df[f"{p}_value_tempered"].max())
                                    if temper_all else np.nan,
                                    "qhat_rows": int(n)})
                    if opts.get("policy_dir") and p in selected_vectors:  # for the cross-arm pick diagnostics
                        assert int(best["trial"]) == selected_vectors[p][0][1], "saving a different trial"
                        save_selected_policy(Path(opts["policy_dir"]) / f"{label}_n{n}{POLICY_SUFFIX}",
                                             *selected_vectors[p][1], offset=selected_vectors[p][2], arm=label,
                                             trial=int(best["trial"]), value_greedy=v_greedy, value=v_soft)
                    summaries.setdefault(label, []).append(row)
                    trials_by_label.setdefault(label, []).append(trials_df.assign(method=label, prediction=p))
    out = {}
    for label, rows in summaries.items():
        df = pd.DataFrame(rows).set_index("train_size")
        out[label] = (df, pd.concat(trials_by_label[label], ignore_index=True))
    return out
