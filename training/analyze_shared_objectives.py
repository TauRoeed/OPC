"""The shared-objective study's tables (docs/shared_objective_study.md).

    python -m training.analyze_shared_objectives tune --runs RUN... --out DIR
        the tuning grid's selections, marginals and the pre-registered edge rule (§8)
    python -m training.analyze_shared_objectives oracles --out DIR
        the population optima of the four objectives on the affine-bilinear class (§6), from the saved oracle policies
    python -m training.analyze_shared_objectives compare --runs RUN... --oracles DIR --out DIR
        the main comparison (§9-§10): per world and arm, pooled by bias, paired contrasts, the three selections, and the
        decomposition into objective mismatch, finite-sample training gap and selection gap

Values are the true greedy CTR; gains are in CTR points over the logger's greedy CTR. Every interval is a 95% t-interval
over worlds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_cause_fair import BIAS_NAMES, BIAS_ORDER, _tags, mean_ci
from training.run_state import read_trials_long

WORLD = ["dataset", "bias", "seed"]
ARMS = ("shared_likelihood", "shared_iw_likelihood", "shared_iw_likelihood_clip10", "shared_opc")
ARM_NAMES = {"shared_likelihood": "penalized likelihood", "shared_iw_likelihood": "IW likelihood (raw)",
             "shared_iw_likelihood_clip10": "IW likelihood (clip 10)", "shared_opc": "OPC objective",
             "opc": "OPC (historical arm)", "dm_own": "DM-only (own range)", "cap_c": "CausE-cap-C, ρ = 0",
             "blob_l10_nq": "BLOB-Pnorm-NQ (P₀ = 10)"}
# the population objective each arm's training objective estimates (§6): its optimum is the arm's own target
POPULATION_OF = {"shared_likelihood": "likelihood", "shared_iw_likelihood": "uniform_likelihood",
                 "shared_iw_likelihood_clip10": "clip10_likelihood", "shared_opc": "value"}
POPULATION_NAMES = {"likelihood": "θ_log*", "uniform_likelihood": "θ_uniform*", "clip10_likelihood": "θ_clip10*",
                    "value": "θ_value*"}
# the native selection: the column of trials_long and the direction (§5)
NATIVE = {"shared_likelihood": ("diag_val_nll", "min"), "shared_iw_likelihood": ("diag_val_iw_nll", "min"),
          "shared_iw_likelihood_clip10": ("diag_val_iw_nll_clip", "min"), "shared_opc": ("value", "max")}
COMMON = "diag_dr_greedy_low"  # the common selector: the DR lower bound of the greedy policy
ORACLE_RUNS = {"value": Path("artifacts/full_study/run_class_oracles_20261005"),
               "likelihood": Path("artifacts/full_study/run_class_oracles_20261005"),
               "uniform_likelihood": Path("artifacts/full_study/run_shared_oracles_20261006"),
               "clip10_likelihood": Path("artifacts/full_study/run_shared_oracles_20261006")}
COMPARATORS = Path("artifacts/full_study/blob_prior_calibration/compare/table_conditions.csv")
COMPARATOR_ARMS = ("opc", "dm_own", "cap_c", "blob_l10_nq")
BLOB_RUN = Path("artifacts/full_study/run_blob_main_25k_nq")  # the data-identity reference (train / val click sums)
EDGE_MARGIN = 0.25  # CTR points (the BLOB study's rule, §8)
EVAL_USERS = 20_000


# ---------------------------------------------------------------------------------------------------------- trials
def load_trials(*run_dirs) -> pd.DataFrame:
    """Every trial of the shared arms: world tags, the logger's greedy value and each trial's greedy gain."""
    frames = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "trials_long.csv").exists() or not (cond / "summary_metrics.csv").exists():
                continue
            t = read_trials_long(cond / "trials_long.csv")
            t = t[t["method"].isin(ARMS)].copy()
            if t.empty:
                continue
            s = pd.read_csv(cond / "summary_metrics.csv")
            v0 = float(s.loc[s["train_size"] == 0, "policy_rewards_greedy"].iloc[0])  # the logger's (train size 0)
            v0s = float(s.loc[s["train_size"] == 0, "policy_rewards"].iloc[0])
            tags = _tags(cond.name)
            t = t.assign(dataset=tags["dataset"], bias=tags["bias"], seed=int(tags["seed"]), run_tag=Path(run).name,
                         V_logger_greedy=v0, V_logger=v0s)
            t["gain_greedy"] = 100 * (t["actual_reward_greedy"] - v0)
            frames.append(t)
    if not frames:
        raise FileNotFoundError(f"no shared-arm trials under {run_dirs}")
    return pd.concat(frames, ignore_index=True)


def load_summaries(*run_dirs) -> pd.DataFrame:
    """The shared arms' summary rows at their train size (one per world and arm), with world tags."""
    frames = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "summary_metrics.csv").exists():
                continue
            s = pd.read_csv(cond / "summary_metrics.csv")
            s = s[s["method"].isin(ARMS) & (s["train_size"] > 0)]
            if s.empty:
                continue
            tags = _tags(cond.name)
            frames.append(s.assign(dataset=tags["dataset"], bias=tags["bias"], seed=int(tags["seed"])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def selections(t: pd.DataFrame) -> pd.DataFrame:
    """Per world and arm (and run, when several runs are pooled) the three selected trials (§5): native (the trainer's
    choice, checked against the native column), common (the highest DR lower bound of the greedy policy) and
    oracle-best (the highest true greedy value, diagnosis only)."""
    rows = []
    keys = WORLD + (["run_tag"] if "run_tag" in t else []) + ["method"]
    for key, g in t.groupby(keys):
        arm = key[-1]
        col, how = NATIVE[arm]
        native = g[g["is_best_in_run"].astype(bool)]
        if len(native) != 1:
            raise ValueError(f"{key}: {len(native)} natively selected trials")
        native = native.iloc[0]
        expected = g.loc[g[col].idxmin()] if how == "min" else g.loc[g[col].idxmax()]
        if int(expected["trial_number"]) != int(native["trial_number"]):
            raise ValueError(f"{key}: the trainer selected trial {native['trial_number']}, the native column "
                             f"{col} gives {expected['trial_number']}")
        common = g.loc[g[COMMON].idxmax()]
        best = g.loc[g["actual_reward_greedy"].idxmax()]
        r = dict(zip(keys[:-1] + ["arm"], key))
        r.update({"trials": len(g), "diverged": int(g.get("diverged", pd.Series(False, index=g.index)).astype(bool).sum()),
                  "V_logger_greedy": float(g["V_logger_greedy"].iloc[0]),
                  "V_logger": float(g["V_logger"].iloc[0]) if "V_logger" in g else np.nan})
        for name, trial in (("native", native), ("common", common), ("best", best)):
            r[f"{name}_trial"] = int(trial["trial_number"])
            r[f"{name}_V_greedy"] = float(trial["actual_reward_greedy"])
            r[f"{name}_gain"] = float(trial["gain_greedy"])
            r[f"{name}_V"] = float(trial["actual_reward"])
            r[f"{name}_lambda"] = float(trial["param_anchor_lambda"])
        for col_ in ("diag_anchor_R", "diag_logit_scale_s", "diag_changed_share", "diag_changed_gain", "diag_changed_loss",
                     "diag_val_nll", "diag_dr_greedy", "diag_dr_greedy_low", "param_lr", "param_num_epochs",
                     "param_batch_size", "param_lr_decay"):
            if col_ in native.index:
                r[f"native_{col_.replace('diag_', '').replace('param_', '')}"] = float(native[col_])
        r["common_changed_share"] = float(common.get("diag_changed_share", np.nan))
        # the candidates as a whole, not only the best of 20 (a maximum favours the arm whose trials spread more)
        r["trial_mean_gain"] = float(g["gain_greedy"].mean())
        r["trial_median_gain"] = float(g["gain_greedy"].median())
        r["regret_native"] = r["best_gain"] - r["native_gain"]
        r["regret_common"] = r["best_gain"] - r["common_gain"]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- tuning
def _bins(t: pd.DataFrame, dim: str) -> tuple[pd.Series, list]:
    """Each trial's bin of ``dim`` (§8): lr per half-decade, epochs in 4 bins, lr decay in 2, λ and batch per value."""
    if dim == "lr":
        lo, hi = np.log10(t["param_lr"].min()), np.log10(t["param_lr"].max())
        edges = np.arange(np.floor(lo * 2) / 2, hi + 0.5, 0.5)
        edges[-1] = max(edges[-1], hi + 1e-9)
        labels = [f"[{10 ** a:.2g}, {10 ** b:.2g})" for a, b in zip(edges[:-1], edges[1:])]
        return pd.cut(np.log10(t["param_lr"]), edges, labels=labels, include_lowest=True, right=False).astype(str), labels
    if dim == "epochs":  # 5-30, and 31-60 when the edge rule extended the range
        edges, labels = [4.5, 11.5, 17.5, 24.5, 30.5], ["5-11", "12-17", "18-24", "25-30"]
        if t["param_num_epochs"].max() > 30:
            edges, labels = edges + [45.5, 60.5], labels + ["31-45", "46-60"]
        return pd.cut(t["param_num_epochs"], edges, labels=labels).astype(str), labels
    if dim == "lr_decay":
        labels = ["[0.8, 0.9)", "[0.9, 1.0]"]
        return pd.cut(t["param_lr_decay"], [0.0, 0.9, 1.0 + 1e-9], labels=labels, right=False).astype(str), labels
    col = {"batch_size": "param_batch_size", "anchor_lambda": "param_anchor_lambda"}[dim]
    values = sorted(t[col].unique())
    return t[col].map(lambda v: f"{v:g}"), [f"{v:g}" for v in values]


EDGE_DIMENSIONS = ("lr", "epochs", "lr_decay", "batch_size", "anchor_lambda")
EXTENDABLE = {"lr": ("low", "high"), "epochs": ("high",), "anchor_lambda": ("high",)}  # §8


def marginal_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per arm, dimension and bin: trials, distinct configurations and the mean gap of a trial's greedy gain below its
    world's best trial of the same arm (CTR points; 0 = always the best), over every run given (a supplementary round
    is pooled with the first, §8). The paired sampler draws one set of configurations per seed and run, shared by every
    world of that seed: a configuration is (run, seed, trial number)."""
    rows = []
    for arm, g in t.groupby("method"):
        g = g.copy()
        g["gap"] = g["gain_greedy"] - g.groupby(WORLD)["gain_greedy"].transform("max")
        g["config"] = (g["run_tag"].astype(str) if "run_tag" in g else "") + ":" + g["seed"].astype(str) + ":" \
            + g["trial_number"].astype(str)
        for dim in EDGE_DIMENSIONS:
            b, order = _bins(g, dim)
            m = g.groupby(b).agg(mean=("gap", "mean"), size=("gap", "size"), configs=("config", "nunique"))
            for pos, label in enumerate(order):
                if label in m.index:
                    rows.append({"arm": arm, "dimension": dim, "value": label, "position": pos, "of": len(order),
                                 "trials": int(m.loc[label, "size"]), "configs": int(m.loc[label, "configs"]),
                                 "below_best_mean": float(m.loc[label, "mean"])})
    return pd.DataFrame(rows)


def edge_table(marg: pd.DataFrame) -> pd.DataFrame:
    """The edge rule (§8): per arm and dimension, the bin with the smallest mean gap; ``extend`` when it is at an
    extendable edge of the range and beats its neighbour by more than ``EDGE_MARGIN`` points."""
    rows = []
    for (arm, dim), g in marg.groupby(["arm", "dimension"]):
        g = g.sort_values("position")
        if len(g) < 2:
            continue
        best = g.loc[g["below_best_mean"].idxmax()]
        first, last = int(g["position"].min()), int(g["position"].max())
        side = "low" if int(best["position"]) == first else "high" if int(best["position"]) == last else None
        neighbour = None
        if side == "low":
            neighbour = g[g["position"] > best["position"]].iloc[0]
        elif side == "high":
            neighbour = g[g["position"] < best["position"]].iloc[-1]
        margin = float(best["below_best_mean"] - neighbour["below_best_mean"]) if neighbour is not None else np.nan
        extendable = side is not None and side in EXTENDABLE.get(dim, ())
        rows.append({"arm": arm, "dimension": dim, "best_value": best["value"], "best_configs": int(best["configs"]),
                     "best_below_best": float(best["below_best_mean"]),
                     "edge": side or "", "neighbour": "" if neighbour is None else neighbour["value"], "margin": margin,
                     "extendable": extendable, "extend": bool(extendable and margin > EDGE_MARGIN)})
    return pd.DataFrame(rows)


def selection_summary(sel: pd.DataFrame) -> pd.DataFrame:
    """Per arm (and run; all worlds pooled, and per bias): mean gains of the three selections and the regrets."""
    rows = []
    by = (["run_tag"] if "run_tag" in sel else []) + ["arm"]
    for panel, g in [("all", sel)] + [(b, sel[sel["bias"] == b]) for b in BIAS_ORDER if (sel["bias"] == b).any()]:
        for key, ga in g.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            r = {"panel": panel, **dict(zip(by, key)), "worlds": len(ga), "diverged_trials": int(ga["diverged"].sum())}
            for col in ("native_gain", "common_gain", "best_gain", "trial_mean_gain", "regret_native", "regret_common",
                        "native_anchor_R", "native_changed_share"):
                if col in ga:
                    m, lo, hi, _n = mean_ci(ga[col])
                    r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            for lam, n in ga["native_lambda"].value_counts().sort_index().items():
                r[f"native_lambda={lam:g}"] = int(n)
            rows.append(r)
    return pd.DataFrame(rows)


def tune_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load_trials(*args.runs)
    t.to_csv(out / "tuning_trials_long.csv.gz", index=False, float_format="%.8g")
    sel = selections(t)
    sel.to_csv(out / "tuning_selected.csv", index=False, float_format="%.6g")
    selection_summary(sel).to_csv(out / "tuning_summary.csv", index=False, float_format="%.6g")
    marg = marginal_table(t)
    marg.to_csv(out / "tuning_marginals.csv", index=False, float_format="%.6g")
    edges = edge_table(marg)
    edges.to_csv(out / "tuning_edges.csv", index=False, float_format="%.6g")
    s = load_summaries(*args.runs)
    if not s.empty:
        weights = [c for c in s.columns if c.startswith(("train_w", "train_wclip", "val_w", "val_wclip"))]
        s[WORLD + ["method"] + weights].to_csv(out / "tuning_weights.csv", index=False, float_format="%.6g")
    print(f"wrote {out}: {len(t)} trials, {t[WORLD].drop_duplicates().shape[0]} worlds")
    fired = edges[edges["extend"]]
    print("edge rule: " + ("no extension" if fired.empty else "extend " + ", ".join(
        f"{r.arm}:{r.dimension} past {r.best_value} ({r.edge})" for r in fired.itertuples())))


# ---------------------------------------------------------------------------------------------------------- oracles
def _oracle_policy(objective: str, world: tuple) -> Path:
    ds, bias, seed = world
    return (ORACLE_RUNS[objective] / ds / "policies" / f"dataset={ds}__bias={bias}__seed={seed}"
            / f"oracle_affine_bilinear_{objective}_n0_selected_policy.npz")


def _oracle_csv_value(objective: str, world: tuple, column: str) -> float:
    ds, bias, seed = world
    o = pd.read_csv(ORACLE_RUNS[objective] / ds / "class_oracles.csv")
    row = o[(o["class"] == "affine_bilinear") & (o["objective"] == objective) & (o["bias"] == bias) & (o["seed"] == seed)]
    if len(row) != 1:
        raise ValueError(f"{world} {objective}: {len(row)} oracle rows")
    return float(row[column].iloc[0])


def _oracle_ready(world: tuple) -> bool:
    """Whether every optimum of ``world`` has its saved policy and its row in the oracle run's table."""
    ds, bias, seed = world
    for obj in POPULATION_OF.values():
        table = ORACLE_RUNS[obj] / ds / "class_oracles.csv"
        if not _oracle_policy(obj, world).exists() or not table.exists():
            return False
        o = pd.read_csv(table)
        if not ((o["class"] == "affine_bilinear") & (o["objective"] == obj) & (o["bias"] == bias)
                & (o["seed"] == seed)).any():
            return False
    return True


def population_losses(dataset: dict, ux, ia, offset, users: np.ndarray, *, recalibrate: bool = False,
                      batch_users: int = 2048) -> dict:
    """The three likelihood objectives (§6) of the click model σ(ux·ia + offset) on ``users`` (every item, exact
    clicks): L_log (logger-weighted), L_uniform and L_clip10. With ``recalibrate``, first the affine map α f + β that
    minimizes L_log (a value oracle's scores are policy logits, not click log-odds)."""
    import torch

    from training.class_oracles import LIKELIHOOD_OBJECTIVES, likelihood_weights
    from training.oracle_repair import true_q_rows
    from training.trainer_trials import _policy_temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = float(_policy_temperature(dataset))
    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)
    UX, IA, OFF = t(ux), t(ia), t(offset)
    X, A = t(dataset["our_x"]), t(dataset["our_a"])
    clean = t(np.asarray(dataset["env"].emb_a))
    batches = [torch.as_tensor(users[s:s + batch_users], device=device) for s in range(0, len(users), batch_users)]
    alpha, beta = torch.ones((), device=device), torch.zeros((), device=device)
    if recalibrate:
        alpha.requires_grad_(True)
        beta.requires_grad_(True)
        opt = torch.optim.LBFGS([alpha, beta], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = 0.0
            for b in batches:
                f = alpha * (UX[b] @ IA.T + OFF[b][:, None]) + beta
                ce = torch.nn.functional.binary_cross_entropy_with_logits(f, true_q_rows(dataset, b, clean), reduction="none")
                loss = (likelihood_weights("likelihood", X[b], A, T) * ce).sum() / len(users)
                loss.backward()
                total += float(loss.detach())
            return torch.tensor(total)

        opt.step(closure)
    out = {obj: 0.0 for obj in LIKELIHOOD_OBJECTIVES}
    with torch.no_grad():
        for b in batches:
            f = alpha * (UX[b] @ IA.T + OFF[b][:, None]) + beta
            ce = torch.nn.functional.binary_cross_entropy_with_logits(f, true_q_rows(dataset, b, clean), reduction="none")
            for obj in LIKELIHOOD_OBJECTIVES:
                out[obj] += float((likelihood_weights(obj, X[b], A, T) * ce).sum())
    res = {f"L_{obj}": v / len(users) for obj, v in out.items()}
    if recalibrate:
        res.update({"recal_alpha": float(alpha.detach()), "recal_beta": float(beta.detach())})
    return res


def correction_size(x: np.ndarray, ux: np.ndarray, ia: np.ndarray) -> dict:
    """M from the ranking vectors (ux = [x M, 1], ia = [a, item term]): its relative distance from its best multiple
    of I and the item term's spread relative to the bilinear scores' spread (§6)."""
    K = x.shape[1]
    M = np.linalg.lstsq(np.asarray(x, np.float64), np.asarray(ux[:, :K], np.float64), rcond=None)[0]
    mbar = np.trace(M) / K
    item = np.asarray(ia[:, K], np.float64)
    rng = np.random.default_rng(0)
    users = rng.choice(len(x), size=min(2048, len(x)), replace=False)
    scores = (np.asarray(x, np.float64)[users] @ M) @ np.asarray(ia[:, :K], np.float64).T
    return {"M_dev": float(np.linalg.norm(M - mbar * np.eye(K)) / (abs(mbar) * np.sqrt(K))),
            "M_trace_mean": float(mbar), "item_share": float(item.std() / np.sqrt(scores.var(axis=1).mean())),
            "M": M}


def oracle_rows(world: tuple, emb_dir: Path = Path("BPR/embeddings"), ctr: float = 0.05) -> tuple[list, list]:
    """One world: per optimum its values, losses and size; per pair of optima their distances."""
    from training.run_full_study import build_condition_world
    from training.trainer_trials import _policy_greedy_reward_from_embeddings, _policy_reward_from_embeddings
    from training.policy_diagnostics import load_selected_policy
    from utils.seeding import derive_seed, seed_everything
    from utils.simulation_utils import _normalized_prior, calc_greedy_reward, calc_reward
    from types import SimpleNamespace

    ds, bias, seed = world
    seed_everything(int(seed))
    dataset, *_ = build_condition_world(ds, emb_dir, bias, ctr, int(seed), world_options={})
    prior = _normalized_prior(dataset)
    users = np.random.default_rng(derive_seed(int(seed), "shared_oracle_eval")).choice(
        int(dataset["n_users"]), size=EVAL_USERS, p=prior)
    v0g = _policy_greedy_reward_from_embeddings(dataset, dataset["our_x"], dataset["our_a"])
    v0 = _policy_reward_from_embeddings(dataset, dataset["our_x"], dataset["our_a"])
    rows, picks, Ms = [], {}, {}
    for obj in POPULATION_OF.values():
        pol = load_selected_policy(_oracle_policy(obj, world))
        ux, ia = pol["ux"], pol["ia"]
        offset = pol["offset"] if pol["offset"] is not None else np.zeros(len(ux), np.float32)
        vg, picks[obj] = calc_greedy_reward(dataset, ux, ia, return_picks=True)
        csv_vg = _oracle_csv_value(obj, world, "greedy")
        if abs(vg - csv_vg) > 1e-7:
            raise AssertionError(f"{world} {obj}: greedy value {vg} here, {csv_vg} in the oracle run")
        r = {"dataset": ds, "bias": bias, "seed": int(seed), "optimum": obj, "V_greedy": vg, "V_logger_greedy": v0g,
             "V_logger": v0, "gain_greedy": 100 * (vg - v0g), "fit_objective": _oracle_csv_value(obj, world, "fit_objective"),
             "lr": _oracle_csv_value(obj, world, "lr")}
        if obj == "value":
            r["V_stochastic"] = float(calc_reward(dataset, SimpleNamespace(user_emb=ux, item_emb=ia, temperature=1.0,
                                                                           action_chunk=8192)))
        r.update(population_losses(dataset, ux, ia, offset, users, recalibrate=obj == "value"))
        size = correction_size(np.asarray(dataset["our_x"]), ux, ia)
        Ms[obj] = size.pop("M")
        r.update(size)
        rows.append(r)
    pairs = []
    names = list(POPULATION_OF.values())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = Ms[a].ravel(), Ms[b].ravel()
            pairs.append({"dataset": ds, "bias": bias, "seed": int(seed), "a": a, "b": b,
                          "M_cosine_distance": float(1 - ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb))),
                          "top1_agreement": float(prior[picks[a] == picks[b]].sum())})
    return rows, pairs


def oracle_summary(o: pd.DataFrame) -> pd.DataFrame:
    """Per bias (and the biased worlds pooled): V(θ_value*) − V(θ_obj*) for each likelihood optimum, in points, with
    its CI and the worlds where the value optimum is higher (§6, H1-H3)."""
    wide = o.pivot_table(index=WORLD, columns="optimum", values="V_greedy").reset_index()
    rows = []
    panels = [(b, wide[wide["bias"] == b]) for b in BIAS_ORDER if (wide["bias"] == b).any()]
    panels.append(("biased (pooled)", wide[wide["bias"] != "none"]))
    for panel, g in panels:
        for obj in ("likelihood", "uniform_likelihood", "clip10_likelihood"):
            d = 100 * (g["value"] - g[obj])
            m, lo, hi, n = mean_ci(d)
            rows.append({"bias": panel, "quantity": f"V(θ_value*) − V({POPULATION_NAMES[obj]})", "mean": m,
                         "ci_lo": lo, "ci_hi": hi, "worlds": n, "value_higher": int((d > 1e-6).sum())})
        for a, b in (("uniform_likelihood", "likelihood"), ("clip10_likelihood", "likelihood"),
                     ("uniform_likelihood", "clip10_likelihood")):
            d = 100 * (g[a] - g[b])
            m, lo, hi, n = mean_ci(d)
            rows.append({"bias": panel, "quantity": f"V({POPULATION_NAMES[a]}) − V({POPULATION_NAMES[b]})", "mean": m,
                         "ci_lo": lo, "ci_hi": hi, "worlds": n, "value_higher": int((d > 1e-6).sum())})
    return pd.DataFrame(rows)


PROFILE_COLUMNS = ("gain_greedy", "L_likelihood", "L_uniform_likelihood", "L_clip10_likelihood", "M_dev", "item_share")


def oracle_profile(o: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per bias (and the biased worlds pooled) and optimum: the greedy gain over the logger, the three likelihood
    losses (the value optimum's after its recalibration), the correction's size (M's relative distance from a multiple
    of I, the item term's share) and the value optimum's stochastic value; per pair of optima their distances (§6)."""
    rows, drows = [], []
    for panel, g in _panels(o):
        for obj, go in g.groupby("optimum", sort=False):
            r = {"bias": panel, "optimum": obj, "worlds": len(go)}
            for col in PROFILE_COLUMNS + (("V_stochastic",) if obj == "value" else ()):
                m, lo, hi, _n = mean_ci(go[col] if col != "V_stochastic" else 100 * (go[col] - go["V_logger"]))
                key = col if col != "V_stochastic" else "stochastic_gain"
                r[key], r[key + "_lo"], r[key + "_hi"] = m, lo, hi
            rows.append(r)
    for panel, g in _panels(pairs):
        for (a, b), gp in g.groupby(["a", "b"], sort=False):
            r = {"bias": panel, "a": a, "b": b, "worlds": len(gp)}
            for col in ("M_cosine_distance", "top1_agreement"):
                m, lo, hi, _n = mean_ci(gp[col])
                r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            drows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(drows)


def oracles_main(args) -> None:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path, pairs_path = out / "oracles.csv", out / "oracle_distances.csv"
    done = set()
    if rows_path.exists():
        prev = pd.read_csv(rows_path)
        done = set(zip(prev["dataset"], prev["bias"], prev["seed"]))
    worlds = [(ds, b, s) for s in args.seeds for ds in args.datasets for b in BIAS_ORDER]
    missing = []
    for world in worlds:
        if world in done:
            continue
        if not _oracle_ready(world):
            missing.append(world)  # an oracle run still in progress: the next invocation picks the world up
            continue
        rows, pairs = oracle_rows(world, Path(args.emb_dir))
        pd.DataFrame(rows).to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False, float_format="%.8g")
        pd.DataFrame(pairs).to_csv(pairs_path, mode="a", header=not pairs_path.exists(), index=False,
                                   float_format="%.8g")
        print(json.dumps({"world": world, **{r["optimum"]: round(r["gain_greedy"], 3) for r in rows}}), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    o = pd.read_csv(rows_path)
    oracle_summary(o).to_csv(out / "oracle_summary.csv", index=False, float_format="%.6g")
    profile, distances = oracle_profile(o, pd.read_csv(pairs_path))
    profile.to_csv(out / "oracle_profile.csv", index=False, float_format="%.6g")
    distances.to_csv(out / "oracle_distance_summary.csv", index=False, float_format="%.6g")
    print(f"wrote {out}: {o[WORLD].drop_duplicates().shape[0]} worlds"
          + (f"; {len(missing)} without every optimum yet: {missing}" if missing else ""))


# ---------------------------------------------------------------------------------------------------------- compare
def world_ceilings(path: Path = COMPARATORS) -> pd.DataFrame:
    """Per world the target-best greedy value (each user's truly best item, ``analyze_cause_fair.world_references``)."""
    c = pd.read_csv(path)
    return c[WORLD + ["ceiling"]].drop_duplicates(WORLD)


def comparator_rows(path: Path = COMPARATORS) -> pd.DataFrame:
    """The reused comparator arms per world (§7): greedy and stochastic values and gains, not rerun. A click model's
    stochastic value is its DR-tempered softmax (``V_tempered``, the CausE and BLOB studies' fair sharpening); OPC's and
    DM-only's is their own softmax."""
    c = pd.read_csv(path)
    c = c[c["arm"].isin(COMPARATOR_ARMS)].copy()
    c["V"] = c["V_tempered"].where(c["V_tempered"].notna(), c["V"])
    return c[WORLD + ["arm", "V_greedy", "V", "gain_greedy", "V_logger_greedy", "V_logger"]].rename(
        columns={"V_greedy": "native_V_greedy", "V": "native_V", "gain_greedy": "native_gain"})


def data_identity(s: pd.DataFrame, blob_run: Path = BLOB_RUN, sel: pd.DataFrame | None = None,
                  comps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per world: whether the four shared arms trained and validated on the same rows (hash, click and propensity
    sums), whether those match the BLOB run's training and validation click sums of the same world (§4), and whether
    the world's logger has the same greedy value as in the reused comparator rows (the same world)."""
    blob = load_reference_sums(blob_run)
    logger = sel.groupby(WORLD)["V_logger_greedy"].agg(["min", "max"]) if sel is not None else None
    comp_logger = comps.groupby(WORLD)["V_logger_greedy"].agg(["min", "max"]) if comps is not None else None
    rows = []
    for world, g in s.groupby(WORLD):
        r = dict(zip(WORLD, world))
        r["arms"] = len(g)
        for col in ("train_rows_sha1", "val_rows_sha1", "train_click_sum", "val_click_sum", "train_pscore_sum",
                    "val_pscore_sum", "train_rows", "val_rows"):
            r[f"{col}_identical"] = bool(g[col].nunique() == 1)
        b = blob.get(world)
        r["train_click_sum"] = float(g["train_click_sum"].iloc[0])
        r["matches_blob_rows"] = (b is not None and abs(b[0] - r["train_click_sum"]) < 1e-9
                                  and abs(b[1] - float(g["val_click_sum"].iloc[0])) < 1e-9)
        if logger is not None and comp_logger is not None and world in logger.index and world in comp_logger.index:
            lo = min(logger.loc[world, "min"], comp_logger.loc[world, "min"])
            hi = max(logger.loc[world, "max"], comp_logger.loc[world, "max"])
            r["logger_matches_comparators"] = bool(hi - lo <= 1e-7 * abs(hi))  # the comparator table keeps 8 digits
        rows.append(r)
    return pd.DataFrame(rows)


def load_reference_sums(run: Path) -> dict:
    out = {}
    for cond in sorted(Path(run).glob("dataset=*")):
        p = cond / "summary_metrics.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        s = s[s["method"].astype(str).str.startswith("blob") & (s["train_size"] > 0)]
        if s.empty or "train_click_sum" not in s:
            continue
        tags = _tags(cond.name)
        out[(tags["dataset"], tags["bias"], int(tags["seed"]))] = (float(s["train_click_sum"].iloc[0]),
                                                                  float(s["val_click_sum"].iloc[0]))
    return out


def condition_table(sel: pd.DataFrame, s: pd.DataFrame, oracles: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    """One row per world and arm: the three selections, the population optima of the arm's objective and the value,
    the decomposition, the share of the value oracle's gap recovered, and the weight diagnostics; then the reused
    comparator arms with their native values."""
    o = oracles.pivot_table(index=WORLD, columns="optimum", values="V_greedy").reset_index()
    t = sel.merge(o, on=WORLD, how="left")
    t["V_value_star"] = t["value"]
    t["V_objective_star"] = [r[POPULATION_OF[r["arm"]]] for _, r in t.iterrows()]
    pts = lambda a, b: 100 * (a - b)
    t["objective_mismatch"] = pts(t["V_value_star"], t["V_objective_star"])
    t["training_gap"] = pts(t["V_objective_star"], t["best_V_greedy"])
    t["selection_gap_native"] = pts(t["best_V_greedy"], t["native_V_greedy"])
    t["selection_gap_common"] = pts(t["best_V_greedy"], t["common_V_greedy"])
    t["total_gap_native"] = pts(t["V_value_star"], t["native_V_greedy"])
    t = t.merge(world_ceilings(), on=WORLD, how="left")
    t["capacity_gap"] = pts(t["ceiling"], t["V_value_star"])  # beyond the global class: what more capacity could reach
    span = (t["V_value_star"] - t["V_logger_greedy"]).where(t["bias"] != "none")  # no gap to recover without bias
    t["frac_value_gap_native"] = (t["native_V_greedy"] - t["V_logger_greedy"]) / span
    t["native_stoch_gain"] = pts(t["native_V"], t["V_logger"])  # OPC's softmax; for a likelihood arm a convention
    t["frac_value_gap_common"] = (t["common_V_greedy"] - t["V_logger_greedy"]) / span
    keep = [c for c in s.columns if c.startswith(("train_w", "train_wclip", "val_w", "val_wclip", "head_"))]
    t = t.merge(s[WORLD + ["method"] + keep].rename(columns={"method": "arm"}), on=WORLD + ["arm"], how="left")
    if comps.empty:
        return t
    comps = comps.merge(o[WORLD + ["value"]], on=WORLD, how="left").rename(columns={"value": "V_value_star"})
    comps["frac_value_gap_native"] = ((comps["native_V_greedy"] - comps["V_logger_greedy"])
                                      / (comps["V_value_star"] - comps["V_logger_greedy"]).where(comps["bias"] != "none"))
    comps["native_stoch_gain"] = 100 * (comps["native_V"] - comps["V_logger"])
    return pd.concat([t, comps], ignore_index=True, sort=False)


def _panels(t: pd.DataFrame):
    for b in BIAS_ORDER:
        if (t["bias"] == b).any():
            yield b, t[t["bias"] == b]
    yield "biased (pooled)", t[t["bias"] != "none"]


SUMMARY_COLUMNS = ("native_gain", "native_stoch_gain", "common_gain", "best_gain", "trial_mean_gain",
                   "trial_median_gain", "regret_native", "regret_common",
                   "capacity_gap",
                   "frac_value_gap_native",
                   "frac_value_gap_common", "objective_mismatch", "training_gap", "selection_gap_native",
                   "selection_gap_common", "total_gap_native", "native_anchor_R", "native_changed_share",
                   "native_changed_gain", "native_changed_loss", "native_val_nll", "native_logit_scale_s")


def summary_table(t: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over worlds per bias × arm, and over the 24 biased worlds pooled."""
    rows = []
    for bias, tb in _panels(t):
        for arm, g in tb.groupby("arm"):
            r = {"bias": bias, "arm": arm, "worlds": len(g)}
            for col in SUMMARY_COLUMNS:
                if col in g and g[col].notna().any():
                    m, lo, hi, _n = mean_ci(g[col])
                    r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def dataset_table(t: pd.DataFrame, cols=("native_gain", "best_gain")) -> pd.DataFrame:
    """Per dataset (its 8 biased worlds) and arm: mean and 95% CI over worlds, to see whether a pooled result holds on
    every dataset."""
    rows = []
    for (ds, arm), g in t[t["bias"] != "none"].groupby(["dataset", "arm"]):
        r = {"dataset": ds, "arm": arm, "worlds": len(g)}
        for col in cols:
            if col in g and g[col].notna().any():
                m, lo, hi, _n = mean_ci(g[col])
                r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired_table(t: pd.DataFrame, pairs, col: str = "native_gain") -> pd.DataFrame:
    """a − b paired by world: mean, 95% CI and the worlds where a is higher, per bias and pooled over biased worlds."""
    rows = []
    for a, b in pairs:
        va = t[t["arm"] == a].set_index(WORLD)[col]
        vb = t[t["arm"] == b].set_index(WORLD)[col]
        d = (va - vb).dropna()
        frame = d.reset_index()
        frame.columns = WORLD + ["d"]
        for bias, g in _panels(frame):
            if g.empty:
                continue
            m, lo, hi, n = mean_ci(g["d"])
            rows.append({"a": a, "b": b, "col": col, "bias": bias, "a_minus_b": m, "ci_lo": lo, "ci_hi": hi,
                         "worlds": n, "a_higher": int((g["d"] > 0).sum())})
    return pd.DataFrame(rows)


PAIRS = (("shared_opc", "shared_likelihood"), ("shared_iw_likelihood", "shared_likelihood"),
         ("shared_iw_likelihood_clip10", "shared_likelihood"), ("shared_iw_likelihood_clip10", "shared_iw_likelihood"),
         ("shared_opc", "shared_iw_likelihood"), ("shared_opc", "shared_iw_likelihood_clip10"),
         ("shared_opc", "opc"), ("shared_likelihood", "cap_c"), ("shared_opc", "cap_c"))


# ---------------------------------------------------------------------------------------------------------- figures
# Okabe-Ito, as the repository's other reports: OPC blue, the likelihood family in pink, the weighted likelihoods in
# vermillion / orange (validated for colour-vision deficiency; direct labels and the CSVs carry identity)
COLORS = {"shared_likelihood": "#CC79A7", "shared_iw_likelihood": "#D55E00", "shared_iw_likelihood_clip10": "#E69F00",
          "shared_opc": "#0072B2", "opc": "#6B6B6B", "dm_own": "#6B6B6B", "cap_c": "#6B6B6B", "blob_l10_nq": "#6B6B6B"}
OBJECTIVE_COLORS = {"likelihood": "#CC79A7", "uniform_likelihood": "#D55E00", "clip10_likelihood": "#E69F00"}
PANEL_NAMES = {**BIAS_NAMES, "biased (pooled)": "biased (24 worlds)"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.edgecolor": "#8A8A8A",
                         "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#333333"})
    return plt


def _save(fig, out: Path, name: str, data: pd.DataFrame) -> None:
    data.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def _dot_panels(summary: pd.DataFrame, arms, col: str, title: str, xlabel: str, out: Path, name: str,
                reference: dict | None = None) -> None:
    """One panel per bias (and the biased worlds pooled): each arm's mean and 95% CI of ``col``, arms as labelled
    rows; ``reference``: a vertical dashed line per panel (e.g. the value oracle's gain)."""
    plt = _plt()
    panels = [p for p in list(BIAS_ORDER) + ["biased (pooled)"] if (summary["bias"] == p).any()]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    rows = []
    for ax, panel in zip(axes.ravel(), panels):
        g = summary[summary["bias"] == panel].set_index("arm")
        for i, arm in enumerate(arms):
            if arm not in g.index or pd.isna(g.loc[arm].get(col)):
                continue
            r = g.loc[arm]
            ax.errorbar([r[col]], [i], xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], fmt="o",
                        color=COLORS[arm], markersize=5, capsize=2, linewidth=1.2)
            rows.append({"panel": panel, "arm": arm, col: r[col], "lo": r[col + "_lo"], "hi": r[col + "_hi"]})
        if reference and panel in reference:
            ax.axvline(reference[panel], color="#333333", linestyle="--", linewidth=0.9)
        ax.axvline(0, color="#8A8A8A", linewidth=0.8)
        ax.set_title(PANEL_NAMES.get(panel, panel), fontsize=9)
        ax.set_yticks(range(len(arms)), [ARM_NAMES[a] for a in arms])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, out, name, pd.DataFrame(rows))
    plt.close(fig)


def fig_population(osum: pd.DataFrame, out: Path) -> None:
    """V(θ_value*) − V(θ_obj*) per bias for the three likelihood optima (CTR points, 95% CI over worlds)."""
    plt = _plt()
    objs = ("likelihood", "uniform_likelihood", "clip10_likelihood")
    panels = [p for p in list(BIAS_ORDER) + ["biased (pooled)"] if (osum["bias"] == p).any()]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rows = []
    for j, obj in enumerate(objs):
        q = f"V(θ_value*) − V({POPULATION_NAMES[obj]})"
        g = osum[osum["quantity"] == q].set_index("bias")
        for i, panel in enumerate(panels):
            if panel not in g.index:
                continue
            r = g.loc[panel]
            y = i + (j - 1) * 0.22
            ax.errorbar([r["mean"]], [y], xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o",
                        color=OBJECTIVE_COLORS[obj], markersize=5, capsize=2, linewidth=1.2,
                        label=POPULATION_NAMES[obj] if i == 0 else None)
            rows.append({"bias": panel, "optimum": obj, "mean": r["mean"], "lo": r["ci_lo"], "hi": r["ci_hi"]})
    ax.axvline(0, color="#8A8A8A", linewidth=0.8)
    ax.set_yticks(range(len(panels)), [PANEL_NAMES.get(p, p) for p in panels])
    ax.invert_yaxis()
    ax.set_xlabel("V(θ_value*) − V(optimum), greedy CTR points")
    ax.set_title("Objective mismatch in the population: the value optimum minus each likelihood optimum", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig3_population_mismatch", pd.DataFrame(rows))
    plt.close(fig)


def fig_decomposition(summary: pd.DataFrame, out: Path, panel: str = "biased (pooled)") -> None:
    """Per arm, the three parts of V(θ_value*) − V(native) in one bias panel: objective mismatch, finite-sample training
    gap and selection gap (CTR points, 95% CI over worlds), one row per part."""
    plt = _plt()
    parts = (("objective_mismatch", "objective mismatch"), ("training_gap", "training gap"),
             ("selection_gap_native", "selection gap (native)"), ("selection_gap_common", "selection gap (common DR)"),
             ("total_gap_native", "total (native)"))
    g = summary[summary["bias"] == panel].set_index("arm")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    rows = []
    for j, arm in enumerate(ARMS):
        if arm not in g.index:
            continue
        for i, (col, _label) in enumerate(parts):
            r = g.loc[arm]
            if pd.isna(r.get(col)):
                continue
            y = i + (j - 1.5) * 0.17
            ax.errorbar([r[col]], [y], xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], fmt="o",
                        color=COLORS[arm], markersize=4.5, capsize=2, linewidth=1.1, label=ARM_NAMES[arm] if i == 0 else None)
            rows.append({"panel": panel, "arm": arm, "part": col, "mean": r[col], "lo": r[col + "_lo"],
                         "hi": r[col + "_hi"]})
    ax.axvline(0, color="#8A8A8A", linewidth=0.8)
    ax.set_yticks(range(len(parts)), [label for _c, label in parts])
    ax.invert_yaxis()
    ax.set_xlabel("greedy CTR points below the value optimum θ_value*")
    ax.set_title(f"Where each objective loses value, {PANEL_NAMES.get(panel, panel)}", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "fig2_decomposition", pd.DataFrame(rows))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- tables.md
def _fmt(m, lo=None, hi=None, digits=2, signed=True) -> str:
    if m is None or pd.isna(m):
        return "—"
    sign = "+" if signed else ""
    s = f"{m:{sign}.{digits}f}"
    if lo is not None and not pd.isna(lo):
        s += f" [{lo:{sign}.{digits}f}, {hi:{sign}.{digits}f}]"
    return s


def tables_md(summary: pd.DataFrame, paired: pd.DataFrame, osum: pd.DataFrame, ident: pd.DataFrame,
              cond: pd.DataFrame, profile: pd.DataFrame | None = None, distances: pd.DataFrame | None = None,
              per_dataset: pd.DataFrame | None = None) -> str:
    panels = [p for p in list(BIAS_ORDER) + ["biased (pooled)"] if (summary["bias"] == p).any()]
    head = "| arm | " + " | ".join(PANEL_NAMES.get(p, p) for p in panels) + " |\n|---|" + "---|" * len(panels) + "\n"
    lines = ["# Shared-objective study: tables", "",
             "Greedy CTR points; mean and 95% CI over worlds (6 per bias, 24 biased pooled). DEVELOPMENT worlds.", ""]

    lines += ["## 1. Population optima (§6): V(θ_value*) − V(θ_obj*)", "",
              "| quantity | " + " | ".join(PANEL_NAMES.get(p, p) for p in panels) + " |",
              "|---|" + "---|" * len(panels)]
    for q in osum["quantity"].unique():
        g = osum[osum["quantity"] == q].set_index("bias")
        cells = [(_fmt(g.loc[p, "mean"], g.loc[p, "ci_lo"], g.loc[p, "ci_hi"]) + f" ({int(g.loc[p, 'value_higher'])}/"
                  f"{int(g.loc[p, 'worlds'])})") if p in g.index else "—" for p in panels]
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    if profile is not None and not profile.empty:
        lines += ["", "## 1b. Population optima: greedy gain, losses (the value optimum recalibrated under L_log), "
                  "correction size", "",
                  "| bias | optimum | greedy gain | L_log | L_uniform | L_clip10 | M_dev | item share |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in profile.itertuples():
            lines.append(f"| {PANEL_NAMES.get(r.bias, r.bias)} | {POPULATION_NAMES[r.optimum]} | {r.gain_greedy:+.2f} | "
                         f"{r.L_likelihood:.4f} | {r.L_uniform_likelihood:.4f} | {r.L_clip10_likelihood:.4f} | "
                         f"{r.M_dev:.3f} | {r.item_share:.3f} |")
        v = profile[profile["optimum"] == "value"]
        if "stochastic_gain" in v:
            lines += ["", "The value optimum's stochastic gain (its softmax policy): " + "; ".join(
                f"{PANEL_NAMES.get(r.bias, r.bias)} {r.stochastic_gain:+.2f}" for r in v.itertuples()) + "."]
    if distances is not None and not distances.empty:
        lines += ["", "## 1c. Distances between the optima: cosine distance of M; prior-weighted share of users with the "
                  "same greedy item", "", "| bias | pair | M cosine distance | same top item |", "|---|---|---|---|"]
        for r in distances.itertuples():
            lines.append(f"| {PANEL_NAMES.get(r.bias, r.bias)} | {POPULATION_NAMES[r.a]} – {POPULATION_NAMES[r.b]} | "
                         f"{r.M_cosine_distance:.3f} | {r.top1_agreement:.3f} |")

    def block(title, col, arms, note="", signed=True, digits=2):
        out = ["", f"## {title}", ""] + ([note, ""] if note else []) + [head.rstrip("\n")]
        for arm in arms:
            g = summary[summary["arm"] == arm].set_index("bias")
            if g.empty or col not in g:
                continue
            cells = [_fmt(g.loc[p, col], g.loc[p, col + "_lo"], g.loc[p, col + "_hi"], digits, signed)
                     if p in g.index else "—" for p in panels]
            out.append(f"| {ARM_NAMES.get(arm, arm)} | " + " | ".join(cells) + " |")
        return out

    lines += block("2. Finite samples at 25k: native-selected greedy gain over the logger", "native_gain",
                   ARMS + COMPARATOR_ARMS, "Comparators are reused rows (not rerun), each with its own selection.")
    if per_dataset is not None and not per_dataset.empty:
        found = list(dict.fromkeys(per_dataset["dataset"]))
        dsets = [d for d in ("ml", "kuairand", "anime") if d in found] + [d for d in found if d not in ("ml", "kuairand", "anime")]
        lines += ["", "## 2c. Native-selected greedy gain per dataset (its 8 biased worlds)", "",
                  "| arm | " + " | ".join(dsets) + " |", "|---|" + "---|" * len(dsets)]
        for arm in ARMS + COMPARATOR_ARMS:
            g = per_dataset[per_dataset["arm"] == arm].set_index("dataset")
            if g.empty:
                continue
            cells = [_fmt(g.loc[d, "native_gain"], g.loc[d, "native_gain_lo"], g.loc[d, "native_gain_hi"])
                     if d in g.index else "—" for d in dsets]
            lines.append(f"| {ARM_NAMES.get(arm, arm)} | " + " | ".join(cells) + " |")
    lines += block("2b. Stochastic value of the native-selected policy over the logger's", "native_stoch_gain",
                   ("shared_opc",) + COMPARATOR_ARMS, "OPC's softmax policy is its deployable stochastic policy; the "
                   "CausE-cap and BLOB rows are their DR-tempered softmax. The likelihood arms are left out: their "
                   "softmax at the click model's scale is not a policy anyone would deploy, and they were not tempered "
                   "(§1); their greedy policy is the comparison.")
    lines += block("3. Common selector (DR lower bound of the greedy policy): greedy gain", "common_gain", ARMS)
    lines += block("4. Best available trial (oracle choice among the 20, diagnosis only): greedy gain", "best_gain", ARMS)
    lines += block("4b. Mean greedy gain of the 20 trials (the candidates as a whole)", "trial_mean_gain", ARMS)
    lines += block("5. Objective mismatch V(θ_value*) − V(θ_obj*)", "objective_mismatch", ARMS)
    lines += block("6. Finite-sample training gap V(θ_obj*) − V(best trial)", "training_gap", ARMS)
    lines += block("7. Selection gap V(best) − V(native)", "selection_gap_native", ARMS)
    lines += block("8. Selection gap V(best) − V(common DR)", "selection_gap_common", ARMS)
    lines += block("9. Share of the value oracle's gap recovered (native)", "frac_value_gap_native", ARMS + COMPARATOR_ARMS,
                   "(V(selected) − V(logger)) / (V(θ_value*) − V(logger)), greedy; biased worlds only (no gap without "
                   "bias).", signed=False)
    lines += block("9b. Capacity gap beyond the global class: target-best − V(θ_value*)", "capacity_gap", ARMS[-1:],
                   "The same in every arm's row (a property of the world); shown once.")
    lines += block("10. Correction size R(θ) of the native-selected trial", "native_anchor_R", ARMS, signed=False, digits=3)
    lines += block("11. Share of users whose greedy item changes from the logger's (native)", "native_changed_share", ARMS,
                   signed=False)

    lines += ["", "## 12. Paired contrasts (a − b by world; worlds where a is higher)", "",
              "| a − b | selection | " + " | ".join(PANEL_NAMES.get(p, p) for p in panels) + " |",
              "|---|---|" + "---|" * len(panels)]
    for (a, b, col), g in paired.groupby(["a", "b", "col"], sort=False):
        g = g.set_index("bias")
        cells = [(_fmt(g.loc[p, "a_minus_b"], g.loc[p, "ci_lo"], g.loc[p, "ci_hi"]) + f" ({int(g.loc[p, 'a_higher'])}/"
                  f"{int(g.loc[p, 'worlds'])})") if p in g.index else "—" for p in panels]
        lines.append(f"| {ARM_NAMES.get(a, a)} − {ARM_NAMES.get(b, b)} | {col.replace('_gain', '')} | "
                     + " | ".join(cells) + " |")

    w = cond[cond["arm"] == "shared_iw_likelihood"]
    if not w.empty and "train_w_ess_share" in w:
        lines += ["", "## 13. Uniform-reference weights w = 1/(P p) on the training rows (mean of the two seeds)", "",
                  "| dataset | bias | ESS (rows) | ESS share | max w | 99.9% quantile | share above 10 | mass above 10 | "
                  "ESS share, clipped at 10 |", "|---|---|---|---|---|---|---|---|---|"]
        order = {d: i for i, d in enumerate(("ml", "kuairand", "anime"))}
        border = {b: i for i, b in enumerate(BIAS_ORDER)}
        keys = sorted(w.groupby(["dataset", "bias"]).groups, key=lambda k: (order.get(k[0], 9), border.get(k[1], 9)))
        for ds, bias in keys:
            g = w[(w["dataset"] == ds) & (w["bias"] == bias)]
            lines.append(f"| {ds} | {PANEL_NAMES.get(bias, bias)} | {g['train_w_ess'].mean():.0f} | "
                         f"{g['train_w_ess_share'].mean():.4f} | {g['train_w_max'].mean():.0f} | "
                         f"{g['train_w_q99.9'].mean():.1f} | {g['train_w_clip_share'].mean():.4f} | "
                         f"{g['train_w_clip_mass_share'].mean():.3f} | {g['train_wclip_ess_share'].mean():.3f} |")
    lines += ["", "## 14. Data identity", "",
              f"- Worlds: {len(ident)}; the four arms' training and validation rows identical (hash, clicks, "
              f"propensities) in {int(ident[[c for c in ident if c.endswith('_identical')]].all(axis=1).sum())}; "
              f"the same training and validation clicks as the BLOB run of the world in "
              f"{int(ident['matches_blob_rows'].sum())}; the logger's greedy value equal to the comparator rows' in "
              f"{int(ident.get('logger_matches_comparators', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}.",
              ""]
    return "\n".join(lines)


def compare_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(*args.runs)
    trials.to_csv(out / "trials_long.csv.gz", index=False, float_format="%.8g")
    sel = selections(trials)
    s = load_summaries(*args.runs)
    oracles = pd.read_csv(Path(args.oracles) / "oracles.csv")
    t = condition_table(sel, s, oracles, comparator_rows())
    t.to_csv(out / "table_conditions.csv", index=False, float_format="%.8g")
    summary_table(t).to_csv(out / "table_summary.csv", index=False, float_format="%.6g")
    paired = pd.concat([paired_table(t, PAIRS, "native_gain"), paired_table(t, PAIRS[:6], "common_gain"),
                        paired_table(t, PAIRS[:6], "best_gain"), paired_table(t, PAIRS[:6], "trial_mean_gain")],
                       ignore_index=True)
    paired.to_csv(out / "table_paired.csv", index=False, float_format="%.6g")
    ident = data_identity(s, sel=sel, comps=comparator_rows())
    ident.to_csv(out / "table_data_identity.csv", index=False)
    summ = pd.read_csv(out / "table_summary.csv")
    osum = oracle_summary(oracles)
    osum.to_csv(out / "table_population.csv", index=False, float_format="%.6g")
    profile, distances = oracle_profile(oracles, pd.read_csv(Path(args.oracles) / "oracle_distances.csv"))
    profile.to_csv(out / "table_population_profile.csv", index=False, float_format="%.6g")
    distances.to_csv(out / "table_population_distances.csv", index=False, float_format="%.6g")
    one_row_per_world = t[t["arm"] == "shared_opc"]  # the value oracle's gain per world, averaged per panel
    ref = {panel: float(100 * (g["V_value_star"] - g["V_logger_greedy"]).mean()) for panel, g in _panels(one_row_per_world)}
    _dot_panels(summ, ARMS + COMPARATOR_ARMS, "native_gain", "Native-selected greedy gain over the logger (25k); "
                "dashed: the value oracle θ_value*", "greedy CTR points", out, "fig1_native_gain", reference=ref)
    _dot_panels(summ, ARMS, "common_gain", "Common-selected (DR lower bound of the greedy policy) greedy gain",
                "greedy CTR points", out, "fig1b_common_gain", reference=ref)
    fig_decomposition(summ, out)
    fig_population(osum, out)
    per_dataset = dataset_table(t)
    per_dataset.to_csv(out / "table_per_dataset.csv", index=False, float_format="%.6g")
    (out / "tables.md").write_text(tables_md(summ, paired, osum, ident, t, profile, distances, per_dataset),
                                   encoding="utf-8")
    print(f"wrote {out}: {len(trials)} trials, {trials[WORLD].drop_duplicates().shape[0]} worlds")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tune = sub.add_parser("tune", help="the tuning grid and its edge rule (§8)")
    tune.add_argument("--runs", nargs="+", required=True)
    tune.add_argument("--out", required=True)
    orc = sub.add_parser("oracles", help="the population optima (§6), from the saved oracle policies")
    orc.add_argument("--datasets", nargs="+", default=["ml", "kuairand", "anime"])
    orc.add_argument("--seeds", nargs="+", type=int, default=[100, 101])
    orc.add_argument("--emb-dir", default="BPR/embeddings")
    orc.add_argument("--out", required=True)
    comp = sub.add_parser("compare", help="the main comparison (§9-§10)")
    comp.add_argument("--runs", nargs="+", required=True)
    comp.add_argument("--oracles", required=True)
    comp.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"tune": tune_main, "oracles": oracles_main, "compare": compare_main}[args.cmd](args)


if __name__ == "__main__":
    main()
