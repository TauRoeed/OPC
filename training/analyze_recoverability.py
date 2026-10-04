"""Recoverability tables for the representation-bias study (development runs).

Stage 1 (structural): from ``training.oracle_repair`` rows, per condition (dataset, bias, seed). The learner's class
(``repair``) is the linear repair with a learnable logit scale; both oracle fits (scale fixed and scale learned) are
policies in it, so its bound is the better of the two (value and greedy value separately):
  V_logger, V_logger_greedy        the biased logger's exact true value and its ranking's (greedy) value
  V_clean, V_clean_greedy          the matched no-bias reference: the same dataset and seed with bias ``none``
                                   (its greedy value is the ceiling: each user's truly best item)
  V_oracle[_greedy] per class      the best exact value the learner's class reaches trained on the truth
  representation_loss              V_clean - V_logger          (and the greedy version)
  oracle_repair_gain               V_oracle - V_logger         (and the greedy version)
  structural_recoverability        gain / loss, not clipped; NaN when |loss| < MIN_LOSS (then read the raw values)

The stochastic values mix ranking and sharpness: the clean reference is the clean logger at its own exploration
level, while the repair class can also sharpen (a linear map can scale the vectors). The greedy values compare
rankings only and are the primary structural measure.

Stage 2 (learned): ``learned_recovery`` joins learned runs to the oracle rows: learned_gain = V_method - V_logger
and fraction_of_oracle_repair = learned_gain / (V_oracle - V_logger) (NaN when the denominator is below MIN_LOSS).
``stage2_tables`` summarizes them (fractions per arm, paired arm differences with 95% t-intervals over the
dataset × seed conditions, selection diagnostics) and ``paired_runs`` compares two runs of one arm condition by
condition (e.g. the shrink:100 robustness slice against the main harmonic:0.1 run).

CLI (writes CSV tables):
  python -m training.analyze_recoverability stage1 ORACLE_ROOT --out DIR
  python -m training.analyze_recoverability stage2 ORACLE_ROOT --runs RUN [RUN ...] [--su RUN] --out DIR
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.representation_bias import bias_label, parse_bias

MIN_LOSS = 1e-3  # CTR points / 100: below this a ratio is not reported
CLASSES = ("repair", "linear", "linear+scale", "scale")
_CONFIGS = (("none", "no bias"), ("medium", "combined medium"), ("high", "combined high"),
            ("high/none/none", "warp only (high)"), ("none/high/none", "group only (high)"), ("none/none/high", "vector only (high)"))
BIAS_ORDER = tuple(bias_label(parse_bias(c)) for c, _ in _CONFIGS)  # the labels the runners record
BIAS_NAMES = {bias_label(parse_bias(c)): name for c, name in _CONFIGS}


def load_oracle(root) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(Path(root) / "**" / "oracle_repair.csv"), recursive=True))]
    if not frames:
        raise FileNotFoundError(f"no oracle_repair.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def _ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan)
    ok = np.abs(den) >= MIN_LOSS
    out[ok] = num[ok] / den[ok]
    return out


def derive(rows: pd.DataFrame) -> pd.DataFrame:
    """Adds the clean reference (bias none of the same dataset and seed) and the derived quantities."""
    df = rows.copy()
    df["bias"] = [bias_label(parse_bias(b)) for b in df["bias"]]
    if {"oracle_linear_value", "oracle_linear+scale_value"} <= set(df.columns):  # the learner's class: best of both fits
        df["oracle_repair_value"] = df[["oracle_linear_value", "oracle_linear+scale_value"]].max(axis=1)
        df["oracle_repair_greedy"] = df[["oracle_linear_greedy", "oracle_linear+scale_greedy"]].max(axis=1)
    clean = df[df["bias"] == bias_label(parse_bias("none"))].set_index(["dataset", "seed"])[["logger_value", "logger_greedy"]]
    clean = clean.rename(columns={"logger_value": "V_clean", "logger_greedy": "V_clean_greedy"})
    df = df.join(clean, on=["dataset", "seed"])
    df["V_logger"], df["V_logger_greedy"], df["ceiling"] = df["logger_value"], df["logger_greedy"], df["logger_ceiling"]
    df["representation_loss"] = df["V_clean"] - df["V_logger"]
    df["representation_loss_greedy"] = df["V_clean_greedy"] - df["V_logger_greedy"]
    for cls in CLASSES:
        if f"oracle_{cls}_value" not in df:
            continue
        df[f"gain_{cls}"] = df[f"oracle_{cls}_value"] - df["V_logger"]
        df[f"gain_{cls}_greedy"] = df[f"oracle_{cls}_greedy"] - df["V_logger_greedy"]
        df[f"recoverability_{cls}"] = _ratio(df[f"gain_{cls}"], df["representation_loss"])
        df[f"recoverability_{cls}_greedy"] = _ratio(df[f"gain_{cls}_greedy"], df["representation_loss_greedy"])
    return df


def summary(df: pd.DataFrame, cls: str = "repair") -> pd.DataFrame:
    """Per bias configuration: means over datasets and seeds (CTR in %), and the per-dataset greedy recoverability."""
    cols = {"V_logger": "V_logger %", "V_logger_greedy": "V_logger greedy %", "V_clean": "V_clean %",
            "ceiling": "ceiling %", f"oracle_{cls}_value": "V_oracle %", f"oracle_{cls}_greedy": "V_oracle greedy %",
            "representation_loss": "loss %", "representation_loss_greedy": "loss greedy %",
            f"gain_{cls}": "gain %", f"gain_{cls}_greedy": "gain greedy %"}
    g = df.groupby("bias")
    out = pd.DataFrame({new: 100 * g[old].mean() for old, new in cols.items()})
    out["recoverability"] = g[f"recoverability_{cls}"].mean()
    out["recoverability greedy"] = g[f"recoverability_{cls}_greedy"].mean()
    out["recoverability greedy min"] = g[f"recoverability_{cls}_greedy"].min()
    out["recoverability greedy max"] = g[f"recoverability_{cls}_greedy"].max()
    per_ds = df.pivot_table(index="bias", columns="dataset", values=f"recoverability_{cls}_greedy", aggfunc="mean")
    out = out.join(per_ds.add_prefix("rec. greedy "))
    out["n"] = g.size()
    order = [b for b in BIAS_ORDER if b in out.index] + [b for b in out.index if b not in BIAS_ORDER]
    out = out.loc[order]
    out.insert(0, "bias type", [BIAS_NAMES.get(b, b) for b in out.index])
    return out


def learned_recovery(learned: pd.DataFrame, oracle: pd.DataFrame, cls: str = "repair") -> pd.DataFrame:
    """``learned``: one row per (dataset, bias, seed, method, train_size) with columns V_method and V_method_greedy
    (true stochastic and greedy CTR of the selected policy). Joins the oracle rows of the same world."""
    o = derive(oracle)[["dataset", "bias", "seed", "V_logger", "V_logger_greedy", f"oracle_{cls}_value",
                        f"oracle_{cls}_greedy"]]
    learned = learned.assign(bias=[bias_label(parse_bias(b)) for b in learned["bias"]])
    m = learned.merge(o, on=["dataset", "bias", "seed"], how="left", validate="many_to_one")
    m["learned_gain"] = m["V_method"] - m["V_logger"]
    m["learned_gain_greedy"] = m["V_method_greedy"] - m["V_logger_greedy"]
    m["fraction_of_oracle_repair"] = _ratio(m["learned_gain"], m[f"oracle_{cls}_value"] - m["V_logger"])
    m["fraction_of_oracle_repair_greedy"] = _ratio(m["learned_gain_greedy"], m[f"oracle_{cls}_greedy"] - m["V_logger_greedy"])
    return m


def _tags(folder: str) -> dict:
    return dict(part.split("=", 1) for part in folder.split("__") if "=" in part)


def load_learned(*run_dirs) -> pd.DataFrame:
    """One row per (condition, method, train size) of study runs: the selected policy's true stochastic and greedy
    CTR (V_method, V_method_greedy), its learned logit scale, the tempered logger's value in the same cell, and the
    selected trial's diagnostics (raw-weight ESS, share of weights > 10, largest weight, selection-estimate error of
    the DR point estimate and of the lower bound) and the true selection regret over the arm's trials (development
    diagnostic)."""
    out = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "summary_metrics.csv").exists():
                continue
            tags = _tags(cond.name)
            s = pd.read_csv(cond / "summary_metrics.csv")
            t = pd.read_csv(cond / "trials_long.csv")
            s = s[s["train_size"] > 0]
            temp = s[s["method"] == "tempered_logger"].set_index("train_size")["policy_rewards"]
            for _, r in s.iterrows():
                g = t[(t["method"] == r["method"]) & (t["train_size"] == r["train_size"])]
                best = g[g["is_best_in_run"].astype(bool)].iloc[0]
                assert abs(float(best["actual_reward"]) - float(r["policy_rewards"])) < 1e-9
                out.append(dict(
                    run=Path(run).name, dataset=tags["dataset"], bias=tags["bias"], seed=int(tags["seed"]),
                    method=r["method"], train_size=int(r["train_size"]), V_method=float(r["policy_rewards"]),
                    V_method_greedy=float(r.get("policy_rewards_greedy", np.nan)),  # absent in runs before c1b6dfb
                    logit_scale=float(best.get("logit_scale", np.nan)),
                    V_tempered=float(temp.get(r["train_size"], np.nan)),
                    ess_raw=float(best.get("ess_raw", np.nan)), w_share_gt10=float(best.get("diag_w_share_gt10", np.nan)),
                    w_max=float(best.get("diag_w_max", np.nan)),
                    sel_error_point=float(best["r_hat"] - best["actual_reward"]),
                    sel_error_lower=float(best["value"] - best["actual_reward"]),
                    regret=float(g["actual_reward"].max() - best["actual_reward"]), n_trials=len(g)))
    return pd.DataFrame(out)


ARMS = ("opc", "dm", "no_propensity", "tempered_logger")
_KEYS = ["dataset", "bias", "seed", "train_size"]


def mean_ci(x) -> tuple[float, float, float, int]:
    """Mean and 95% t-interval over paired conditions (the interval is NaN below two)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    if n < 2:
        return mean, np.nan, np.nan, n
    half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half, n


DATASET_ORDER = ("ml", "kuairand", "anime")


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """The study's bias order, then dataset, train size and arm (rows keep their order otherwise), with the bias name."""
    if df.empty or "bias" not in df:  # e.g. no OPC rows yet, so no OPC contrasts
        return df
    df = df.copy()
    df["_o"] = df["bias"].map({b: i for i, b in enumerate(BIAS_ORDER)}).fillna(len(BIAS_ORDER))
    sort = ["_o"]
    if "dataset" in df:
        df["_d"] = df["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(len(DATASET_ORDER))
        sort.append("_d")
    if "train_size" in df:
        sort.append("train_size")
    if "method" in df:
        df["_m"] = df["method"].map({a: i for i, a in enumerate(ARMS)}).fillna(len(ARMS))
        sort.append("_m")
    df = df.sort_values(sort, kind="stable").drop(columns=[c for c in ("_o", "_d", "_m") if c in df])
    df.insert(df.columns.get_loc("bias") + 1, "bias type", df["bias"].map(lambda b: BIAS_NAMES.get(b, b)))
    return df.reset_index(drop=True)


def stage2_tables(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stage 2 summaries from ``learned_recovery`` rows (CTR values and differences in points):
      fractions    per bias × train size × arm, means over the dataset × seed conditions: the fraction of the
                   oracle repair recovered (greedy and stochastic), the learned gain, the true CTR, and the
                   greedy fraction per dataset;
      paired       per bias × train size: OPC minus each other arm in true CTR (stochastic and greedy), mean and
                   95% t-interval over the paired conditions (the arms share configurations and seeds);
      diagnostics  per bias × train size × arm, means: the selected trial's raw-weight ESS, share of weights
                   above 10, largest weight, learned logit scale, selection-estimate errors (DR point and
                   lower bound minus the truth) and the true selection regret over the arm's trials."""
    g = m.groupby(["bias", "train_size", "method"])
    fractions = pd.DataFrame({
        "n": g.size(),
        "fraction greedy": g["fraction_of_oracle_repair_greedy"].mean(),
        "fraction": g["fraction_of_oracle_repair"].mean(),
        "gain greedy %": 100 * g["learned_gain_greedy"].mean(),
        "gain %": 100 * g["learned_gain"].mean(),
        "V greedy %": 100 * g["V_method_greedy"].mean(),
        "V %": 100 * g["V_method"].mean(),
    })
    per_ds = m.pivot_table(index=["bias", "train_size", "method"], columns="dataset",
                           values="fraction_of_oracle_repair_greedy", aggfunc="mean")
    fractions = fractions.join(per_ds.add_prefix("fraction greedy ")).reset_index()

    rows = []
    for measure, col in (("stochastic", "V_method"), ("greedy", "V_method_greedy")):
        wide = m.pivot_table(index=_KEYS, columns="method", values=col)
        others = list(ARMS[1:]) + sorted(c for c in wide.columns if c not in ARMS)  # robustness arms last
        for (bias, size), cell in wide.groupby(level=["bias", "train_size"]):
            for other in others:
                if "opc" not in cell or other not in cell:
                    continue
                mean, lo, hi, n = mean_ci(100 * (cell["opc"] - cell[other]))
                rows.append(dict(bias=bias, train_size=size, contrast=f"opc - {other}", measure=measure,
                                 mean=mean, ci_low=lo, ci_high=hi, n=n))
    paired = pd.DataFrame(rows)

    d = m.groupby(["bias", "train_size", "method"])
    diagnostics = pd.DataFrame({
        "n": d.size(), "ess_raw": d["ess_raw"].mean(), "w>10 %": 100 * d["w_share_gt10"].mean(),
        "w_max": d["w_max"].mean(), "logit_scale": d["logit_scale"].mean(),
        "sel error point %": 100 * d["sel_error_point"].mean(), "sel error lower %": 100 * d["sel_error_lower"].mean(),
        "regret %": 100 * d["regret"].mean(), "n_trials": d["n_trials"].mean(),
    }).reset_index()
    return {"fractions": _ordered(fractions), "paired": _ordered(paired), "diagnostics": _ordered(diagnostics)}


def paired_runs(a: pd.DataFrame, b: pd.DataFrame, method: str = "opc") -> pd.DataFrame:
    """``a`` minus ``b`` for one arm, condition by condition (``learned_recovery`` rows of two runs that share
    worlds, configurations and seeds): per bias × train size, mean and 95% t-interval of the difference in true
    CTR (stochastic, greedy; points), in the fraction of the oracle repair (greedy), and in the raw-weight ESS."""
    cols = ["V_method", "V_method_greedy", "fraction_of_oracle_repair_greedy", "ess_raw", "w_share_gt10"]
    x = a[a["method"] == method].set_index(_KEYS)[cols]
    y = b[b["method"] == method].set_index(_KEYS)[cols]
    both = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
    rows = []
    for (bias, size), cell in both.groupby(level=["bias", "train_size"]):
        for measure, col, scale in (("V %", "V_method", 100), ("V greedy %", "V_method_greedy", 100),
                                    ("fraction greedy", "fraction_of_oracle_repair_greedy", 1),
                                    ("ess_raw", "ess_raw", 1), ("w>10 %", "w_share_gt10", 100)):
            mean, lo, hi, n = mean_ci(scale * (cell[f"{col}_a"] - cell[f"{col}_b"]))
            rows.append(dict(bias=bias, train_size=size, measure=measure, a=scale * cell[f"{col}_a"].mean(),
                             b=scale * cell[f"{col}_b"].mean(), diff=mean, ci_low=lo, ci_high=hi, n=n))
    return _ordered(pd.DataFrame(rows))


REPAIR = ("linear", "linear+scale")
_WORLD = ["dataset", "bias", "seed"]


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(bias=[bias_label(parse_bias(b)) for b in df["bias"]])


def validated_oracle(oracle: pd.DataFrame, candidates: pd.DataFrame, base_steps: int = 3000) -> pd.DataFrame:
    """Stage 1 rows with each repair class re-taken over the validation candidates (``training.oracle_validation``)
    as well. The rule is Stage 1's: per class, the candidate with the best exact stochastic value, and that
    candidate's greedy value. The Stage 1 winner is pooled with the candidates, so a bound can only rise. The
    Stage 1 values are kept as ``stage1_oracle_{cls}_{value,greedy}``. The new winner's rate and budget are
    ``oracle_{cls}_{lr,steps}``, and ``oracle_{cls}_greedy_max`` is the best greedy value over every candidate of
    the class (a sensitivity: the ranking bound without the stochastic selection rule)."""
    out = _canonical(oracle)
    cand = _canonical(candidates)
    for cls in REPAIR:
        s1 = out[_WORLD + [f"oracle_{cls}_{k}" for k in ("value", "greedy", "logit_scale", "lr")]]
        s1 = s1.rename(columns=lambda c: c.replace(f"oracle_{cls}_", "")).assign(steps=base_steps)
        pool = pd.concat([s1, cand.loc[cand["cls"] == cls, _WORLD + ["value", "greedy", "logit_scale", "lr", "steps"]]],
                         ignore_index=True)
        best = pool.loc[pool.groupby(_WORLD)["value"].idxmax()].set_index(_WORLD)
        idx = pd.MultiIndex.from_frame(out[_WORLD])
        for k in ("value", "greedy"):
            out[f"stage1_oracle_{cls}_{k}"] = out[f"oracle_{cls}_{k}"]
        for k in ("value", "greedy", "logit_scale", "lr", "steps"):
            out[f"oracle_{cls}_{k}"] = best[k].reindex(idx).to_numpy()
        out[f"oracle_{cls}_greedy_max"] = pool.groupby(_WORLD)["greedy"].max().reindex(idx).to_numpy()
        out[f"oracle_{cls}_candidates"] = pool.groupby(_WORLD).size().reindex(idx).to_numpy()
    return out


def oracle_validation_table(validated: pd.DataFrame) -> pd.DataFrame:
    """Per world: the greedy repair bound and structural recoverability under Stage 1's search and the widened one
    (same rule), their changes, the sensitivity without the selection rule, and the winning rates and budgets."""
    new = derive(validated)
    old = derive(validated.assign(**{f"oracle_{c}_{k}": validated[f"stage1_oracle_{c}_{k}"] for c in REPAIR
                                     for k in ("value", "greedy")}))
    rows = new[_WORLD + ["V_logger_greedy", "ceiling", "representation_loss_greedy"]].copy()
    rows["old bound greedy"] = old["oracle_repair_greedy"].to_numpy()
    rows["new bound greedy"] = new["oracle_repair_greedy"].to_numpy()
    rows["old recoverability greedy"] = old["recoverability_repair_greedy"].to_numpy()
    rows["new recoverability greedy"] = new["recoverability_repair_greedy"].to_numpy()
    rows["change"] = rows["new recoverability greedy"] - rows["old recoverability greedy"]
    gmax = validated[[f"oracle_{c}_greedy_max" for c in REPAIR]].max(axis=1).to_numpy()
    rows["recoverability greedy, best greedy candidate"] = _ratio(gmax - new["V_logger_greedy"], new["representation_loss_greedy"])
    rows["old bound value"] = old["oracle_repair_value"].to_numpy()
    rows["new bound value"] = new["oracle_repair_value"].to_numpy()
    for c in REPAIR:
        rows[f"{c} lr"] = validated[f"oracle_{c}_lr"].to_numpy()
        rows[f"{c} steps"] = validated[f"oracle_{c}_steps"].to_numpy()
    return _ordered(rows[rows["bias"] != "none"])


def oracle_validation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """``oracle_validation_table`` averaged per bias (over datasets and seeds) and per dataset × bias (over seeds)."""
    cols = {"old recoverability greedy": "mean", "new recoverability greedy": "mean", "change": "mean",
            "recoverability greedy, best greedy candidate": "mean"}
    parts = []
    for keys in (["bias"], ["dataset", "bias"]):
        g = table.groupby(keys).agg(cols).join(table.groupby(keys)["change"].agg(max_change="max", n="size"))
        parts.append(g.reset_index().assign(dataset=lambda f: f["dataset"] if "dataset" in f else "all"))
    out = pd.concat(parts, ignore_index=True)
    out = _ordered(out)
    return out[["bias", "bias type", "dataset", "n", *cols, "max_change"]]


def stage1_by_dataset(d: pd.DataFrame, cls: str = "repair") -> pd.DataFrame:
    """Per dataset × bias (``derive`` rows; CTR points, mean over the seeds): the logger's ranking loss (ceiling minus
    its greedy value), the oracle repair's ranking gain and the structural recoverability (greedy), with its range."""
    x = d[d["bias"] != "none"]
    g = x.groupby(["dataset", "bias"])
    out = pd.DataFrame({
        "n": g.size(), "logger ranking loss %": 100 * g["representation_loss_greedy"].mean(),
        "oracle repair gain %": 100 * g[f"gain_{cls}_greedy"].mean(),
        "structural recoverability": g[f"recoverability_{cls}_greedy"].mean(),
        "recoverability min": g[f"recoverability_{cls}_greedy"].min(), "recoverability max": g[f"recoverability_{cls}_greedy"].max(),
    }).reset_index()
    return _ordered(out)


_DIFFS = {"OPC-DM greedy": ("V_method_greedy", "opc", "dm"), "OPC-no-prop greedy": ("V_method_greedy", "opc", "no_propensity"),
          "OPC-DM": ("V_method", "opc", "dm"), "OPC-no-prop": ("V_method", "opc", "no_propensity")}


def stage2_by_dataset(m: pd.DataFrame) -> pd.DataFrame:
    """Per dataset × bias × train size (``learned_recovery`` rows; CTR points, mean over the seeds):
      - each arm's true gain over the logger, greedy (ranking; the tempered logger's is 0) and stochastic;
      - OPC's and DM-only's fraction of the oracle ranking repair;
      - OPC minus DM-only and minus no-propensity, greedy and stochastic, with the seeds' min and max."""
    keys = ["dataset", "bias", "train_size"]
    piv = lambda col: m.pivot_table(index=keys + ["seed"], columns="method", values=col)
    gain_g, gain_s, frac, vg, vs = (piv(c) for c in ("learned_gain_greedy", "learned_gain", "fraction_of_oracle_repair_greedy",
                                                      "V_method_greedy", "V_method"))
    per_seed = pd.DataFrame({
        "OPC gain greedy %": 100 * gain_g["opc"], "DM gain greedy %": 100 * gain_g["dm"],
        "no-prop gain greedy %": 100 * gain_g["no_propensity"],
        "OPC gain %": 100 * gain_s["opc"], "DM gain %": 100 * gain_s["dm"], "no-prop gain %": 100 * gain_s["no_propensity"],
        "tempered gain %": 100 * gain_s["tempered_logger"],
        "OPC fraction greedy": frac["opc"], "DM fraction greedy": frac["dm"],
        **{k: 100 * ((vg if col == "V_method_greedy" else vs)[a] - (vg if col == "V_method_greedy" else vs)[b])
           for k, (col, a, b) in _DIFFS.items()},
    })
    g = per_seed.groupby(level=keys)
    out = g.mean().join(g.min()[list(_DIFFS)].add_suffix(" min")).join(g.max()[list(_DIFFS)].add_suffix(" max"))
    out.insert(0, "n", g.size())
    return _ordered(out.reset_index())


GAP_COLUMNS = ("V_target_best", "V_logger", "V_oracle_repair", "V_OPC", "representation_loss",
               "structurally_recoverable_gain", "learned_repair_gain", "structural_gap", "learning_gap")


def gap_decomposition(learned: pd.DataFrame, oracle: pd.DataFrame, method: str = "opc") -> pd.DataFrame:
    """Greedy (ranking) decomposition of the target gap, one row per condition × train size. V_target_best is the
    ceiling: each user's truly best item, the clean ranking's greedy value, the same for every bias configuration of
    a dataset and seed. V_logger is the logger's greedy value, V_oracle_repair the repair class's greedy bound
    (``derive``), and V_OPC the selected policy's greedy value. representation_loss = V_target_best − V_logger =
    structural_gap (V_target_best − V_oracle_repair) + learning_gap (V_oracle_repair − V_OPC) + learned_repair_gain
    (V_OPC − V_logger). The shares divide by the representation loss (NaN below MIN_LOSS)."""
    o = derive(oracle)[_WORLD + ["ceiling", "V_logger_greedy", "oracle_repair_greedy"]]
    x = _canonical(learned[learned["method"] == method]).merge(o, on=_WORLD, how="inner", validate="many_to_one")
    out = x[_WORLD + ["train_size"]].copy()
    out["V_target_best"], out["V_logger"] = x["ceiling"], x["V_logger_greedy"]
    out["V_oracle_repair"], out["V_OPC"] = x["oracle_repair_greedy"], x["V_method_greedy"]
    out["representation_loss"] = out["V_target_best"] - out["V_logger"]
    out["structurally_recoverable_gain"] = out["V_oracle_repair"] - out["V_logger"]
    out["learned_repair_gain"] = out["V_OPC"] - out["V_logger"]
    out["structural_gap"] = out["V_target_best"] - out["V_oracle_repair"]
    out["learning_gap"] = out["V_oracle_repair"] - out["V_OPC"]
    for part in ("structural_gap", "learning_gap", "learned_repair_gain"):
        out[f"{part} share"] = _ratio(out[part], out["representation_loss"])
    return out


def gap_summary(gaps: pd.DataFrame, by_dataset: bool = False) -> pd.DataFrame:
    """Means of ``gap_decomposition`` rows per bias × train size (and dataset), values in CTR points."""
    keys = (["dataset"] if by_dataset else []) + ["bias", "train_size"]
    g = gaps.groupby(keys)
    out = pd.DataFrame({"n": g.size()})
    for c in GAP_COLUMNS:
        out[f"{c} %"] = 100 * g[c].mean()
    for part in ("structural_gap", "learning_gap", "learned_repair_gain"):
        out[f"{part} share"] = g[f"{part} share"].mean()
    out["min learning_gap %"] = 100 * g["learning_gap"].min()  # negative would mean OPC beat the class bound
    return _ordered(out.reset_index())


def gap_figure(summary: pd.DataFrame, path) -> None:
    """Stacked bars of the representation loss per bias (rows) and train size (panels): recovered by OPC, expressible
    but not learned (learning gap), not expressible by the repair class (structural gap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary[summary["bias"] != "none"]
    sizes = sorted(s["train_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.2), sharey=True, squeeze=False)
    parts = (("learned_repair_gain %", "learned by OPC", "#2b6cb0"), ("learning_gap %", "learning gap", "#90cdf4"),
             ("structural_gap %", "structural gap", "#a0aec0"))
    for ax, size in zip(axes[0], sizes):
        t = s[s["train_size"] == size].iloc[::-1]
        left = pd.Series(0.0, index=t.index)
        for col, label, color in parts:
            ax.barh(t["bias type"], t[col], left=left, color=color, label=label, edgecolor="white", linewidth=0.8)
            left = left + t[col]
        ax.set_title(f"train {int(size):,}", fontsize=10)
        ax.set_xlabel("greedy CTR points below the target best")
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Representation loss = learned + learning gap + structural gap (greedy, mean over datasets × seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["stage1", "stage2", "followup"])
    ap.add_argument("oracle_root")
    ap.add_argument("--runs", nargs="*", default=[], help="stage2: study run folders (the four arms)")
    ap.add_argument("--su", default=None, help="stage2: the shrink:100 robustness run (OPC only), paired with --runs")
    ap.add_argument("--robust", action="append", default=[], metavar="LABEL=DIR",
                    help="stage2: an OPC-only robustness run paired with --runs (repeatable): reported as the arm "
                         "opc_LABEL next to the others, and as LABEL minus the default OPC per bias x train size")
    ap.add_argument("--candidates", default=None, help="followup: the oracle validation folder (oracle_candidates.csv)")
    ap.add_argument("--use-validated", action="store_true",
                    help="followup: build the per-dataset and gap tables on the validated bounds (default: Stage 1's)")
    ap.add_argument("--out", default=None, help="folder for the CSV tables (default: print only)")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    oracle = load_oracle(a.oracle_root)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    if a.stage == "followup":
        tables = {}
        if a.candidates:
            cand = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(Path(a.candidates) / "**" / "oracle_candidates.csv"),
                                                                        recursive=True))], ignore_index=True)
            validated = validated_oracle(oracle, cand)
            tables["oracle_validation_by_world"] = oracle_validation_table(validated)
            tables["oracle_validation_summary"] = oracle_validation_summary(tables["oracle_validation_by_world"])
            tables["oracle_validation_candidates"] = _ordered(_canonical(cand))
        learned = load_learned(*a.runs)
        base = validated if (a.candidates and a.use_validated) else oracle
        m = learned_recovery(learned, base)
        tables["stage1_by_dataset"] = stage1_by_dataset(derive(base))
        tables["stage2_by_dataset"] = stage2_by_dataset(m)
        gaps = gap_decomposition(learned, base)
        tables["gap_decomposition_rows"] = _ordered(gaps)
        tables["gap_decomposition"] = gap_summary(gaps)
        tables["gap_decomposition_by_dataset"] = gap_summary(gaps, by_dataset=True)
        if a.candidates and not a.use_validated:  # sensitivity: the same decomposition on the validated bounds
            tables["gap_decomposition_validated_bound"] = gap_summary(gap_decomposition(learned, validated))
        if out:
            gap_figure(tables["gap_decomposition"], out / "gap_decomposition.png")
        for name in ("oracle_validation_by_world", "stage1_by_dataset", "gap_decomposition"):
            if name in tables:
                print(f"\n===== {name}\n{tables[name].round(4).to_string()}")
    elif a.stage == "stage1":
        tables = {"stage1_oracle_rows": derive(oracle).round(6), "stage1_recoverability_by_bias": summary(derive(oracle)).round(4)}
        for cls in CLASSES:
            print(f"\n===== class {cls}")
            print(summary(derive(oracle), cls).round(3).to_string())
    else:
        m = learned_recovery(load_learned(*a.runs), oracle)
        robust = {}
        for spec in a.robust:
            label, path = spec.split("=", 1)
            robust[label] = learned_recovery(load_learned(path), oracle)
        m_all = pd.concat([m] + [r[r["method"] == "opc"].assign(method=f"opc_{label}") for label, r in robust.items()],
                          ignore_index=True)
        tables = {f"stage2_{k}": v for k, v in stage2_tables(m_all).items()}
        tables["stage2_learned_rows"] = m_all
        for label, r in robust.items():
            tables[f"stage2_robust_{label}_minus_default"] = paired_runs(r, m)
        if a.su:
            su = learned_recovery(load_learned(a.su), oracle)
            tables["stage2_su_harmonic_minus_shrink100"] = paired_runs(m, su)
        for name, t in tables.items():
            if name != "stage2_learned_rows":
                print(f"\n===== {name}\n{t.round(3).to_string()}")
    if out:
        for name, t in tables.items():
            t.to_csv(out / f"{name}.csv", index=name == "stage1_recoverability_by_bias")
        print(f"\nwrote {len(tables)} tables to {out}")


if __name__ == "__main__":
    main()
