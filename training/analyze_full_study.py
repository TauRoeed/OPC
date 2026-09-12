import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.metrics_utils import pct_change


_LEVEL_ORDER = ["low", "medium", "high", "extreme", "brutal", "catastrophic"]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _parse_condition_dirname(name: str):
    out = {}
    for part in name.split("__"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _load_summary(path: Path):
    df = pd.read_csv(path)
    numeric_cols = [
        "train_size",
        "policy_rewards",
        "conv_dr",
        "seed",
        "ctr",
        "initial_reward",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "noise_axis" not in df.columns:
        df["noise_axis"] = "combined"
    if "ctr" not in df.columns:
        df["ctr"] = np.nan
    return df


def _condition_data_paths(run_dir: Path, filename: str):
    for p in sorted(run_dir.rglob(filename)):
        if p.parent.name.startswith("dataset="):
            yield p


def _load_long(run_dir: Path, filename: str):
    frames = []
    for p in _condition_data_paths(run_dir, filename):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        tags = _parse_condition_dirname(p.parent.name)
        mapping = {
            "dataset": "dataset",
            "noise": "noise_mode",
            "axis": "noise_axis",
            "comp": "noise_component",
            "level": "noise_level",
            "seed": "seed",
            "ctr": "ctr",
            "val": "val_size",
        }
        for src, dst in mapping.items():
            if src in tags and dst not in df.columns:
                df[dst] = tags[src]
            elif src in tags and dst in ("val_size", "ctr", "seed"):
                tagged = pd.to_numeric(tags[src], errors="coerce")
                current = pd.to_numeric(df[dst], errors="coerce")
                df[dst] = current.fillna(tagged)
        if "noise_component" not in df.columns:
            df["noise_component"] = "combined"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in ("train_size", "run", "seed", "initial_reward", "val_size"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "ctr" in out.columns:
        out["ctr"] = pd.to_numeric(out["ctr"], errors="coerce")
    if "noise_axis" not in out.columns:
        out["noise_axis"] = "combined"
    if "noise_component" not in out.columns:
        out["noise_component"] = "combined"
    return out


def _load_trials_long_union(run_dir: Path) -> pd.DataFrame:
    """Merge per-condition trial logs (unified or split OPC / no-prop filenames)."""
    parts = []
    for name in ("trials_long.csv", "opc_trials_long.csv", "no_prop_trials_long.csv"):
        df = _load_long(run_dir, name)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    dedupe_cols = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "ctr",
            "seed",
            "method",
            "train_size",
            "val_size",
            "run",
            "trial_number",
        )
        if c in out.columns
    ]
    if len(dedupe_cols) >= 5:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    return out


def _squeeze_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a trained policy (exclude train_size==0 baselines)."""
    if df.empty or "train_size" not in df.columns:
        return df
    return df[pd.to_numeric(df["train_size"], errors="coerce") > 0].copy()


def _resolve_sweep_axis(df: pd.DataFrame) -> tuple[str, str]:
    """Pick x-axis dimension: prefer validation size when train size is fixed."""
    work = _eval_rows(df)
    candidates: list[tuple[str, str, int]] = []
    for col, label in (
        ("val_size", "validation size"),
        ("val_size_config", "validation size"),
        ("train_size", "train_size"),
    ):
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").dropna().unique()
        if len(vals) > 1:
            candidates.append((col, label, len(vals)))
    if candidates:
        candidates.sort(key=lambda item: (0 if item[0].startswith("val") else 1, -item[2]))
        return candidates[0][0], candidates[0][1]
    if "val_size" in df.columns:
        return "val_size", "validation size"
    return "train_size", "train_size"


def _load_runs_union(run_dir: Path) -> pd.DataFrame:
    runs = _load_long(run_dir, "runs_long.csv")
    if runs.empty:
        runs = _load_long(run_dir, "opc_runs_long.csv")
        r2 = _load_long(run_dir, "no_prop_runs_long.csv")
        if not r2.empty:
            runs = pd.concat([runs, r2], ignore_index=True) if not runs.empty else r2
    return runs


def _apply_slim_reward_fallback(df: pd.DataFrame, reward_col: str = "policy_rewards") -> pd.DataFrame:
    """Use actual_reward_selected when slim runs skip post-hoc policy_rewards."""
    if df.empty or reward_col not in df.columns:
        return df
    out = df.copy()
    pr = pd.to_numeric(out[reward_col], errors="coerce")
    if pr.notna().any():
        return out
    for fb in ("actual_reward_selected", "actual_reward"):
        if fb not in out.columns:
            continue
        fill = pd.to_numeric(out[fb], errors="coerce")
        if fill.notna().any():
            out[reward_col] = fill
            break
    return out


def _enrich_reward_pct_columns(df: pd.DataFrame, reward_col: str = "policy_rewards") -> pd.DataFrame:
    out = df.copy()
    if reward_col not in out.columns:
        return out
    pr = pd.to_numeric(out[reward_col], errors="coerce")
    ir = pd.to_numeric(out.get("initial_reward"), errors="coerce")
    ctr = pd.to_numeric(out.get("ctr"), errors="coerce")
    out["reward_pct_vs_initial"] = [
        pct_change(float(a), float(b)) if np.isfinite(a) and np.isfinite(b) else float("nan")
        for a, b in zip(pr, ir)
    ]
    out["reward_pct_vs_ctr"] = [
        pct_change(float(a), float(b)) if np.isfinite(a) and np.isfinite(b) else float("nan")
        for a, b in zip(pr, ctr)
    ]
    return out


def _backfill_summary_from_runs(summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or runs.empty:
        return summary
    runs = _apply_slim_reward_fallback(runs)
    if pd.to_numeric(summary.get("policy_rewards"), errors="coerce").notna().any():
        return summary
    key = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "seed",
            "method",
            "train_size",
            "val_size",
        )
        if c in summary.columns and c in runs.columns
    ]
    if len(key) < 4:
        return summary
    fill = runs[key + ["policy_rewards"]].drop_duplicates(subset=key)
    out = summary.copy()
    fill = fill.copy()
    for k in key:
        if k in ("seed", "train_size", "val_size", "val_size_config", "ctr"):
            out[k] = pd.to_numeric(out[k], errors="coerce")
            fill[k] = pd.to_numeric(fill[k], errors="coerce")
        else:
            out[k] = out[k].astype(str)
            fill[k] = fill[k].astype(str)
    out = out.merge(fill, on=key, how="left", suffixes=("", "_run"))
    if "policy_rewards_run" in out.columns:
        pr = pd.to_numeric(out["policy_rewards"], errors="coerce")
        fr = pd.to_numeric(out["policy_rewards_run"], errors="coerce")
        out["policy_rewards"] = pr.where(pr.notna(), fr)
        out = out.drop(columns=["policy_rewards_run"])
    return out


def _load_meta_ctr(run_dir: Path):
    rows = []
    for p in _condition_data_paths(run_dir, "run_meta.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = _parse_condition_dirname(p.parent.name)
        rows.append(
            {
                "dataset": tags.get("dataset", obj.get("dataset")),
                "noise_mode": tags.get("noise", obj.get("noise_mode")),
                "noise_axis": tags.get("axis", obj.get("noise_axis", "combined")),
                "noise_level": tags.get("level", obj.get("noise_level")),
                "seed": pd.to_numeric(tags.get("seed", obj.get("seed")), errors="coerce"),
                "ctr": pd.to_numeric(tags.get("ctr", obj.get("ctr", np.nan)), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pivot_delta(df: pd.DataFrame, metric: str):
    idx_cols = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "seed",
        "train_size",
        "ctr",
        "val_size",
        "val_size_config",
    ]
    idx_cols = [c for c in idx_cols if c in df.columns]
    if "ctr" in idx_cols and pd.to_numeric(df["ctr"], errors="coerce").notna().sum() == 0:
        idx_cols = [c for c in idx_cols if c != "ctr"]
    if "method" not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    piv = df.pivot_table(index=idx_cols, columns="method", values=metric, aggfunc="mean")
    if piv.empty or not {"opc", "no_propensity"}.issubset(piv.columns):
        return pd.DataFrame()
    tmp = piv.reset_index()
    tmp["delta"] = tmp["opc"] - tmp["no_propensity"]
    return tmp.dropna(subset=["opc", "no_propensity"])


def _ctr_ref(part: pd.DataFrame):
    vals = pd.to_numeric(part.get("ctr"), errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _initial_reward_ref(part: pd.DataFrame):
    if "initial_reward" not in part.columns:
        return np.nan
    vals = pd.to_numeric(part["initial_reward"], errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _ordered_levels(values):
    present = {v for v in values if pd.notna(v)}
    ordered = [level for level in _LEVEL_ORDER if level in present]
    extras = sorted(present - set(_LEVEL_ORDER))
    return ordered + extras


def _fmt_ctr(value):
    return f"{float(value):g}"


def _ctr_pct_vs_initial(ctr: float, initial_reward: float) -> float:
    return pct_change(ctr, initial_reward)


def _plot_trainsize_method_panels(
    agg: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    title_prefix: str,
    out_dir: Path,
    filename_prefix: str,
    scale: str,
    sweep_col: str,
    sweep_label: str,
):
    """Shared layout for oracle / selected policy curves (absolute or % vs initial)."""
    if agg.empty or y_col not in agg.columns or sweep_col not in agg.columns:
        return
    group_cols = ["dataset", "noise_mode", "noise_axis", "ctr"]
    group_cols = [c for c in group_cols if c in agg.columns]
    for key, part in agg.groupby(group_cols):
        if len(group_cols) == 4:
            dataset, noise_mode, axis, ctr = key
        else:
            continue
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        if not levels:
            continue
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), sharey=True)
        if len(levels) == 1:
            axes = [axes]

        ref_base = ref_df
        for c, v in zip(group_cols, key):
            if c in ref_base.columns:
                ref_base = ref_base[ref_base[c] == v]

        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level]
            ref_sub = ref_base[ref_base["noise_level"] == level] if "noise_level" in ref_base.columns else ref_base
            ir = _initial_reward_ref(ref_sub)
            ctr_line = _ctr_ref(ref_sub)

            if scale == "pct":
                y_ref = 0.0
                ctr_ref = _ctr_pct_vs_initial(ctr_line, ir) if np.isfinite(ctr_line) and np.isfinite(ir) else float("nan")
            else:
                y_ref = ir
                ctr_ref = ctr_line

            for method, mm in pp.groupby("method"):
                mm = mm.sort_values(sweep_col)
                x = pd.to_numeric(mm[sweep_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(mm[y_col], errors="coerce").to_numpy(dtype=float)
                se = pd.to_numeric(mm.get("se", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y, se = x[ok], y[ok], se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=method)
                if np.any(se > 0):
                    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.2)
            if scale == "pct":
                ax.axhline(0.0, color="gray", ls=":", lw=1.2, label="initial (0%)")
                if np.isfinite(ctr_ref):
                    ax.axhline(ctr_ref, color="tab:red", ls="--", lw=1.2, label="ctr vs initial")
            else:
                if np.isfinite(y_ref):
                    ax.axhline(y_ref, color="gray", ls=":", lw=1.2, label="initial_reward")
                if np.isfinite(ctr_ref):
                    ax.axhline(ctr_ref, color="tab:red", ls="--", lw=1.2, label="max (ctr)")
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel(sweep_label)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(y_label)
        h, lab = axes[-1].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="upper center", ncol=max(1, len(set(lab))))
        ctr_label = _fmt_ctr(ctr)
        suffix = "_pct" if scale == "pct" else ""
        fig.suptitle(
            f"{title_prefix} ({dataset}, {noise_mode}, axis={axis}, ctr={ctr_label}, by {sweep_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"{filename_prefix}_{dataset}_{noise_mode}_{axis}_ctr_{ctr_label}{suffix}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_curves_per_axis(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "policy_rewards",
    suffix: str = "",
    *,
    sweep_col: str,
    sweep_label: str,
):
    df = _eval_rows(df)
    if metric not in df.columns or df[metric].notna().sum() == 0:
        return
    grp_cols = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "ctr",
            "noise_level",
            "method",
            sweep_col,
        )
        if c in df.columns
    ]
    grp = (
        df.groupby(grp_cols)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
    levels = _LEVEL_ORDER

    for (dataset, noise_mode, noise_axis, ctr), part in grp.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level]
            for method, mm in pp.groupby("method"):
                mm = mm.sort_values(sweep_col)
                x = pd.to_numeric(mm[sweep_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(mm["mean"], errors="coerce").to_numpy(dtype=float)
                se = pd.to_numeric(mm["se"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x = x[ok]
                y = y[ok]
                se = se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=method)
                ax.fill_between(
                    x,
                    y - 1.96 * se,
                    y + 1.96 * se,
                    alpha=0.2,
                )
            if np.isfinite(ctr):
                ax.axhline(ctr, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel(sweep_label)
        ylab = metric if not suffix else f"{metric} (% vs initial)"
        axes[0].set_ylabel(ylab)
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        ctr_label = _fmt_ctr(ctr)
        title = "Curves" if not suffix else "Curves (% improvement vs initial)"
        fig.suptitle(
            f"{title} ({dataset}, {noise_mode}, axis={noise_axis}, ctr={ctr_label}, by {sweep_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"curves_{dataset}_{noise_mode}_{noise_axis}_ctr_{ctr_label}{suffix}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_pct_improvement_curves(
    df: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str
):
    enriched = _enrich_reward_pct_columns(_apply_slim_reward_fallback(df))
    _plot_curves_per_axis(
        enriched,
        out_dir,
        metric="reward_pct_vs_initial",
        suffix="_pct",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )


def _selection_with_ess(trials_long: pd.DataFrame, runs_long: pd.DataFrame, qs):
    if trials_long.empty or runs_long.empty:
        return pd.DataFrame()
    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        "train_size",
        "val_size",
        "run",
    ]
    key = [c for c in key if c in trials_long.columns and c in runs_long.columns]
    runs = runs_long.copy()
    if "val_size" not in runs.columns:
        runs["val_size"] = np.nan
    runs["val_size"] = pd.to_numeric(runs["val_size"], errors="coerce")

    t = trials_long.copy()
    t["ess"] = pd.to_numeric(t.get("ess"), errors="coerce")
    t["value"] = pd.to_numeric(t.get("value"), errors="coerce")
    if "val_size" not in t.columns:
        t = t.merge(runs[key + ["val_size"]].drop_duplicates(subset=key), on=key, how="left")
    else:
        t["val_size"] = pd.to_numeric(t["val_size"], errors="coerce")
    t["ess_ratio"] = t["ess"] / t["val_size"].clip(lower=1)

    outs = []
    gcols = [c for c in key if c != "run"]
    for q in qs:
        tt = t[t["ess_ratio"] >= float(q)].copy()
        if tt.empty:
            continue
        sel = tt.sort_values("value", ascending=False).groupby(gcols, as_index=False).first()
        chosen = runs.merge(sel[gcols + ["run"]], on=gcols + ["run"], how="inner")
        chosen["ess_threshold"] = float(q)
        outs.append(chosen)
    if not outs:
        return pd.DataFrame()
    return pd.concat(outs, ignore_index=True)


def _plot_ess_threshold_curves(
    selection_with_ess: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str
):
    if selection_with_ess.empty:
        return
    selection_with_ess = _apply_slim_reward_fallback(selection_with_ess)
    if selection_with_ess["policy_rewards"].notna().sum() == 0:
        return
    grp_cols = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "ctr",
            "noise_level",
            "method",
            "ess_threshold",
            sweep_col,
        )
        if c in selection_with_ess.columns
    ]
    grp = (
        selection_with_ess.groupby(grp_cols)["policy_rewards"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"].clip(lower=1))

    for (dataset, noise_mode, noise_axis, ctr, method), part in grp.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr", "method"]
    ):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.2), sharey=True)
        if len(levels) == 1:
            axes = [axes]
        for ax, level in zip(axes, levels):
            pp = part[part["noise_level"] == level]
            for q, qq in pp.groupby("ess_threshold"):
                qq = qq.sort_values(sweep_col)
                x = pd.to_numeric(qq[sweep_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(qq["mean"], errors="coerce").to_numpy(dtype=float)
                se = pd.to_numeric(qq["se"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
                x = x[ok]
                y = y[ok]
                se = se[ok]
                if len(x) == 0:
                    continue
                ax.plot(x, y, marker="o", label=f"ess>={q:.1f}")
                ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.18)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel(sweep_label)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("policy_rewards")
        h, lab = axes[-1].get_legend_handles_labels()
        if h:
            fig.legend(h, lab, loc="upper center", ncol=max(1, len(set(lab))))
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(
            f"ESS-filtered selection ({dataset}, {noise_mode}, axis={noise_axis}, {method}, ctr={ctr_label}, by {sweep_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"ess_curves_{dataset}_{noise_mode}_{noise_axis}_{method}_ctr_{ctr_label}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_delta_with_ci(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "policy_rewards",
    suffix: str = "",
    *,
    sweep_col: str,
    sweep_label: str,
):
    delta = _pivot_delta(_eval_rows(df), metric)
    if delta.empty or sweep_col not in delta.columns:
        return
    agg = (
        delta.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", sweep_col])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    levels = _LEVEL_ORDER

    for (dataset, noise_mode, noise_axis, ctr), part in agg.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, level in enumerate(levels):
            ax = axes[i]
            pp = part[part["noise_level"] == level].sort_values(sweep_col)
            if not pp.empty:
                ax.errorbar(pp[sweep_col], pp["mean"], yerr=1.96 * pp["se"], marker="o")
            ax.axhline(0.0, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel(sweep_label)
        ylab = "opc - no_propensity" if not suffix else "opc - no_propensity (% pts vs initial)"
        axes[0].set_ylabel(ylab)
        ctr_label = _fmt_ctr(ctr)
        title = "Delta with CI" if not suffix else "Delta with CI (% vs initial)"
        fig.suptitle(
            f"{title} ({dataset}, {noise_mode}, axis={noise_axis}, ctr={ctr_label}, by {sweep_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"delta_{dataset}_{noise_mode}_{noise_axis}_ctr_{ctr_label}{suffix}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_delta_pct_with_ci(df: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str):
    enriched = _enrich_reward_pct_columns(_apply_slim_reward_fallback(df))
    _plot_delta_with_ci(
        enriched,
        out_dir,
        metric="reward_pct_vs_initial",
        suffix="_pct",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )


def _plot_opc_pct_improvement_over_nop(
    df: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str
):
    """Relative % lift of OPC over no-propensity: 100*(opc-nop)/|nop|."""
    base = _apply_slim_reward_fallback(_eval_rows(df))
    if base.empty or "policy_rewards" not in base.columns:
        return
    idx = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "seed",
            sweep_col,
            "ctr",
        )
        if c in base.columns
    ]
    idx = list(dict.fromkeys(idx))
    piv = base.pivot_table(index=idx, columns="method", values="policy_rewards", aggfunc="mean")
    if piv.empty or not {"opc", "no_propensity"}.issubset(piv.columns):
        return
    tmp = piv.reset_index()
    nop = pd.to_numeric(tmp["no_propensity"], errors="coerce")
    opc = pd.to_numeric(tmp["opc"], errors="coerce")
    tmp["opc_lift_pct"] = 100.0 * (opc - nop) / nop.abs().clip(lower=1e-12)
    tmp = tmp[np.isfinite(tmp["opc_lift_pct"])]
    if tmp.empty:
        return
    agg = (
        tmp.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level", sweep_col])["opc_lift_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    for (dataset, noise_mode, noise_axis, ctr), part in agg.groupby(
        ["dataset", "noise_mode", "noise_axis", "ctr"]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for i, level in enumerate(_LEVEL_ORDER):
            ax = axes[i]
            pp = part[part["noise_level"] == level].sort_values(sweep_col)
            if not pp.empty:
                ax.errorbar(pp[sweep_col], pp["mean"], yerr=1.96 * pp["se"], marker="o")
            ax.axhline(0.0, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(level)
            ax.set_xlabel(sweep_label)
        axes[0].set_ylabel("OPC lift over no-prop (%)")
        ctr_label = _fmt_ctr(ctr)
        fig.suptitle(
            f"OPC % lift vs no-prop ({dataset}, {noise_mode}, axis={noise_axis}, ctr={ctr_label}, by {sweep_label})"
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"opc_lift_pct_{dataset}_{noise_mode}_{noise_axis}_ctr_{ctr_label}.png",
            dpi=180,
        )
        plt.close(fig)


def _plot_seed_strip(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "policy_rewards",
    suffix: str = "",
    *,
    sweep_col: str,
    sweep_label: str,
):
    df = _eval_rows(df)
    for (dataset, noise_axis), part in df.groupby(["dataset", "noise_axis"]):
        levels = _ordered_levels(part["noise_level"].dropna().unique().tolist())
        modes = sorted(part["noise_mode"].dropna().unique().tolist())
        fig, axes = plt.subplots(
            len(modes), len(levels), figsize=(5 * len(levels), 3.5 * len(modes)), squeeze=False
        )
        for i, mode in enumerate(modes):
            for j, level in enumerate(levels):
                ax = axes[i][j]
                pp = part[(part["noise_mode"] == mode) & (part["noise_level"] == level)]
                for method, mm in pp.groupby("method"):
                    x = np.log10(pd.to_numeric(mm[sweep_col], errors="coerce").clip(lower=1))
                    jitter = (
                        pd.to_numeric(mm.get("seed"), errors="coerce").fillna(0).astype(float) % 7 - 3
                    ) * 0.01
                    yv = pd.to_numeric(mm[metric], errors="coerce")
                    ax.scatter(x + jitter, yv, alpha=0.7, label=method, s=18)
                ax.set_title(f"{mode} | {level}")
                ax.set_xlabel(f"log10({sweep_label})")
                if j == 0:
                    ax.set_ylabel(metric if not suffix else f"{metric} (% vs initial)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(set(labels))))
        title = f"Per-seed strip ({dataset}, axis={noise_axis})"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / f"seeds_{dataset}_{noise_axis}{suffix}.png", dpi=180)
        plt.close(fig)


def _plot_calibration(df: pd.DataFrame, out_dir: Path):
    for (dataset, noise_axis), part in df.groupby(["dataset", "noise_axis"]):
        plt.figure(figsize=(6, 5))
        for method, p in part.groupby("method"):
            x = pd.to_numeric(p["conv_dr"], errors="coerce").to_numpy()
            y = pd.to_numeric(p["policy_rewards"], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if len(x) == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            bins = np.array_split(np.arange(len(x)), min(20, len(x)))
            xb = np.array([x[b].mean() for b in bins if len(b) > 0])
            yb = np.array([y[b].mean() for b in bins if len(b) > 0])
            plt.plot(xb, yb, marker="o", label=method)
        lo, hi = plt.xlim()
        plt.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        plt.xlabel("estimated (conv_dr)")
        plt.ylabel("true (policy_rewards)")
        plt.title(f"Calibration ({dataset}, axis={noise_axis})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_{dataset}_{noise_axis}.png", dpi=180)
        plt.close()


def _plot_oracle_best_actual(
    trials_long: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str
):
    """Max actual_reward over hyperparam trials per setting; ref lines from logs (initial_reward, ctr)."""
    if trials_long.empty or "actual_reward" not in trials_long.columns:
        return

    t = _eval_rows(trials_long.copy())
    for c in (
        "train_size",
        "seed",
        "run",
        "trial_number",
        "actual_reward",
        "initial_reward",
        "ctr",
    ):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        sweep_col,
    ]
    key = [c for c in key if c in t.columns]
    if len(key) < 5:
        return

    oracle = (
        t.dropna(subset=["actual_reward"])
        .groupby(key, as_index=False)["actual_reward"]
        .max()
        .rename(columns={"actual_reward": "best_actual_reward"})
    )
    if "initial_reward" in t.columns:
        ir_map = t.groupby(key, as_index=False)["initial_reward"].median()
        oracle = oracle.merge(ir_map, on=key, how="left")
    oracle.to_csv(out_dir / "oracle_best_actual_by_setting.csv", index=False)

    gcols = [
        c
        for c in ("dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "method", sweep_col)
        if c in oracle.columns
    ]
    agg = (
        oracle.groupby(gcols, as_index=False)
        .agg(mean_best_actual=("best_actual_reward", "mean"), std=("best_actual_reward", "std"), n=("best_actual_reward", "count"))
        .reset_index(drop=True)
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg.to_csv(out_dir / "oracle_best_actual_aggregated.csv", index=False)

    oracle_pct = oracle.copy()
    oracle_pct["oracle_pct_vs_initial"] = [
        pct_change(float(y), float(ir))
        for y, ir in zip(
            pd.to_numeric(oracle_pct["best_actual_reward"], errors="coerce"),
            pd.to_numeric(oracle_pct.get("initial_reward"), errors="coerce"),
        )
    ]
    agg_pct = (
        oracle_pct.groupby(gcols, as_index=False)
        .agg(
            mean_oracle_pct=("oracle_pct_vs_initial", "mean"),
            std=("oracle_pct_vs_initial", "std"),
            n=("oracle_pct_vs_initial", "count"),
        )
        .reset_index(drop=True)
    )
    agg_pct["se"] = agg_pct["std"] / np.sqrt(agg_pct["n"].clip(lower=1))

    _plot_trainsize_method_panels(
        agg.rename(columns={"mean_best_actual": "mean_y"}),
        t,
        y_col="mean_y",
        y_label="oracle best actual_reward",
        title_prefix="Oracle best actual",
        out_dir=out_dir,
        filename_prefix="oracle_best_actual",
        scale="absolute",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )
    _plot_trainsize_method_panels(
        agg_pct.rename(columns={"mean_oracle_pct": "mean_y"}),
        t,
        y_col="mean_y",
        y_label="oracle best (% vs initial)",
        title_prefix="Oracle best actual",
        out_dir=out_dir,
        filename_prefix="oracle_best_actual",
        scale="pct",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )


def _plot_selected_policy_reward(
    df: pd.DataFrame, out_dir: Path, *, sweep_col: str, sweep_label: str
):
    """Selected policy reward over train_size; same layout as oracle-best plots."""
    df = _apply_slim_reward_fallback(df)
    if df.empty or "policy_rewards" not in df.columns:
        return

    selected = df.copy()
    for c in ("train_size", "seed", "policy_rewards", "initial_reward", "ctr"):
        if c in selected.columns:
            selected[c] = pd.to_numeric(selected[c], errors="coerce")
    selected = _eval_rows(selected).dropna(subset=["policy_rewards"])
    selected = _enrich_reward_pct_columns(selected)

    key = [
        "dataset",
        "noise_mode",
        "noise_axis",
        "noise_level",
        "ctr",
        "seed",
        "method",
        sweep_col,
    ]
    key = [c for c in key if c in selected.columns]
    if len(key) < 5:
        return

    setting = selected.groupby(key, as_index=False).agg(
        selected_policy_reward=("policy_rewards", "mean"),
        initial_reward=("initial_reward", "median"),
        ctr=("ctr", "median"),
    )
    setting.to_csv(out_dir / "selected_policy_reward_by_setting.csv", index=False)

    gcols = [
        c
        for c in ("dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "method", sweep_col)
        if c in setting.columns
    ]
    agg = (
        setting.groupby(gcols, as_index=False)
        .agg(
            mean_selected_policy_reward=("selected_policy_reward", "mean"),
            std=("selected_policy_reward", "std"),
            n=("selected_policy_reward", "count"),
        )
        .reset_index(drop=True)
    )
    agg["se"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg.to_csv(out_dir / "selected_policy_reward_aggregated.csv", index=False)

    setting_pct = setting.copy()
    setting_pct["selected_pct_vs_initial"] = [
        pct_change(float(y), float(ir))
        for y, ir in zip(
            pd.to_numeric(setting_pct["selected_policy_reward"], errors="coerce"),
            pd.to_numeric(setting_pct.get("initial_reward"), errors="coerce"),
        )
    ]
    agg_pct = (
        setting_pct.groupby(gcols, as_index=False)
        .agg(
            mean_selected_pct=("selected_pct_vs_initial", "mean"),
            std=("selected_pct_vs_initial", "std"),
            n=("selected_pct_vs_initial", "count"),
        )
        .reset_index(drop=True)
    )
    agg_pct["se"] = agg_pct["std"] / np.sqrt(agg_pct["n"].clip(lower=1))

    _plot_trainsize_method_panels(
        agg.rename(columns={"mean_selected_policy_reward": "mean_y"}),
        selected,
        y_col="mean_y",
        y_label="selected policy reward",
        title_prefix="Selected policy reward",
        out_dir=out_dir,
        filename_prefix="selected_policy_reward",
        scale="absolute",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )
    _plot_trainsize_method_panels(
        agg_pct.rename(columns={"mean_selected_pct": "mean_y"}),
        selected,
        y_col="mean_y",
        y_label="selected policy (% vs initial)",
        title_prefix="Selected policy reward",
        out_dir=out_dir,
        filename_prefix="selected_policy_reward",
        scale="pct",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )


def _selected_trial_actuals(trials_long: pd.DataFrame) -> pd.DataFrame:
    """True actual_reward of the validation-selected trial per setting."""
    t = _eval_rows(trials_long)
    if t.empty or "actual_reward" not in t.columns:
        return pd.DataFrame()
    key = [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "ctr",
            "seed",
            "method",
            "val_size",
            "train_size",
            "run",
        )
        if c in t.columns
    ]
    if "is_best_in_run" in t.columns and t["is_best_in_run"].any():
        sel = t[t["is_best_in_run"]].copy()
    elif len(key) >= 5:
        sel = t.sort_values("value", ascending=False).groupby(key, as_index=False).first()
    else:
        return pd.DataFrame()
    sel["selected_actual_reward"] = pd.to_numeric(sel["actual_reward"], errors="coerce")
    sel["initial_reward"] = pd.to_numeric(sel["initial_reward"], errors="coerce")
    sel["val_size"] = pd.to_numeric(sel.get("val_size"), errors="coerce")
    return sel


def _plot_selected_actual_vs_val_scatter(trials_long: pd.DataFrame, out_dir: Path):
    """Scatter selected-policy true reward vs validation size; green if beats initial."""
    sub = _selected_trial_actuals(trials_long)
    if sub.empty:
        return
    reward_col = "selected_actual_reward"
    sub = sub.dropna(subset=[reward_col, "val_size", "initial_reward", "method"])
    if sub.empty:
        return
    sub["beat_initial"] = sub[reward_col] > sub["initial_reward"]

    panels = (("opc", "OPC"), ("no_propensity", "no propensity"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (method, title) in zip(axes, panels):
        part = sub[sub["method"] == method]
        lose = part[~part["beat_initial"]]
        win = part[part["beat_initial"]]
        if not lose.empty:
            ax.scatter(
                lose["val_size"],
                lose[reward_col],
                c="C0",
                alpha=0.75,
                s=40,
                edgecolors="none",
                label="≤ initial",
            )
        if not win.empty:
            ax.scatter(
                win["val_size"],
                win[reward_col],
                c="tab:green",
                alpha=0.85,
                s=40,
                edgecolors="none",
                label="> initial",
            )
        for vs, grp in part.groupby("val_size"):
            ir = float(grp["initial_reward"].median())
            if np.isfinite(ir):
                ax.axhline(ir, color="gray", ls=":", lw=0.8, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("validation size")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("selected policy true reward")
    fig.suptitle("Selected policy true reward vs validation size (all datasets)")
    fig.tight_layout()
    out = out_dir / "scatter_selected_actual_vs_val_all_datasets.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    for method, title in panels:
        part = sub[sub["method"] == method]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 5))
        lose = part[~part["beat_initial"]]
        win = part[part["beat_initial"]]
        if not lose.empty:
            ax.scatter(lose["val_size"], lose[reward_col], c="C0", alpha=0.75, s=40, label="≤ initial")
        if not win.empty:
            ax.scatter(win["val_size"], win[reward_col], c="tab:green", alpha=0.85, s=40, label="> initial")
        ax.set_xscale("log")
        ax.set_xlabel("validation size")
        ax.set_ylabel("selected policy true reward")
        ax.set_title(f"{title} (all datasets)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_selected_actual_vs_val_{method}.png", dpi=180)
        plt.close(fig)


def _scatter_selected_vs_val_panels(
    sub: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    suptitle: str,
    out_dir: Path,
    filename: str,
    zero_ref_line: bool = False,
):
    panels = (("opc", "OPC"), ("no_propensity", "no propensity"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (method, title) in zip(axes, panels):
        part = sub[sub["method"] == method]
        lose = part[~part["beat_initial"]]
        win = part[part["beat_initial"]]
        if not lose.empty:
            ax.scatter(
                lose["val_size"],
                lose[y_col],
                c="C0",
                alpha=0.75,
                s=40,
                edgecolors="none",
                label="≤ initial",
            )
        if not win.empty:
            ax.scatter(
                win["val_size"],
                win[y_col],
                c="tab:green",
                alpha=0.85,
                s=40,
                edgecolors="none",
                label="> initial",
            )
        if zero_ref_line:
            ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
        else:
            for _, grp in part.groupby("val_size"):
                ir = float(grp["initial_reward"].median())
                if np.isfinite(ir):
                    ax.axhline(ir, color="gray", ls=":", lw=0.8, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("validation size")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel(y_label)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _prepare_selected_pct_df(trials_long: pd.DataFrame) -> pd.DataFrame:
    sub = _selected_trial_actuals(trials_long)
    if sub.empty or "dataset" not in sub.columns:
        return pd.DataFrame()
    sub = sub.dropna(subset=["selected_actual_reward", "val_size", "initial_reward", "method", "dataset"])
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    sub["pct_improvement"] = [
        pct_change(float(a), float(b))
        for a, b in zip(sub["selected_actual_reward"], sub["initial_reward"])
    ]
    sub = sub[np.isfinite(sub["pct_improvement"])]
    if sub.empty:
        return pd.DataFrame()
    sub["beat_initial"] = sub["pct_improvement"] > 0
    return sub


def _mean_pct_improvement_vs_val_panels(
    sub: pd.DataFrame,
    *,
    suptitle: str,
    out_dir: Path,
    filename: str,
):
    panels = (("opc", "OPC"), ("no_propensity", "no propensity"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (method, title) in zip(axes, panels):
        part = sub[sub["method"] == method]
        agg = (
            part.groupby("val_size")["pct_improvement"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        agg = agg.sort_values("val_size")
        x = pd.to_numeric(agg["val_size"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(agg["mean"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(agg["se"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se)
        x, y, se = x[ok], y[ok], se[ok]
        if len(x):
            ax.errorbar(x, y, yerr=1.96 * se, marker="o", capsize=4, lw=1.5)
        ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("validation size")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("mean % improvement vs initial (across seeds)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _plot_selected_pct_improvement_vs_val_scatter(trials_long: pd.DataFrame, out_dir: Path):
    """Scatter % improvement vs validation size; aggregate + per dataset."""
    sub = _prepare_selected_pct_df(trials_long)
    if sub.empty:
        return

    _scatter_selected_vs_val_panels(
        sub,
        y_col="pct_improvement",
        y_label="% improvement vs initial",
        suptitle="Selected policy % improvement vs validation size (all datasets)",
        out_dir=out_dir,
        filename="scatter_selected_pct_vs_val_all_datasets.png",
        zero_ref_line=True,
    )
    for dataset in sorted(sub["dataset"].dropna().unique()):
        part = sub[sub["dataset"] == dataset]
        if part.empty:
            continue
        _scatter_selected_vs_val_panels(
            part,
            y_col="pct_improvement",
            y_label="% improvement vs initial",
            suptitle=f"Selected policy % improvement vs validation size ({dataset})",
            out_dir=out_dir,
            filename=f"scatter_selected_pct_vs_val_{dataset}.png",
            zero_ref_line=True,
        )


def _plot_mean_selected_pct_improvement_vs_val(trials_long: pd.DataFrame, out_dir: Path):
    """Mean % improvement vs validation size averaged across seeds."""
    sub = _prepare_selected_pct_df(trials_long)
    if sub.empty:
        return

    _mean_pct_improvement_vs_val_panels(
        sub,
        suptitle="Mean selected policy % improvement vs validation size (all datasets, across seeds)",
        out_dir=out_dir,
        filename="mean_selected_pct_vs_val_all_datasets.png",
    )
    for dataset in sorted(sub["dataset"].dropna().unique()):
        part = sub[sub["dataset"] == dataset]
        if part.empty:
            continue
        _mean_pct_improvement_vs_val_panels(
            part,
            suptitle=f"Mean selected policy % improvement vs validation size ({dataset}, across seeds)",
            out_dir=out_dir,
            filename=f"mean_selected_pct_vs_val_{dataset}.png",
        )


def _robustness_ranking(df: pd.DataFrame):
    delta = _pivot_delta(_apply_slim_reward_fallback(df), "policy_rewards")
    if delta.empty:
        return pd.DataFrame()
    return (
        delta.groupby(["dataset", "noise_mode", "noise_axis", "ctr", "noise_level"])["delta"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize full study outputs.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing dataset=*/...")
    parser.add_argument("--summary-csv", default=None, help="Optional explicit summary CSV.")
    parser.add_argument("--out-dir", default=None, help="Optional explicit output directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None and args.summary_csv is None:
        run_dir = Path("artifacts/full_study")
    if run_dir is not None and (
        list(run_dir.rglob("opc_trials_long.csv"))
        or list(run_dir.rglob("trials_long.csv"))
    ):
        from training.rebuild_study_summary import rebuild_run_summaries

        rebuild_run_summaries(run_dir)
    if args.summary_csv is not None:
        summary_csv = Path(args.summary_csv)
    else:
        summary_csv = run_dir / "all_summary_metrics.csv"
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif run_dir is not None:
        out_dir = run_dir / "figures"
    else:
        out_dir = Path("artifacts/full_study/figures")
    _ensure_dir(out_dir)

    df = _load_summary(summary_csv)
    if run_dir is None:
        run_dir = summary_csv.parent

    meta_ctr = _load_meta_ctr(run_dir)
    if not meta_ctr.empty:
        key_cols = ["dataset", "noise_mode", "noise_axis", "noise_level", "seed"]
        existing = [c for c in key_cols if c in df.columns and c in meta_ctr.columns]
        if existing:
            df = df.merge(meta_ctr[existing + ["ctr"]], on=existing, how="left", suffixes=("", "_meta"))
            if "ctr_meta" in df.columns:
                df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce").fillna(
                    pd.to_numeric(df["ctr_meta"], errors="coerce")
                )
                df = df.drop(columns=["ctr_meta"])

    trials_long = _squeeze_duplicate_cols(_load_trials_long_union(run_dir))
    runs_long = _squeeze_duplicate_cols(_load_runs_union(run_dir))
    runs_long = _apply_slim_reward_fallback(runs_long)
    df = _squeeze_duplicate_cols(_backfill_summary_from_runs(df, runs_long))
    df = _apply_slim_reward_fallback(df)
    sweep_col, sweep_label = _resolve_sweep_axis(df)
    print(f"Sweep axis: {sweep_col} ({sweep_label})")

    selection_with_ess = _selection_with_ess(trials_long, runs_long, qs=(0.0, 0.1, 0.3, 0.5, 0.7))
    if not selection_with_ess.empty:
        selection_with_ess.to_csv(out_dir / "selection_with_ess.csv", index=False)

    _plot_curves_per_axis(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_pct_improvement_curves(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_delta_with_ci(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_delta_pct_with_ci(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_opc_pct_improvement_over_nop(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_seed_strip(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_seed_strip(
        _enrich_reward_pct_columns(df),
        out_dir,
        metric="reward_pct_vs_initial",
        suffix="_pct",
        sweep_col=sweep_col,
        sweep_label=sweep_label,
    )
    _plot_calibration(df, out_dir)
    _plot_ess_threshold_curves(
        selection_with_ess, out_dir, sweep_col=sweep_col, sweep_label=sweep_label
    )
    _plot_oracle_best_actual(
        trials_long, out_dir, sweep_col=sweep_col, sweep_label=sweep_label
    )
    _plot_selected_policy_reward(df, out_dir, sweep_col=sweep_col, sweep_label=sweep_label)
    _plot_selected_actual_vs_val_scatter(trials_long, out_dir)
    _plot_selected_pct_improvement_vs_val_scatter(trials_long, out_dir)
    _plot_mean_selected_pct_improvement_vs_val(trials_long, out_dir)

    robust = _robustness_ranking(df)
    if not robust.empty:
        robust.to_csv(out_dir / "robustness_ranking.csv", index=False)

    delta = _pivot_delta(df, "policy_rewards")
    if not delta.empty:
        sig = (
            delta.groupby(
                [c for c in ("dataset", "noise_mode", "noise_axis", "ctr", "noise_level", sweep_col) if c in delta.columns]
            )["delta"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        sig["se"] = sig["std"] / np.sqrt(sig["count"].clip(lower=1))
        sig["ci_low"] = sig["mean"] - 1.96 * sig["se"]
        sig["ci_high"] = sig["mean"] + 1.96 * sig["se"]
        sig["significant_positive"] = sig["ci_low"] > 0
        sig.to_csv(out_dir / "delta_significance.csv", index=False)


if __name__ == "__main__":
    main()
