#!/usr/bin/env python3
"""Build a self-contained hybrid/misspec results PDF (LaTeX → pdflatex).

Reads hybrid_by_cell CSVs and writes:
  hybrid_results_by_setup.tex / .pdf
with math exposition + tables separated by dataset × misspec × γ × method,
including DiD with 95% CIs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/full_study/misspec_experiment")
STUDY = Path("artifacts/full_study")
CELL = ROOT / "hybrid_by_cell"
OUT_TEX = ROOT / "hybrid_results_by_setup.tex"
OUT_PDF = ROOT / "hybrid_results_by_setup.pdf"
INVENTORY = CELL / "study_inventory.json"
TARGET_N_GRID = 40

METHOD_ORDER = [
    "opc",
    "nll",
    "nll_dr",
    "hybrid_seq_f0.25",
    "hybrid_seq_f0.5",
    "hybrid_seq_f0.75",
    "hybrid_seq_adapt",
]
METHOD_LABEL = {
    "opc": "OPC/DR",
    "nll": "NLL",
    "nll_dr": "NLL$\\to$DR (ref)",
    "hybrid_seq_f0.25": "hybrid $f{=}0.25$",
    "hybrid_seq_f0.5": "hybrid $f{=}0.5$",
    "hybrid_seq_f0.75": "hybrid $f{=}0.75$",
    "hybrid_seq_adapt": "hybrid adapt",
}
CONTRAST_ORDER = [
    "OPC-NLL",
    "H25-NLL",
    "H-NLL",
    "H75-NLL",
    "A-NLL",
    "H-OPC",
    "A-OPC",
    "A-H",
]
CONTRAST_LABEL = {
    "OPC-NLL": "$V_{\\mathrm{OPC}}-V_{\\mathrm{NLL}}$",
    "H25-NLL": "$V_{f0.25}-V_{\\mathrm{NLL}}$",
    "H-NLL": "$V_{f0.5}-V_{\\mathrm{NLL}}$",
    "H75-NLL": "$V_{f0.75}-V_{\\mathrm{NLL}}$",
    "A-NLL": "$V_{\\mathrm{adapt}}-V_{\\mathrm{NLL}}$",
    "H-OPC": "$V_{f0.5}-V_{\\mathrm{OPC}}$",
    "A-OPC": "$V_{\\mathrm{adapt}}-V_{\\mathrm{OPC}}$",
    "A-H": "$V_{\\mathrm{adapt}}-V_{f0.5}$",
}
PRIMARY_DID_CONTRASTS = ["OPC-NLL", "H25-NLL", "H-NLL", "H75-NLL", "A-NLL"]
GRID_METHODS = [
    "opc",
    "nll",
    "hybrid_seq_f0.25",
    "hybrid_seq_f0.5",
    "hybrid_seq_f0.75",
    "hybrid_seq_adapt",
]


def dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("n").drop_duplicates(keys, keep="last").reset_index(drop=True)


def fmt(x, signed=False, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def ci_cell(mean, lo, hi, signed=False, nd=2):
    return f"{fmt(mean, signed=signed, nd=nd)} [{fmt(lo, signed=signed, nd=nd)}, {fmt(hi, signed=signed, nd=nd)}]"


def esc(s: str) -> str:
    s = str(s)
    repl = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "τ": "$\\tau$",
        "→": "$\\to$",
        "×": "$\\times$",
        "γ": "$\\gamma$",
        "φ": "$\\phi$",
        "−": "--",
        "–": "--",
        "—": "---",
        "≤": "$\\le$",
        "≥": "$\\ge$",
        "≈": "$\\approx$",
        "·": "$\\cdot$",
        "’": "'",
        "‘": "'",
        "“": "``",
        "”": "''",
        "°": "$^\\circ$",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    # bare * after letters → math star; keep file globs simple
    s = s.replace("*", "$^*$")
    return s


def esc_para(s: str) -> str:
    """Escape free text that may contain unicode math-ish glyphs."""
    return esc(s)


def _ci_arr(x: np.ndarray, alpha: float = 0.05):
    from scipy import stats as st

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(x.mean())
    if n == 1:
        return mean, mean, mean, 1
    se = float(x.std(ddof=1) / np.sqrt(n))
    t = float(st.t.ppf(1 - alpha / 2, n - 1))
    return mean, mean - t * se, mean + t * se, n


def load_tag_method_ctr(tag: str) -> dict[str, np.ndarray]:
    """Seed-level greedy CTR (fraction) per method from summary_metrics.csv."""
    import re

    run = STUDY / f"run_{tag}"
    out: dict[str, dict[int, float]] = {}
    if not run.is_dir():
        return {}
    for sm in run.glob("dataset=*/summary_metrics.csv"):
        seed_m = re.search(r"seed=(\d+)", sm.parent.name)
        if not seed_m:
            continue
        seed = int(seed_m.group(1))
        df = pd.read_csv(sm)
        for _, r in df.iterrows():
            meth = str(r.get("method", ""))
            if meth not in METHOD_ORDER:
                continue
            g = r.get("policy_rewards_greedy")
            if g is None or (isinstance(g, float) and not np.isfinite(g)):
                continue
            out.setdefault(meth, {})[seed] = float(g)
    return {m: np.array(list(seeds.values()), dtype=float) for m, seeds in out.items()}


def load():
    levels = dedupe(pd.read_csv(CELL / "levels_ci.csv"), ["dataset", "misspec", "gamma", "method"])
    contrasts = dedupe(pd.read_csv(CELL / "contrasts_ci.csv"), ["dataset", "misspec", "gamma", "contrast"])
    did = dedupe(pd.read_csv(CELL / "did_ci.csv"), ["dataset", "misspec", "g0", "g1", "contrast"])
    when = json.loads((CELL / "when_to_use.json").read_text())
    ratios = None
    rp = CELL / "reward_ratios_summary.csv"
    if rp.exists():
        ratios = pd.read_csv(rp)
    inv = None
    if INVENTORY.exists():
        inv = json.loads(INVENTORY.read_text())
    return levels, contrasts, did, when, ratios, inv


def enrich_levels_from_inventory(levels: pd.DataFrame, inv: dict | None) -> pd.DataFrame:
    """Ensure every inventory (dataset, misspec, gamma, method) has a row; fill from run dirs if needed."""
    if not inv:
        return levels
    rows = levels.to_dict("records")
    have = {(r["dataset"], r["misspec"], float(r["gamma"]), r["method"]) for r in rows}
    for cell in inv.get("cells", []):
        ds, misspec, g = cell["dataset"], cell["misspec"], float(cell["gamma"])
        tag = cell["tag"]
        target = int(cell.get("target_n", TARGET_N_GRID))
        live = None
        for m in cell.get("methods", GRID_METHODS):
            if m not in GRID_METHODS:
                continue
            key = (ds, misspec, g, m)
            n_file = int(cell.get("method_n", {}).get(m, 0))
            if key in have:
                # annotate later; do not duplicate
                continue
            if n_file == 0:
                rows.append(
                    dict(
                        dataset=ds,
                        misspec=misspec,
                        gamma=g,
                        method=m,
                        n=0,
                        mean_ctr_pct=float("nan"),
                        ci_lo=float("nan"),
                        ci_hi=float("nan"),
                        status="did_not_run",
                        target_n=target,
                        tag=tag,
                    )
                )
                have.add(key)
                continue
            if live is None:
                live = load_tag_method_ctr(tag)
            arr = live.get(m, np.array([]))
            mean, lo, hi, n = _ci_arr(100.0 * arr)
            st = "complete" if n >= target else ("partial" if n > 0 else "did_not_run")
            rows.append(
                dict(
                    dataset=ds,
                    misspec=misspec,
                    gamma=g,
                    method=m,
                    n=n,
                    mean_ctr_pct=mean,
                    ci_lo=lo,
                    ci_hi=hi,
                    status=st,
                    target_n=target,
                    tag=tag,
                )
            )
            have.add(key)
    out = pd.DataFrame(rows)
    if "status" not in out.columns:
        out["status"] = np.nan
    if "target_n" not in out.columns:
        out["target_n"] = TARGET_N_GRID
    # annotate existing analyze rows with status
    inv_n = {}
    for cell in inv.get("cells", []):
        for m, n in cell.get("method_n", {}).items():
            inv_n[(cell["dataset"], cell["misspec"], float(cell["gamma"]), m)] = (
                int(n),
                int(cell.get("target_n", TARGET_N_GRID)),
                cell["tag"],
            )
    statuses = []
    targets = []
    tags = []
    for _, r in out.iterrows():
        key = (r.dataset, r.misspec, float(r.gamma), r.method)
        if isinstance(r.get("status"), str) and r.get("status") in {"did_not_run", "partial", "complete"}:
            statuses.append(r["status"])
            targets.append(int(r.get("target_n", TARGET_N_GRID)))
            tags.append(r.get("tag", ""))
            continue
        if key in inv_n:
            n, tgt, tag = inv_n[key]
            targets.append(tgt)
            tags.append(tag)
            if int(r.n) <= 0 and n <= 0:
                statuses.append("did_not_run")
            elif int(r.n) >= tgt:
                statuses.append("complete")
            else:
                statuses.append("partial")
        else:
            # non-grid locked cell
            targets.append(int(r.n) if int(r.n) >= 20 else 20)
            tags.append("")
            statuses.append("complete" if int(r.n) >= 10 else ("partial" if int(r.n) > 0 else "did_not_run"))
    out["status"] = statuses
    out["target_n"] = targets
    out["tag"] = tags
    return out

def preamble() -> str:
    return r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{array}
\usepackage{multirow}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Hybrid misspec results by setup}
\fancyhead[R]{\small OPC / \today}
\fancyfoot[C]{\thepage}
\renewcommand{\arraystretch}{1.15}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\title{\textbf{Hybrid training under residual misspecification}\\
\large Results by dataset $\times$ misspec family $\times$ $\gamma$ $\times$ method\\
\normalsize Exact-greedy CTR (\%), paired contrasts, and DiD with 95\% $t$ CIs}
\author{OPC project}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
\noindent
This note explains the experiment from first principles and then reports
\textbf{every locked cell with numbers}: absolute CTR levels, method contrasts
(hybrid / OPC vs NLL), and difference-in-differences (DiD) with 95\% confidence
intervals. Tables are stratified by \textbf{dataset}, \textbf{misspec family},
\textbf{$\gamma$}, and \textbf{method}. The full hybrid grid
(\texttt{nlres\_grid\_*}, $n{=}40$, methods including $f\in\{0.25,0.5,0.75\}$ and
adapt) is still running for non-ML catalogs; ML SoftAND / SoftThresh / $R^\star$
grids and earlier multi-dataset $R^\star$ locks are included.
\end{abstract}
\tableofcontents
\newpage
"""


def math_sections() -> str:
    return r"""
\section{Problem setup (for a reader not in the loop)}

\subsection{Offline policy learning with logged bandit feedback}
We have users $x$, items / actions $a\in\mathcal{A}$, and binary rewards
$r\in\{0,1\}$ (click). A logging policy $\pi_0$ collected a dataset
$\mathcal{D}=\{(x_i,a_i,r_i,p_i)\}$ with propensities $p_i=\pi_0(a_i\mid x_i)$.
A learner outputs a new policy $\pi_\theta$. We evaluate with the
\textbf{exact-greedy CTR} of $\pi_\theta$ in a simulated world whose click
probabilities $q(x,a)=\Pr(r{=}1\mid x,a)$ are known to the evaluator
(but not used as training labels for the policy, except through the usual
reward model / IPS machinery of each method).

Reported metric (percentage points of CTR):
\begin{equation}
V(\pi)=100\cdot\mathbb{E}_{x\sim\nu}\!\left[q\!\bigl(x,\arg\max_a \pi(a\mid x)\bigr)\right].
\end{equation}

\subsection{Ground-truth world and the misspecification dial $\gamma$}
The environment logit is an affine-bilinear ``clean'' score plus a residual:
\begin{equation}
\label{eq:logit}
\ell_\gamma(x,a)=\underbrace{b+u_x^\top v_a}_{\text{GT bilinear}}
+\gamma\cdot\alpha\cdot\phi(x,a),\qquad
\mathrm{sd}(\phi)=1,
\end{equation}
and $q=\sigma(\ell_\gamma)$ after a global offset that freezes a target
reference CTR (here $5\%$ under the logger). The scalar $\gamma\ge 0$ is the
\textbf{dial}: $\gamma=0$ is well-specified bilinear truth; large $\gamma$ makes
the residual $\phi$ dominate ranking.

Three residual families $\phi$ are used (unit-SD by construction after
normalization):
\begin{itemize}[leftmargin=1.2em]
\item \textbf{$R^\star$} (even map on a second random projection;
  $s{=}1,m{=}4,\psi{=}\tanh$). Empirically $\mathrm{sd}(\gamma\alpha\phi)/\mathrm{sd}(\text{GT})\approx 2.4\gamma$
  (overshoots the nominal dial).
\item \textbf{SoftAND$^\star$} (multiplicative gate; tracks $\mathrm{sd\_ratio}\approx\gamma$).
\item \textbf{SoftThresh$\tau{=}1.5$} (thresholded residual; also tracks $\mathrm{sd\_ratio}\approx\gamma$).
\end{itemize}
A useful summary of ``how loud is $\phi$'' is the logit variance share
$\mathrm{Var}(\gamma\alpha\phi)/\mathrm{Var}(\ell_\gamma)$. SoftAND / SoftThresh at
$\gamma{=}0.5$ keep $\approx 20\%$ variance on $\phi$; $R^\star$ at the same nominal
$\gamma$ is louder ($\approx 59\%$).

\subsection{Learners}
All primary learners are matched linear policies with Optuna-tuned LR / epochs
($n_{\mathrm{train}}{=}10^5$, $n_{\mathrm{val}}{=}2{\cdot}10^4$, 20 trials, bias cell
\texttt{high}). Methods in the tables:
\begin{itemize}[leftmargin=1.2em]
\item \textbf{NLL}: maximum likelihood on logged clicks (no IPS correction).
\item \textbf{OPC/DR}: doubly robust offline policy learning (direct method + IPS).
\item \textbf{hybrid\_seq\_$f$}: train with NLL for the first fraction $f$ of the
  (selected) epoch budget, then switch to DR for the remainder. Locked default
  $f^\star{=}0.5$; the full grid also reports $f\in\{0.25,0.75\}$.
\item \textbf{hybrid\_seq\_adapt}: start with NLL, switch to DR when validation NLL
  plateaus (patience 2), with switch fraction clipped to $[0.25,0.75]$.
\end{itemize}

\subsection{Estimands: levels, contrasts, DiD}
For a fixed dataset, misspec family, and $\gamma$, write $V_m(\gamma)$ for method $m$.
Paired over seeds $s=1,\dots,n$:
\begin{align}
\Delta_{m,m'}(\gamma)
&= V_m(\gamma)-V_{m'}(\gamma),\\
\mathrm{DiD}_{m,m'}(\gamma_1;\gamma_0)
&= \Delta_{m,m'}(\gamma_1)-\Delta_{m,m'}(\gamma_0).
\end{align}
Primary contrast vs likelihood: $\Delta_{\mathrm{H},\mathrm{NLL}}$ with
$\mathrm{H}=\texttt{hybrid\_seq\_f0.5}$. DiD asks: \emph{does the hybrid advantage
over NLL grow when we turn up misspecification?}
CIs are seed-level Student-$t$:
$\bar x \pm t_{0.975,n-1}\,\mathrm{SE}$, with $\mathrm{SE}=s/\sqrt{n}$.

\subsection{Decision regret (no $\gamma$ oracle)}
Across locked cells, cell regret of a rule that picks hybrid or NLL is the
forgone $|\Delta_{\mathrm{H},\mathrm{NLL}}|$ when the pick disagrees with the sign of
the estimated mean. Empirically, \textbf{always-hybrid} has far smaller summed
regret than always-NLL when $\gamma$ is unknown.

\newpage
\section{Status snapshot}
\begin{itemize}[leftmargin=1.2em]
\item \textbf{ML full hybrid grid} (\texttt{nlres\_grid\_ml\_*}): SoftAND complete
  across $\gamma$; SoftThresh / $R^\star$ largely complete or near-complete ($n$ up to 40;
  some mid-cells still filling $f\in\{0.25,0.75,\mathrm{adapt}\}$).
\item \textbf{Anime} grid: in flight (partial seeds).
\item \textbf{Myket / KuaiRand / KuaiRec} full hybrid grids: queued; earlier Phase-1
  $R^\star$ locks ($n{=}20$) are included below.
\item SoftAND $n{=}40$ catchup on ML (seeds 320--339) is complete for prior tags.
\end{itemize}
"""


def ratios_section(ratios: pd.DataFrame | None) -> str:
    if ratios is None or ratios.empty:
        return ""
    lines = [
        r"\section{Dial calibration (MovieLens reward ratios)}",
        r"Uniform $(x,a)$ pairs; designed $\mathrm{sd}(\phi)=1$ so SoftAND/SoftThresh track "
        r"$\mathrm{sd\_ratio}\approx\gamma$. $R^\star$ overshoots ($\approx 2.4\times$).",
        r"\begin{center}\small\begin{tabular}{llrrrrr}",
        r"\toprule family & $\gamma$ & sd\_ratio & var\_share$_\phi$ & reward\_ratio & CTR lift$_\phi$ \\",
        r"\midrule",
    ]
    # expect columns from summary csv - be flexible
    cols = {c.lower(): c for c in ratios.columns}
    fam_c = cols.get("family") or cols.get("kind") or list(ratios.columns)[0]
    for _, r in ratios.iterrows():
        fam = r.get("family", r.get("kind", "?"))
        g = r.get("gamma", r.get("g", float("nan")))
        sd = r.get("sd_ratio_mean", r.get("sd_ratio", float("nan")))
        vs = r.get("var_share_phi_mean", r.get("var_share_phi", float("nan")))
        rr = r.get(
            "reward_ratio_full_over_gt_mean",
            r.get("reward_ratio_mean", r.get("reward_ratio", float("nan"))),
        )
        lift = r.get("ctr_lift_phi_mean", r.get("ctr_lift_phi", float("nan")))
        if isinstance(vs, float) and np.isfinite(vs):
            vs_s = f"{100 * vs:.1f}\\%" if vs <= 1.5 else f"{vs:.1f}\\%"
        else:
            vs_s = "---"
        # skip insane reward ratios at extreme gamma in table body — still show but compact
        rr_s = fmt(rr, nd=2) if np.isfinite(rr) and rr < 1e3 else (f"{rr:.2e}" if np.isfinite(rr) else "---")
        lines.append(
            f"{esc(str(fam))} & {fmt(float(g), nd=2)} & {fmt(float(sd), nd=3)} & {vs_s} & {rr_s} & {fmt(float(lift), signed=True, nd=2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}\end{center}", ""]
    return "\n".join(lines)


def when_section(when: dict) -> str:
    dr = when.get("decision_rule") or {}
    lines = [
        r"\section{When to use hybrid (aggregate)}",
        esc_para(when.get("recommendation", "")),
        "",
    ]
    if dr:
        lines += [
            r"\paragraph{Best regret rule.} "
            f"\\texttt{{{esc(dr.get('name',''))}}}. "
            f"Summed regret (pt): always-NLL ${fmt(dr.get('sum_regret_pt_always_NLL'), nd=2)}$, "
            f"always-hybrid ${fmt(dr.get('sum_regret_pt_always_hybrid'), nd=2)}$, "
            f"rule ${fmt(dr.get('sum_regret_pt_rule'), nd=2)}$. "
            f"Sign agreement with H$-$NLL: {dr.get('agreement_with_sign_H_NLL')}; "
            f"clear wins caught: {dr.get('sensitivity_clear_hybrid_wins_caught')}/{dr.get('n_clear_hybrid_wins')}.",
            "",
        ]
    return "\n".join(lines)


def method_sort_key(m):
    try:
        return METHOD_ORDER.index(m)
    except ValueError:
        return 100


def contrast_sort_key(c):
    try:
        return CONTRAST_ORDER.index(c)
    except ValueError:
        return 100


def inventory_section(inv: dict | None) -> str:
    if not inv:
        return r"\section{Full-grid study matrix}\emph{Inventory missing; run \_inventory\_grid.py.}"
    lines = [
        r"\section{Full-grid study matrix (complete picture)}",
        (
            r"Planned confirmatory grid: \textbf{"
            f"{inv['n_planned_grid']}"
            r"} cells "
            r"(dataset$\times$misspec$\times\gamma$), each with methods "
            r"OPC/DR, NLL, hybrid $f\in\{0.25,0.5,0.75\}$, adapt; target $n{=}40$ seeds. "
            f"Status now: \\textbf{{{inv['n_complete']}}} complete, "
            f"\\textbf{{{inv['n_partial']}}} partial, "
            f"\\textbf{{{inv['n_did_not_run']}}} did not run. "
            r"Partial cells still contribute seed-level estimates below (with current $n$ and CIs). "
            r"Did-not-run cells appear explicitly as ``did not run''."
        ),
        r"{\footnotesize\begin{longtable}{llrrll}",
        r"\toprule dataset & misspec & $\gamma$ & status & $n$ range (methods) & tag \\",
        r"\midrule\endfirsthead",
        r"\toprule dataset & misspec & $\gamma$ & status & $n$ range (methods) & tag \\",
        r"\midrule\endhead\bottomrule\endfoot",
    ]
    for c in sorted(inv["cells"], key=lambda x: (x["dataset"], x["misspec"], x["gamma"])):
        st = c["status"].replace("_", " ")
        nlo, nhi = c["n_min"], c["n_max"]
        n_s = f"{nlo}--{nhi}" if nlo != nhi else str(nlo)
        if c["status"] == "did_not_run":
            n_s = "0"
        lines.append(
            f"{esc(c['dataset'])} & {esc(c['misspec'])} & {c['gamma']:g} & {esc(st)} & {n_s} & "
            f"\\texttt{{{esc(c['tag'])}}} \\\\"
        )
    lines += [r"\end{longtable}}", r"\newpage"]
    return "\n".join(lines)


def _bf(text: str, yes: bool) -> str:
    return rf"\textbf{{{text}}}" if yes else text


def _best_idx(values: list[float], higher_is_better: bool = True) -> set[int]:
    """Indices of finite best values (ties all marked)."""
    finite = [(i, v) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if not finite:
        return set()
    target = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
    return {i for i, v in finite if v == target}


def emit_setup_blocks(levels, contrasts, did, inv) -> str:
    chunks = [
        r"\section{Results by setup}",
        r"One small table per setup $=(\textbf{dataset},\textbf{misspec},\boldsymbol{\gamma})$. "
        r"\textbf{Bold} marks the best finite entry in that table "
        r"(highest CTR for levels; highest mean for contrasts / DiD). "
        r"Partial runs keep their current $n$; missing methods are ``did not run''.",
        "",
    ]

    keys = set(map(tuple, levels[["dataset", "misspec"]].drop_duplicates().values.tolist()))
    if inv:
        for c in inv["cells"]:
            keys.add((c["dataset"], c["misspec"]))

    for ds, misspec in sorted(keys):
        lv = levels[(levels.dataset == ds) & (levels.misspec == misspec)].copy()
        ct = contrasts[(contrasts.dataset == ds) & (contrasts.misspec == misspec)].copy()
        dd = did[(did.dataset == ds) & (did.misspec == misspec)].copy()

        inv_cells = [
            c for c in (inv or {}).get("cells", []) if c["dataset"] == ds and c["misspec"] == misspec
        ]
        planned_gammas = (
            sorted({float(c["gamma"]) for c in inv_cells}) if inv_cells else sorted(lv.gamma.unique())
        )
        target_n = int(inv_cells[0]["target_n"]) if inv_cells else TARGET_N_GRID
        methods_here = list(GRID_METHODS) if inv_cells else sorted(lv.method.unique(), key=method_sort_key)

        n_complete = sum(1 for c in inv_cells if c["status"] == "complete")
        n_partial = sum(1 for c in inv_cells if c["status"] == "partial")
        n_miss = sum(1 for c in inv_cells if c["status"] == "did_not_run")

        chunks.append(rf"\subsection{{{esc(ds)} / {esc(misspec)}}}")
        if inv_cells:
            chunks.append(
                rf"Grid cells: {len(inv_cells)} "
                rf"(complete {n_complete}, partial {n_partial}, did not run {n_miss}); "
                rf"target $n={target_n}$."
            )
        else:
            chunks.append(r"Legacy / non-grid locked tags.")

        lv_idx = {(float(r.gamma), r.method): r for _, r in lv.iterrows()}
        ct_idx = {(float(r.gamma), r.contrast): r for _, r in ct.iterrows()}
        dd_idx = {(float(r.g0), float(r.g1), r.contrast): r for _, r in dd.iterrows()}

        for g in planned_gammas:
            # cell-level status from inventory if present
            cell_st = next((c["status"] for c in inv_cells if float(c["gamma"]) == float(g)), None)
            chunks.append(rf"\subsubsection{{{esc(ds)} / {esc(misspec)} / $\gamma={g:g}$}}")
            if cell_st:
                chunks.append(rf"\emph{{Cell status: {cell_st.replace('_', ' ')}}}.")

            # ---- Levels table (methods)
            rows = []
            vals = []
            for m in methods_here:
                r = lv_idx.get((float(g), m))
                if r is None or int(getattr(r, "n", 0) or 0) <= 0:
                    rows.append((m, 0, "did not run", None, None, None))
                    vals.append(float("nan"))
                else:
                    st = str(r.get("status", "complete")).replace("_", " ")
                    rows.append((m, int(r.n), st, float(r.mean_ctr_pct), float(r.ci_lo), float(r.ci_hi)))
                    vals.append(float(r.mean_ctr_pct))
            best = _best_idx(vals, higher_is_better=True)

            chunks.append(r"\paragraph{Levels (greedy CTR \%).}")
            chunks.append(r"{\centering\small\begin{tabular}{llrrrr}")
            chunks.append(r"\toprule method & status & $n$ & mean & CI lo & CI hi \\")
            chunks.append(r"\midrule")
            for i, (m, n, st, mean, lo, hi) in enumerate(rows):
                is_best = i in best
                mlab = METHOD_LABEL.get(m, esc(m))
                if mean is None or not np.isfinite(mean):
                    chunks.append(f"{mlab} & did not run & 0 & --- & --- & --- \\\\")
                else:
                    chunks.append(
                        f"{_bf(mlab, is_best)} & {esc(st)} & {_bf(str(n), is_best)} & "
                        f"{_bf(fmt(mean), is_best)} & {_bf(fmt(lo), is_best)} & {_bf(fmt(hi), is_best)} \\\\"
                    )
            chunks.append(r"\bottomrule\end{tabular}\par}")

            # ---- Contrasts table
            crow = []
            cvals = []
            extras = [
                r.contrast
                for _, r in ct[ct.gamma == g].iterrows()
                if r.contrast not in PRIMARY_DID_CONTRASTS
            ]
            contrast_list = list(PRIMARY_DID_CONTRASTS) + sorted(set(extras), key=contrast_sort_key)
            for contrast in contrast_list:
                r = ct_idx.get((float(g), contrast))
                if r is None or int(r.n) <= 0:
                    crow.append((contrast, 0, "did not run", None, None, None))
                    cvals.append(float("nan"))
                else:
                    st = "partial" if int(r.n) < target_n else "complete"
                    crow.append(
                        (contrast, int(r.n), st, float(r.mean_pt), float(r.ci_lo), float(r.ci_hi))
                    )
                    cvals.append(float(r.mean_pt))
            cbest = _best_idx(cvals, higher_is_better=True)

            chunks.append(r"\paragraph{Contrasts (CTR points).}")
            chunks.append(r"{\centering\small\begin{tabular}{llrrrr}")
            chunks.append(r"\toprule contrast & status & $n$ & mean & CI lo & CI hi \\")
            chunks.append(r"\midrule")
            for i, (contrast, n, st, mean, lo, hi) in enumerate(crow):
                lab = CONTRAST_LABEL.get(contrast, esc(contrast))
                is_best = i in cbest
                if mean is None or not np.isfinite(mean):
                    chunks.append(f"{lab} & did not run & 0 & --- & --- & --- \\\\")
                else:
                    chunks.append(
                        f"{_bf(lab, is_best)} & {st} & {_bf(str(n), is_best)} & "
                        f"{_bf(fmt(mean, signed=True), is_best)} & "
                        f"{_bf(fmt(lo, signed=True), is_best)} & "
                        f"{_bf(fmt(hi, signed=True), is_best)} \\\\"
                    )
            chunks.append(r"\bottomrule\end{tabular}\par}")

            # ---- DiD table (only for γ>0)
            if float(g) != 0.0:
                drow = []
                dvals = []
                dextras = [
                    r.contrast
                    for _, r in dd[(dd.g0 == 0.0) & (dd.g1 == g)].iterrows()
                    if r.contrast not in PRIMARY_DID_CONTRASTS
                ]
                dlist = list(PRIMARY_DID_CONTRASTS) + sorted(set(dextras), key=contrast_sort_key)
                for contrast in dlist:
                    r = dd_idx.get((0.0, float(g), contrast))
                    if r is None or int(r.n) <= 0:
                        drow.append((contrast, 0, "did not run", None, None, None))
                        dvals.append(float("nan"))
                    else:
                        st = "partial" if int(r.n) < target_n else "complete"
                        drow.append(
                            (contrast, int(r.n), st, float(r.DiD_pt), float(r.ci_lo), float(r.ci_hi))
                        )
                        dvals.append(float(r.DiD_pt))
                dbest = _best_idx(dvals, higher_is_better=True)

                chunks.append(
                    rf"\paragraph{{DiD vs $\gamma=0$ (CTR points).}} "
                    r"$\mathrm{DiD}=\Delta(\gamma)-\Delta(0)$."
                )
                chunks.append(r"{\centering\small\begin{tabular}{llrrrr}")
                chunks.append(r"\toprule contrast & status & $n$ & DiD & CI lo & CI hi \\")
                chunks.append(r"\midrule")
                for i, (contrast, n, st, mean, lo, hi) in enumerate(drow):
                    lab = CONTRAST_LABEL.get(contrast, esc(contrast))
                    is_best = i in dbest
                    if mean is None or not np.isfinite(mean):
                        chunks.append(f"{lab} & did not run & 0 & --- & --- & --- \\\\")
                    else:
                        chunks.append(
                            f"{_bf(lab, is_best)} & {st} & {_bf(str(n), is_best)} & "
                            f"{_bf(fmt(mean, signed=True), is_best)} & "
                            f"{_bf(fmt(lo, signed=True), is_best)} & "
                            f"{_bf(fmt(hi, signed=True), is_best)} \\\\"
                        )
                chunks.append(r"\bottomrule\end{tabular}\par}")

            chunks.append(r"\bigskip")

        chunks.append(r"\newpage")
    return "\n".join(chunks)
def summary_did_table(did: pd.DataFrame) -> str:
    """Compact highlight: primary DiDs with any n>0 (partial included)."""
    dd = did[did.contrast.isin(PRIMARY_DID_CONTRASTS)].copy()
    dd = dd[dd.n >= 1]
    if dd.empty:
        return ""
    lines = [
        r"\section{Compact DiD index (primary contrasts; includes partial $n$)}",
        r"All estimable DiDs with $n\ge 1$. Smaller $n$ are partial runs --- interpret CIs cautiously.",
        r"{\footnotesize\begin{longtable}{llrrlrrrl}",
        r"\toprule dataset & misspec & $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI & status \\",
        r"\midrule\endfirsthead",
        r"\toprule dataset & misspec & $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI & status \\",
        r"\midrule\endhead\bottomrule\endfoot",
    ]
    dd["_k"] = dd.contrast.map(contrast_sort_key)
    dd = dd.sort_values(["dataset", "misspec", "g1", "_k"])
    for _, r in dd.iterrows():
        st = "partial" if int(r.n) < TARGET_N_GRID else "complete"
        # non-grid often n=20
        if int(r.n) in (10, 20) and int(r.n) >= 10:
            st = "complete" if int(r.n) >= 20 else "partial"
        lines.append(
            f"{esc(r.dataset)} & {esc(r.misspec)} & {r.g0:g} & {r.g1:g} & "
            f"{CONTRAST_LABEL.get(r.contrast, esc(r.contrast))} & {int(r.n)} & "
            f"{fmt(r.DiD_pt, signed=True)} & "
            f"[{fmt(r.ci_lo, signed=True)}, {fmt(r.ci_hi, signed=True)}] & {st} \\\\"
        )
    lines += [r"\end{longtable}}", r"\newpage"]
    return "\n".join(lines)


def main():
    levels, contrasts, did, when, ratios, inv = load()
    levels = enrich_levels_from_inventory(levels, inv)
    tex = []
    tex.append(preamble())
    tex.append(math_sections())
    tex.append(inventory_section(inv))
    tex.append(when_section(when))
    tex.append(ratios_section(ratios))
    tex.append(summary_did_table(did))
    tex.append(emit_setup_blocks(levels, contrasts, did, inv))
    tex.append(
        r"""
\section{Reproducibility}
Tags live under \texttt{artifacts/full\_study/run\_*}.
Full-grid queue: \texttt{logs/full\_hybrid\_grid\_queue.txt}.
Inventory: \texttt{\_inventory\_grid.py} $\to$ \texttt{hybrid\_by\_cell/study\_inventory.json}.
Analysis: \texttt{analyze\_hybrid\_by\_cell.py} $\to$ \texttt{hybrid\_by\_cell/*.csv}.
This PDF: \texttt{build\_hybrid\_results\_pdf.py}.
\end{document}
"""
    )
    OUT_TEX.write_text("\n".join(tex))
    print(f"Wrote {OUT_TEX} ({OUT_TEX.stat().st_size} bytes)")
    tex_only = "--tex-only" in __import__("sys").argv
    if tex_only:
        return
    # compile twice for TOC
    for i in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", OUT_TEX.name],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            log = ROOT / "hybrid_results_by_setup.log"
            print(proc.stdout[-2000:] if proc.stdout else "")
            print(proc.stderr[-1000:] if proc.stderr else "")
            if log.exists():
                print(log.read_text(errors="ignore")[-3000:])
            raise SystemExit(f"pdflatex failed (pass {i+1})")
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
