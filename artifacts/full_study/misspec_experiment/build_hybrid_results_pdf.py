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
CELL = ROOT / "hybrid_by_cell"
OUT_TEX = ROOT / "hybrid_results_by_setup.tex"
OUT_PDF = ROOT / "hybrid_results_by_setup.pdf"


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


def load():
    levels = dedupe(pd.read_csv(CELL / "levels_ci.csv"), ["dataset", "misspec", "gamma", "method"])
    contrasts = dedupe(pd.read_csv(CELL / "contrasts_ci.csv"), ["dataset", "misspec", "gamma", "contrast"])
    did = dedupe(pd.read_csv(CELL / "did_ci.csv"), ["dataset", "misspec", "g0", "g1", "contrast"])
    when = json.loads((CELL / "when_to_use.json").read_text())
    ratios = None
    rp = CELL / "reward_ratios_summary.csv"
    if rp.exists():
        ratios = pd.read_csv(rp)
    return levels, contrasts, did, when, ratios


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


def emit_setup_blocks(levels, contrasts, did) -> str:
    chunks = [r"\section{Results by setup}", r"Each subsection is one (\textbf{dataset}, \textbf{misspec}) pair. "
              r"Within it: absolute levels by $\gamma$ and method, then paired contrasts at each $\gamma$, "
              r"then DiD vs the family's $\gamma{=}0$ baseline (with 95\% CI).", ""]

    keys = sorted(levels.groupby(["dataset", "misspec"]).groups.keys())
    for ds, misspec in keys:
        lv = levels[(levels.dataset == ds) & (levels.misspec == misspec)].copy()
        ct = contrasts[(contrasts.dataset == ds) & (contrasts.misspec == misspec)].copy()
        dd = did[(did.dataset == ds) & (did.misspec == misspec)].copy()
        if lv.empty:
            continue
        chunks.append(rf"\subsection{{{esc(ds)} / {esc(misspec)}}}")
        gammas = sorted(lv.gamma.unique())
        chunks.append(
            rf"Gammas present: ${', '.join(f'{g:g}' for g in gammas)}$. "
            rf"Seeds $n$ vary by method/cell (see tables)."
        )

        # ---- Levels longtable
        chunks.append(r"\subsubsection*{Absolute greedy CTR (\%) by $\gamma$ and method}")
        chunks.append(r"{\footnotesize\begin{longtable}{rrlrrr}")
        chunks.append(
            r"\toprule $\gamma$ & $n$ & method & mean CTR\% & 95\% CI lo & 95\% CI hi \\"
        )
        chunks.append(r"\midrule\endfirsthead")
        cont = esc(f"{ds} / {misspec} levels")
        chunks.append(rf"\toprule\multicolumn{{6}}{{l}}{{\emph{{continued: {cont}}}}} \\")
        chunks.append(
            r"\toprule $\gamma$ & $n$ & method & mean CTR\% & 95\% CI lo & 95\% CI hi \\"
        )
        chunks.append(r"\midrule\endhead\bottomrule\endfoot")
        for g in gammas:
            sub = lv[lv.gamma == g].copy()
            sub["_k"] = sub.method.map(method_sort_key)
            sub = sub.sort_values(["_k", "method"])
            for _, r in sub.iterrows():
                chunks.append(
                    f"{r.gamma:g} & {int(r.n)} & {METHOD_LABEL.get(r.method, esc(r.method))} & "
                    f"{fmt(r.mean_ctr_pct)} & {fmt(r.ci_lo)} & {fmt(r.ci_hi)} \\\\"
                )
        chunks.append(r"\end{longtable}}")

        # ---- Contrasts by gamma
        if not ct.empty:
            chunks.append(r"\subsubsection*{Paired contrasts (CTR points) by $\gamma$}")
            chunks.append(r"{\footnotesize\begin{longtable}{rlrrrr}")
            chunks.append(
                r"\toprule $\gamma$ & contrast & $n$ & mean & 95\% CI lo & 95\% CI hi \\"
            )
            chunks.append(r"\midrule\endfirsthead")
            cont = esc(f"{ds} / {misspec} contrasts")
            chunks.append(rf"\toprule\multicolumn{{6}}{{l}}{{\emph{{continued: {cont}}}}} \\")
            chunks.append(
                r"\toprule $\gamma$ & contrast & $n$ & mean & 95\% CI lo & 95\% CI hi \\"
            )
            chunks.append(r"\midrule\endhead\bottomrule\endfoot")
            for g in sorted(ct.gamma.unique()):
                sub = ct[ct.gamma == g].copy()
                sub["_k"] = sub.contrast.map(contrast_sort_key)
                sub = sub.sort_values("_k")
                for _, r in sub.iterrows():
                    chunks.append(
                        f"{r.gamma:g} & {CONTRAST_LABEL.get(r.contrast, esc(r.contrast))} & {int(r.n)} & "
                        f"{fmt(r.mean_pt, signed=True)} & {fmt(r.ci_lo, signed=True)} & {fmt(r.ci_hi, signed=True)} \\\\"
                    )
            chunks.append(r"\end{longtable}}")

        # ---- DiD
        if not dd.empty:
            chunks.append(
                r"\subsubsection*{DiD vs $\gamma{=}0$ (CTR points) with 95\% CI}"
            )
            chunks.append(
                r"Definition: $\mathrm{DiD}=\Delta(\gamma_1)-\Delta(\gamma_0)$ for the named contrast, "
                r"paired over seeds present at both $\gamma$."
            )
            chunks.append(r"{\footnotesize\begin{longtable}{rrlrrrr}")
            chunks.append(
                r"\toprule $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI lo & 95\% CI hi \\"
            )
            chunks.append(r"\midrule\endfirsthead")
            cont = esc(f"{ds} / {misspec} DiD")
            chunks.append(rf"\toprule\multicolumn{{7}}{{l}}{{\emph{{continued: {cont}}}}} \\")
            chunks.append(
                r"\toprule $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI lo & 95\% CI hi \\"
            )
            chunks.append(r"\midrule\endhead\bottomrule\endfoot")
            dd = dd.copy()
            dd["_k"] = dd.contrast.map(contrast_sort_key)
            dd = dd.sort_values(["g1", "_k"])
            for _, r in dd.iterrows():
                chunks.append(
                    f"{r.g0:g} & {r.g1:g} & {CONTRAST_LABEL.get(r.contrast, esc(r.contrast))} & {int(r.n)} & "
                    f"{fmt(r.DiD_pt, signed=True)} & {fmt(r.ci_lo, signed=True)} & {fmt(r.ci_hi, signed=True)} \\\\"
                )
            chunks.append(r"\end{longtable}}")

        chunks.append(r"\newpage")
    return "\n".join(chunks)


def summary_did_table(did: pd.DataFrame) -> str:
    """Compact highlight: primary DiDs H-NLL / OPC-NLL / A-NLL with n>=10."""
    dd = did[did.contrast.isin(["H-NLL", "OPC-NLL", "A-NLL", "H25-NLL", "H75-NLL"])].copy()
    dd = dd[dd.n >= 10]
    if dd.empty:
        return ""
    lines = [
        r"\section{Compact DiD index (primary contrasts, $n\ge 10$)}",
        r"{\footnotesize\begin{longtable}{llrrlrrr}",
        r"\toprule dataset & misspec & $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI \\",
        r"\midrule\endfirsthead",
        r"\toprule dataset & misspec & $\gamma_0$ & $\gamma_1$ & contrast & $n$ & DiD & 95\% CI \\",
        r"\midrule\endhead\bottomrule\endfoot",
    ]
    dd["_k"] = dd.contrast.map(contrast_sort_key)
    dd = dd.sort_values(["dataset", "misspec", "g1", "_k"])
    for _, r in dd.iterrows():
        lines.append(
            f"{esc(r.dataset)} & {esc(r.misspec)} & {r.g0:g} & {r.g1:g} & "
            f"{CONTRAST_LABEL.get(r.contrast, esc(r.contrast))} & {int(r.n)} & "
            f"{fmt(r.DiD_pt, signed=True)} & "
            f"[{fmt(r.ci_lo, signed=True)}, {fmt(r.ci_hi, signed=True)}] \\\\"
        )
    lines += [r"\end{longtable}}", r"\newpage"]
    return "\n".join(lines)


def main():
    levels, contrasts, did, when, ratios = load()
    tex = []
    tex.append(preamble())
    tex.append(math_sections())
    tex.append(when_section(when))
    tex.append(ratios_section(ratios))
    tex.append(summary_did_table(did))
    tex.append(emit_setup_blocks(levels, contrasts, did))
    tex.append(
        r"""
\section{Reproducibility}
Tags live under \texttt{artifacts/full\_study/run\_*}.
Analysis: \texttt{artifacts/full\_study/misspec\_experiment/analyze\_hybrid\_by\_cell.py}
$\to$ \texttt{hybrid\_by\_cell/*.csv}. This PDF:
\texttt{build\_hybrid\_results\_pdf.py}.
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
