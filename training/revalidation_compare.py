"""Old (buggy-simulator) vs new (fixed-simulator) results, paired by world (docs/simulator_fix_opc_revalidation_20261004.md,
Phase 4).

The old and the new Stage 2 / Stage 3 runs use the same worlds (dataset × bias × seed, the same truth and logger)
and the same train sizes; only the logged rows differ (and, for OPC, whatever the re-tuning changed). Every
comparison here is paired by world: the difference new − old is formed per world and summarized with a 95%
t-interval over the worlds of a cell.

Classification of a finding (a mean contrast over the worlds of a cell, e.g. OPC − DM-only at combined high, 25k),
from its old and new 95% intervals and the paired interval of the change:
  - unchanged: the same sign and the same significance (interval excluding 0 or not), and the change's interval
    includes 0;
  - same direction, different magnitude: significant before and after with the same sign, and the change's
    interval excludes 0;
  - weakened: significant before; after, the same sign but smaller, with the interval including 0;
  - unchanged size, less precise: significant before; after, the same sign and at least as large, but the interval
    includes 0 (the effect did not shrink; the new evidence is noisier);
  - unsupported: significant before; after, the opposite sign but the interval includes 0;
  - reversed: significant after with the opposite sign of the old mean (significant or not before);
  - new: not significant before, significant after with the old mean's sign (or the old mean 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["dataset", "bias", "seed", "train_size"]


def mean_ci(x) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def _significant(lo, hi) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def classify(old, new, change) -> str:
    """``old``, ``new``, ``change``: (mean, lo, hi) triples."""
    om, olo, ohi = old
    nm, nlo, nhi = new
    _, clo, chi = change
    old_sig, new_sig = _significant(olo, ohi), _significant(nlo, nhi)
    same_sign = np.sign(om) == np.sign(nm)
    if new_sig and not same_sign and np.sign(om) != 0:
        return "reversed"
    if old_sig and not new_sig:
        if not same_sign:
            return "unsupported"
        return "unchanged size, less precise" if abs(nm) >= abs(om) else "weakened"
    if not old_sig and new_sig:
        return "new"
    if old_sig and new_sig and _significant(clo, chi):
        return "same direction, different magnitude"
    return "unchanged"


def paired_contrast(rows: pd.DataFrame, a: str, b: str | None, value: str) -> pd.DataFrame:
    """Per world: ``value`` of arm a minus arm b (b None: a alone), in CTR points when ``value`` is a CTR."""
    pa = rows[rows["method"] == a].set_index(KEYS)[value]
    if b is None:
        return pa.rename("x").reset_index()
    pb = rows[rows["method"] == b].set_index(KEYS)[value]
    return (pa - pb).rename("x").dropna().reset_index()


def old_new_table(old_rows: pd.DataFrame, new_rows: pd.DataFrame, contrasts: dict, scale: float = 100.0,
                  by=("bias", "train_size")) -> pd.DataFrame:
    """For each contrast {label: (arm a, arm b or None, value column)}: per cell of ``by``, the old and the new mean
    with 95% intervals over the worlds, the paired change (new − old) with its interval, and the classification.
    ``scale`` converts CTRs to points (use 1 for fractions)."""
    out = []
    for label, (a, b, value) in contrasts.items():
        sc = scale if not value.startswith("fraction") else 1.0
        o = paired_contrast(old_rows, a, b, value).rename(columns={"x": "old"})
        n = paired_contrast(new_rows, a, b, value).rename(columns={"x": "new"})
        j = o.merge(n, on=KEYS, how="inner")
        for cell, g in j.groupby(list(by)):
            old = mean_ci(sc * g["old"])
            new = mean_ci(sc * g["new"])
            ch = mean_ci(sc * (g["new"] - g["old"]))
            out.append(dict(contrast=label, **dict(zip(by, cell if isinstance(cell, tuple) else (cell,))),
                            worlds=old[3], old=old[0], old_lo=old[1], old_hi=old[2], new=new[0], new_lo=new[1],
                            new_hi=new[2], change=ch[0], change_lo=ch[1], change_hi=ch[2],
                            verdict=classify(old[:3], new[:3], ch[:3])))
    return pd.DataFrame(out)
