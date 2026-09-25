"""Importance-weight transforms shared by training losses, trial selection and post-hoc estimates.

A spec is ``none`` (raw weights w = pi_e / pi_b), ``clip:M`` (``min(w, M)``) or ``shrink:lam``
(Su et al. 2020: ``lam * w / (w**2 + lam)``, at most sqrt(lam) / 2, and falling back toward 0
past w = sqrt(lam)).
"""

from __future__ import annotations

import math

import numpy as np

WEIGHT_MODES = ("none", "clip", "shrink")


def parse_weight_spec(spec) -> tuple[str, float]:
    """``'none' | 'clip:M' | 'shrink:lam'`` (or a (mode, param) pair) -> (mode, param)."""
    if isinstance(spec, (tuple, list)):
        mode, param = str(spec[0]).lower(), float(spec[1])
    else:
        text = str(spec).strip().lower()
        mode, _, value = text.partition(":")
        if mode in ("none", "raw"):
            return "none", math.inf
        if not value:
            raise ValueError(f"weight spec {spec!r}: use none, clip:M or shrink:lambda")
        param = float(value)
    if mode in ("none", "raw"):
        return "none", math.inf
    if mode not in WEIGHT_MODES:
        raise ValueError(f"weight spec {spec!r}: mode must be one of {WEIGHT_MODES}")
    if not param > 0:
        raise ValueError(f"weight spec {spec!r}: the parameter must be > 0")
    if mode == "clip" and math.isinf(param):
        return "none", math.inf
    if mode == "shrink" and math.isinf(param):
        return "none", math.inf  # lam -> inf leaves every weight unchanged
    return mode, param


def weight_spec_label(spec) -> str:
    """Canonical text of a spec: 'none', 'clip:100', 'shrink:10000'."""
    mode, param = parse_weight_spec(spec)
    return "none" if mode == "none" else f"{mode}:{param:g}"


def transform_weights(iw, spec) -> np.ndarray:
    """Apply a weight spec to raw importance weights (numpy)."""
    mode, param = parse_weight_spec(spec)
    iw = np.asarray(iw)
    if mode == "clip":
        return np.minimum(iw, param)
    if mode == "shrink":
        return (param * iw) / (param + iw * iw)
    return iw


def effective_sample_size(w) -> float:
    """(sum w)^2 / sum w^2, in the weights' own dtype (the trainer's float32 arithmetic)."""
    w = np.asarray(w)
    return float((w.sum() ** 2) / ((w**2).sum() + 1e-12))
