"""SNR and related metrics for GT vs noised embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np


def rmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    diff = np.asarray(x_hat, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def snr_db(x: np.ndarray, x_hat: np.ndarray, *, eps: float = 1e-12) -> float:
    """Frobenius SNR in dB: 10 log10(||X||^2 / ||X_hat - X||^2)."""
    x64 = np.asarray(x, dtype=np.float64)
    diff = np.asarray(x_hat, dtype=np.float64) - x64
    signal = float(np.sum(x64 * x64))
    noise = float(np.sum(diff * diff))
    return float(10.0 * np.log10((signal + eps) / (noise + eps)))


def cosine_retention(x: np.ndarray, x_hat: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean row-wise cosine similarity between GT and noised embeddings."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(x_hat, dtype=np.float64)
    if x64.shape != y64.shape:
        raise ValueError(f"shape mismatch: {x64.shape} vs {y64.shape}")
    num = np.sum(x64 * y64, axis=1)
    den = np.linalg.norm(x64, axis=1) * np.linalg.norm(y64, axis=1)
    return float(np.mean(num / (den + eps)))


def signal_frac(epsilons: list[float] | tuple[float, ...]) -> float:
    return float(1.0 - float(np.sum(epsilons)))


def embedding_noise_metrics(
    x_gt: np.ndarray,
    x_noisy: np.ndarray,
    *,
    epsilons: list[float] | tuple[float, ...] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "snr_db": snr_db(x_gt, x_noisy),
        "cosine_retention": cosine_retention(x_gt, x_noisy),
        "rmse": rmse(x_gt, x_noisy),
    }
    if epsilons is not None:
        out["signal_frac"] = signal_frac(epsilons)
    return out


def dataset_snr_report(
    dataset: dict[str, Any],
    *,
    eps1: float,
    eps2: float,
    eps_meta: float,
) -> dict[str, Any]:
    """Build SNR summary from a generated bandit dataset dict."""
    emb_a = dataset["emb_a"]
    emb_x = dataset["emb_x"]
    our_a = dataset["our_a"]
    our_x = dataset["our_x"]
    item_eps = [eps1, eps2]
    user_eps = [eps1, eps2]
    if float(eps_meta) > 0.0:
        # Match generate_dataset: meta may apply to items, users, or both.
        # Use nominal eps_meta in signal_frac when any meta noise is present.
        item_eps = item_eps + [eps_meta]
        user_eps = user_eps + [eps_meta]

    action = embedding_noise_metrics(emb_a, our_a, epsilons=item_eps)
    context = embedding_noise_metrics(emb_x, our_x, epsilons=user_eps)
    return {
        "action": action,
        "context": context,
        "eps1": float(eps1),
        "eps2": float(eps2),
        "eps_meta": float(eps_meta),
        "snr_db_mean": float(0.5 * (action["snr_db"] + context["snr_db"])),
        "cosine_mean": float(
            0.5 * (action["cosine_retention"] + context["cosine_retention"])
        ),
    }


def isolate_component_metrics(
    x_gt: np.ndarray,
    noise_vecs: list[np.ndarray],
    epsilons: list[float],
    *,
    component_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Mix one noise component at a time and report metrics vs GT."""
    from utils.simulation_utils import mix_ground_truth_with_noises

    if len(noise_vecs) != len(epsilons):
        raise ValueError("noise_vecs and epsilons must have same length")
    names = component_names or [f"comp_{i}" for i in range(len(noise_vecs))]
    if len(names) != len(noise_vecs):
        raise ValueError("component_names length mismatch")

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        eps_iso = [0.0] * len(epsilons)
        eps_iso[i] = float(epsilons[i])
        if eps_iso[i] <= 0.0:
            out[name] = {
                "snr_db": float("inf"),
                "cosine_retention": 1.0,
                "rmse": 0.0,
                "signal_frac": 1.0,
                "eps": 0.0,
            }
            continue
        mixed = mix_ground_truth_with_noises(x_gt, noise_vecs, eps_iso)
        metrics = embedding_noise_metrics(x_gt, mixed, epsilons=eps_iso)
        metrics["eps"] = float(epsilons[i])
        out[name] = metrics
    return out
