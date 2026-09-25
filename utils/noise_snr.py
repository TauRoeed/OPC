"""SNR and related metrics for clean vs biased embeddings."""

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


def embedding_noise_metrics(x_gt: np.ndarray, x_noisy: np.ndarray) -> dict[str, float]:
    return {
        "snr_db": snr_db(x_gt, x_noisy),
        "cosine_retention": cosine_retention(x_gt, x_noisy),
        "rmse": rmse(x_gt, x_noisy),
    }


def dataset_snr_report(dataset: dict[str, Any]) -> dict[str, Any]:
    """Vector-level distance between the clean (emb_*) and biased (our_*) vectors.

    The score-level measure the bias levels are calibrated on is
    ``dataset['world']['signal_kept']`` (see utils.representation_bias). Taste parts only: the
    popularity column (if any) is not part of the representation bias.
    """
    d = int(dataset["emb_dim"]) if dataset.get("pop_column") else None
    action = embedding_noise_metrics(dataset["emb_a"][:, :d], dataset["our_a"][:, :d])
    context = embedding_noise_metrics(dataset["emb_x"][:, :d], dataset["our_x"][:, :d])
    return {
        "action": action,
        "context": context,
        "snr_db_mean": float(0.5 * (action["snr_db"] + context["snr_db"])),
        "cosine_mean": float(0.5 * (action["cosine_retention"] + context["cosine_retention"])),
    }
