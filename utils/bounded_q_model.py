"""Reward model with controlled L-infinity error via convex mix with a bad predictor."""

from __future__ import annotations

import numpy as np


class ConstantRewardModel:
    """Predict constant q for every (x,a) — duck-types AnalyticRewardModel surface."""

    def __init__(self, value: float, *, n_actions: int, kind: str = "constant"):
        self.value = float(np.clip(value, 0.0, 1.0))
        self.n_actions = int(n_actions)
        self.len_list = 1
        self.kind = str(kind)

    def predict_pairs(
        self, context: np.ndarray, action: np.ndarray, pos: int = 0
    ) -> np.ndarray:
        _ = pos
        context = np.asarray(context)
        n = context.shape[0] if context.ndim > 1 else 1
        return np.full(n, self.value, dtype=np.float32)

    def predict_user_action_block(
        self, context: np.ndarray, action_start: int, action_end: int
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        b = context.shape[0]
        a = int(action_end) - int(action_start)
        return np.full((b, a, 1), self.value, dtype=np.float32)

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.predict_user_action_block(context, 0, self.n_actions)


class BoundedErrorRewardModel:
    """
    q_hat = clip((1-eps)*q_good + eps*q_bad, 0, 1).

    If ||q_good - q_bad||_inf <= 1 then ||q_hat - q_good||_inf <= eps.
    """

    def __init__(self, q_good, q_bad, *, eps: float, n_actions: int):
        self.q_good = q_good
        self.q_bad = q_bad
        self.eps = float(np.clip(eps, 0.0, 1.0))
        self.n_actions = int(n_actions)
        self.len_list = 1
        self.kind = f"bounded_eps={self.eps:g}"

    def predict_pairs(
        self, context: np.ndarray, action: np.ndarray, pos: int = 0
    ) -> np.ndarray:
        g = self.q_good.predict_pairs(context, action, pos=pos)
        b = self.q_bad.predict_pairs(context, action, pos=pos)
        return np.clip((1.0 - self.eps) * g + self.eps * b, 0.0, 1.0).astype(
            np.float32
        )

    def predict_user_action_block(
        self, context: np.ndarray, action_start: int, action_end: int
    ) -> np.ndarray:
        g = self.q_good.predict_user_action_block(context, action_start, action_end)
        b = self.q_bad.predict_user_action_block(context, action_start, action_end)
        return np.clip((1.0 - self.eps) * g + self.eps * b, 0.0, 1.0).astype(
            np.float32
        )

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.predict_user_action_block(context, 0, self.n_actions)


def max_pairwise_q_error(q_a, q_b, user_context: np.ndarray, n_actions: int) -> float:
    """Empirical max |q_a - q_b| on a random user subset x all actions."""
    ctx = np.asarray(user_context, dtype=np.float32)
    if ctx.ndim != 2:
        raise ValueError("user_context must be 2D")
    q1 = q_a.predict_user_action_block(ctx, 0, n_actions)
    q2 = q_b.predict_user_action_block(ctx, 0, n_actions)
    return float(np.max(np.abs(q1 - q2)))
