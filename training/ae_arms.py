"""The arms of docs/opc_adaptive_estimator_study.md §5: the rank-4 adapter's OPC arm with family A's weights, family B's
support terms, family C's CRM terms and family D's reward models. One registry for the
runner (``run_full_study.SHARED_OBJECTIVE_ARMS``), the run state and the analysis.

- A1: harmonic λ fixed; A2 / A3: the adaptive λ (``opc_extension.weights``).
- B (support) and C (CRM) keep raw DR, the unbiased estimator, and act on the policy (B) or the estimate's variance
  (C); C2 (harmonic CRM) and C3 (delta-CRM) are gated (§5.3).
"""
from __future__ import annotations

A1_LAMBDAS = (0.005, 0.01, 0.03, 0.2)
S2_BETAS = (0.0003, 0.001, 0.003, 0.01)
S2_CAPS = (5, 10, 20, 50)
KL_BETAS = (0.001, 0.003)
CRM_GAMMAS = (0.5, 1, 2)


def ae_arms() -> dict:
    base = {"shared_objective": "opc", "shared_adapter": "lowrank"}
    arms = {f"shared_lr_opc_h{lam:g}": {**base, "train_weights": f"harmonic:{lam:g}"} for lam in A1_LAMBDAS}
    arms["shared_lr_opc_a2"] = {**base, "opc_extension": {"weights": "a2"}}
    arms["shared_lr_opc_a3"] = {**base, "opc_extension": {"weights": "a3"}}
    raw = {**base, "train_weights": "none"}
    arms.update({f"shared_lr_opc_s2b{b:g}": {**raw, "opc_extension": {"support": ["s2", b]}} for b in S2_BETAS})
    arms.update({f"shared_lr_opc_s2c{c:g}": {**raw, "opc_extension": {"support": ["s2_hard", c]}} for c in S2_CAPS})
    arms.update({f"shared_lr_opc_klb{b:g}": {**raw, "opc_extension": {"support": ["kl", b]}} for b in KL_BETAS})
    arms.update({f"shared_lr_opc_crm{g:g}": {**raw, "opc_extension": {"crm": [g, "abs"]}} for g in CRM_GAMMAS})
    arms.update({f"shared_lr_opc_hcrm{g:g}": {**base, "opc_extension": {"crm": [g, "abs"]}} for g in CRM_GAMMAS})
    arms.update({f"shared_lr_opc_dcrm{g:g}": {**raw, "opc_extension": {"crm": [g, "delta"]}} for g in CRM_GAMMAS})
    # D: harmonic λ = 0.1 OPC with a richer reward model in training and native selection (§5.4)
    arms.update({f"shared_lr_opc_{name}": {**base, "reward_model_kind": kind}
                 for name, kind in (("d1", "lowrank"), ("d2", "hetero"))})
    return arms


AE_ARMS = ae_arms()
