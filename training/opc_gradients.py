"""OPC's gradient against the population value gradient (docs/opc_gradient_regime_study.md §2-§4).

- ``exact_value``: the population value V(θ) = Σ_u prior(u) Σ_j π_θ(j|u) q(u, j) of the shared policy model
  (``SharedCorrectionModel``, OPC's class) and its gradient g*(θ), by autograd over every user and item with the
  simulator's q. A simulator oracle: never used for a deployable result.
- ``fit_value_path`` / ``fit_likelihood_population`` / ``policy_states``: the population optima in this
  parameterization, fit from the source (the logger) as ``training/class_oracles.py`` fits its oracles, and the four
  policy states of §3.
- ``estimator_gradients``: the logged-data estimators G1-G5 of §2, each the gradient of the training loss's own
  per-row surrogate (``models.custom_losses.dr_sndr_surrogate`` with the direct gradient and no normalizer, what
  ``DRPolicyLoss`` averages), with q̂ = 0, the true q or the cross-fitted q̂, and raw or harmonic:0.1 weights; plus
  the exact conditional bias of G5 given its q̂ and the minibatch gradients.
- ``logged_rows`` / ``crossfit_qhat``: one logged dataset of the world and its cross-fitted reward model, drawn and
  fit exactly as the study draws and fits them.

Gradients are ascent directions of the value estimates (the negated training-loss gradient), float64.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from models.custom_losses import dr_sndr_surrogate
from models.shared_objectives import SharedCorrectionModel
from utils.importance_weights import parse_weight_spec
from utils.seeding import derive_seed, seed_everything

# estimator -> (weight spec, reward-model source)
ESTIMATORS = {"G1": ("none", "zero"), "G2": ("none", "oracle"), "G3": ("none", "qhat"),
              "G4": ("harmonic:0.1", "oracle"), "G5": ("harmonic:0.1", "qhat")}
ESTIMATOR_NAMES = tuple(ESTIMATORS)
STATES = ("source", "likelihood", "mid", "mid_greedy", "value")
VALUE_LRS = (0.003, 0.01, 0.03)  # §3, as training/class_oracles.py
FIT_STEPS = 3000
FIT_USERS = 20_000
BATCH_USERS = 2048
CHECKPOINT_EVERY = 25
CHECKPOINT_DENSE = 200  # every step up to here: the softmax value moves fastest at the start (the scale learns first)
HARMONIC = 0.1


def _device(device=None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _prior(dataset: dict) -> np.ndarray:
    from utils.simulation_utils import _normalized_prior

    return _normalized_prior(dataset)


# ------------------------------------------------------------------------------------------------------- the model
def build_model(dataset: dict, *, mode: str = "policy", device=None, dtype=torch.float32) -> SharedCorrectionModel:
    """OPC's policy class at θ = 0 (the logger): the shared model on the frozen biased vectors, the learned scale
    starting at 1, the logger's temperature. ``mode='click'`` adds the click intercept (the likelihood's head)."""
    from training.trainer_trials import _policy_temperature

    t = lambda z: torch.as_tensor(np.asarray(z, dtype=np.float32))
    k = int(np.asarray(dataset["our_x"]).shape[1])
    model = SharedCorrectionModel(int(dataset["n_users"]), int(dataset["n_actions"]), k,
                                  initial_user_embeddings=t(dataset["our_x"]),
                                  initial_actions_embeddings=t(dataset["our_a"]),
                                  temperature=_policy_temperature(dataset), mode=mode)
    for emb in (model.user_embeddings, model.actions_embeddings):
        emb.weight.requires_grad_(False)
    model = model.to(_device(device))
    return model.double() if dtype == torch.float64 else model


def policy_params(model) -> list:
    """θ = (D_u, b_u, D_a, b_a, θ_s) in a fixed order: the policy's parameters (the click intercept excluded)."""
    u, a = model.user_transform, model.action_transform
    return [u.delta, u.bias, a.delta, a.bias, model.log_logit_scale]


def get_theta(model) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1).double().cpu() for p in policy_params(model)]).numpy()


def set_theta(model, theta) -> None:
    theta = np.asarray(theta, dtype=np.float64)
    pos = 0
    with torch.no_grad():
        for p in policy_params(model):
            n = p.numel()
            p.copy_(torch.as_tensor(theta[pos:pos + n].reshape(p.shape), dtype=p.dtype, device=p.device))
            pos += n
    if pos != theta.size:
        raise ValueError(f"θ has {theta.size} entries, the model {pos}")


def _flat_grad(grads) -> np.ndarray:
    return torch.cat([g.reshape(-1).double() for g in grads]).cpu().numpy()


def _policy(model, users) -> torch.Tensor:
    out = model(users)
    return out[:, :, 0] if out.dim() == 3 else out


# --------------------------------------------------------------------------------------------- the simulator's q, π0
class WorldTensors:
    """The world's fixed tensors on one device: the clean items (true q), the logger's vectors and temperature."""

    def __init__(self, dataset: dict, device=None, dtype=torch.float32):
        from training.trainer_trials import _logging_uniform_mix, _policy_temperature

        self.device = _device(device)
        self.dataset = dataset
        env = dataset["env"]
        self.clean_x = torch.as_tensor(np.asarray(env.emb_x), dtype=dtype, device=self.device)
        self.clean_a = torch.as_tensor(np.asarray(env.emb_a), dtype=dtype, device=self.device)
        self.scale, self.offset = float(env.scale), float(env.offset)
        self.our_x = torch.as_tensor(np.asarray(dataset["our_x"]), dtype=dtype, device=self.device)
        self.our_a = torch.as_tensor(np.asarray(dataset["our_a"]), dtype=dtype, device=self.device)
        self.temperature = float(_policy_temperature(dataset))
        self.mix = float(_logging_uniform_mix(dataset))
        self.n_items = int(dataset["n_actions"])

    def q(self, users: torch.Tensor) -> torch.Tensor:
        """q(u, ·) for every item: sigmoid(scale · x_u·a + offset) on the clean vectors (``oracle_repair.true_q_rows``)."""
        return torch.sigmoid(self.scale * (self.clean_x[users] @ self.clean_a.T) + self.offset)

    def logger(self, users: torch.Tensor) -> torch.Tensor:
        """π0(· | u): the logger's softmax at its temperature, with its uniform mix (``utils.policies.Policy``)."""
        p = torch.softmax((self.our_x[users] @ self.our_a.T) / max(self.temperature, 1e-8), dim=1)
        if self.mix > 0.0:
            p = (1.0 - self.mix) * p + self.mix / self.n_items
        return p


# ------------------------------------------------------------------------------------- the exact value and gradient
def exact_value(model, dataset: dict, *, world: WorldTensors | None = None, grad: bool = True, chunk: int = 2048,
                users: np.ndarray | None = None) -> tuple[float, np.ndarray | None]:
    """(V(θ), g*(θ)): Σ_u prior(u) Σ_j π_θ(j|u) q(u, j) over every user, and its gradient over θ (None with
    ``grad=False``). With ``users`` (a sample drawn from the prior, as the fits draw theirs) the same with equal
    weights: an unbiased estimate of V. Float64 sums."""
    world = world or WorldTensors(dataset, next(model.parameters()).device)
    if users is None:
        users = np.arange(int(dataset["n_users"]))
        weights = _prior(dataset)
    else:
        users = np.asarray(users, dtype=np.int64)
        weights = np.full(len(users), 1.0 / len(users))
    params = policy_params(model)
    total, g = 0.0, None
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(users[s:s + chunk], device=world.device)
        wt = torch.as_tensor(weights[s:s + chunk], device=world.device, dtype=torch.float64)
        with torch.set_grad_enabled(grad):
            val = ((_policy(model, u) * world.q(u)).sum(dim=1).double() * wt).sum()
            if grad:
                part = _flat_grad(torch.autograd.grad(val, params))
                g = part if g is None else g + part
        total += float(val.detach())
    return total, g


@torch.no_grad()
def sample_greedy(model, dataset: dict, world: WorldTensors, users: np.ndarray, chunk: int = 2048) -> float:
    """The greedy value on a prior-drawn sample of users (equal weights): each user's top item's true q."""
    total = 0.0
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(users[s:s + chunk], device=world.device)
        top = model.policy_logits(u).argmax(dim=1)
        total += float(world.q(u)[torch.arange(len(u), device=world.device), top].double().sum())
    return total / len(users)


def exact_greedy(model, dataset: dict) -> float:
    """The true value of each user's top item under the model (the study's ``calc_greedy_reward``)."""
    from utils.simulation_utils import calc_greedy_reward

    x, a = model.get_params()
    return float(calc_greedy_reward(dataset, x.detach().cpu().numpy(), a.detach().cpu().numpy()))


# ------------------------------------------------------------------------------------------- population optima
def _fit_users(dataset: dict, seed: int, label: str, n: int = FIT_USERS) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(int(seed), "opc_gradients_users", label))
    return rng.choice(int(dataset["n_users"]), size=int(n), p=_prior(dataset))


def fit_value_path(dataset: dict, *, lr: float, seed: int, steps: int = FIT_STEPS, fit_users: int = FIT_USERS,
                   batch_users: int = BATCH_USERS, checkpoint_every: int = CHECKPOINT_EVERY, device=None,
                   world: WorldTensors | None = None) -> tuple:
    """Adam ascent of the true value from the source, as ``class_oracles`` / ``oracle_repair`` fit theirs: a fixed
    prior-weighted sample of users (equal weights), ``batch_users`` per step, every item, a cosine schedule.
    Returns (model, [(step, θ)] every ``checkpoint_every`` steps, from step 0)."""
    device = _device(device)
    world = world or WorldTensors(dataset, device)
    seed_everything(derive_seed(int(seed), "opc_gradients_value", f"{lr:g}"))
    model = build_model(dataset, device=device)
    users = _fit_users(dataset, seed, "value", fit_users)
    rng = np.random.default_rng(derive_seed(int(seed), "opc_gradients_value_order", f"{lr:g}"))
    params = policy_params(model)
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    order, pos, path = rng.permutation(len(users)), 0, [(0, get_theta(model))]
    for step in range(1, int(steps) + 1):
        if pos + batch_users > len(order):
            order, pos = rng.permutation(len(users)), 0
        b = torch.as_tensor(users[order[pos:pos + batch_users]], device=device)
        pos += batch_users
        loss = -(_policy(model, b) * world.q(b)).sum(dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step <= CHECKPOINT_DENSE or step % int(checkpoint_every) == 0 or step == int(steps):
            path.append((step, get_theta(model)))
    return model, path


def _bce(z, q):
    return torch.nn.functional.binary_cross_entropy_with_logits(z, q, reduction="none")


def population_head(dataset: dict, users: np.ndarray, *, world: WorldTensors, chunk: int = 2048) -> tuple[float, float]:
    """(θ_s, c) minimizing the logging-weighted population NLL of σ(s g0/T + c) with the maps at the identity, over
    ``users`` with every item and the true q: the population version of ``fit_click_head``. LBFGS on (log s, c)."""
    from models.models import LOGIT_SCALE_SPEED

    log_s = torch.zeros((), dtype=torch.float64, device=world.device, requires_grad=True)
    c = torch.tensor(math.log(0.05 / 0.95), dtype=torch.float64, device=world.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_s, c], lr=1.0, max_iter=60, line_search_fn="strong_wolfe")
    blocks = [torch.as_tensor(users[s:s + chunk], device=world.device) for s in range(0, len(users), chunk)]

    def closure():
        opt.zero_grad()
        total = 0.0
        for u in blocks:
            g0 = ((world.our_x[u] @ world.our_a.T) / max(world.temperature, 1e-8)).double()
            z = torch.exp(log_s) * g0 + c
            loss = (world.logger(u).double() * _bce(z, world.q(u).double())).sum() / len(users)
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total)

    opt.step(closure)
    return float(log_s.detach()) / LOGIT_SCALE_SPEED, float(c.detach())


def fit_likelihood_population(dataset: dict, *, lr: float, seed: int, steps: int = FIT_STEPS,
                              fit_users: int = FIT_USERS, batch_users: int = BATCH_USERS, device=None,
                              world: WorldTensors | None = None, head=None):
    """Adam descent of the logging-weighted population NLL Σ_u Σ_j π0(j|u) CE(q(u, j), σ(s g/T + c)) from the
    source with the head at its population fit (``population_head``): the click model of the shared likelihood arm
    at infinite data. Returns the click-mode model."""
    device = _device(device)
    world = world or WorldTensors(dataset, device)
    seed_everything(derive_seed(int(seed), "opc_gradients_likelihood", f"{lr:g}"))
    model = build_model(dataset, mode="click", device=device)
    users = _fit_users(dataset, seed, "likelihood", fit_users)
    theta_s, c0 = head if head is not None else population_head(dataset, users, world=world)
    with torch.no_grad():
        model.log_logit_scale.fill_(theta_s)
        model.click_intercept.fill_(c0)
    rng = np.random.default_rng(derive_seed(int(seed), "opc_gradients_likelihood_order", f"{lr:g}"))
    params = policy_params(model) + [model.click_intercept]
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps))
    order, pos = rng.permutation(len(users)), 0
    for _ in range(int(steps)):
        if pos + batch_users > len(order):
            order, pos = rng.permutation(len(users)), 0
        b = torch.as_tensor(users[order[pos:pos + batch_users]], device=device)
        pos += batch_users
        z = model(b)[:, :, 0]
        loss = (world.logger(b) * _bce(z, world.q(b))).sum(dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def exact_likelihood(model, dataset: dict, *, world: WorldTensors, chunk: int = 2048) -> float:
    """The logging-weighted population NLL of a click-mode model, over every user (prior-weighted)."""
    prior = _prior(dataset)
    total = 0.0
    for s in range(0, int(dataset["n_users"]), chunk):
        u = torch.arange(s, min(s + chunk, int(dataset["n_users"])), device=world.device)
        nll = (world.logger(u) * _bce(model(u)[:, :, 0], world.q(u))).sum(dim=1).double()
        total += float((nll * torch.as_tensor(prior[s:s + chunk], device=world.device)).sum())
    return total


def value_optimal_scale(model, dataset: dict, *, world: WorldTensors, users: np.ndarray,
                        grid=np.linspace(-0.25, 0.25, 51)) -> float:
    """θ_s maximizing the value of the model's maps (its click intercept is irrelevant to the policy), by a grid on
    θ_s (s = exp(30 θ_s) from 5e-4 to 1.8e3) and a golden-section refinement, on ``users``."""
    def value_at(ts: float) -> float:
        with torch.no_grad():
            model.log_logit_scale.fill_(float(ts))
        v, _ = exact_value(model, dataset, world=world, grad=False, users=users)
        return v

    vals = [value_at(t) for t in grid]
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    phi = (math.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = value_at(a), value_at(b)
    for _ in range(30):
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = value_at(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = value_at(b)
    best = (a + b) / 2
    value_at(best)
    return float(best)


def policy_states(dataset: dict, *, seed: int, lrs=VALUE_LRS, steps: int = FIT_STEPS, fit_users: int = FIT_USERS,
                  batch_users: int = BATCH_USERS, device=None, log=print) -> dict:
    """The policy states of §3 as θ vectors, with their exact and greedy values and the fits' diagnostics:
    ``source`` (θ = 0), ``value`` (θ_value*), ``mid`` / ``mid_greedy`` (the θ_value* run's checkpoints closest to
    halfway in softmax and in greedy value), ``likelihood`` (θ_log*'s maps at their value-optimal scale) and
    ``likelihood_click`` (θ_log* at its click scale)."""
    device = _device(device)
    world = WorldTensors(dataset, device)
    src = build_model(dataset, device=device)
    v_src, _ = exact_value(src, dataset, world=world, grad=False)
    out = {"source": {"theta": get_theta(src), "value": v_src, "greedy": exact_greedy(src, dataset)}}

    best = None
    for lr in lrs:
        model, path = fit_value_path(dataset, lr=lr, seed=seed, steps=steps, fit_users=fit_users,
                                     batch_users=batch_users, device=device, world=world)
        v, _ = exact_value(model, dataset, world=world, grad=False)
        log(f"[states] value lr={lr:g}: V={v:.6f}")
        if best is None or v > best[0]:
            best = (v, lr, model, path)
    v_star, lr_star, model, path = best
    out["value"] = {"theta": get_theta(model), "value": v_star, "greedy": exact_greedy(model, dataset), "lr": lr_star}

    # the path's checkpoints are valued on a fixed prior-drawn sample of users (an unbiased estimate, to locate the
    # halfway points); the chosen states are then valued exactly
    probe = build_model(dataset, device=device)
    eval_users = _fit_users(dataset, seed, "path_eval", fit_users)
    path_values = []
    for step, theta in path:
        set_theta(probe, theta)
        path_values.append((step, exact_value(probe, dataset, world=world, grad=False, users=eval_users)[0],
                            sample_greedy(probe, dataset, world, eval_users)))
    # mid: halfway in the softmax value (§3); mid_greedy: halfway in the greedy value, where the ranking itself is
    # half repaired (the softmax value's halfway point comes within a few steps, mostly from sharpening)
    ends = {1: (path_values[0][1], path_values[-1][1]), 2: (path_values[0][2], path_values[-1][2])}
    for name, col in (("mid", 1), ("mid_greedy", 2)):
        half = sum(ends[col]) / 2
        k = int(np.argmin([abs(row[col] - half) for row in path_values]))
        set_theta(probe, path[k][1])
        out[name] = {"theta": path[k][1], "value": exact_value(probe, dataset, world=world, grad=False)[0],
                     "greedy": exact_greedy(probe, dataset), "step": int(path[k][0]), "halfway_sample": half}
    out["value_path"] = np.array(path_values)

    users = _fit_users(dataset, seed, "likelihood", fit_users)
    head = population_head(dataset, users, world=world)
    best = None
    for lr in lrs:
        lik = fit_likelihood_population(dataset, lr=lr, seed=seed, steps=steps, fit_users=fit_users,
                                        batch_users=batch_users, device=device, world=world, head=head)
        nll = exact_likelihood(lik, dataset, world=world)
        log(f"[states] likelihood lr={lr:g}: L_log={nll:.6f}")
        if best is None or nll < best[0]:
            best = (nll, lr, lik)
    nll, lr_lik, lik = best
    click_theta = get_theta(lik)
    lik.mode = "policy"
    v_click, _ = exact_value(lik, dataset, world=world, grad=False)
    out["likelihood_click"] = {"theta": click_theta, "value": v_click, "greedy": exact_greedy(lik, dataset),
                               "lr": lr_lik, "L_log": nll, "head_theta_s": head[0], "head_c": head[1],
                               "click_intercept": float(lik.click_intercept.detach())}
    ts = value_optimal_scale(lik, dataset, world=world, users=_fit_users(dataset, seed, "scale", fit_users))
    v_lik, _ = exact_value(lik, dataset, world=world, grad=False)
    out["likelihood"] = {"theta": get_theta(lik), "value": v_lik, "greedy": exact_greedy(lik, dataset), "theta_s": ts}
    return out


# ------------------------------------------------------------------------------------------- logged data and q̂
def logged_rows(dataset: dict, n: int, seed: int) -> dict:
    """n logged rows of the world (users from the prior, actions from the logger with its exact propensities, clicks
    from q), drawn by the study's own simulator (``_simulate_from_embedding_policy``)."""
    from training.trainer_trials import _simulate_from_embedding_policy
    from utils.simulation_utils import get_train_data

    sim = _simulate_from_embedding_policy(dataset, dataset["our_x"], dataset["our_a"], int(n), random_state=int(seed))
    return get_train_data(int(dataset["n_actions"]), int(n), sim, np.arange(int(n)), dataset["our_x"])


def study_user_fold(dataset: dict, seed: int, folds: int = 5) -> np.ndarray:
    """The study's fixed user folds of a world (``run_full_study._run_condition``)."""
    return np.random.default_rng(derive_seed(int(seed), "crossfit", "user_fold")).integers(0, int(folds),
                                                                                          int(dataset["n_users"]))


def crossfit_qhat(dataset: dict, rows: dict, user_fold: np.ndarray, *, folds: int = 5, device=None):
    """The cross-fitted reward model of ``rows`` as the training loss reads it: one logistic model per user fold, fit
    on the rows of the other folds (``run_full_study._crossfit_bundles``), looked up per user
    (``CrossFitScoresLookup``; the first fold's lookup supplies the shared attributes)."""
    import contextlib
    import io

    from training.run_full_study import _crossfit_bundles
    from training.trainer_trials import CrossFitScoresLookup, _scores_lookup_from_bundle

    with contextlib.redirect_stdout(io.StringIO()):
        bundles = _crossfit_bundles(dataset, rows, user_fold, folds, reward_model="regression",
                                    reward_features="interaction")
    dev = _device(device)
    lookups = [_scores_lookup_from_bundle(b, dev) for b in bundles]
    return CrossFitScoresLookup(lookups, user_fold, lookups[0])


# ------------------------------------------------------------------------------------------------- the estimators
def _surrogate(spec: str, scores, prob, actions, rewards, pscore):
    mode, param = parse_weight_spec(spec)
    return dr_sndr_surrogate(scores, prob, actions, rewards, pscore, use_iw=True, use_log_trick=False, iw_mode=mode,
                             iw_param=param, normalizer="none")


def estimator_gradients(model, rows: dict, *, world: WorldTensors, qhat=None, estimators=ESTIMATOR_NAMES,
                        chunk: int = 2048, row_weight: np.ndarray | None = None, conditional_bias: bool = False,
                        diagnostics: dict | None = None) -> dict:
    """{estimator: the gradient of its value estimate at the model's θ}, over the logged ``rows`` (a
    ``get_train_data`` dict). Each estimate is Σ_i ρ_i f_i, f_i the per-row DR surrogate of the training loss
    (``dr_sndr_surrogate``), ρ_i = 1/n (or ``row_weight``, e.g. outcome probabilities in an enumeration).

    ``conditional_bias``: also ``'bias5'``, the gradient of Σ_i ρ_i Σ_j π0(j|x_i)(h(w_ij) − w_ij)(q − q̂)(x_i, j),
    whose expectation is E[G5] − g* given q̂ (§2). ``diagnostics``: filled with the raw weights' profile and the
    reward model's logging- and target-weighted RMSE on the rows."""
    params = policy_params(model)
    n = len(rows["r"])
    rho_all = np.full(n, 1.0 / n) if row_weight is None else np.asarray(row_weight, dtype=np.float64)
    users_all = np.asarray(rows["x_idx"], dtype=np.int64)
    acts_all = np.asarray(rows["a"], dtype=np.int64)
    grads = {e: None for e in estimators}
    if conditional_bias:
        grads["bias5"] = None
    need_qhat = any(ESTIMATORS[e][1] == "qhat" for e in estimators) or conditional_bias
    if need_qhat and qhat is None:
        raise ValueError("a q̂ estimator needs the cross-fitted q̂")
    w_raw, sq_log, sq_tgt = [], 0.0, 0.0
    for s in range(0, n, chunk):
        u = torch.as_tensor(users_all[s:s + chunk], device=world.device)
        a = torch.as_tensor(acts_all[s:s + chunk], device=world.device)
        r = torch.as_tensor(np.asarray(rows["r"][s:s + chunk]), dtype=torch.float64, device=world.device)
        p = torch.as_tensor(np.asarray(rows["pscore"][s:s + chunk]), dtype=torch.float64, device=world.device)
        rho = torch.as_tensor(rho_all[s:s + chunk], dtype=torch.float64, device=world.device)
        prob = _policy(model, u)
        q = world.q(u)
        qh = qhat[u].to(prob.dtype) if need_qhat else None
        sources = {"zero": torch.zeros_like(prob), "oracle": q, "qhat": qh}
        for e in estimators:
            spec, src = ESTIMATORS[e]
            val = (_surrogate(spec, sources[src], prob, a, r, p) * rho).sum()
            part = _flat_grad(torch.autograd.grad(val, params, retain_graph=True))
            grads[e] = part if grads[e] is None else grads[e] + part
        if conditional_bias:
            p0 = world.logger(u)
            wj = prob / p0.clamp(min=1e-10)
            mode, lam = parse_weight_spec("harmonic:0.1")
            hw = wj / (1.0 - lam + lam * wj)
            val = (((p0 * (hw - wj) * (q - qh).detach()).sum(dim=1)).double() * rho).sum()
            part = _flat_grad(torch.autograd.grad(val, params, retain_graph=True))
            grads["bias5"] = part if grads["bias5"] is None else grads["bias5"] + part
        if diagnostics is not None:
            with torch.no_grad():
                rows_t = torch.arange(len(u), device=world.device)
                w_raw.append((prob[rows_t, a].double() / p).cpu().numpy())
                if need_qhat:
                    err2 = (qh - q).double() ** 2
                    sq_log += float(((world.logger(u).double() * err2).sum(dim=1) * rho).sum())
                    sq_tgt += float(((prob.double() * err2).sum(dim=1) * rho).sum())
    if diagnostics is not None:
        w = np.concatenate(w_raw) if w_raw else np.zeros(0)
        diagnostics.update(weight_profile(w))
        if need_qhat:
            diagnostics.update({"qhat_rmse_logging": math.sqrt(max(sq_log, 0.0) / rho_all.sum()),
                                "qhat_rmse_target": math.sqrt(max(sq_tgt, 0.0) / rho_all.sum())})
    return grads


def weight_profile(w: np.ndarray) -> dict:
    """Raw ratios π_θ(a_i|x_i)/p_i of logged rows: mean, sd, quantiles, maximum, ESS and its share."""
    if len(w) == 0:
        return {}
    out = {"w_mean": float(w.mean()), "w_sd": float(w.std()), "w_max": float(w.max())}
    for qq in (0.5, 0.9, 0.99, 0.999):
        out[f"w_q{qq * 100:g}"] = float(np.quantile(w, qq))
    ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))
    out.update({"w_ess": ess, "w_ess_share": ess / len(w)})
    return out


def minibatch_gradients(model, rows: dict, *, world: WorldTensors, qhat, estimators, batch: int, count: int,
                        seed: int) -> dict:
    """{estimator: [count gradients]}: each the gradient of one training minibatch's loss (``batch`` rows drawn
    without replacement, the mean of the rows' surrogates), as one optimizer step sees it."""
    rng = np.random.default_rng(derive_seed(int(seed), "opc_gradients_minibatch"))
    n = len(rows["r"])
    out = {e: [] for e in estimators}
    for _ in range(int(count)):
        idx = rng.choice(n, size=min(int(batch), n), replace=False)
        sub = {k: (np.asarray(v)[idx] if k in ("x", "a", "r", "x_idx", "pscore") else v) for k, v in rows.items()}
        g = estimator_gradients(model, sub, world=world, qhat=qhat, estimators=estimators, chunk=len(idx))
        for e in estimators:
            out[e].append(g[e])
    return out
