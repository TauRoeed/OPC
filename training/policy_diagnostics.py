"""Exact pick-level diagnostics of greedy policies against the simulator's truth and the logger
(docs/blob_controlled_integration.md §5).

For a greedy policy a*(u) = argmax_a ux_u · ia_a (ties to the first item, as ``calc_greedy_reward``):
- **where it recommends,** relative to the logger π0 = (1 − m) softmax(x·a / T) + m / P:
  - the shares of users whose pick is the logger's top item or in its top 10, the median rank of the pick under the
    logger's scores and the share ranked 100 or lower;
  - the share whose pick the logger shows less often than uniformly, π0(a*|u) < 1/P;
  - the mean log10(P π0(a*|u));
- **what the picks are worth:** the true click probability at the picks that are and are not the logger's top item,
  inside and outside its top 10, and above and below uniform propensity;
- **concentration:** the number of distinct items picked and the share of the most-picked one;
- **for a click model σ(ux·ia + offset):**
  - the optimism at its picks, Σ prior (σ(f(u, a*)) − q(u, a*));
  - its prediction error over every (u, a) pair, by how often the logger shows the pair (bins of P π0(a|u)), with
    each bin's share of the pairs and of the logged rows;
  - the infinite-data likelihood objective Σ prior Σ π0 CE(q, σ(f)).

Every sum is over users weighted by the prior and is exact (no sampling).

Usage (every saved selected policy of the given runs, per world):
    python -m training.policy_diagnostics --runs <run dirs> --emb-dir BPR/embeddings --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.simulation_utils import _exact_reward_device, _normalized_prior, ensure_exact_env_q_cache

TOP_K = 10
PI0_BINS = (0.1, 1.0, 10.0)  # P π0(a|u): below a tenth of uniform, below uniform, below 10× uniform, above
BIN_NAMES = ("lt0.1", "0.1-1", "1-10", "ge10")
BLOCK_CELLS = 8 * 1024 * 1024
POLICY_SUFFIX = "_selected_policy.npz"


def _device(dataset: dict) -> torch.device:
    d = _exact_reward_device(dataset)
    return d if d is not None else torch.device("cpu")


def _rows(dataset: dict) -> int:
    return max(1, min(int(dataset["n_users"]), BLOCK_CELLS // max(int(dataset["n_actions"]), 1)))


class _Truth:
    """Blocks of q(u, ·) and of the logger's log π0(·|u) on one device."""

    def __init__(self, dataset: dict, device: torch.device):
        from training.trainer_trials import _logging_uniform_mix, _policy_temperature

        ensure_exact_env_q_cache(dataset)
        t = lambda a: torch.as_tensor(np.require(a, dtype=np.float32, requirements="W"), device=device)
        self.q_cache = dataset.get("q_x_a")
        if self.q_cache is None:
            env = dataset["env"]
            self.env_x, self.env_a_t = t(env.emb_x), t(env.emb_a).T.contiguous()
            self.scale, self.offset = float(env.scale), float(env.offset)
        else:
            self.q_all = t(self.q_cache)
        self.x, self.a_t = t(dataset["our_x"]), t(dataset["our_a"]).T.contiguous()
        self.temperature = max(_policy_temperature(dataset), 1e-8)
        self.mix = _logging_uniform_mix(dataset)
        self.n_actions = int(dataset["n_actions"])

    def q(self, u0: int, u1: int) -> torch.Tensor:
        if self.q_cache is not None:
            return self.q_all[u0:u1]
        return torch.sigmoid((self.env_x[u0:u1] @ self.env_a_t) * self.scale + self.offset)

    def logger_scores(self, u0: int, u1: int) -> torch.Tensor:
        return self.x[u0:u1] @ self.a_t

    def pi0(self, u0: int, u1: int, scores: torch.Tensor | None = None) -> torch.Tensor:
        z = self.logger_scores(u0, u1) if scores is None else scores
        p = torch.softmax(z / self.temperature, dim=1)
        return p if self.mix <= 0 else (1.0 - self.mix) * p + self.mix / self.n_actions


@torch.no_grad()
def logger_reference(dataset: dict) -> dict:
    """Per user: the logger's top item and top-10 items."""
    device = _device(dataset)
    truth = _Truth(dataset, device)
    n, rows = int(dataset["n_users"]), _rows(dataset)
    top = np.empty((n, TOP_K), dtype=np.int64)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for u0 in range(0, n, rows):
            u1 = min(u0 + rows, n)
            top[u0:u1] = torch.topk(truth.x[u0:u1] @ truth.a_t, TOP_K, dim=1).indices.cpu().numpy()
    finally:
        torch.set_float32_matmul_precision(prev)
    return {"top": top}


@torch.no_grad()
def greedy_pick_diagnostics(dataset: dict, ux: np.ndarray, ia: np.ndarray, *, offset=None, ref: dict | None = None,
                            return_picks: bool = False) -> dict:
    """The diagnostics above for one greedy policy; ``offset`` (a scalar or one value per user) makes ux·ia + offset
    a click model and adds its prediction diagnostics."""
    device = _device(dataset)
    truth = _Truth(dataset, device)
    ref = ref or logger_reference(dataset)
    n, rows = int(dataset["n_users"]), _rows(dataset)
    P = int(dataset["n_actions"])
    prior = _normalized_prior(dataset)
    t = lambda a: torch.as_tensor(np.require(a, dtype=np.float32, requirements="W"), device=device)
    pol_x, pol_a_t = t(ux), t(ia).T.contiguous()
    click = offset is not None
    if click:
        off = np.broadcast_to(np.asarray(offset, dtype=np.float32), (n,))
        off_t = t(np.ascontiguousarray(off))
        edges = torch.as_tensor(PI0_BINS, dtype=torch.float32, device=device)
        nb = len(PI0_BINS) + 1
        err_sum, bias_sum, pairs, logged = (np.zeros(nb) for _ in range(4))
        ce_total, optimism = 0.0, 0.0
    picks = np.empty(n, dtype=np.int64)
    rank_pick = np.empty(n, dtype=np.int64)
    q_pick = np.empty(n)
    pi0_pick = np.empty(n)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for u0 in range(0, n, rows):
            u1 = min(u0 + rows, n)
            scores = pol_x[u0:u1] @ pol_a_t
            best = scores.argmax(dim=1)
            q = truth.q(u0, u1)
            z = truth.logger_scores(u0, u1)
            pi0 = truth.pi0(u0, u1, z)
            picks[u0:u1] = best.cpu().numpy()
            # the pick's rank under the logger's scores (0 = the logger's own top item)
            rank_pick[u0:u1] = (z > z.gather(1, best[:, None])).sum(dim=1).cpu().numpy()
            q_pick[u0:u1] = q.gather(1, best[:, None])[:, 0].double().cpu().numpy()
            pi0_pick[u0:u1] = pi0.gather(1, best[:, None])[:, 0].double().cpu().numpy()
            if click:
                w = torch.as_tensor(prior[u0:u1], device=device)
                f = scores + off_t[u0:u1, None]
                p = torch.sigmoid(f)
                optimism += float((w * (p.gather(1, best[:, None])[:, 0] - q.gather(1, best[:, None])[:, 0])).sum())
                # CE(q, σ(f)) = softplus(f) − q f
                ce = torch.nn.functional.softplus(f) - q * f
                ce_total += float((w * (pi0 * ce).sum(dim=1, dtype=torch.float64)).sum())
                b = torch.bucketize(pi0 * P, edges)
                d = p - q
                for k in range(nb):
                    m = (b == k)
                    pairs[k] += float((w * m.sum(dim=1, dtype=torch.float64)).sum())
                    logged[k] += float((w * (pi0 * m).sum(dim=1, dtype=torch.float64)).sum())
                    err_sum[k] += float((w * (d.abs() * m).sum(dim=1, dtype=torch.float64)).sum())
                    bias_sum[k] += float((w * (d * m).sum(dim=1, dtype=torch.float64)).sum())
    finally:
        torch.set_float32_matmul_precision(prev)
        pol_x = pol_a_t = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    top = ref["top"]
    is_top1 = picks == top[:, 0]
    in_top = (top == picks[:, None]).any(axis=1)
    below = pi0_pick * P < 1.0
    counts = np.bincount(picks, weights=prior, minlength=P)

    def mean_where(x, mask):
        w = float(prior[mask].sum())
        return float(np.dot(prior[mask], x[mask]) / w) if w > 0 else np.nan

    order = np.argsort(rank_pick, kind="stable")
    cum = np.cumsum(prior[order])
    out = {"value_greedy": float(np.dot(prior, q_pick)), "agree_logger_top1": float(prior[is_top1].sum()),
           "pick_rank_median": float(rank_pick[order][np.searchsorted(cum, 0.5)]),
           "pick_rank_ge100": float(prior[rank_pick >= 100].sum()),
           "q_at_picks_top1": mean_where(q_pick, is_top1), "q_at_picks_not_top1": mean_where(q_pick, ~is_top1),
           "in_logger_top10": float(prior[in_top].sum()), "pick_below_uniform": float(prior[below].sum()),
           "pick_log10_rel_pi0": float(np.dot(prior, np.log10(np.maximum(pi0_pick * P, 1e-300)))),
           "q_at_picks_in_top10": mean_where(q_pick, in_top), "q_at_picks_off_top10": mean_where(q_pick, ~in_top),
           "q_at_picks_below_uniform": mean_where(q_pick, below),
           "q_at_picks_above_uniform": mean_where(q_pick, ~below),
           "distinct_items": int((counts > 0).sum()), "top_item_share": float(counts.max())}
    if click:
        out.update({"optimism_at_pick": optimism, "logged_ce": ce_total})
        for k, name in enumerate(BIN_NAMES):
            out[f"pairs_{name}"] = pairs[k] / P
            out[f"logged_{name}"] = logged[k]
            out[f"mae_{name}"] = err_sum[k] / pairs[k] if pairs[k] > 0 else np.nan
            out[f"bias_{name}"] = bias_sum[k] / pairs[k] if pairs[k] > 0 else np.nan
    if return_picks:
        out["_picks"], out["_q_pick"], out["_pi0_pick"] = picks, q_pick, pi0_pick
    return out


def pairwise_table(prior: np.ndarray, picks: dict, P: int) -> pd.DataFrame:
    """Per ordered pair of policies (A, B): the users where they agree, and on the users where they disagree, the
    value difference V_A − V_B (it all comes from them) split by which pick the logger shows less often."""
    rows = []
    names = sorted(picks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = picks[a], picks[b]
            agree = pa["_picks"] == pb["_picks"]
            dq = pa["_q_pick"] - pb["_q_pick"]
            a_rarer = (pa["_pi0_pick"] < pb["_pi0_pick"]) & ~agree
            b_rarer = (pb["_pi0_pick"] < pa["_pi0_pick"]) & ~agree
            rows.append({"a": a, "b": b, "agree": float(prior[agree].sum()),
                         "value_diff": float(np.dot(prior, dq)),
                         "users_a_rarer": float(prior[a_rarer].sum()),
                         "value_diff_a_rarer": float(np.dot(prior[a_rarer], dq[a_rarer])),
                         "users_b_rarer": float(prior[b_rarer].sum()),
                         "value_diff_b_rarer": float(np.dot(prior[b_rarer], dq[b_rarer]))})
    return pd.DataFrame(rows)


def save_selected_policy(path: Path, ux, ia, *, offset=None, **meta) -> None:
    """One selected policy: its ranking vectors, the click model's offset when it has one, and scalar metadata."""
    arrays = {"ux": np.asarray(ux, dtype=np.float32), "ia": np.asarray(ia, dtype=np.float32)}
    if offset is not None:
        arrays["offset"] = np.asarray(offset, dtype=np.float32)
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))


def load_selected_policy(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {"ux": z["ux"], "ia": z["ia"], "offset": z["offset"] if "offset" in z.files else None}
    out.update(json.loads(str(z["meta"])))
    return out


def _world_key(cond: Path) -> tuple:
    from training.analyze_cause_fair import _tags

    tags = _tags(cond.name)
    return tags["dataset"], tags["bias"], int(tags["seed"])


def _world_options(run: Path) -> dict:
    """The world options a run was made with: its manifest (study runs) or its settings (class oracles, whose
    policies sit in OUT/policies)."""
    for d in (run, run.parent):
        for name in ("run_manifest.json", "class_oracle_settings.json"):
            if (d / name).exists():
                return json.loads((d / name).read_text())["world_options"]
    # a study run still in progress has no manifest yet: its conditions' run_meta.json record the same options
    metas = sorted(run.glob("dataset=*/run_meta.json"))
    if metas:
        params = json.loads(metas[0].read_text())["params"]
        return {k: v for k, v in params.items() if k not in ("bias", "ctr", "logging_uniform_mix")}
    raise FileNotFoundError(f"no run_manifest.json, class_oracle_settings.json or run_meta.json for {run}")


def main(argv=None) -> None:
    from training.run_full_study import build_condition_world
    from utils.seeding import seed_everything

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run directories holding *_selected_policy.npz files")
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--ctr", type=float, default=0.05)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="only these datasets (the output is resumable by world, so a dataset can be added later)")
    ap.add_argument("--expect", type=int, default=None,
                    help="skip a world with fewer saved policies than this (an arm still running)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worlds: dict[tuple, list] = {}
    options = {}
    for run in args.runs:
        options[run] = _world_options(Path(run))
        for cond in sorted(Path(run).glob("dataset=*")):
            for path in sorted(cond.glob("*" + POLICY_SUFFIX)):
                worlds.setdefault(_world_key(cond), []).append((Path(run).name, path))
    world_options = next(iter(options.values()))
    if any(o != world_options for o in options.values()):
        raise ValueError(f"the runs were made with different world options: {options}")
    diag_path, pair_path = out / "policy_diagnostics.csv", out / "policy_pairs.csv"
    done = set()
    if diag_path.exists():
        prev = pd.read_csv(diag_path)
        done = set(zip(prev["dataset"], prev["bias"], prev["seed"]))
    for (ds, bias, seed), items in sorted(worlds.items()):
        if (ds, bias, seed) in done or (args.datasets and ds not in args.datasets):
            continue
        if args.expect is not None and len(items) < args.expect:
            print(f"{ds} {bias} {seed}: {len(items)} policies, expected {args.expect}; skipped", flush=True)
            continue
        t0 = time.time()
        seed_everything(seed)
        dataset, *_ = build_condition_world(ds, Path(args.emb_dir), bias, args.ctr, seed, world_options=world_options)
        ref = logger_reference(dataset)
        prior = _normalized_prior(dataset)
        rows, picks = [], {}
        for run, path in items:
            pol = load_selected_policy(path)
            arm = path.name[: -len(POLICY_SUFFIX)]
            # a value oracle's logits rank and sharpen; they are not a calibrated click model
            offset = None if (arm.startswith("oracle_") and "_value_" in arm) else pol["offset"]
            d = greedy_pick_diagnostics(dataset, pol["ux"], pol["ia"], offset=offset, ref=ref, return_picks=True)
            if "value_greedy" in pol:  # the run's own value of the same policy: the world and replay check
                d["run_value_greedy"] = float(pol["value_greedy"])
                if abs(d["run_value_greedy"] - d["value_greedy"]) > 1e-7:
                    raise AssertionError(f"{path}: value {d['value_greedy']:.8f} here, {d['run_value_greedy']:.8f} in the run")
            picks[f"{run}/{arm}"] = d
            rows.append({"dataset": ds, "bias": bias, "seed": seed, "run": run, "arm": arm,
                         **{k: v for k, v in d.items() if not k.startswith("_")}})
        logger = greedy_pick_diagnostics(dataset, dataset["our_x"], dataset["our_a"], ref=ref, return_picks=True)
        picks["logger"] = logger
        rows.append({"dataset": ds, "bias": bias, "seed": seed, "run": "", "arm": "logger",
                     **{k: v for k, v in logger.items() if not k.startswith("_")}})
        # BLOB-supplied-source at its prior mean (zero deviation, zero intercepts): its ranking is the source's after
        # the released column normalization of Ψ, the point BLOB's training starts from
        from models.blob import prepare_psi
        from training.blob_trials import supplied_source

        omega, psi = supplied_source(dataset)
        start = greedy_pick_diagnostics(dataset, omega, prepare_psi(psi)[0], ref=ref, return_picks=True)
        picks["blob_prior_mean"] = start
        rows.append({"dataset": ds, "bias": bias, "seed": seed, "run": "", "arm": "blob_prior_mean",
                     **{k: v for k, v in start.items() if not k.startswith("_")}})
        frame = pd.DataFrame(rows).assign(world_seconds=time.time() - t0)
        frame.to_csv(diag_path, mode="a", header=not diag_path.exists(), index=False, float_format="%.8g")
        pairs = pairwise_table(prior, picks, int(dataset["n_actions"])).assign(dataset=ds, bias=bias, seed=seed)
        pairs.to_csv(pair_path, mode="a", header=not pair_path.exists(), index=False, float_format="%.8g")
        print(f"{ds} {bias} {seed}: {len(items)} policies in {time.time() - t0:.0f}s", flush=True)
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
