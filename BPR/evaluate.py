"""Test-set quality of the BPR recipes: Recall@5 / @20, NDCG@5 / @20 and MPR.

For each user with at least ``--min-positives`` liked items, one liked item is held out as the
test item (seeded, and separate from the early-stopping validation item). The dataset's recipe
is then trained exactly as ``BPR.generate_artifacts`` trains it (early stopping on one validation
item per user, then a refit) on the remaining interactions, and each test item is ranked among
the items its user did not interact with in training:

    Recall@k = [r < k]     NDCG@k = [r < k] / log2(r + 2)     PR = r / (candidates - 1)

with r the 0-based rank of the test item (ties count half). MPR is the mean PR: 0 when the test
item always comes first, 0.5 at random. The embeddings in BPR/embeddings are refit on all
interactions, so these numbers describe the recipe, not those exact vectors.

    python -m BPR.evaluate --dataset ml --root datasets/ml-1m                  # train + evaluate
    python -m BPR.evaluate --dataset ml --skip-train --backends numpy cuda     # evaluate again
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from BPR.bpr_config import load_bpr_dataset_config, resolve_bpr_params
from BPR.bpr_minibatch import BPRConfig, MiniBatchBPR, _binary_csr, data_fingerprint, holdout_split
from utils.seeding import DEFAULT_CPU_THREADS, pin_cpu_threads

KS = (5, 20)
# user rows x items scored at once: 64 MB of fp32 for numpy, 512 MB for torch (few large GPU blocks)
BLOCK_CELLS = {"numpy": 16 * 1024 * 1024, "torch": 128 * 1024 * 1024}


def split_test_items(X, config: BPRConfig, min_positives: int = 4):
    """(X_rest, users, items): one held-out liked item per user with >= min_positives likes.

    The default 4 leaves every test user with >= 3 likes, so the recipe's validation split
    (``val_min_positives`` = 3) can still hold one of them out."""
    seed = int(np.random.SeedSequence([int(config.random_state), 5]).generate_state(1)[0])
    return holdout_split(X, min_positives=min_positives, seed=seed)


def held_out_ranks(U, V, b, X_train, users, items, *, backend: str = "numpy", block_cells: int | None = None):
    """0-based rank of each held-out item among its user's candidates (items not in X_train),
    ties counting half, and the number of candidates. ``backend``: numpy, cpu or cuda (torch)."""
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    X_train = _binary_csr(X_train)
    n_items = int(V.shape[0])
    cells = block_cells or BLOCK_CELLS["numpy" if backend == "numpy" else "torch"]
    rows = max(1, int(cells) // max(n_items, 1))
    candidates = n_items - np.diff(X_train.indptr)[users]
    ranks = np.empty(len(users), dtype=np.float64)
    U = np.asarray(U, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    b = None if b is None else np.asarray(b, dtype=np.float32)
    if backend == "numpy":
        for s in range(0, len(users), rows):
            uu, ii = users[s : s + rows], items[s : s + rows]
            S = U[uu] @ V.T
            if b is not None:
                S += b[None, :]
            sub = X_train[uu]
            S[np.repeat(np.arange(len(uu)), np.diff(sub.indptr)), sub.indices] = -np.inf
            t = S[np.arange(len(uu)), ii][:, None]
            ranks[s : s + len(uu)] = (S > t).sum(axis=1) + 0.5 * ((S == t).sum(axis=1) - 1)
        return ranks, candidates
    import torch

    device = torch.device(backend)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # no TF32: same fp32 scores as numpy up to rounding
    try:
        Ut, Vt = torch.as_tensor(U, device=device), torch.as_tensor(V, device=device)
        bt = None if b is None else torch.as_tensor(b, device=device)
        for s in range(0, len(users), rows):
            uu, ii = users[s : s + rows], items[s : s + rows]
            S = Ut[torch.as_tensor(uu, device=device)] @ Vt.T
            if bt is not None:
                S += bt[None, :]
            sub = X_train[uu]
            r = torch.as_tensor(np.repeat(np.arange(len(uu)), np.diff(sub.indptr)), device=device)
            S[r, torch.as_tensor(sub.indices.astype(np.int64), device=device)] = -torch.inf
            t = S[torch.arange(len(uu), device=device), torch.as_tensor(ii, device=device)][:, None]
            rk = (S > t).sum(dim=1).double() + 0.5 * ((S == t).sum(dim=1).double() - 1)
            ranks[s : s + len(uu)] = rk.cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return ranks, candidates
    finally:
        torch.set_float32_matmul_precision(prev)


def ranking_metrics(ranks, candidates, ks=KS) -> dict:
    """Recall@k, NDCG@k and MPR from 0-based ranks (one held-out item per user)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {"users": int(len(ranks))}
    for k in ks:
        hit = ranks < k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float((hit / np.log2(ranks + 2.0)).mean())
    out["mpr"] = float(np.mean(ranks / np.maximum(np.asarray(candidates, dtype=np.float64) - 1.0, 1.0)))
    return out


def popularity_scores(X_train) -> np.ndarray:
    """Every user's scores when ranking items by their number of likes in training."""
    return np.asarray(_binary_csr(X_train).sum(axis=0), dtype=np.float32).ravel()


def _paths(out_dir: Path, dataset: str) -> dict:
    return {k: out_dir / f"{dataset}_test_{k}" for k in
            ("user_factors.npy", "item_factors.npy", "item_bias.npy", "train.npz", "split.npz", "meta.json", "metrics.json")}


def train(dataset: str, root: str, out_dir: Path, *, min_positives: int, download: bool = True) -> dict:
    from BPR.generate_artifacts import _dataset_bundle

    ds_cfg = load_bpr_dataset_config(dataset)
    config = BPRConfig(**resolve_bpr_params(dataset))
    _, _, data = _dataset_bundle(dataset, root, data_cfg=ds_cfg["data"], download=download)
    X = _binary_csr(data.X)
    X_rest, users, items = split_test_items(X, config, min_positives)
    print(f"{dataset}: {X.shape[0]:,} users, {X.shape[1]:,} items, {X.nnz:,} interactions; "
          f"{len(users):,} test users (>= {min_positives} likes)", flush=True)
    model = MiniBatchBPR(config).fit(X_rest)
    p = _paths(out_dir, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_embeddings(str(p["user_factors.npy"]), str(p["item_factors.npy"]), str(p["item_bias.npy"]))
    sparse.save_npz(p["train.npz"], X_rest)
    np.savez(p["split.npz"], users=users, items=items)
    meta = {"dataset": dataset, "min_positives": int(min_positives), "data_fingerprint": data_fingerprint(X),
            "n_users": int(X.shape[0]), "n_items": int(X.shape[1]), "n_interactions": int(X.nnz),
            "test_users": int(len(users)), **model.summary()}
    p["meta.json"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def evaluate(dataset: str, out_dir: Path, backends, *, popularity: bool = True) -> dict:
    p = _paths(out_dir, dataset)
    U, V = np.load(p["user_factors.npy"]), np.load(p["item_factors.npy"])
    b = np.load(p["item_bias.npy"]) if p["item_bias.npy"].exists() else None
    X_train = sparse.load_npz(p["train.npz"]).tocsr()
    split = np.load(p["split.npz"])
    users, items = split["users"], split["items"]
    result = {"dataset": dataset, "backends": {}}
    for backend in backends:
        if backend == "cuda":
            import torch

            torch.zeros(1, device="cuda").sum().item()  # CUDA start-up is not part of the timing
        t0 = time.time()
        ranks, cand = held_out_ranks(U, V, b, X_train, users, items, backend=backend)
        result["backends"][backend] = {"seconds": round(time.time() - t0, 2), **ranking_metrics(ranks, cand)}
        print(f"{dataset} [{backend}] " + ", ".join(
            f"{k} {v:.4f}" for k, v in result["backends"][backend].items() if k not in ("users", "seconds"))
            + f" ({len(users):,} users, {result['backends'][backend]['seconds']:.1f}s)", flush=True)
    if popularity:
        zu = np.zeros((U.shape[0], 1), dtype=np.float32)
        zi = np.zeros((V.shape[0], 1), dtype=np.float32)
        ranks, cand = held_out_ranks(zu, zi, popularity_scores(X_train), X_train, users, items, backend=backends[-1])
        result["popularity"] = ranking_metrics(ranks, cand)
    p["metrics.json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=["ml", "myket", "anime", "lastfm", "msd", "kuairec", "kuairand"])
    parser.add_argument("--root", default=None, help="Dataset root (needed unless --skip-train).")
    parser.add_argument("--out-dir", type=Path, default=Path("BPR/embeddings/test_split"))
    parser.add_argument("--min-positives", type=int, default=4)
    parser.add_argument("--skip-train", action="store_true", help="Evaluate the saved test-split model.")
    parser.add_argument("--backends", nargs="+", default=["numpy"], choices=["numpy", "cpu", "cuda"],
                        help="numpy (CPU), or torch on cpu / cuda; the last one also scores the popularity baseline.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    args = parser.parse_args()
    pin_cpu_threads(args.cpu_threads)
    if not args.skip_train:
        if args.root is None:
            parser.error("--root is required unless --skip-train")
        train(args.dataset, args.root, args.out_dir, min_positives=args.min_positives, download=args.download)
    evaluate(args.dataset, args.out_dir, args.backends)


if __name__ == "__main__":
    main()
