"""The value gradient restricted to the logged data's visible pairs (docs/opc_gradient_regime_study.md §12.3), for the
gradient benchmark's low-overlap cells, from ``training.opc_gradient_benchmark`` run folders.

    python -m training.opc_visible_gradients --runs RUN... [--ess-below 1e-4] [--min-count 1]

Where the population ESS share 1 / E_π0[w²] of a state is below ``--ess-below``, the target policy can put much of its
mass on (user, item) pairs that the R × N logged rows of all the replicates are expected to contain less than once.
Unbiased estimators then average to the gradient over the other pairs, and their Monte Carlo floor misses the rest. Per
such world and state this writes the gradient of Σ_u prior Σ_j π q 1[R N prior(u) π0(j|u) ≥ min_count] and the target
mass on the unseen pairs (gstar_visible.npz / .json), R being the replicates written so far; also at the boundaries
``SENSITIVITY`` (keys ``state@c``), a diagnostic of how sharp the supported region's edge is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.opc_gradient_benchmark import build_world
from training.opc_gradients import WorldTensors, build_model, set_theta, visible_value_gradient
from utils.seeding import enable_determinism, pin_cpu_threads

ESS_BELOW = 1e-4
MIN_COUNT = 1.0  # the supported region used by the check (§12.3)
SENSITIVITY = (0.1, 10.0)  # other boundaries, reported as a diagnostic only


def _tags(name: str) -> dict:
    return dict(part.partition("=")[::2] for part in name.split("__"))


def low_overlap_states(wdir: Path, ess_below: float = ESS_BELOW) -> list[str]:
    """The benchmarked states of a world whose population ESS share is below ``ess_below``."""
    cfg = json.loads((wdir / "config.json").read_text())
    meta = json.loads((wdir / "gstar.json").read_text())
    return [s for s in cfg["states"] if s in meta and meta[s]["pop_ess_share"] < ess_below]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ess-below", type=float, default=ESS_BELOW)
    ap.add_argument("--min-count", type=float, default=MIN_COUNT)
    ap.add_argument("--emb-dir", default="BPR/embeddings")
    ap.add_argument("--cpu-threads", type=int, default=4, help="the world build depends on it (as the benchmark's)")
    args = ap.parse_args(argv)
    enable_determinism(True)
    pin_cpu_threads(int(args.cpu_threads))
    for run in args.runs:
        for wdir in sorted(Path(run).glob("dataset=*")):
            if not (wdir / "gstar.json").exists() or not (wdir / "config.json").exists():
                continue
            states = low_overlap_states(wdir, args.ess_below)
            reps = len(list((wdir / "rep").glob("rep_*.json")))
            if not states or reps == 0:
                continue
            cfg = json.loads((wdir / "config.json").read_text())
            rows = reps * int(cfg["train_size"])
            prev = json.loads((wdir / "gstar_visible.json").read_text()) if (wdir / "gstar_visible.json").exists() else {}
            if all(prev.get(s, {}).get("rows") == rows and prev[s].get("min_count") == args.min_count
                   and all(f"{s}@{c:g}" in prev for c in SENSITIVITY) for s in states):
                continue
            t = _tags(wdir.name)
            dataset, label = build_world(t["dataset"], t["bias"], int(t["seed"]), Path(args.emb_dir), cfg["world_options"])
            assert label == t["bias"], (label, wdir.name)
            world = WorldTensors(dataset)
            model = build_model(dataset, device=world.device)
            z = np.load(wdir / "states.npz")
            arrays, meta = {}, {}
            for s in states:
                set_theta(model, z[s])
                g, hidden = visible_value_gradient(model, dataset, world=world, rows=rows, min_count=args.min_count)
                arrays[s] = g
                meta[s] = {"rows": rows, "replicates": reps, "min_count": args.min_count, "hidden_mass": hidden,
                           "gvis_norm": float(np.linalg.norm(g))}
                for c in SENSITIVITY:
                    gc, hc = visible_value_gradient(model, dataset, world=world, rows=rows, min_count=c)
                    arrays[f"{s}@{c:g}"] = gc
                    meta[f"{s}@{c:g}"] = {"rows": rows, "replicates": reps, "min_count": c, "hidden_mass": hc,
                                          "gvis_norm": float(np.linalg.norm(gc))}
            np.savez(wdir / "gstar_visible.npz", **arrays)
            (wdir / "gstar_visible.json").write_text(json.dumps(meta, indent=2))
            print(json.dumps({"world": wdir.name, **{k: round(m["hidden_mass"], 4) for k, m in meta.items()}}), flush=True)
            del dataset, world, model


if __name__ == "__main__":
    main()
