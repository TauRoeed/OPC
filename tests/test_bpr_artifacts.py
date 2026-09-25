"""BPR v2 artifacts: settings resolution, the meta file, the staleness check the runners use,
and generate_artifacts end to end on the KuaiRec fixture."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from BPR.bpr_config import (
    build_bpr_meta,
    bpr_artifact_status,
    bpr_meta_path,
    load_bpr_dataset_config,
    resolve_bpr_params,
)
from BPR.bpr_minibatch import BPRConfig, MiniBatchBPR, data_fingerprint

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_resolve_bpr_params_merges_overrides_and_defaults():
    base = resolve_bpr_params("ml")
    assert set(base) == set(BPRConfig.__dataclass_fields__)  # every setting is resolved
    over = resolve_bpr_params("ml", overrides={"factors": 16, "negatives": "popularity", "patience": None})
    assert over["factors"] == 16 and over["negatives"] == "popularity" and over["patience"] == base["patience"]
    with pytest.raises(ValueError, match="unknown BPR settings"):
        resolve_bpr_params("ml", overrides={"samples_per_epoch": 5})


def _tiny_model(X, **cfg):
    return MiniBatchBPR(BPRConfig(**{"factors": 4, "batch_size": 64, "early_stopping": False, "epochs": 1, **cfg})).fit(X, log=None)


def test_meta_record_and_status(tmp_path):
    rng = np.random.default_rng(0)
    X = csr_matrix((rng.random((30, 12)) < 0.3).astype(np.float32))
    data_cfg = load_bpr_dataset_config("kuairec")["data"]
    settings = resolve_bpr_params("kuairec")
    model = MiniBatchBPR(BPRConfig(**{**settings, "max_epochs": settings["max_epochs"]}))
    model.fit(X, log=None)
    meta = build_bpr_meta("kuairec", model, data_cfg, X)
    json.dumps(meta)  # JSON-able
    assert meta["settings"] == settings and meta["data_fingerprint"] == data_fingerprint(X)
    assert meta["item_bias_file"] is True and meta["n_interactions"] == X.nnz
    assert {"git_commit", "created_at", "history", "validation", "best_epoch", "epochs_trained"} <= set(meta)

    assert bpr_artifact_status(tmp_path, "kuairec")["status"] == "missing"
    bpr_meta_path(tmp_path, "kuairec").write_text(json.dumps(meta))
    ok = bpr_artifact_status(tmp_path, "kuairec")
    assert ok["status"] == "ok" and ok["differences"] == {}

    stale = dict(meta, settings={**settings, "negatives": "popularity"})
    bpr_meta_path(tmp_path, "kuairec").write_text(json.dumps(stale))
    st = bpr_artifact_status(tmp_path, "kuairec")
    assert st["status"] == "stale" and list(st["differences"]) == ["negatives"]

    bpr_meta_path(tmp_path, "kuairec").write_text(json.dumps(dict(meta, data={"watch_ratio_min": 3.0})))
    assert "data" in bpr_artifact_status(tmp_path, "kuairec")["differences"]

    bpr_meta_path(tmp_path, "toy").write_text(json.dumps(meta))
    assert bpr_artifact_status(tmp_path, "toy")["status"] == "stale"  # not in the config


def test_no_held_out_users_falls_back_to_fixed_epochs():
    X = csr_matrix(np.eye(6, dtype=np.float32))  # one liked item per user: nothing to hold out
    lines = []
    m = MiniBatchBPR(BPRConfig(factors=2, batch_size=8, epochs=2)).fit(X, log=lines.append)
    assert m.epochs_trained == 2 and m.best_epoch is None
    assert any("fixed epochs" in line for line in lines)


def _generate(emb_dir, *extra):
    cmd = [sys.executable, "-m", "BPR.generate_artifacts", "--dataset", "kuairec", "--root", str(FIXTURES / "kuairec"),
           "--emb-dir", str(emb_dir), "--no-download", "--batch-size", "8", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_generate_artifacts_end_to_end(tmp_path):
    r = _generate(tmp_path, "--no-early-stopping", "--epochs", "2")
    assert r.returncode == 0, r.stderr[-2000:]
    for name in ("user_factors", "item_factors", "item_bias", "item_metadata", "user_interaction_counts"):
        assert (tmp_path / f"kuairec_{name}.npy").exists(), name
    meta = json.loads(bpr_meta_path(tmp_path, "kuairec").read_text())
    x = np.load(tmp_path / "kuairec_user_factors.npy"); a = np.load(tmp_path / "kuairec_item_factors.npy")
    b = np.load(tmp_path / "kuairec_item_bias.npy")
    assert x.shape == (meta["n_users"], 32) and a.shape == (meta["n_items"], 32) and b.shape == (meta["n_items"],)
    assert meta["settings"]["epochs"] == 2 and meta["settings"]["early_stopping"] is False
    st = bpr_artifact_status(tmp_path, "kuairec")
    assert st["status"] == "stale" and set(st["differences"]) == {"batch_size", "early_stopping", "epochs"}

    r = _generate(tmp_path, "--no-early-stopping", "--epochs", "2", "--no-item-bias")
    assert r.returncode == 0, r.stderr[-2000:]
    assert not (tmp_path / "kuairec_item_bias.npy").exists()  # no stale bias next to bias-free vectors
    assert json.loads(bpr_meta_path(tmp_path, "kuairec").read_text())["item_bias_file"] is False
