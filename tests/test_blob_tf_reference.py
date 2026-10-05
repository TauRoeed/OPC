"""models/blob.py against the authors' unmodified BLOB TensorFlow graph (criteo-research/blob @ e15cb38).

The fixture (scripts/blob_reference/make_tf_fixture.py, run in the standalone TF 1.15 audit environment) executes the
released graph-building block verbatim, feeds fixed user embeddings, actions and clicks in a fixed batch order, and
records each step's noise, losses and the final variables. Here the PyTorch port replays the same steps."""
from pathlib import Path

import numpy as np
import pytest
import torch

from models.blob import BlobBandit, BlobPriors, TFAdam, prepare_psi, tf_param_order

FIXTURE = Path(__file__).parent / "fixtures" / "blob_tf_reference.npz"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="TF reference fixture not generated")
FAMILY = {"mnq_released": "mnq", "nq_released": "nq", "mnq_nonorm": "mnq", "nq_wide_prior": "nq",
          "mnq_wide_prior": "mnq"}


@pytest.fixture(scope="module")
def ref():
    return np.load(FIXTURE, allow_pickle=False)


def _priors(vals) -> BlobPriors:
    wa_m, wb_m, wc_m, wa_s, wb_s, wc_s, kappa_s = (float(v) for v in vals)
    return BlobPriors(wa_m=wa_m, wb_m=wb_m, wc_m=wc_m, wa_s=wa_s, wb_s=wb_s, wc_s=wc_s, kappa_s=kappa_s)


def _replay(ref, case):
    pre = case + "/"
    model = BlobBandit(ref["psi"], family=FAMILY[case], priors=_priors(ref[pre + "prior"]), norm=bool(ref[pre + "norm"]))
    params = tf_param_order(model)
    names = [str(n) for n in ref[pre + "var_names"]]
    assert len(names) == len(params)
    for n, p in zip(names, params):  # the released initial values (deterministic: priors, zeros)
        np.testing.assert_allclose(p.detach().numpy().reshape(-1), ref[pre + "init/" + n].reshape(-1), rtol=0, atol=0)
    X = torch.as_tensor(ref["X"])
    A = torch.as_tensor(ref["A"].astype(np.int64))
    Y = torch.as_tensor(ref["Y"])
    noise = torch.as_tensor(ref[pre + "noise"])
    opt = TFAdam(params, lr=1e-3)
    n, batch = X.shape[0], int(ref["batch"])
    losses, row = [], 0
    for order in ref["orders"]:
        for s in range(0, n, batch):
            idx = torch.as_tensor(order[s:s + batch].astype(np.int64))
            b = idx.shape[0]
            nz = noise[row:row + b]
            row += b
            loss, _nll, _kl = model.neg_elbo(X[idx], A[idx], Y[idx], n, {"wa": nz[:, 0:1], "wb": nz[:, 1:2],
                                                                          "band": nz[:, 2:3], "bias": nz[:, 3:4]})
            losses.append(float(loss.detach()))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, params, names, np.array(losses)


@pytest.mark.parametrize("case", list(FAMILY))
def test_port_matches_the_released_graph_step_by_step(ref, case):
    pre = case + "/"
    model, params, names, losses = _replay(ref, case)
    np.testing.assert_allclose(losses, ref[pre + "losses"], rtol=2e-5, atol=1e-6)
    for n, p in zip(names, params):
        np.testing.assert_allclose(p.detach().numpy().reshape(-1), ref[pre + "final/" + n].reshape(-1), rtol=1e-4,
                                   atol=2e-6, err_msg=n)
    beta, kappa = model.point_beta()
    np.testing.assert_allclose(beta.numpy(), ref[pre + "beta"], rtol=1e-4, atol=2e-6)
    np.testing.assert_allclose(kappa.numpy(), ref[pre + "kappa"].reshape(-1), rtol=1e-4, atol=2e-6)


def test_released_normalization_aliases_the_prior_mean(ref):
    """With norm the released graph divides Ψ's columns in place, so the prior mean uses the normalized Ψ too."""
    for case in ("mnq_released", "mnq_nonorm"):
        pre = case + "/"
        loc, cov, chol = prepare_psi(ref["psi"], norm=bool(ref[pre + "norm"]))
        np.testing.assert_allclose(loc, ref[pre + "psi_loc"], rtol=1e-6)
        np.testing.assert_allclose(cov, ref[pre + "psi_cov"], rtol=1e-6)
        np.testing.assert_allclose(chol, ref[pre + "L"], rtol=1e-5, atol=1e-7)
    assert not np.allclose(ref["mnq_released/psi_loc"], ref["psi"])  # the aliasing is real
    np.testing.assert_allclose(ref["mnq_nonorm/psi_loc"], ref["psi"])


def test_fixture_provenance_is_the_unmodified_release(ref):
    prov = " ".join(str(x) for x in ref["provenance"])
    assert "e15cb38616a4b29a8ae8b8828975cd98dc62ea50" in prov
    assert "sha256 d2110c7f946f729ce74134c513a160c6ca43d30aec1a952382c306c46c7468d4" in prov
    assert "models dirty: no" in prov and "tensorflow 1.15" in prov
