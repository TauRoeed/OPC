"""Reference fixture from the authors' unmodified BLOB TensorFlow graph (criteo-research/blob @ e15cb38).

Runs ONLY in the standalone audit environment (Python 3.6, TensorFlow 1.15.2, tensorflow-probability 0.7), outside OPC:

    ~/code/BLOB_audit/envs/blob-py36/bin/python scripts/blob_reference/make_tf_fixture.py \
        --blob_root ~/code/BLOB --out tests/fixtures/blob_tf_reference.npz

For each case it executes, verbatim, the graph-building block of the authors' ``do_training`` (from
``K, P = self.K, self.P`` to ``init_op = tf.global_variables_initializer()`` in models/models_organic_bandit.py, for
``RecoModelRTAEWithBanditTF`` = BLOB-MNQ and ``RecoModelRTAEWithBanditTF_Full`` = BLOB-NQ). Nothing in the graph is
changed. The released loop's DataLoader and organic encoder are bypassed: fixed user embeddings X, actions and clicks
are fed in a fixed batch order, and the four noise tensors of each step are fetched with ``train_op``. The point
estimate is evaluated with the authors' own expression. tests/test_blob_tf_reference.py replays the same steps with
models/blob.py.
"""
from __future__ import print_function

import argparse
import hashlib
import inspect
import os
import subprocess
import sys
import textwrap
import types

import numpy as np

P, K, N = 12, 4, 200
BATCH = 64
EPOCHS = 3
SEED = 7

CASES = [
    # name, class, norm, prior overrides
    ("mnq_released", "RecoModelRTAEWithBanditTF", True, {}),
    ("nq_released", "RecoModelRTAEWithBanditTF_Full", True, {}),
    ("mnq_nonorm", "RecoModelRTAEWithBanditTF", False, {}),
    ("nq_wide_prior", "RecoModelRTAEWithBanditTF_Full", True, {"wb_m": 0.0, "kappa_s": 0.5, "wc_m": -1.0}),
    ("mnq_wide_prior", "RecoModelRTAEWithBanditTF", True, {"wb_m": 0.0, "kappa_s": 0.5, "wc_m": -1.0}),
]
RELEASED = dict(wa_m=-1.0, wb_m=-6.0, wc_m=-4.5, wa_s=1.0, wb_s=1.0, wc_s=10.0, kappa_s=0.01)


def graph_block(cls):
    """The verbatim graph-building source lines of ``cls.do_training``."""
    src = inspect.getsource(cls.do_training).splitlines()
    start = next(i for i, l in enumerate(src) if l.strip().replace(" ", "") == "K,P=self.K,self.P")
    end = next(i for i, l in enumerate(src) if "init_op = tf.global_variables_initializer()" in l)
    block = src[start:end + 1]
    base = len(block[0]) - len(block[0].lstrip())  # the method body's indentation
    out = []
    for line in block:  # the released block has one comment line indented one space less (``# tf.reset_...``)
        if line[:base].strip() == "":
            out.append(line[base:])
        elif line.lstrip().startswith("#"):
            out.append(line.lstrip())
        else:
            raise ValueError("unexpected indentation in the released block: %r" % line)
    return "\n".join(out)


def point_estimate_exprs(cls):
    """The authors' point-estimate expressions (the arguments of the two sess.run calls after training)."""
    src = inspect.getsource(cls.do_training)
    beta_line = src[src.index("self.p_beta = torch.Tensor(sess.run(") + len("self.p_beta = torch.Tensor(sess.run("):]
    depth, out = 1, []
    for ch in beta_line:  # up to the matching ')' of sess.run(
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return " ".join("".join(out).split()), "kappa_means"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.blob_root)
    import tensorflow as tf
    import torch
    from tensorflow_probability import distributions as tfd
    from tensorflow.math import softplus as sp
    import models.models_organic_bandit as mob

    rng = np.random.RandomState(SEED)
    psi = rng.rand(P, K).astype(np.float32)  # like the released torch.rand initialization of Psi
    X = rng.randn(N, K).astype(np.float32)
    A = rng.randint(0, P, size=N).astype(np.int32)
    true_logit = 0.8 * (X * psi[A]).sum(1) - 1.0
    Y = (rng.rand(N) < 1.0 / (1.0 + np.exp(-true_logit))).astype(np.float32)
    orders = [rng.permutation(N) for _ in range(EPOCHS)]

    data = {"psi": psi, "X": X, "A": A, "Y": Y, "orders": np.stack(orders), "batch": np.array(BATCH),
            "cases": np.array([c[0] for c in CASES])}
    for name, cls_name, norm, override in CASES:
        cls = getattr(mob, cls_name)
        block = graph_block(cls)
        beta_expr, kappa_expr = point_estimate_exprs(cls)
        tf.reset_default_graph()
        tf.set_random_seed(SEED)
        prior = dict(RELEASED, **override)
        ns = {"tf": tf, "tfd": tfd, "sp": sp, "np": np, "KL_multivariate": mob.KL_multivariate,
              "self": types.SimpleNamespace(K=K, P=P, p_Psi=torch.tensor(psi.copy()),
                                            args=dict(norm=norm, **prior)),
              "dataloader_bandit": types.SimpleNamespace(dataset=list(range(N)))}
        exec(compile(block, "<released graph block %s>" % cls_name, "exec"), ns)
        fetch_noise = [ns["wa_noise"], ns["wb_noise"], ns["band_noise"], ns["bias_noise"]]
        var_names = [v.name.split(":")[0] for v in tf.global_variables() if "Adam" not in v.name and "power" not in v.name]
        with tf.Session() as sess:
            sess.run(ns["init_op"])
            init_vals = {v.name.split(":")[0]: sess.run(v) for v in tf.global_variables() if v.name.split(":")[0] in var_names}
            losses, nlls, kls, noises, batches = [], [], [], [], []
            for order in orders:
                for s in range(0, N, BATCH):
                    idx = order[s:s + BATCH]
                    out = sess.run([ns["train_op"], ns["neg_ELBO"], ns["neg_log_prob"], ns["kl_div"]] + fetch_noise,
                                   feed_dict={ns["X"]: X[idx], ns["Y"]: Y[idx].reshape(-1, 1), ns["A"]: A[idx]})
                    losses.append(out[1]); nlls.append(out[2]); kls.append(out[3])
                    noises.append(np.concatenate([o.reshape(-1, 1) for o in out[4:8]], axis=1))
                    batches.append(idx)
            final_vals = {n: sess.run(tf.get_default_graph().get_tensor_by_name(n + ":0")) for n in var_names}
            beta = sess.run(eval(beta_expr, ns))
            kappa = sess.run(ns[kappa_expr])
            psi_loc, psi_cov, L = sess.run([ns["tf_psi_loc"], ns["tf_psi_cov"], ns["tf_L"]])
        pre = name + "/"
        data[pre + "var_names"] = np.array(var_names)
        for n in var_names:
            data[pre + "init/" + n] = init_vals[n]
            data[pre + "final/" + n] = final_vals[n]
        data[pre + "losses"] = np.array(losses, dtype=np.float64)
        data[pre + "nlls"] = np.array(nlls, dtype=np.float64)
        data[pre + "kls"] = np.array(kls, dtype=np.float64)
        data[pre + "noise"] = np.concatenate(noises, axis=0).astype(np.float32)  # rows in step order; cols wa, wb, band, bias
        data[pre + "beta"], data[pre + "kappa"] = beta, kappa
        data[pre + "psi_loc"], data[pre + "psi_cov"], data[pre + "L"] = psi_loc, psi_cov, L
        data[pre + "norm"] = np.array(norm)
        data[pre + "prior"] = np.array([prior[k] for k in ("wa_m", "wb_m", "wc_m", "wa_s", "wb_s", "wc_s", "kappa_s")])
        print(name, "steps", len(losses), "loss first/last", losses[0], losses[-1], "beta norm", np.linalg.norm(beta))

    src_file = os.path.join(args.blob_root, "models", "models_organic_bandit.py")
    sha = hashlib.sha256(open(src_file, "rb").read()).hexdigest()
    commit = subprocess.check_output(["git", "-C", args.blob_root, "rev-parse", "HEAD"]).decode().strip()
    dirty = subprocess.check_output(["git", "-C", args.blob_root, "status", "--porcelain", "--", "models"]).decode().strip()
    data["provenance"] = np.array(["criteo-research/blob models/models_organic_bandit.py (unmodified)",
                                   "checkout HEAD " + commit, "sha256 " + sha, "models dirty: " + (dirty or "no"),
                                   "tensorflow " + tf.__version__])
    np.savez_compressed(args.out, **data)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
