"""Reference fixture from the authors' unmodified CausE TensorFlow graph (criteo-research/CausE @ 957e556).

Runs ONLY in the standalone audit environment (Python 3.6, TensorFlow 1.9), outside OPC:

    ~/code/CausE/repro/.local/envs/cause-py36-tf19/bin/python scripts/cause_reference/make_tf_fixture.py \
        --cause_root ~/code/CausE --out tests/fixtures/cause_tf_reference.npz

For each case it builds the authors' model class (`src/models.py`; for the momentum cases the audit's
paper-described variant `repro/paper_variant/paper_models.py`), sets the initial scale alpha (the only
override), feeds a fixed sequence of tiny batches through `model.apply_grads` (the training op; the tf.data
loader is bypassed so the batch order is fixed), and saves the initial parameters, the per-step losses and
the final parameters. tests/test_cause_tf_reference.py replays the same batches with models/cause.py.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

N_USERS = 7
N_PRODUCTS = 6  # the --num_products flag; id 0 is the free id (CausE-avg's pooled product), items 1..5
DIM = 4
SEED = 123
BATCH_SIZES = [8, 8, 8, 5, 8, 8, 8, 5, 8, 8]
VARS = ["user_embeddings", "product_embeddings", "user_b", "prod_b", "global_bias", "alpha"]

CASES = [
    # name, variant, optimizer, lr, l2_pen, cf_pen, alpha0 (None = released 1e-8), symmetric
    ("prod_sgd_released", "prod", "sgd", 1.0, 0.0, 1.0, None, False),
    ("prod_sgd_alpha", "prod", "sgd", 0.5, 0.0, 1.0, 0.7, False),
    ("prod_sgd_l2", "prod", "sgd", 0.5, 1e-2, 3.0, 0.7, False),
    ("avg_sgd_released", "avg", "sgd", 1.0, 0.0, 1.0, None, False),
    ("avg_sgd_alpha", "avg", "sgd", 0.5, 0.0, 1.0, 0.7, False),
    ("avg_sgd_l2", "avg", "sgd", 0.5, 1e-2, 3.0, 0.7, False),
    ("prod_mom", "prod", "momentum_decay", 0.3, 0.0, 1.0, 0.7, False),
    ("prod_mom_sym", "prod", "momentum_decay", 0.3, 0.0, 1.0, 0.7, True),
    ("prod_mom_l2", "prod", "momentum_decay", 0.3, 1e-2, 1.0, 0.7, False),
    ("avg_mom", "avg", "momentum_decay", 0.3, 0.0, 1.0, 0.7, False),
    ("avg_mom_sym", "avg", "momentum_decay", 0.3, 0.0, 3.0, 0.7, True),
]


class Flags(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_batches(variant, seed):
    """Fixed tiny batches in the released file conventions (items 1..5; S_t rows: +N for prod, 0 for avg)."""
    rng = np.random.RandomState(seed)
    out = []
    for bs in BATCH_SIZES:
        users = rng.randint(0, N_USERS, size=bs).astype(np.int32)
        items = rng.randint(1, N_PRODUCTS, size=bs).astype(np.int32)
        treat = rng.rand(bs) < 0.3
        labels = (rng.rand(bs) < 0.4).astype(np.float32).reshape(-1, 1)
        if variant == "prod":
            products = np.where(treat, items + N_PRODUCTS, items).astype(np.int32)
        else:
            products = np.where(treat, 0, items).astype(np.int32)
        out.append((users, products, labels))
    return out


def run_case(case, cause_root):
    import tensorflow as tf

    sys.path.insert(0, os.path.join(cause_root, "src"))
    sys.path.insert(0, os.path.join(cause_root, "repro", "paper_variant"))
    import models  # authors' code, unmodified
    import utils as ut  # authors' code, unmodified

    name, variant, optimizer, lr, l2_pen, cf_pen, alpha0, symmetric = case
    batches = make_batches(variant, seed=7 if variant == "prod" else 11)
    flags = Flags(num_users=N_USERS, num_products=N_PRODUCTS * (2 if variant == "prod" else 1),
                  embedding_size=DIM, l2_pen=l2_pen, learning_rate=lr, plot_gradients=False,
                  cf_pen=cf_pen, cf_distance="l1", total_steps=len(batches), momentum=0.9,
                  symmetric_tie=symmetric)
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(SEED)
        if optimizer == "sgd":
            assert not symmetric, "the released classes have the one-way tie only"
            model = models.CausalProd2Vec2i(flags) if variant == "prod" else models.CausalProd2Vec(flags)
        else:
            import paper_models  # the audit's paper-described variant (momentum + linear decay, optional symmetric tie)

            model = (paper_models.PaperCausalProd2Vec2i(flags) if variant == "prod"
                     else paper_models.PaperCausalProd2Vec(flags))
        set_alpha = model.alpha.assign(alpha0) if alpha0 is not None else None
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            if set_alpha is not None:
                sess.run(set_alpha)
            tf_vars = [getattr(model, v) for v in ("user_embeddings", "product_embeddings", "user_b", "prod_b",
                                                    "global_bias", "alpha")]
            init = sess.run(tf_vars)
            # the tie's in-graph difference (before abs), to expose the subgradient signs TF actually uses
            tie_diff = graph.get_tensor_by_name("counter_factual/Sub:0")
            losses = []
            diffs = []
            for users, products, labels in batches:
                feed = {model.user_list_placeholder: users, model.product_list_placeholder: products,
                        model.label_list_placeholder: labels}
                if variant == "prod":
                    feed[model.reg_list_placeholder] = ut.compute_2i_regularization_id(products, N_PRODUCTS)
                d = sess.run(tie_diff, feed_dict=feed)  # the parameters before this step's update
                _, loss, log_loss = sess.run([model.apply_grads, model.loss, model.log_loss], feed_dict=feed)
                losses.append((loss, log_loss))
                diffs.append(d)
            final = sess.run(tf_vars)
    out = {"meta": np.array([variant, optimizer, str(lr), str(l2_pen), str(cf_pen), str(alpha0), str(symmetric)])}
    for v, a in zip(VARS, init):
        out["init_" + v] = np.asarray(a)
    for v, a in zip(VARS, final):
        out["final_" + v] = np.asarray(a)
    out["losses"] = np.asarray(losses, dtype=np.float64)
    for i, (u, p, y) in enumerate(batches):
        out["batch%d_tie_diff" % i] = np.asarray(diffs[i])
        out["batch%d_users" % i] = u
        out["batch%d_products" % i] = p
        out["batch%d_labels" % i] = y.reshape(-1)
    return {name + "/" + k: v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cause_root", default=os.path.expanduser("~/code/CausE"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import subprocess

    import tensorflow as tf

    commit = subprocess.check_output(["git", "-C", args.cause_root, "rev-parse", "HEAD"]).decode().strip()
    src_diff = subprocess.check_output(["git", "-C", args.cause_root, "diff", "--stat", "957e556", "--", "src"]).decode()
    if src_diff.strip():
        raise SystemExit("src/ differs from upstream 957e556; refusing to build the reference")
    data = {}
    for case in CASES:
        data.update(run_case(case, args.cause_root))
        print("done", case[0])
    data["provenance"] = np.array(["criteo-research/CausE src @ 957e556 (unmodified)", "checkout HEAD " + commit,
                                   "tensorflow " + tf.__version__, "python " + sys.version.split()[0],
                                   "cases: " + ",".join(c[0] for c in CASES)])
    np.savez_compressed(args.out, **data)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
