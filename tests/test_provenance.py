"""Every run records the code that produced it (utils/provenance.py)."""
import re

import utils.provenance as provenance


def test_code_commit_names_this_checkout():
    out = provenance.code_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", out["commit"]) and isinstance(out["dirty"], bool)


def test_code_commit_outside_a_checkout_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_ROOT", tmp_path)
    assert provenance.code_commit() == {"commit": None, "dirty": None}
