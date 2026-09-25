"""Default datasets stay in sync: the runners, the launch scripts' study step, and their BPR step
(which builds every dataset with a recipe; the embeddings are stored, so it is a one-off)."""

import json
import re
import sys
from pathlib import Path

import pytest

from BPR.bpr_config import DEFAULT_CONFIG_PATH, DEFAULT_DATASETS

ROOT = Path(__file__).resolve().parent.parent
ALL_DATASETS = set(json.loads(DEFAULT_CONFIG_PATH.read_text()))
LAUNCH_SCRIPTS = ("run_from_scratch.sh", "run_clone_branch_install_run.sh", "run_from_scratch_slurm.sh")


def test_defaults_are_known_datasets():
    assert ALL_DATASETS == {"ml", "myket", "kuairec", "kuairand", "anime", "lastfm", "msd"}
    assert DEFAULT_DATASETS == ("ml", "myket", "kuairec", "kuairand", "anime", "msd")
    assert set(DEFAULT_DATASETS) <= ALL_DATASETS


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel", "run_h1_study", "characterize_world"])
def test_runner_default_datasets(module, monkeypatch, capsys):
    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    assert "Default: " + " ".join(DEFAULT_DATASETS) + "." in " ".join(capsys.readouterr().out.split())


def _script(name):
    return (ROOT / "scripts" / name).read_text()


@pytest.mark.parametrize("name", LAUNCH_SCRIPTS)
def test_launch_scripts_build_bpr_for_every_dataset(name):
    built = re.findall(r"^\s*run_bpr\s+(\w+)\s", _script(name), flags=re.M)
    assert set(built) == ALL_DATASETS, built
    assert len(built) - built.count("ml") == len(ALL_DATASETS) - 1  # ml may repeat in the smoke branch


def test_launch_scripts_study_default_datasets():
    text = _script("run_from_scratch.sh")
    assert re.search(r'STUDY_DATASETS="\$\{STUDY_DATASETS:-([a-z ]+)\}"', text).group(1).split() == list(DEFAULT_DATASETS)
    text = _script("run_h1_study.sh")
    assert re.search(r'DATASETS="\$\{DATASETS:-([a-z ]+)\}"', text).group(1).split() == list(DEFAULT_DATASETS)
    for name in ("run_clone_branch_install_run.sh", "run_from_scratch_slurm.sh"):
        lists = [m.split() for m in re.findall(r"--datasets\s+([a-z ]+?)\s*(?:\\)?$", _script(name), flags=re.M)]
        full = [ds for ds in lists if ds != ["ml"]]  # the smoke branch runs ml only
        assert full == [list(DEFAULT_DATASETS)], (name, lists)
