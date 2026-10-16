import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_tree


def test_repo_tree_ok():
    assert check_tree.check(Path(__file__).resolve().parents[2]) == []


def test_detects_extra_and_missing(tmp_path):
    (tmp_path / "rogue").mkdir()
    errs = check_tree.check(tmp_path)
    assert "unlisted top-level folder: rogue" in errs
    assert any(e.startswith("missing") for e in errs)
