import pytest

from fairxai.config import load_config
from fairxai.data.ingest import MemoryBatchStore, ingest_batch

CFG = load_config()


@pytest.fixture
def good(synthetic_clean, tmp_path):
    p = tmp_path / "good.csv"
    synthetic_clean.head(500).to_csv(p, index=False)
    return p


def test_accept_and_idempotent(good, tmp_path):
    store, root = MemoryBatchStore(), tmp_path / "ing"
    assert ingest_batch(good, "b1", store, root, CFG, do_notify=False) == "ACCEPTED"
    assert (root / "accepted" / "b1.csv").exists()
    assert ingest_batch(good, "b1-again", store, root, CFG, do_notify=False) == "SKIPPED"
    assert not (root / "accepted" / "b1-again.csv").exists()


def test_rejected_never_reaches_accepted(synthetic_clean, tmp_path):
    bad = tmp_path / "bad.csv"
    synthetic_clean.head(500).drop(columns=["AGE"]).to_csv(bad, index=False)
    store, root = MemoryBatchStore(), tmp_path / "ing"
    assert ingest_batch(bad, "bad1", store, root, CFG, do_notify=False) == "REJECTED"
    assert not list((root / "accepted").glob("*"))
    assert (root / "rejected" / "bad1.csv").exists()
    assert (root / "reports" / "bad1.json").exists()
    assert store.rows["bad1"]["status"] == "REJECTED"
    # a rejected batch is retried, not skipped
    assert ingest_batch(bad, "bad1", store, root, CFG, do_notify=False) == "REJECTED"


def test_unreadable_batch_rejected(tmp_path):
    p = tmp_path / "junk.csv"
    p.write_bytes(b"")
    store = MemoryBatchStore()
    assert ingest_batch(p, "junk", store, tmp_path / "ing", CFG, do_notify=False) == "REJECTED"
