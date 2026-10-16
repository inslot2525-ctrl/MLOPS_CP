import pandas as pd

from fairxai.config import load_config
from fairxai.data.split import BATCH_ROWS, N_BATCHES, run, split_frame

CFG = load_config()


def test_sizes_and_disjoint(synthetic_clean):
    df = synthetic_clean.reset_index(drop=True)
    parts = split_frame(df, CFG)
    assert {k: len(v) for k, v in parts.items()} == {
        "train": 18000,
        "eval": 4500,
        "test": 3000,
        "stream": 4500,
    }
    idx = [set(p.index) for p in parts.values()]
    assert set.union(*idx) == set(df.index)
    assert sum(len(i) for i in idx) == len(df)  # no overlap


def test_stratified_on_target_and_sex(synthetic_clean):
    df = synthetic_clean.reset_index(drop=True)
    parts = split_frame(df, CFG)
    key = df.default.astype(str) + df.SEX.astype(str)
    overall = key.value_counts(normalize=True)
    for name, p in parts.items():
        share = key.loc[p.index].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.01, name


def test_deterministic(synthetic_clean):
    a = split_frame(synthetic_clean, CFG)["eval"].index
    b = split_frame(synthetic_clean, CFG)["eval"].index
    assert list(a) == list(b)


def test_run_writes_nine_batches(synthetic_clean, tmp_path):
    clean = tmp_path / "clean.csv"
    synthetic_clean.to_csv(clean, index=False)
    run(str(clean), str(tmp_path / "p"), str(tmp_path / "s"), CFG)
    files = sorted((tmp_path / "s").glob("batch_*.csv"))
    assert len(files) == N_BATCHES
    assert all(len(pd.read_csv(f)) == BATCH_ROWS for f in files)
    assert (tmp_path / "p" / "reference_stats.json").exists()
