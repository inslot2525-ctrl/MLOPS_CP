"""split stage (N-06): stratified on target x SEX, seed 42; train/eval/test/stream, stream in 9 batches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fairxai.config import Config, load_config
from fairxai.data.load import CONTINUOUS

N_BATCHES = 9
BATCH_ROWS = 500
FRACTIONS = {"train": 0.60, "eval": 0.15, "test": 0.10, "stream": 0.15}


def split_frame(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    seed, target = cfg.base.seed, cfg.data.target
    n = len(df)
    sizes = {k: round(n * v) for k, v in FRACTIONS.items()}
    strat = df[target].astype(str) + "_" + df["SEX"].astype(str)
    rest, test = train_test_split(df, test_size=sizes["test"], stratify=strat, random_state=seed)
    strat = rest[target].astype(str) + "_" + rest["SEX"].astype(str)
    rest, stream = train_test_split(rest, test_size=sizes["stream"], stratify=strat, random_state=seed)
    strat = rest[target].astype(str) + "_" + rest["SEX"].astype(str)
    train, ev = train_test_split(rest, test_size=sizes["eval"], stratify=strat, random_state=seed)
    return {"train": train, "eval": ev, "test": test, "stream": stream}


def reference_stats(train: pd.DataFrame, cfg: Config) -> dict:
    """IQR bounds per continuous column and f_ref (flagged-cell fraction) on the train partition."""
    k = cfg.data.outlier.iqr_multiplier
    cols: dict[str, dict[str, float]] = {}
    flagged = 0
    for c in CONTINUOUS:
        q1, q3 = float(train[c].quantile(0.25)), float(train[c].quantile(0.75))
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        cols[c] = {"q1": q1, "q3": q3, "lo": lo, "hi": hi}
        flagged += int(((train[c] < lo) | (train[c] > hi)).sum())
    return {"columns": cols, "f_ref": flagged / (len(train) * len(CONTINUOUS)), "n_train": len(train)}


def run(clean_csv: str, processed_dir: str, stream_dir: str, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    parts = split_frame(pd.read_csv(clean_csv), cfg)
    out, sdir = Path(processed_dir), Path(stream_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "eval", "test"):
        parts[name].to_csv(out / f"{name}.csv", index=False)
    stream = parts["stream"]
    assert len(stream) == N_BATCHES * BATCH_ROWS, f"stream has {len(stream)} rows"
    for i in range(N_BATCHES):
        stream.iloc[i * BATCH_ROWS : (i + 1) * BATCH_ROWS].to_csv(
            sdir / f"batch_{i + 1:04d}.csv", index=False
        )
    stats = reference_stats(parts["train"], cfg)
    (out / "reference_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {k: len(v) for k, v in parts.items()} | {"f_ref": stats["f_ref"]}


if __name__ == "__main__":
    print(run("data/processed/clean.csv", "data/processed", "data/stream"))
