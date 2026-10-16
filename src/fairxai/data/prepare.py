"""prepare stage: raw canonical CSV -> clean.csv (drops ID only; no category remap, SPEC D.2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairxai.config import Config, load_config
from fairxai.data.load import canonicalize


def prepare(raw_csv: str | Path, out_csv: str | Path, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = canonicalize(pd.read_csv(raw_csv), cfg)
    clean = df.drop(columns=cfg.data.drop_columns)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    return clean


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1], sys.argv[2])
