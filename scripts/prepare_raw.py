"""Convert the team's raw file (Kaggle CSV or UCI .xls) to the canonical CSV data/raw/credit_default.csv."""

from __future__ import annotations

import sys
from pathlib import Path

from fairxai.config import load_config
from fairxai.data.load import canonicalize, read_raw


def main(src: str, dst: str = "data/raw/credit_default.csv") -> None:
    df = canonicalize(read_raw(src), load_config())
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"wrote {dst}: {df.shape[0]} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main(*sys.argv[1:])
