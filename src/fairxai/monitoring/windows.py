"""Windows are defined by prediction-log order (the Taiwan data has no timestamp; A-10)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fairxai.data.load import FEATURES


@dataclass
class Window:
    model_version: int | None
    window_end: object | None  # ts of the newest prediction in the window
    frame: pd.DataFrame  # FEATURES + request_id, ts, prediction, probability, label (NaN if unlabelled)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def labelled(self) -> pd.DataFrame:
        return self.frame[self.frame["label"].notna()]


def _to_frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["request_id", "ts", "model_version", "prediction", "probability", "features", "label"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=[*FEATURES, *cols[:5], "label"])
    feats = pd.DataFrame(list(df["features"]))[FEATURES]
    return pd.concat([feats, df.drop(columns=["features"])], axis=1)


def latest_model_version(conn) -> int | None:
    row = conn.execute(
        "SELECT model_version FROM predictions ORDER BY ts DESC, request_id LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else None


def prediction_window(conn, n_rows: int, model_version: int | None = None) -> Window:
    """The last `n_rows` logged predictions of a model version (default: the most recent version)."""
    version = model_version if model_version is not None else latest_model_version(conn)
    rows = conn.execute(
        "SELECT p.request_id, p.ts, p.model_version, p.prediction, p.probability, p.features, l.label "
        "FROM predictions p LEFT JOIN labels l USING (request_id) WHERE p.model_version = %s "
        "ORDER BY p.ts DESC, p.request_id LIMIT %s",
        (version, n_rows),
    ).fetchall()
    frame = _to_frame(rows)
    return Window(version, frame["ts"].max() if len(frame) else None, frame)


def labelled_window(conn, n_rows: int, model_version: int) -> pd.DataFrame:
    """The most recent `n_rows` labelled predictions of a model version."""
    rows = conn.execute(
        "SELECT p.request_id, p.ts, p.model_version, p.prediction, p.probability, p.features, l.label "
        "FROM predictions p JOIN labels l USING (request_id) WHERE p.model_version = %s "
        "ORDER BY p.ts DESC, p.request_id LIMIT %s",
        (model_version, n_rows),
    ).fetchall()
    return _to_frame(rows)
