"""Request/response schemas. Ranges come from configs/data.yaml; unknown fields are rejected."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, create_model

from fairxai.config import Config
from fairxai.data.load import BILL_COLS, FEATURES, PAYAMT_COLS, column_range

CONTINUOUS = {"LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_features_model(cfg: Config) -> type[BaseModel]:
    fields = {}
    for col in FEATURES:
        lo, hi = column_range(col, cfg)
        typ = float if col in CONTINUOUS else int
        kw = {}
        if lo is not None:
            kw["ge"] = lo
        if hi is not None:
            kw["le"] = hi
        fields[col] = (typ, Field(..., strict=False, **kw))
    return create_model("Features", __config__=ConfigDict(extra="forbid"), **fields)


def build_request_models(cfg: Config):
    features = build_features_model(cfg)
    predict = create_model(
        "PredictRequest",
        __config__=ConfigDict(extra="forbid"),
        request_id=(uuid.UUID | None, None),
        features=(features, ...),
    )
    return features, predict


class LabelItem(Strict):
    request_id: uuid.UUID
    default: int = Field(..., ge=0, le=1)


class FeedbackRequest(Strict):
    labels: list[LabelItem] = Field(..., min_length=1, max_length=1000)


class LoadRequest(Strict):
    model_version: int = Field(..., ge=1)
