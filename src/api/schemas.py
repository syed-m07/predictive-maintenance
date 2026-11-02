"""Request and response shapes of the RUL API (pydantic validates every request against them).

Input = what a monitoring system would actually have: the RAW readings of one engine, one row
per cycle in time order, each row the 24 values [op1, op2, op3, s1 ... s21] exactly as in the
C-MAPSS files. The server applies the training pipeline itself (drop columns -> the model's own
scaler -> 238 features or the last 30-cycle window), so a client can never send features built
differently from training.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

from src.config import RAW_FEATURE_COLS, SEQ_LEN

N_RAW = len(RAW_FEATURE_COLS)       # 24
MAX_ROWS = 5000                     # longest FD001-FD004 engine is 543 cycles; caps request size


class SensorWindow(BaseModel):
    unit_id: str | None = Field(None, description="Optional engine id, echoed back.")
    readings: list[list[float]] = Field(
        ..., min_length=1, max_length=MAX_ROWS,
        description=f"One row per cycle, oldest first; each row = {N_RAW} raw values "
                    f"[{', '.join(RAW_FEATURE_COLS[:4])}, ..., {RAW_FEATURE_COLS[-1]}]. "
                    f">= {SEQ_LEN} rows recommended: shorter histories are padded with the first "
                    "reading (as for short test engines) and predictions are less reliable.")

    @field_validator("readings")
    @classmethod
    def rows_are_complete_and_finite(cls, rows: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(rows):
            if len(row) != N_RAW:
                raise ValueError(f"row {i} has {len(row)} values; expected {N_RAW} ({', '.join(RAW_FEATURE_COLS)})")
            if not all(math.isfinite(x) for x in row):
                raise ValueError(f"row {i} contains NaN or infinity")
        return rows


class PredictionResponse(BaseModel):
    predicted_rul: float = Field(..., description="Remaining useful life in cycles, clipped to [0, 125].")
    model_name: str
    model_version: str
    timestamp: str = Field(..., description="UTC time of the prediction, ISO 8601.")
    unit_id: str | None = None
    input_cycles: int = Field(..., description="Number of rows received.")
    padded: bool = Field(..., description=f"True if fewer than {SEQ_LEN} rows were sent.")
    latency_ms: float = Field(..., description="Server time for preprocessing + prediction.")


class CompareResponse(BaseModel):
    unit_id: str | None = None
    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    input_kind: str
    is_default: bool
    trained_on: str
    test_rmse: float | None = None
    test_rmse_clip125: float | None = None
    nasa_score: float | None = None
    git_commit: str | None = None
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    models: list[str]
    default_model: str | None
