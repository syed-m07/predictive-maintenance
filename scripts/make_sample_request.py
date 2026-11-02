"""Write scripts/sample_request.json: the raw readings of one real FD001 test engine.

    python -m scripts.make_sample_request [unit_id]      # default: engine 76 (true RUL 10)

Then, with the API running:
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
         -d @scripts/sample_request.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import RAW_FEATURE_COLS
from src.data_loader import load_cmapss, load_rul_truth

OUT = Path(__file__).resolve().parent / "sample_request.json"


def build(unit: int = 76) -> dict:
    test = load_cmapss("FD001", "test")
    eng = test[test["unit_id"] == unit].sort_values("cycle")
    if eng.empty:
        raise SystemExit(f"FD001 test has no engine {unit}")
    truth = int(load_rul_truth("FD001")[unit - 1])
    return {"unit_id": f"FD001-test-{unit} (true RUL {truth})",
            "readings": eng[RAW_FEATURE_COLS].round(4).values.tolist()}


if __name__ == "__main__":
    req = build(int(sys.argv[1]) if len(sys.argv) > 1 else 76)
    OUT.write_text(json.dumps(req))
    print(f"wrote {OUT}: {req['unit_id']}, {len(req['readings'])} cycles x {len(req['readings'][0])} values")
