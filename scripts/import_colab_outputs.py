"""Bring a Colab training session's outputs into the local repo (docs/COLAB.md, step 8).

    python scripts/import_colab_outputs.py path/to/colab_outputs_<stamp>.zip

The zip (built by notebooks/colab_train.ipynb) contains:
  trained_models/<bundle dirs>/...   configs/*.json   results/*.png|csv   registry_new_rows.csv
Bundles, configs and results are extracted in place; the registry rows are APPENDED (never
overwriting local rows). Refuses: paths outside those folders, bundles that already exist
locally, and registry rows whose (model_name, version) already exists.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.registry import COLUMNS, REGISTRY_PATH, append_entry, read_registry  # noqa: E402

ALLOWED_TOP = {"trained_models", "configs", "results"}
ROWS_FILE = "registry_new_rows.csv"


def main(zip_path: str, root: Path = ROOT, registry: Path = REGISTRY_PATH) -> None:
    zf = zipfile.ZipFile(zip_path)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    files = [n for n in names if n != ROWS_FILE]
    bad = [n for n in files if Path(n).is_absolute() or ".." in Path(n).parts or Path(n).parts[0] not in ALLOWED_TOP]
    if bad:
        sys.exit(f"refusing: unexpected paths in zip: {bad[:5]}")
    bundles = {Path(n).parts[1] for n in files if Path(n).parts[0] == "trained_models" and len(Path(n).parts) > 2}
    clash = [b for b in bundles if (root / "trained_models" / b).exists()]
    if clash:
        sys.exit(f"refusing: bundles already exist locally: {clash}")

    rows = pd.read_csv(io.BytesIO(zf.read(ROWS_FILE))) if ROWS_FILE in names else pd.DataFrame(columns=COLUMNS)
    local = read_registry(registry)
    have = set(zip(local["model_name"], local["version"].astype(int))) if len(local) else set()
    dup = [(r.model_name, int(r.version)) for r in rows.itertuples() if (r.model_name, int(r.version)) in have]
    if dup:
        sys.exit(f"refusing: registry already has {dup} - rename/re-version on Colab first")

    for n in files:
        target = root / n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(n))
    for r in rows.to_dict("records"):
        append_entry({k: ("" if pd.isna(v) else v) for k, v in r.items() if k in COLUMNS}, registry)
    print(f"imported {len(files)} files, bundles {sorted(bundles)}, {len(rows)} registry rows -> {registry}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
