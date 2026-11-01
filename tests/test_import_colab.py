"""scripts/import_colab_outputs.py appends registry rows and refuses anything unsafe."""
import importlib.util
import zipfile

import pandas as pd
import pytest

from src.registry import COLUMNS, append_entry, read_registry

spec = importlib.util.spec_from_file_location("imp", "scripts/import_colab_outputs.py")
imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)


def make_zip(path, rows, files):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("registry_new_rows.csv", pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False))
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_appends_rows_and_extracts_files(tmp_path):
    reg = tmp_path / "trained_models" / "model_registry.csv"
    append_entry({"model_name": "xgboost_baseline", "version": 1, "subset": "FD001"}, reg)
    z = make_zip(tmp_path / "o.zip", [{"model_name": "tcn", "version": 1, "subset": "FD001", "test_rmse": 15.0}],
                 {"trained_models/tcn_v1/model.pt": b"w", "configs/tcn_best_params.json": b"{}"})
    imp.main(str(z), root=tmp_path, registry=reg)
    r = read_registry(reg)
    assert r["model_name"].tolist() == ["xgboost_baseline", "tcn"]            # appended, not replaced
    assert (tmp_path / "trained_models/tcn_v1/model.pt").read_bytes() == b"w"


@pytest.mark.parametrize("files,rows", [
    ({"src/evil.py": b"x"}, []),                                                  # outside allowed folders
    ({"trained_models/../../evil.txt": b"x"}, []),                                # path traversal
])
def test_refuses_unsafe_paths(tmp_path, files, rows):
    z = make_zip(tmp_path / "o.zip", rows, files)
    with pytest.raises(SystemExit):
        imp.main(str(z), root=tmp_path, registry=tmp_path / "reg.csv")


def test_refuses_existing_bundle_and_duplicate_rows(tmp_path):
    reg = tmp_path / "reg.csv"
    (tmp_path / "trained_models/tcn_v1").mkdir(parents=True)
    z = make_zip(tmp_path / "a.zip", [], {"trained_models/tcn_v1/model.pt": b"w"})
    with pytest.raises(SystemExit):
        imp.main(str(z), root=tmp_path, registry=reg)
    append_entry({"model_name": "tcn_tuned", "version": 1, "subset": "FD001"}, reg)
    z = make_zip(tmp_path / "b.zip", [{"model_name": "tcn_tuned", "version": 1, "subset": "FD001"}], {})
    with pytest.raises(SystemExit):
        imp.main(str(z), root=tmp_path, registry=reg)
