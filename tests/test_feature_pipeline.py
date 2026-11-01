"""Tests for src/feature_engineering.py - correctness AND the three leakage rules:
  1. no engine in both train and validation,
  2. the scaler only learns from training engines,
  3. every feature/window at cycle t uses only cycles <= t of the same engine.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_RAW_DIR, MAX_RUL, SENSORS, SEQ_LEN
from src.feature_engineering import (apply_scaler, build_tabular_features, clip_rul, fit_scaler,
                                     generate_sequences, last_row_per_engine,
                                     last_window_per_engine, load_scaler, pad_to_length,
                                     prepare_data, save_scaler, split_by_engine,
                                     tabular_feature_names)

needs_data = pytest.mark.skipif(
    not (DATA_RAW_DIR / "train_FD001.txt").exists(), reason="raw C-MAPSS data not downloaded"
)
SENS = ["s2", "s3"]


def toy(lengths=(40, 35), seed=0):
    """Engines with random-walk sensors; rul = last cycle - cycle."""
    rng = np.random.default_rng(seed)
    parts = []
    for u, n in enumerate(lengths, start=1):
        d = pd.DataFrame({"unit_id": u, "cycle": np.arange(1, n + 1)})
        for s in SENS:
            d[s] = rng.normal(size=n).cumsum()
        d["rul"] = n - d["cycle"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ labels & split
def test_clip_rul_caps_only_above_max():
    df = pd.DataFrame({"rul": [300, 126, 125, 50, 0]})
    assert clip_rul(df)["rul"].tolist() == [125, 125, 125, 50, 0]
    assert df["rul"].tolist() == [300, 126, 125, 50, 0]                    # input untouched


def test_split_by_engine_is_disjoint_complete_and_deterministic():
    df = toy(lengths=[30] * 50)
    tr, va = split_by_engine(df, 0.2, seed=42)
    tr_u, va_u = set(tr.unit_id), set(va.unit_id)
    assert tr_u.isdisjoint(va_u)                                             # leakage rule 1
    assert tr_u | va_u == set(df.unit_id) and len(va_u) == 10
    assert len(tr) + len(va) == len(df)
    tr2, va2 = split_by_engine(df, 0.2, seed=42)
    assert set(va2.unit_id) == va_u                                          # same seed, same split
    assert set(split_by_engine(df, 0.2, seed=7)[1].unit_id) != va_u


# ------------------------------------------------------------------ scaler
def test_scaler_learns_from_training_rows_only():
    train = pd.DataFrame({"s2": [0.0, 10.0], "s3": [5.0, 15.0]})
    val = pd.DataFrame({"s2": [100.0], "s3": [-50.0]})                      # extreme, unseen
    sc = fit_scaler(train, ["s2", "s3"])
    assert sc.data_min_.tolist() == [0.0, 5.0] and sc.data_max_.tolist() == [10.0, 15.0]  # rule 2
    assert apply_scaler(val, sc)["s2"].iloc[0] == pytest.approx(10.0)       # outside [0,1] is fine
    tr = apply_scaler(train, sc)
    assert tr.min().min() == 0.0 and tr.max().max() == 1.0


def test_scaler_roundtrip(tmp_path):
    sc = fit_scaler(toy(), SENS)
    save_scaler(sc, tmp_path / "scaler.joblib")
    sc2 = load_scaler(tmp_path / "scaler.joblib")
    x = toy(seed=1)
    pd.testing.assert_frame_equal(apply_scaler(x, sc), apply_scaler(x, sc2))


# ------------------------------------------------------------------ tabular features
def test_feature_names_count_and_no_nans():
    f = build_tabular_features(toy(), sensors=SENS)
    names = tabular_feature_names(SENS)
    assert len(names) == len(SENS) * 17 and len(tabular_feature_names()) == 238
    assert list(f.columns) == ["unit_id", "cycle", "rul"] + names
    assert not f.isna().any().any()


def test_rolling_lag_ema_values_by_hand():
    df = pd.DataFrame({"unit_id": 1, "cycle": [1, 2, 3, 4], "rul": [3, 2, 1, 0],
                       "s2": [1.0, 2.0, 4.0, 8.0]})
    f = build_tabular_features(df, sensors=["s2"], windows=[2], lags=[1, 3], alpha=0.5)
    assert f["s2_rollmean_2"].tolist() == [1.0, 1.5, 3.0, 6.0]
    assert f["s2_rollmax_2"].tolist() == [1.0, 2.0, 4.0, 8.0]
    assert f["s2_rollstd_2"].iloc[0] == 0.0                                   # one value -> 0
    assert f["s2_lag_1"].tolist() == [1.0, 1.0, 2.0, 4.0]                     # first value fills
    assert f["s2_lag_3"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert f["s2_ema"].iloc[1] == pytest.approx((2.0 + 0.5 * 1.0) / 1.5)     # adjusted EWM


def test_engines_never_mix():
    df = toy()
    f = build_tabular_features(df, sensors=SENS)
    first_of_2 = f[f.unit_id == 2].iloc[0]
    raw = df[df.unit_id == 2].iloc[0]
    for s in SENS:                       # at an engine's first cycle every stat = its own value
        assert first_of_2[f"{s}_rollmean_30"] == pytest.approx(raw[s])
        assert first_of_2[f"{s}_lag_10"] == pytest.approx(raw[s])
        assert first_of_2[f"{s}_ema"] == pytest.approx(raw[s])


def test_tabular_features_are_past_only():
    """Rule 3: changing FUTURE cycles must not change any feature at earlier cycles."""
    df = toy()
    k = 20
    future = (df.unit_id == 1) & (df.cycle > k)
    altered = df.copy()
    altered.loc[future, SENS] += 1000.0
    a = build_tabular_features(df, sensors=SENS)
    b = build_tabular_features(altered, sensors=SENS)
    past = (df.unit_id == 1) & (df.cycle <= k)
    pd.testing.assert_frame_equal(a[past], b[past])
    assert not a[future].equals(b[future])                                  # sanity: it did change


def test_last_30_rows_give_same_features_as_full_history():
    """Serving consistency: the API receives a recent window, not the whole life.

    Rolling (<=30), lag (<=10) and raw features depend only on the last 30 cycles, so they
    must match exactly; EMA has an infinitely long but geometrically vanishing memory
    (0.7^30 ~ 2e-5), so it must match closely.
    """
    df = toy(lengths=(120,))
    full_last = build_tabular_features(df, sensors=SENS).iloc[-1]
    tail = df.tail(SEQ_LEN).copy()
    tail_last = build_tabular_features(tail, sensors=SENS).iloc[-1]
    exact = [c for c in tabular_feature_names(SENS) if not c.endswith("_ema")]
    np.testing.assert_allclose(full_last[exact].astype(float), tail_last[exact].astype(float), rtol=1e-9, atol=1e-9)
    ema = [c for c in tabular_feature_names(SENS) if c.endswith("_ema")]
    scale = df[SENS].abs().max().max()
    np.testing.assert_allclose(full_last[ema].astype(float), tail_last[ema].astype(float), atol=1e-3 * scale)


def test_last_row_per_engine():
    f = build_tabular_features(toy(), sensors=SENS)
    last = last_row_per_engine(f)
    assert last.unit_id.tolist() == [1, 2] and last.cycle.tolist() == [40, 35]


# ------------------------------------------------------------------ sequences
def test_generate_sequences_shapes_targets_and_content():
    df = toy(lengths=(40, 35))
    X, y, u = generate_sequences(df, seq_len=30, feature_cols=SENS)
    assert X.shape == ((40 - 29) + (35 - 29), 30, 2) and X.dtype == np.float32
    assert (u[:11] == 1).all() and (u[11:] == 2).all()                        # no cross-engine window
    e1 = df[df.unit_id == 1]
    np.testing.assert_allclose(X[0], e1[SENS].iloc[:30].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(X[10], e1[SENS].iloc[10:40].to_numpy(), rtol=1e-6)
    assert y[0] == e1.rul.iloc[29] and y[10] == 0                           # RUL at window's last cycle


def test_sequences_are_past_only():
    df = toy(lengths=(40,))
    altered = df.copy()
    altered.loc[altered.cycle > 35, SENS] += 1000.0
    X, _, _ = generate_sequences(df, 30, SENS)
    Xa, _, _ = generate_sequences(altered, 30, SENS)
    np.testing.assert_array_equal(X[:6], Xa[:6])                             # windows ending <= cycle 35


def test_pad_to_length_repeats_first_row():
    d = np.arange(10, dtype=np.float32).reshape(5, 2)
    p = pad_to_length(d, 8)
    assert p.shape == (8, 2)
    np.testing.assert_array_equal(p[:3], np.repeat(d[:1], 3, axis=0))
    np.testing.assert_array_equal(p[3:], d)
    np.testing.assert_array_equal(pad_to_length(d, 3), d[-3:])               # long -> last rows


def test_last_window_per_engine_uses_last_cycles_and_pads_short_engines():
    df = toy(lengths=(40, 12))
    X, units = last_window_per_engine(df, 30, SENS)
    assert X.shape == (2, 30, 2) and units.tolist() == [1, 2]
    np.testing.assert_allclose(X[0], df[df.unit_id == 1][SENS].iloc[-30:].to_numpy(), rtol=1e-6)
    short = df[df.unit_id == 2][SENS].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(X[1][:18], np.repeat(short[:1], 18, axis=0))
    np.testing.assert_array_equal(X[1][18:], short)


# ------------------------------------------------------------------ real data
@needs_data
def test_prepare_data_fd001_end_to_end():
    d = prepare_data("FD001")
    assert set(d.train_units).isdisjoint(d.val_units) and len(d.val_units) == 20
    assert len(d.train_units) + len(d.val_units) == 100
    assert d.train.rul.max() == MAX_RUL and d.val.rul.max() == MAX_RUL
    assert "rul" not in d.test.columns
    assert d.train[SENSORS].min().min() == 0.0 and d.train[SENSORS].max().max() == 1.0
    f = build_tabular_features(d.train)
    assert f.shape[1] == 3 + 238 and not f.isna().any().any()
    X_test, units = last_window_per_engine(d.test)
    assert X_test.shape == (100, SEQ_LEN, 14) and len(d.test_truth) == len(units) == 100
    assert last_row_per_engine(build_tabular_features(d.test)).shape[0] == 100


@needs_data
def test_prepare_data_scaler_saw_only_training_engines():
    """Rule 2 at the pipeline level: the scaler's learned min/max must equal the min/max of
    the TRAINING engines' raw values - not of train+val (which would leak validation)."""
    from src.data_loader import load_cmapss
    d = prepare_data("FD001")
    raw = load_cmapss("FD001", "train")
    train_raw = raw[raw.unit_id.isin(d.train_units)][SENSORS]
    all_raw = raw[SENSORS]
    # guard: the validation engines really do extend the range, so the test can tell them apart
    assert not np.allclose(train_raw.min().to_numpy(), all_raw.min().to_numpy()) or \
           not np.allclose(train_raw.max().to_numpy(), all_raw.max().to_numpy())
    np.testing.assert_allclose(d.scaler.data_min_, train_raw.min().to_numpy())
    np.testing.assert_allclose(d.scaler.data_max_, train_raw.max().to_numpy())


@needs_data
def test_fd002_short_test_engines_get_padded():
    d = prepare_data("FD002")
    lengths = d.test.groupby("unit_id").size()
    assert lengths.min() < SEQ_LEN                                            # 21 cycles
    X, _ = last_window_per_engine(d.test)
    assert X.shape == (259, SEQ_LEN, 14) and not np.isnan(X).any()
