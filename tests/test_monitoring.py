import numpy as np
import pandas as pd
import pytest

from src.monitoring import (compute_feature_drift, compute_ks_drift, compute_psi, drift_summary,
                            flag_drift, load_reference, psi_bin_edges, psi_level, save_reference)

rng = np.random.default_rng(0)


def test_identical_distributions_no_drift():
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    ks = compute_ks_drift(a, b)
    assert not ks["drift"] and ks["p_value"] > 0.05
    assert compute_psi(a, b) < 0.02
    assert compute_psi(a, a) == pytest.approx(0.0, abs=1e-12)


def test_shifted_normal_drifts():
    a, b = rng.normal(size=5000), rng.normal(loc=1.0, size=5000)
    ks = compute_ks_drift(a, b)
    assert ks["drift"] and ks["p_value"] < 1e-10 and ks["ks_stat"] > 0.3
    assert compute_psi(a, b) > 0.2


def test_psi_matches_hand_computation():
    ref = np.arange(1, 11, dtype=float)                     # deciles -> one value per bin
    cur = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 10], dtype=float)
    e = np.full(10, 0.1)
    a = np.array([5, 1, 1, 1, 1, 0, 0, 0, 0, 1]) / 10
    a = np.clip(a, 1e-4, None)
    assert compute_psi(ref, cur) == pytest.approx(np.sum((a - e) * np.log(a / e)))


def test_out_of_range_values_are_counted_not_dropped():
    ref = rng.uniform(0, 1, 2000)
    edges = psi_bin_edges(ref)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert compute_psi(ref, ref + 100) > 5                  # everything lands in the top bin


def test_invariant_to_increasing_transforms():
    """Why raw sensors are fine: min-max scaling (or any increasing map) changes neither test."""
    a, b = rng.normal(size=3000), rng.normal(0.3, 1.2, size=3000)
    f = lambda x: (x - a.min()) / (a.max() - a.min()) * 7 + 3
    assert compute_psi(f(a), f(b)) == pytest.approx(compute_psi(a, b))
    assert compute_ks_drift(f(a), f(b))["ks_stat"] == pytest.approx(compute_ks_drift(a, b)["ks_stat"])


def test_discrete_and_constant_features():
    ref = rng.integers(390, 400, 5000).astype(float)        # like s17: few distinct values
    assert compute_psi(ref, rng.integers(390, 400, 5000)) < 0.02
    assert compute_psi(ref, ref + 5) > 0.2
    const = np.full(100, 3.0)
    assert compute_psi(const, const) == pytest.approx(0.0, abs=1e-12)
    assert compute_psi(const, const + 1) > 1 and compute_psi(const, const - 1) > 1   # both directions


def test_psi_levels():
    assert (psi_level(0.05), psi_level(0.15), psi_level(0.5)) == ("stable", "moderate", "significant")


def frames(shift_cols, n=4000, shift=1.0):
    cols = ["s1", "s2", "s3", "s4", "s5"]
    ref = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    cur = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    for c in shift_cols:
        cur[c] += shift
    return ref, cur, cols


def test_feature_drift_and_alarm_rule():
    ref, cur, cols = frames(["s1", "s2", "s3"])
    df = compute_feature_drift(ref, cur, cols)
    assert df.loc[["s1", "s2", "s3"], "drift"].all() and not df.loc[["s4", "s5"], "drift"].any()
    assert flag_drift(df)                                   # 3 drifting features -> alarm
    assert not flag_drift(df, min_features=4)
    s = drift_summary(df)
    assert s["n_drift"] == 3 and s["alarm"] and s["drifted_features"] == "s1 s2 s3"


def test_alarm_needs_ks_and_psi_together():
    """A KS-significant but tiny shift (large n) must not count: PSI says the effect is small."""
    ref, cur, cols = frames(["s1", "s2", "s3"], n=200_000, shift=0.05)
    df = compute_feature_drift(ref, cur, cols)
    assert df.loc[["s1", "s2", "s3"], "ks_drift"].all()       # KS flags them...
    assert not df["psi_drift"].any()                          # ...PSI does not
    assert not flag_drift(df)


def test_errors():
    with pytest.raises(ValueError):
        compute_psi([], [1.0])
    with pytest.raises(ValueError):
        compute_ks_drift([1.0, np.nan], [1.0])
    ref, cur, cols = frames([])
    with pytest.raises(KeyError):
        compute_feature_drift(ref, cur.drop(columns="s5"), cols)


def test_reference_roundtrip(tmp_path):
    ref, _, cols = frames([])
    p = save_reference(ref, cols[:3], tmp_path / "ref.npz", meta={"subset": "FD001", "units": [1, 2]})
    back, meta = load_reference(p)
    pd.testing.assert_frame_equal(back, ref[cols[:3]])
    assert meta == {"subset": "FD001", "units": [1, 2]}
