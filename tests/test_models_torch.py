"""Tests for the PyTorch sequence models: causality, receptive field, training loop, bundles."""
import numpy as np
import pytest
import torch

from src.models.base import read_metadata
from src.models.tcn_rul import TCNNet, TCNRUL


def make_net(channels=(8, 8, 8), k=3):
    torch.manual_seed(0)
    return TCNNet(n_features=4, channels=list(channels), kernel_size=k, dropout=0.0).eval()


@torch.no_grad()
def test_tcn_is_causal_future_inputs_never_change_past_outputs():
    """The chomp check: perturb cycle t; outputs at cycles < t must be identical."""
    net = make_net()
    x = torch.randn(2, 30, 4)
    base = net.forward_sequence(x)
    t = 17
    x2 = x.clone(); x2[:, t, :] += 10.0
    out = net.forward_sequence(x2)
    torch.testing.assert_close(out[:, :, :t], base[:, :, :t])          # past untouched
    assert not torch.allclose(out[:, :, t:], base[:, :, t:])            # present/future affected


@torch.no_grad()
def test_receptive_field_formula_matches_behaviour():
    net = make_net()                                                    # k=3, dilations 1,2,4
    assert net.receptive_field == 29
    x = torch.randn(1, 40, 4)
    last = net(x)
    x_out = x.clone(); x_out[:, 40 - 29 - 1, :] += 10.0                 # just outside the field
    torch.testing.assert_close(net(x_out), last)
    x_in = x.clone(); x_in[:, 40 - 29, :] += 10.0                       # oldest step inside it
    assert not torch.allclose(net(x_in), last)


def toy_sequences(n=600, seed=0):
    """Target = 60 + 40 * mean of the last 5 steps of feature 0 (learnable, inside [0, 125])."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 30, 4)).astype(np.float32) * 0.5
    y = (60 + 40 * X[:, -5:, 0].mean(axis=1)).astype(np.float32)
    return X, y


def test_tcn_learns_and_bundle_roundtrip(tmp_path):
    X, y = toy_sequences()
    m = TCNRUL(arch={"channels": [16, 16], "dropout": 0.0},
               train_cfg={"epochs": 25, "batch_size": 64, "lr": 3e-3, "patience": 25}, device="cpu")
    m.fit(X[:500], y[:500], X[500:], y[500:], verbose=False)
    h = m.history
    assert h.val_rmse[h.best_epoch] == min(h.val_rmse)                  # best epoch tracked
    assert m.evaluate(X[500:], y[500:])["rmse"] < 0.5 * y.std()          # learned the signal
    m.save(tmp_path / "b", metadata={"note": "x"})
    m2 = TCNRUL.load(tmp_path / "b")
    np.testing.assert_allclose(m.predict(X[:50]), m2.predict(X[:50]), rtol=1e-5, atol=1e-4)
    assert read_metadata(tmp_path / "b")["receptive_field"] == 1 + 2 * 2 * 3


def test_tcn_rejects_wrong_input_shape():
    X, y = toy_sequences(n=80)
    m = TCNRUL(arch={"channels": [4]}, train_cfg={"epochs": 1}, device="cpu").fit(X[:60], y[:60], X[60:], y[60:], verbose=False)
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 30, 5), dtype=np.float32))


# ------------------------------------------------------------------ LSTM + attention (P7)
from src.models.lstm_rul import LSTMAttentionNet, LSTMRUL  # noqa: E402


def make_lstm(bidirectional, hidden=8, layers=2):
    torch.manual_seed(0)
    return LSTMAttentionNet(4, hidden, layers, 0.0, bidirectional, attn_dim=8).eval()


@torch.no_grad()
@pytest.mark.parametrize("bi", [True, False])
def test_attention_weights_are_a_distribution(bi):
    pred, w = make_lstm(bi)(torch.randn(5, 30, 4))
    assert pred.shape == (5,) and w.shape == (5, 30)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(dim=1), torch.ones(5))


@torch.no_grad()
def test_uni_lstm_states_ignore_future_bi_lstm_states_do_not():
    """The bidirectional flag really changes what each time step can see."""
    x = torch.randn(2, 30, 4)
    x2 = x.clone(); x2[:, 25, :] += 5.0
    uni, bi = make_lstm(False), make_lstm(True)
    torch.testing.assert_close(uni.lstm(x2)[0][:, :25], uni.lstm(x)[0][:, :25])   # uni: past untouched
    assert not torch.allclose(bi.lstm(x2)[0][:, :25], bi.lstm(x)[0][:, :25])       # bi: backward pass sees it


@torch.no_grad()
def test_bidirectional_query_uses_correct_final_states():
    """Forward direction finishes at the LAST step, backward at the FIRST step."""
    net = make_lstm(True, hidden=8, layers=2)
    states, (h_n, _) = net.lstm(torch.randn(3, 30, 4))
    torch.testing.assert_close(h_n[-2], states[:, -1, :8])     # forward, last layer, t = T
    torch.testing.assert_close(h_n[-1], states[:, 0, 8:])      # backward, last layer, t = 1


@pytest.mark.parametrize("bi", [True, False])
def test_lstm_learns_and_bundle_roundtrip(tmp_path, bi):
    X, y = toy_sequences()
    # LSTMs first have to learn the ~60 output offset from a start near 0, so they need a
    # bigger toy budget than the TCN test (25 epochs left them at 0.55-0.87 of y.std()).
    m = LSTMRUL(arch={"hidden_size": 16, "bidirectional": bi, "dropout": 0.0},
                train_cfg={"epochs": 80, "batch_size": 64, "lr": 1e-2, "patience": 80}, device="cpu")
    m.fit(X[:500], y[:500], X[500:], y[500:], verbose=False)
    assert m.evaluate(X[500:], y[500:])["rmse"] < 0.5 * y.std()
    w = m.attention_weights(X[:4])
    assert w.shape == (4, 30) and np.allclose(w.sum(axis=1), 1, atol=1e-5)
    m.save(tmp_path / "b")
    m2 = LSTMRUL.load(tmp_path / "b")
    np.testing.assert_allclose(m.predict(X[:50]), m2.predict(X[:50]), rtol=1e-5, atol=1e-4)
    assert read_metadata(tmp_path / "b")["bidirectional"] is bi
    with pytest.raises(ValueError):
        from src.models.tcn_rul import TCNRUL
        TCNRUL.load(tmp_path / "b")                             # wrong model type
