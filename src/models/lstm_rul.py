"""LSTM with Bahdanau (additive) attention - bidirectional or unidirectional (decision P7).

LSTM: reads the 30-cycle window step by step, keeping a memory (cell state) that gates decide
to keep (forget gate), update (input gate + candidate) and expose (output gate). It yields one
hidden vector h_s per cycle.

Bidirectional: a second LSTM reads the window backwards (cycle 30 -> 1) and its vectors are
concatenated with the forward ones. Legitimate here - the whole window is already observed
past data; it only rules out a strict cycle-by-cycle streaming setup. The unidirectional
variant (bidirectional=False) is the ablation that measures whether the backward pass helps.

Bahdanau attention (Bahdanau et al., 2015):
    score_s = v^T tanh(W1 q + W2 h_s)      q = final hidden state(s) = "what the window says"
    alpha   = softmax(score) over the 30 cycles   (weights >= 0, summing to 1)
    context = sum_s alpha_s h_s
The prediction is made from [context, q]. alpha shows WHICH cycles the model leaned on -
the interpretability signal plotted in results/<model>_attention.png.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.torch_base import TorchRULModel

DEFAULT_ARCH = {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "bidirectional": True,
                "attn_dim": 64}


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, attn_dim, bias=False)   # projects the query
        self.W2 = nn.Linear(hidden_dim, attn_dim, bias=True)    # projects every time step
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, states: torch.Tensor):
        """query (B, H), states (B, T, H) -> context (B, H), weights (B, T)."""
        scores = self.v(torch.tanh(self.W1(query).unsqueeze(1) + self.W2(states))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return context, weights


class LSTMAttentionNet(nn.Module):
    """Input (batch, time, features) -> (RUL (batch,), attention weights (batch, time))."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool, attn_dim: int):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        h = hidden_size * (2 if bidirectional else 1)
        self.attention = BahdanauAttention(h, attn_dim)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x):
        states, (h_n, _) = self.lstm(x)                         # states (B, T, H)
        if self.bidirectional:
            # last layer: forward direction ends at cycle T, backward direction ends at cycle 1
            query = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            query = h_n[-1]
        context, weights = self.attention(query, states)
        pred = self.head(torch.cat([context, query], dim=1)).squeeze(-1)
        return pred, weights


class LSTMRUL(TorchRULModel):
    model_type = "lstm"
    DEFAULT_ARCH = DEFAULT_ARCH

    def _make_net(self, n_features: int) -> LSTMAttentionNet:
        a = self.arch
        return LSTMAttentionNet(n_features, int(a["hidden_size"]), int(a["num_layers"]), float(a["dropout"]),
                                bool(a["bidirectional"]), int(a["attn_dim"]))

    def _extra_metadata(self) -> dict:
        return {"bidirectional": bool(self.arch["bidirectional"])}

    @torch.no_grad()
    def attention_weights(self, X) -> np.ndarray:
        """(n, seq_len) attention weights for each input window."""
        X = self._check(X)
        self.net.eval()
        _, w = self.net(torch.as_tensor(X))
        return w.numpy()
