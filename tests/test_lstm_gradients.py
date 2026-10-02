"""Numerical gradient check of the hand-written LSTM backward pass."""
import numpy as np
from src.models.lstm_numpy import LSTMRegressor


def test_gradients_match_finite_differences():
    r = np.random.default_rng(0)
    X, S, t = r.normal(size=(6, 3, 5)), r.normal(size=(6, 4)), r.normal(size=6)
    m = LSTMRegressor(hidden=4, dropout=0.0, seed=1); m._init(5, 4)

    def loss():
        y, _ = m._forward(X, S, False)
        return np.mean((y - t) ** 2)

    y, ctx = m._forward(X, S, False)
    G = m._backward(y, t, ctx)
    eps, worst = 1e-6, 0.0
    for k, W in m.P.items():
        flat = W.reshape(-1)
        for idx in r.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + eps; lp = loss()
            flat[idx] = old - eps; lm = loss()
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = G[k].reshape(-1)[idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-5, worst
