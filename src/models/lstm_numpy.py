"""LSTM regressor implemented from scratch in NumPy (manual back-propagation through time).

Why NumPy and not PyTorch: runs anywhere with zero heavy dependencies and the maths is explicit
(good for explaining in the viva). A gradient-check unit test (tests/test_lstm_gradients.py)
verifies the backward pass.

Input  : sequence X (N, T=3, d) = the last 3 games (oldest -> newest) of both teams, and
         static features S (N, s) = current context (rest, back-to-back, streak, season averages).
Model  : one LSTM layer (hidden H) -> dropout on the last hidden state -> concat [h_T, S] -> linear -> total points.
Note   : the equations printed in the source paper (update/reset gate + candidate) are those of a GRU;
         the problem statement calls it an LSTM, so the standard 4-gate LSTM is implemented here.
"""
import copy
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMRegressor:
    def __init__(self, hidden=20, lr=3e-3, dropout=0.2, weight_decay=1e-4, batch_size=32,
                 max_epochs=60, patience=8, seed=42):
        self.H, self.lr, self.p, self.wd, self.bs = hidden, lr, dropout, weight_decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.history = {"train_mse": [], "val_mse": []}

    # ---------------- parameters ----------------
    def _init(self, d, s):
        r, H = np.random.default_rng(self.seed), self.H
        self.P = {"Wx": r.normal(0, 1 / np.sqrt(d), (d, 4 * H)), "Wh": r.normal(0, 1 / np.sqrt(H), (H, 4 * H)),
                  "b": np.zeros(4 * H), "Wo": r.normal(0, 1 / np.sqrt(H + s), (H + s, 1)), "bo": np.zeros(1)}
        self.P["b"][H:2 * H] = 1.0                       # forget-gate bias = 1
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    # ---------------- forward / backward ----------------
    def _forward(self, X, S, train, rng=None):
        P, H = self.P, self.H
        B, T, _ = X.shape
        h, c = np.zeros((B, H)), np.zeros((B, H))
        cache = []
        for t in range(T):
            z = X[:, t] @ P["Wx"] + h @ P["Wh"] + P["b"]
            i, f, g, o = _sig(z[:, :H]), _sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sig(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((X[:, t], h, c, i, f, g, o, c_new, tc))
            h, c = h_new, c_new
        mask = np.ones_like(h)
        if train and self.p > 0:
            mask = (rng.random(h.shape) >= self.p) / (1 - self.p)
        feat = np.concatenate([h * mask, S], axis=1)
        y = feat @ P["Wo"] + P["bo"]
        return y[:, 0], (cache, mask, feat)

    def _backward(self, y, target, ctx):
        cache, mask, feat = ctx
        P, H = self.P, self.H
        B = len(y)
        dy = (2.0 / B) * (y - target)[:, None]
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["Wo"] = feat.T @ dy; G["bo"] = dy.sum(0)
        dh = (dy @ P["Wo"].T)[:, :H] * mask
        dc_next = np.zeros_like(dh)
        for (x, hp, cp, i, f, g, o, c, tc) in reversed(cache):
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            dz = np.concatenate([dc * g * i * (1 - i), dc * cp * f * (1 - f),
                                 dc * i * (1 - g ** 2), do * o * (1 - o)], axis=1)
            G["Wx"] += x.T @ dz; G["Wh"] += hp.T @ dz; G["b"] += dz.sum(0)
            dh = dz @ P["Wh"].T
            dc_next = dc * f
        return G

    def _adam(self, G):
        self.t += 1
        for k in self.P:
            g = G[k] + (self.wd * self.P[k] if k in ("Wx", "Wh", "Wo") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    # ---------------- public API ----------------
    def fit(self, Xtr, Str, ytr, Xva, Sva, yva):
        self.mu, self.sd = ytr.mean(), ytr.std()
        z = (ytr - self.mu) / self.sd
        self._init(Xtr.shape[2], Str.shape[1])
        rng = np.random.default_rng(self.seed)
        best, best_P, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            perm = rng.permutation(len(Xtr))
            for s in range(0, len(perm), self.bs):
                b = perm[s:s + self.bs]
                y, ctx = self._forward(Xtr[b], Str[b], True, rng)
                self._adam(self._backward(y, z[b], ctx))
            tr = np.mean((self.predict(Xtr, Str) - ytr) ** 2)
            va = np.mean((self.predict(Xva, Sva) - yva) ** 2)
            self.history["train_mse"].append(tr); self.history["val_mse"].append(va)
            if va < best - 1e-6:
                best, best_P, bad = va, copy.deepcopy(self.P), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.P, self.best_epoch, self.best_val = best_P, int(np.argmin(self.history["val_mse"])) + 1, best
        return self

    def predict(self, X, S):
        y, _ = self._forward(X, S, False)
        return y * self.sd + self.mu
