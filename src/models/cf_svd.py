"""Collaborative-filtering style model (low-rank / SVD factorisation of a team x team matrix).

Treat 'home team' as the user and 'away team' as the item, 'total points' as the rating.
For a game on day d we build a sparse matrix A from games of the previous `horizon_days` days
(league-wide), factorise the OBSERVED entries  A ~ mu + U V^T  with regularised alternating least
squares (equivalent to the masked SVD objective of the paper, min sum_{A_ij observed} (A_ij - u_i.v_j)^2),
and predict mu + u_home . v_away.   Parameters (U, V) are re-learned per day; hyperparameters are
n_factors, horizon_days, reg, n_iter.   Only games strictly before day d are used -> no leakage.
"""
import numpy as np
import pandas as pd


def _als(rows, cols, vals, n, k, reg, n_iter, rng):
    U = rng.normal(0, 0.1, (n, k)); V = rng.normal(0, 0.1, (n, k))
    by_row = [np.where(rows == i)[0] for i in range(n)]
    by_col = [np.where(cols == j)[0] for j in range(n)]
    I = reg * np.eye(k)
    for _ in range(n_iter):
        for i in range(n):
            idx = by_row[i]
            if len(idx):
                Vo = V[cols[idx]]
                U[i] = np.linalg.solve(Vo.T @ Vo + I, Vo.T @ vals[idx])
        for j in range(n):
            idx = by_col[j]
            if len(idx):
                Uo = U[rows[idx]]
                V[j] = np.linalg.solve(Uo.T @ Uo + I, Uo.T @ vals[idx])
    return U, V


class CFSVD:
    def __init__(self, n_factors=10, horizon_days=5, reg=1.0, n_iter=8, seed=42):
        self.n_factors, self.horizon_days, self.reg, self.n_iter, self.seed = n_factors, horizon_days, reg, n_iter, seed

    def predict(self, targets: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
        """targets/history need columns date, home, away (+ total for history)."""
        teams = sorted(set(history.home) | set(history.away) | set(targets.home) | set(targets.away))
        ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
        hist = history.assign(h=history.home.map(ix), a=history.away.map(ix)).sort_values("date")
        h_dates = hist.date.values
        out = np.full(len(targets), np.nan)
        rng = np.random.default_rng(self.seed)
        for d, grp in targets.groupby("date"):
            lo = np.searchsorted(h_dates, np.datetime64(d - pd.Timedelta(days=self.horizon_days)), "left")
            hi = np.searchsorted(h_dates, np.datetime64(d), "left")      # strictly before day d
            w = hist.iloc[lo:hi]
            if len(w) < 3:
                out[targets.index.get_indexer(grp.index)] = hist.total.iloc[:hi].mean() if hi else 205.0
                continue
            m = w.groupby(["h", "a"]).total.mean().reset_index()
            mu = m.total.mean()
            U, V = _als(m.h.values, m.a.values, (m.total - mu).values, n, self.n_factors, self.reg, self.n_iter, rng)
            hi_ix = grp.home.map(ix).values; ai_ix = grp.away.map(ix).values
            out[targets.index.get_indexer(grp.index)] = mu + np.einsum("ij,ij->i", U[hi_ix], V[ai_ix])
        return out
