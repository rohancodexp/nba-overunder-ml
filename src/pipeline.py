"""Shared data preparation used by tuning, final training, the demo and the tests."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS


def prepare(df, seq, meta, train_seasons=None):
    train_seasons = train_seasons or TRAIN_SEASONS
    tr, va, te = (df.season.isin(s).values for s in (train_seasons, VAL_SEASONS, TEST_SEASONS))
    cols = meta["num_cols"] + meta["team_cols"]
    X = df[cols].values.astype(float)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    m, s = seq[tr].mean((0, 1)), seq[tr].std((0, 1)) + 1e-8
    Q = (seq - m) / s
    St = df[meta["static_cols"]].values.astype(float)
    ssc = StandardScaler().fit(St[tr]); Ss = ssc.transform(St)
    return {"tr": tr, "va": va, "te": te, "X": X, "Xs": Xs, "Q": Q, "S": Ss, "y": df["total"].values.astype(float),
            "line": df["total_line"].values.astype(float), "cols": cols,
            "scalers": {"x": sc, "seq_mean": m, "seq_std": s, "static": ssc}}
