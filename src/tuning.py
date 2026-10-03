"""Hyperparameter tuning. EVERY candidate is scored on the VALIDATION season only -- never on test.

Parameters (learned by the optimiser)      : tree splits, U/V factors, NN weights, LSTM gate weights.
Hyperparameters (chosen here, before fit)  : see the SPACES below.
Each trial is appended to a log row -> results/tuning_log.csv.
"""
import itertools, json, time
import numpy as np
from .models.rf import make_rf
from .models.cf_svd import CFSVD
from .models.nn import NNRegressor
from .models.lstm_numpy import LSTMRegressor
from .pipeline import prepare

SPACES = {
    "rf":   {"n_estimators": [200, 400], "max_depth": [5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20],
             "max_features": [0.3, 0.5, 1.0]},
    "cf":   {"n_factors": [2, 5, 10], "horizon_days": [3, 5, 10, 20, 30], "reg": [0.3, 1.0, 3.0]},
    "nn":   {"hidden": [(500, 100, 20), (256, 64), (100, 20)], "lr": [3e-4, 1e-3, 3e-3],
             "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0], "batch_size": [32, 64, 128]},
    "lstm": {"hidden": [10, 20, 32], "lr": [1e-3, 3e-3, 1e-2], "dropout": [0.0, 0.2, 0.4],
             "weight_decay": [1e-5, 1e-4, 1e-3], "batch_size": [32, 64]},
}
N_TRIALS = {"rf": 8, "nn": 8, "lstm": 8}            # random-search budget (CF is a full grid)
TRAIN_START_CANDIDATES = ["2007-08", "2009-10", "2011-12", "2012-13", "2013-14"]


def _sample(space, n, seed):
    rng = np.random.default_rng(seed)
    keys = list(space)
    combos = list(itertools.product(*[space[k] for k in keys]))
    pick = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    return [dict(zip(keys, combos[i])) for i in pick]


def _row(model, params, val, secs, **extra):
    return {"model": model, "params": json.dumps(params, default=str), "val_mse": float(val), "seconds": round(secs, 1), **extra}


def tune_train_start(df, seq, meta, seasons_available, log=print):
    """Data-level hyperparameter: first training season (NBA scoring drifts over time)."""
    rows, order = [], sorted(seasons_available)
    final_train = ["2012-13", "2013-14", "2014-15", "2015-16"]
    for start in TRAIN_START_CANDIDATES:
        if start not in order:
            continue
        seasons = [s for s in order if start <= s <= final_train[-1]]
        D = prepare(df, seq, meta, seasons)
        t = time.time()
        rf = make_rf(n_estimators=150, max_depth=8, min_samples_leaf=5, max_features=0.5).fit(D["X"][D["tr"]], D["y"][D["tr"]])
        v = np.mean((rf.predict(D["X"][D["va"]]) - D["y"][D["va"]]) ** 2)
        rows.append(_row("train_start", {"start": start, "n_train": int(D["tr"].sum())}, v, time.time() - t))
        log(f"  train_start={start}  n_train={int(D['tr'].sum())}  val_mse={v:.2f}")
    best = min(rows, key=lambda r: r["val_mse"])
    return json.loads(best["params"])["start"], rows


def tune_all(df, seq, meta, games, train_seasons, quick=False, log=print):
    D = prepare(df, seq, meta, train_seasons)
    Xtr, Xva = D["X"][D["tr"]], D["X"][D["va"]]
    ytr, yva = D["y"][D["tr"]], D["y"][D["va"]]
    rows, best = [], {}
    nt = {k: (2 if quick else v) for k, v in N_TRIALS.items()}

    log("[tune] Random Forest"); res = []
    for p in _sample(SPACES["rf"], nt["rf"], 1):
        t = time.time(); m = make_rf(**p).fit(Xtr, ytr)
        v = np.mean((m.predict(Xva) - yva) ** 2); rows.append(_row("rf", p, v, time.time() - t)); res.append((v, p))
        log(f"  {p}  val_mse={v:.2f}")
    best["rf"] = min(res, key=lambda r: r[0])[1]

    log("[tune] Collaborative filtering (SVD/ALS)"); res = []
    space = {"n_factors": [2, 5], "horizon_days": [5, 10], "reg": [1.0]} if quick else SPACES["cf"]
    vdf = df.loc[D["va"], ["date", "home", "away"]]
    for vals in itertools.product(*space.values()):
        p = dict(zip(space, vals)); t = time.time()
        pr = CFSVD(**p).predict(vdf, games)
        v = np.mean((pr - yva) ** 2); rows.append(_row("cf", p, v, time.time() - t)); res.append((v, p))
    best["cf"] = min(res, key=lambda r: r[0])[1]
    log(f"  best CF {best['cf']}  val_mse={min(res, key=lambda r: r[0])[0]:.2f}")

    log("[tune] Neural network"); res = []
    for p in _sample(SPACES["nn"], nt["nn"], 2):
        t = time.time(); m = NNRegressor(max_epochs=8 if quick else 30, patience=5, **p).fit(D["Xs"][D["tr"]], ytr, D["Xs"][D["va"]], yva)
        rows.append(_row("nn", p, m.best_val, time.time() - t, best_epoch=m.best_epoch)); res.append((m.best_val, p))
        log(f"  {p}  val_mse={m.best_val:.2f} (epoch {m.best_epoch})")
    best["nn"] = min(res, key=lambda r: r[0])[1]

    log("[tune] LSTM"); res = []
    for p in _sample(SPACES["lstm"], nt["lstm"], 3):
        t = time.time(); m = LSTMRegressor(max_epochs=6 if quick else 40, patience=6, **p).fit(
            D["Q"][D["tr"]], D["S"][D["tr"]], ytr, D["Q"][D["va"]], D["S"][D["va"]], yva)
        rows.append(_row("lstm", p, m.best_val, time.time() - t, best_epoch=m.best_epoch)); res.append((m.best_val, p))
        log(f"  {p}  val_mse={m.best_val:.2f} (epoch {m.best_epoch})")
    best["lstm"] = min(res, key=lambda r: r[0])[1]
    return best, rows
