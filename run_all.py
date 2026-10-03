"""One command runs everything:  python run_all.py --data synthetic|real [--quick]

  1 load data -> 2 features -> 3 tune train-start season -> 4 tune hyper-parameters (validation only)
  -> 5 final fit -> 6 evaluate on test ONCE -> 7 save metrics, tables, figures, models.
"""
import argparse, json, sys, time
import joblib
import numpy as np
import pandas as pd
from src import config as C
from src.data_loader import load_all
from src.features import build
from src.pipeline import prepare
from src.evaluate import regression_metrics, ou_metrics, edge_table, BREAK_EVEN
from src.tuning import tune_all, tune_train_start
from src.models.rf import make_rf
from src.models.cf_svd import CFSVD
from src.models.nn import NNRegressor
from src.models.lstm_numpy import LSTMRegressor
from src import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real"], default="real")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny search budget (smoke test)")
    a = ap.parse_args()
    out = C.RESULTS / ("quick" if a.quick else "")
    out.mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    T0 = time.time()
    log = lambda *s: print(*s, flush=True)

    # 1. data ------------------------------------------------------------------------------
    if a.data == "synthetic":
        from src.data_synth import generate
        raw = C.DATA_SYNTH
        if not any(raw.glob("*.xlsx")):
            log("[data] generating SYNTHETIC SBRO-format files (not real NBA data)"); generate(raw)
    else:
        raw = a.raw_dir or C.DATA_RAW
    games = load_all(raw)
    need = C.TRAIN_SEASONS + C.VAL_SEASONS + C.TEST_SEASONS
    miss = [s for s in need if s not in set(games.season)]
    if miss:
        sys.exit(f"Missing seasons in {raw}: {miss}. Put one SBRO .xlsx per season (2007-08 .. 2017-18) in that folder.")
    log(f"[data] {len(games)} games, seasons {games.season.min()}..{games.season.max()}  (source: {a.data})")

    # 2. features --------------------------------------------------------------------------
    df, seq, meta = build(games)
    log(f"[features] {len(df)} usable games | {len(meta['num_cols'])} numeric + {len(meta['team_cols'])} team one-hot features")

    # 3-4. tuning (validation only) ----------------------------------------------------------
    log("[tune] training-start season")
    start, start_rows = tune_train_start(df, seq, meta, set(games.season), log)
    train_seasons = [s for s in sorted(set(games.season)) if start <= s <= C.TRAIN_SEASONS[-1]]
    best, rows = tune_all(df, seq, meta, games, train_seasons, quick=a.quick, log=log)
    tuning = pd.DataFrame(start_rows + rows); tuning.to_csv(out / "tuning_log.csv", index=False)

    # 5. final fit -------------------------------------------------------------------------
    D = prepare(df, seq, meta, train_seasons)
    tr, va, te, y, line = D["tr"], D["va"], D["te"], D["y"], D["line"]
    log(f"[final] train={tr.sum()} val={va.sum()} test={te.sum()}  (train seasons {train_seasons[0]}..{train_seasons[-1]})")
    rf = make_rf(**best["rf"]).fit(D["X"][tr], y[tr])
    nn = NNRegressor(max_epochs=10 if a.quick else 80, patience=8, **best["nn"]).fit(D["Xs"][tr], y[tr], D["Xs"][va], y[va])
    ls = LSTMRegressor(max_epochs=8 if a.quick else 80, patience=8, **best["lstm"]).fit(D["Q"][tr], D["S"][tr], y[tr], D["Q"][va], D["S"][va], y[va])
    cf = CFSVD(**best["cf"])
    P = {"Random Forest (baseline)": lambda m: rf.predict(D["X"][m]),
     "Collaborative Filtering (SVD)": lambda m: cf.predict(df.loc[m, ["date", "home", "away"]], games),
     "Neural Network": lambda m: nn.predict(D["Xs"][m]),
     "LSTM": lambda m: ls.predict(D["Q"][m], D["S"][m]),
     "NN+LSTM+RF Ensemble": lambda m: (
         rf.predict(D["X"][m])
         + nn.predict(D["Xs"][m])
         + ls.predict(D["Q"][m], D["S"][m])
     ) / 3.0}

    # 6. evaluate --------------------------------------------------------------------------
    metrics = {"data_source": a.data, "quick_mode": a.quick, "train_seasons": train_seasons, "val_seasons": C.VAL_SEASONS,
               "test_seasons": C.TEST_SEASONS, "sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
               "n_features": {"numeric": len(meta["num_cols"]), "team_onehot": len(meta["team_cols"])},
               "best_params": best, "best_train_start": start, "break_even": BREAK_EVEN, "models": {}}
    test_preds = df.loc[te, ["date", "home", "away", "total", "total_line"]].copy()
    for name, fn in P.items():
        res = {}
        for sp, m in (("train", tr), ("val", va), ("test", te)):
            p = fn(m); res[sp] = regression_metrics(y[m], p)
            if sp == "test":
                res["ou"] = ou_metrics(p, line[m], y[m]); res["edge_table"] = edge_table(p, line[m], y[m]); test_preds[name] = p
        metrics["models"][name] = res; log(f"[result] {name:32s} train {res['train']['mse']:8.1f}  val {res['val']['mse']:8.1f}  test {res['test']['mse']:8.1f}  O/U {res['ou']['accuracy']*100:.1f}%")
    ok = ~np.isnan(line)
    book = {sp: float(np.mean((line[m & ok] - y[m & ok]) ** 2)) for sp, m in (("train", tr), ("val", va), ("test", te))}
    metrics["sportsbook"] = {"test_mse": book["test"], "val_mse": book["val"], "train_mse": book["train"]}
    metrics["naive_train_mean_test_mse"] = float(np.mean((y[te] - y[tr].mean()) ** 2))
    log(f"[result] Sportsbook line test MSE {book['test']:.1f}; naive train-mean {metrics['naive_train_mean_test_mse']:.1f}")
    best_name = min(metrics["models"], key=lambda k: metrics["models"][k]["val"]["mse"])   # pick by VALIDATION, not test
    metrics["best_model_by_validation"] = best_name
    metrics["history"] = {"nn_best_epoch": nn.best_epoch, "lstm_best_epoch": ls.best_epoch}
    metrics["runtime_sec"] = round(time.time() - T0)

    # 7. save ------------------------------------------------------------------------------
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    test_preds.to_csv(out / "test_predictions.csv", index=False)
    F = out / "figures"
    plots.curves(nn.history, "Neural Net training", F / "nn_curves.png")
    plots.curves(ls.history, "LSTM training", F / "lstm_curves.png")
    plots.cf_horizon(tuning, F / "cf_horizon.png"); plots.tuning_overview(tuning, F / "tuning_overview.png")
    plots.comparison(metrics, F / "model_comparison.png")
    bp = test_preds[best_name].values
    plots.pred_vs_actual(test_preds.total.values, bp, best_name.split(" (")[0], F / "pred_vs_actual.png")
    plots.edge_plot(metrics["models"][best_name]["edge_table"], best_name.split(" (")[0], F / "edge_accuracy.png", BREAK_EVEN)
    plots.importances(D["cols"], rf.feature_importances_, F / "rf_importance.png")
    if not a.quick:
        C.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"rf": rf, "nn": nn, "lstm": ls, "cf_params": best["cf"], "meta": meta, "train_seasons": train_seasons,
                     "scalers": D["scalers"], "best_params": best, "data_source": a.data}, C.MODELS_DIR / "bundle.joblib")
    log(f"[done] {round(time.time() - T0)} s  -> {out}")


if __name__ == "__main__":
    main()
