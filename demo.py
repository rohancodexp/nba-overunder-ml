"""Live demo: predict total points for real games on a given date and compare with the sportsbook line.

  python demo.py --data synthetic --date 2018-03-10
  python demo.py --data real --date 2018-03-10 --home GoldenState
Requires a prior `python run_all.py ...` (it saves models_saved/bundle.joblib).
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from src import config as C
from src.data_loader import load_all
from src.features import build
from src.models.cf_svd import CFSVD

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["synthetic", "real"], default="real")
ap.add_argument("--date", required=True, help="YYYY-MM-DD inside the test season")
ap.add_argument("--home", default=None, help="optional: filter to one home team (e.g. GoldenState)")
a = ap.parse_args()

B = joblib.load(C.MODELS_DIR / "bundle.joblib")
games = load_all(C.DATA_SYNTH if a.data == "synthetic" else C.DATA_RAW)
df, seq, meta = build(games)
m = np.array((df.date == pd.Timestamp(a.date)).values)
if a.home:
    m = m & np.array((df.home == a.home).values)
if not m.any():
    raise SystemExit("No games found for that date/team in the feature table.")

S = B["scalers"]
cols = B["meta"]["num_cols"] + B["meta"]["team_cols"]
X = df.loc[m, cols].values.astype(float)
Q = (seq[m] - S["seq_mean"]) / S["seq_std"]
St = S["static"].transform(df.loc[m, B["meta"]["static_cols"]].values.astype(float))
out = df.loc[m, ["date", "home", "away", "total_line", "total"]].copy()
out["RF"] = B["rf"].predict(X)
out["CF-SVD"] = CFSVD(**B["cf_params"]).predict(df.loc[m, ["date", "home", "away"]], games)
out["NN"] = B["nn"].predict(S["x"].transform(X))
out["LSTM"] = B["lstm"].predict(Q, St)
for k in ("RF", "CF-SVD", "NN", "LSTM"):
    out[k + "_pick"] = np.where(out[k] > out.total_line, "OVER", "UNDER")
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
pd.set_option("display.width", 200)
print(out.round(1).to_string(index=False))
