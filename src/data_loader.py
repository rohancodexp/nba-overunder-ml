"""Load Sportsbook Reviews Online (SBRO) NBA odds Excel files into one game-level DataFrame.

Expected files: one .xlsx per season in data/raw/, file name containing the season, e.g.
  'nba odds 2012-13.xlsx'. Each game = two consecutive rows (V = visitor, H = home) with columns
  Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H.
In one row of each pair 'Close' holds the game TOTAL (Over/Under); in the other the SPREAD.
The total is always the larger of the two numbers, which is how we separate them.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {  # franchise renames / spelling variants -> one canonical key
    "NewJersey": "Brooklyn", "NewOrleansHornets": "NewOrleans", "NewOrleansPelicans": "NewOrleans",
    "CharlotteBobcats": "Charlotte", "Seattle": "OklahomaCity",
    "L.A.Clippers": "LAClippers", "L.A.Lakers": "LALakers",
}


def _canon(name) -> str:
    s = str(name).strip().replace(" ", "")
    return ALIASES.get(s, s)


def _season_from_name(path: Path) -> int:
    m = re.search(r"(\d{4})\s*-\s*(\d{2})", path.name)
    if not m:
        raise ValueError(f"Cannot read season from file name: {path.name} (expected like '2012-13')")
    return int(m.group(1))


def load_season(path) -> pd.DataFrame:
    path = Path(path)
    y0 = _season_from_name(path)
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ["date", "vh", "team", "final", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    df["vh"] = df["vh"].astype(str).str.strip().str.upper().str[0]
    df = df[df["vh"].isin(["V", "H"])].reset_index(drop=True)
    v_idx = [i for i in range(len(df) - 1) if df.at[i, "vh"] == "V" and df.at[i + 1, "vh"] == "H"]
    v = df.loc[v_idx].reset_index(drop=True)
    h = df.loc[[i + 1 for i in v_idx]].reset_index(drop=True)
    num = lambda s: pd.to_numeric(s, errors="coerce")
    mmdd = num(v["date"])
    month, day = (mmdd // 100).astype("float"), (mmdd % 100).astype("float")
    year = np.where(month >= 9, y0, y0 + 1)
    date = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce")
    line = np.fmax(num(v["close"]), num(h["close"]))   # total = the larger number of the pair
    if "open" in v.columns:
        line = np.where(np.isnan(line), np.fmax(num(v["open"]), num(h["open"])), line)
    out = pd.DataFrame({
        "season": f"{y0}-{str(y0 + 1)[-2:]}", "date": date,
        "away": v["team"].map(_canon), "home": h["team"].map(_canon),
        "away_pts": num(v["final"]), "home_pts": num(h["final"]), "total_line": line})
    out["total"] = out["home_pts"] + out["away_pts"]
    out.loc[~out["total_line"].between(150, 280), "total_line"] = np.nan
    return out.dropna(subset=["date", "home_pts", "away_pts"])


def load_all(raw_dir) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files in {raw_dir}. Download SBRO NBA odds files (see README).")
    g = pd.concat([load_season(f) for f in files], ignore_index=True)
    g = g.sort_values(["date", "home"]).reset_index(drop=True)
    g["game_id"] = np.arange(len(g))
    return g
