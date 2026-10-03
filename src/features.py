"""Leakage-free feature engineering.

Every feature of a game is computed from games played STRICTLY BEFORE it.
For each team (home and away) we use:
  * last-3 games: points for, points against, was-home, rest days before that game,
    and the opponent's season-to-date points for/against at that time (opponent strength)
  * current context: rest days, back-to-back flag, consecutive-away-games streak
    (travel-fatigue proxy), and games played in the previous 7 days
  * season-to-date points for / against
plus one-hot team identity (used by NN and LSTM static input).
"""
import numpy as np
import pandas as pd
from .config import LOOKBACK

STEP_FEATS = ["pts", "opp_pts", "home", "rest", "opp_sta_for", "opp_sta_against"]

CTX_FEATS = [
    "rest",
    "b2b",
    "away_streak",
    "sta_for",
    "sta_against",
    "games_last_7d",
]


def _team_log(games: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({
        "game_id": games.game_id,
        "date": games.date,
        "season": games.season,
        "team": games.home,
        "opp": games.away,
        "home": 1,
        "pts": games.home_pts,
        "opp_pts": games.away_pts,
    })

    a = pd.DataFrame({
        "game_id": games.game_id,
        "date": games.date,
        "season": games.season,
        "team": games.away,
        "opp": games.home,
        "home": 0,
        "pts": games.away_pts,
        "opp_pts": games.home_pts,
    })

    log = (
        pd.concat([h, a], ignore_index=True)
        .sort_values(["team", "date", "game_id"])
        .reset_index(drop=True)
    )

    g = log.groupby("team")

    # Rest days before the current game.
    log["rest"] = g["date"].diff().dt.days.clip(upper=7).fillna(7)
    log["b2b"] = (log["rest"] == 1).astype(int)

    # Number of games played by each team in the previous 7 days.
    # The current game is excluded because only earlier rows are counted.
    games_last_7d = np.zeros(len(log), dtype=int)

    for _, idx in log.groupby("team").groups.items():
        positions = np.asarray(list(idx))
        dates = log.loc[positions, "date"].to_numpy()

        for j, pos in enumerate(positions):
            current_date = dates[j]
            games_last_7d[pos] = np.sum(
                (dates[:j] < current_date)
                & (dates[:j] >= current_date - np.timedelta64(7, "D"))
            )

    log["games_last_7d"] = games_last_7d

    # Consecutive away games before this one.
    streak = np.zeros(len(log))

    prev_team, run = None, 0
    for i, (t, hm) in enumerate(
        zip(log["team"].values, log["home"].values)
    ):
        if t != prev_team:
            run, prev_team = 0, t

        streak[i] = run
        run = run + 1 if hm == 0 else 0

    log["away_streak"] = streak

    # Season-to-date averages BEFORE the game.
    gs = log.groupby(["team", "season"])

    for col, name in (
        ("pts", "sta_for"),
        ("opp_pts", "sta_against"),
    ):
        log[name] = gs[col].transform(
            lambda s: s.shift(1).expanding().mean()
        )

    prior = log.groupby("season")["pts"].mean()
    fallback = {
        s: prior.get(p, prior[s])
        for p, s in zip([None] + list(prior.index[:-1]), prior.index)
    }

    for name in ("sta_for", "sta_against"):
        log[name] = log[name].fillna(log["season"].map(fallback))

    # Opponent strength at the time of each game.
    opp = log[
        ["game_id", "team", "sta_for", "sta_against"]
    ].rename(
        columns={
            "team": "opp",
            "sta_for": "opp_sta_for",
            "sta_against": "opp_sta_against",
        }
    )

    log = (
        log.merge(opp, on=["game_id", "opp"], how="left")
        .sort_values(["team", "date", "game_id"])
        .reset_index(drop=True)
    )

    g = log.groupby("team")

    for k in range(1, LOOKBACK + 1):
        for f in STEP_FEATS:
            log[f"{f}_l{k}"] = g[f].shift(k)

    return log


def build(games: pd.DataFrame):
    """Return (frame, seq, meta) where frame has one row per game with features + targets."""
    log = _team_log(games)

    lag_cols = [
        f"{f}_l{k}"
        for k in range(1, LOOKBACK + 1)
        for f in STEP_FEATS
    ]

    keep = ["game_id"] + CTX_FEATS + lag_cols

    h = (
        log[log.home == 1][keep]
        .add_prefix("h_")
        .rename(columns={"h_game_id": "game_id"})
    )

    a = (
        log[log.home == 0][keep]
        .add_prefix("a_")
        .rename(columns={"a_game_id": "game_id"})
    )

    df = games.merge(h, on="game_id").merge(a, on="game_id")

    df = df.dropna(
        subset=[
            c
            for c in df.columns
            if c.startswith(("h_", "a_"))
        ]
    ).reset_index(drop=True)

    num_cols = [
        c for c in df.columns
        if c.startswith(("h_", "a_"))
    ]

    teams = sorted(set(games.home) | set(games.away))

    dummies = pd.DataFrame({
        **{
            f"home_is_{t}": (df.home == t).astype(int)
            for t in teams
        },
        **{
            f"away_is_{t}": (df.away == t).astype(int)
            for t in teams
        },
    })

    df = pd.concat([df, dummies], axis=1)

    team_cols = list(dummies.columns)

    # Sequence tensor for the recurrent model:
    # (N, LOOKBACK, 2 teams * len(STEP_FEATS)), oldest -> newest.
    seq = np.stack([
        np.concatenate([
            df[
                [f"{p}_{f}_l{k}" for f in STEP_FEATS]
            ].values
            for p in ("h", "a")
        ], axis=1)
        for k in range(LOOKBACK, 0, -1)
    ], axis=1).astype(float)

    static_cols = [
        f"{p}_{f}"
        for p in ("h", "a")
        for f in CTX_FEATS
    ]

    meta = {
        "num_cols": num_cols,
        "team_cols": team_cols,
        "static_cols": static_cols,
        "teams": teams,
    }

    return df, seq, meta


def split(df, seasons):
    return (df.season.isin(seasons)).values