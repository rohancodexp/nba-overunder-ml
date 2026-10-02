"""Synthetic NBA-like data written in the *same Excel layout* as Sportsbook Reviews Online (SBRO).

Why this exists: lets the whole pipeline (loader -> features -> models -> report) run and be
unit-tested without internet. It is NOT real data. Numbers produced from it must never be
presented as real NBA results.

Generative story: each team has a persistent 'scoring' and 'strength' latent rating that drifts
between seasons; league scoring rises over time (3-point era); back-to-backs lower totals;
the sportsbook line = true expected total + noise (so the book is good but not perfect).
"""
import numpy as np
import pandas as pd
from pathlib import Path

TEAMS = ["Atlanta", "Boston", "Brooklyn", "Charlotte", "Chicago", "Cleveland", "Dallas", "Denver",
         "Detroit", "GoldenState", "Houston", "Indiana", "LAClippers", "LALakers", "Memphis", "Miami",
         "Milwaukee", "Minnesota", "NewOrleans", "NewYork", "OklahomaCity", "Orlando", "Philadelphia",
         "Phoenix", "Portland", "Sacramento", "SanAntonio", "Toronto", "Utah", "Washington"]


def season_label(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


def _schedule(rng, n):
    """~1230 games: each team hosts 41 games vs balanced opponents, packed into days."""
    games = []
    for h in range(n):
        others = [t for t in range(n) if t != h]
        games += [(h, others[i % 29]) for i in range(41)]
    order = rng.permutation(len(games))
    remaining = [games[i] for i in order]
    days = []
    while remaining:
        busy, today, rest = set(), [], []
        for (h, a) in remaining:
            if len(today) < 9 and h not in busy and a not in busy:
                today.append((h, a)); busy |= {h, a}
            else:
                rest.append((h, a))
        days.append(today); remaining = rest
    return days


def generate(out_dir, start=2007, end=2017, seed=42):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(TEAMS)
    scoring = rng.normal(0, 2.0, n)      # contribution to game totals
    strength = rng.normal(0, 4.0, n)     # point-differential ability
    for y in range(start, end + 1):
        scoring = 0.6 * scoring + rng.normal(0, 1.8, n)
        strength = 0.6 * strength + rng.normal(0, 3.2, n)
        base = 199 + 1.35 * (y - 2007) + rng.normal(0, 0.8)
        last_day, rows, rot = {}, [], 1
        date0 = pd.Timestamp(year=y, month=10, day=25)
        for d, games in enumerate(_schedule(rng, n)):
            date = date0 + pd.Timedelta(days=d)
            for (h, a) in games:
                b2b_h = last_day.get(h, -9) == d - 1
                b2b_a = last_day.get(a, -9) == d - 1
                last_day[h] = last_day[a] = d
                mu_tot = base + scoring[h] + scoring[a] - 1.2 * b2b_h - 1.2 * b2b_a
                mu_mar = 3.0 + strength[h] - strength[a] - 1.5 * b2b_h + 1.5 * b2b_a
                tot = mu_tot + rng.normal(0, 16.5)
                mar = mu_mar + rng.normal(0, 12.0)
                hp, ap = int(round((tot + mar) / 2)), int(round((tot - mar) / 2))
                line = np.round((mu_tot + rng.normal(0, 3.0)) * 2) / 2
                spread = np.round((-mu_mar) * 2) / 2
                fin = {"V": ap, "H": hp}
                for vh, t in (("V", a), ("H", h)):
                    p = rng.dirichlet([40] * 4)
                    q = np.round(p * fin[vh]).astype(int); q[3] = fin[vh] - q[:3].sum()
                    is_fav_row = (vh == "H") == (spread < 0)
                    val = -abs(spread) if is_fav_row else line
                    rows.append({"Date": int(date.month * 100 + date.day), "Rot": rot, "VH": vh,
                                 "Team": TEAMS[t], "1st": q[0], "2nd": q[1], "3rd": q[2], "4th": q[3],
                                 "Final": fin[vh], "Open": val, "Close": val,
                                 "ML": -150 if is_fav_row else 130, "2H": 0})
                    rot += 1
        pd.DataFrame(rows).to_excel(out_dir / f"nba odds {season_label(y)}.xlsx", index=False)
    return out_dir
