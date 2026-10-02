"""Regression metrics + sportsbook-style Over/Under evaluation."""
import numpy as np
from .config import BET_ODDS

PAYOUT = 100 / abs(BET_ODDS)                  # profit per 1 unit staked at -110 -> 0.909
BREAK_EVEN = 1 / (1 + PAYOUT)                 # 52.38 %


def regression_metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    return {"mse": float(np.mean(e ** 2)), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}


def ou_metrics(pred, line, actual):
    """Bet OVER when pred > line else UNDER. Pushes (actual == line) are excluded."""
    pred, line, actual = map(np.asarray, (pred, line, actual))
    ok = ~np.isnan(line) & (actual != line)
    hit = (pred[ok] > line[ok]) == (actual[ok] > line[ok])
    n = int(ok.sum()); acc = float(hit.mean()) if n else float("nan")
    ci = 1.96 * np.sqrt(acc * (1 - acc) / n) if n else float("nan")
    roi = acc * PAYOUT - (1 - acc)
    return {"n_bets": n, "accuracy": acc, "ci95": float(ci), "roi_per_unit": float(roi),
            "break_even": BREAK_EVEN, "beats_break_even": bool(acc > BREAK_EVEN)}


def edge_table(pred, line, actual, thresholds=(0, 1, 2, 3, 4, 6)):
    pred, line, actual = map(np.asarray, (pred, line, actual))
    rows = []
    for t in thresholds:
        m = np.abs(pred - line) >= t
        r = ou_metrics(pred[m], line[m], actual[m]) if m.sum() else {"n_bets": 0, "accuracy": float("nan"), "roi_per_unit": float("nan")}
        rows.append({"min_edge": t, "n_bets": r["n_bets"], "accuracy": r["accuracy"], "roi_per_unit": r["roi_per_unit"]})
    return rows
