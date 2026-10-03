"""Fast tests: loader round-trip, leakage-free features, split integrity, evaluation maths."""
import numpy as np
import pandas as pd
from src.data_synth import generate
from src.data_loader import load_all
from src.features import build
from src.evaluate import ou_metrics, edge_table, BREAK_EVEN
from src import config as C


def _games(tmp_path):
    generate(tmp_path, start=2007, end=2009, seed=1)
    return load_all(tmp_path)


def test_loader_pairs_games_and_total_line(tmp_path):
    g = _games(tmp_path)
    assert len(g) == 3 * 1230
    assert g.total_line.between(150, 280).all()
    assert (g.total == g.home_pts + g.away_pts).all()
    assert abs((g.total - g.total_line).mean()) < 3          # line is a good estimator of total


def test_features_use_only_past_games(tmp_path):
    g = _games(tmp_path)
    df, seq, meta = build(g)
    assert seq.shape == (len(df), 3, 12) and not df[meta["num_cols"]].isna().any().any()
    r = df.iloc[1500]
    prev = g[(g.date < r.date) & ((g.home == r.home) | (g.away == r.home))].sort_values("date").iloc[-1]
    pts = prev.home_pts if prev.home == r.home else prev.away_pts
    assert r.h_pts_l1 == pts                                  # lag-1 feature == previous game of that team
    assert r.h_rest == min(7, (r.date - prev.date).days)


def test_splits_are_chronological_and_disjoint():
    assert set(C.TRAIN_SEASONS).isdisjoint(C.VAL_SEASONS + C.TEST_SEASONS)
    assert max(C.TRAIN_SEASONS) < min(C.VAL_SEASONS) < min(C.TEST_SEASONS)


def test_ou_metrics_excludes_pushes_and_counts_hits():
    r = ou_metrics(np.array([210, 190, 205]), np.array([200, 200, 205]), np.array([215, 205, 205]))
    assert r["n_bets"] == 2 and r["accuracy"] == 0.5          # over hit, under missed, push dropped
    assert abs(BREAK_EVEN - 0.5238) < 1e-3

def test_edge_table_filters_by_prediction_edge():
    pred = np.array([201, 205, 208])
    line = np.array([200, 200, 200])
    actual = np.array([210, 190, 205])

    rows = edge_table(pred, line, actual, thresholds=(0, 4, 7))

    assert [r["n_bets"] for r in rows] == [3, 2, 1]
    assert rows[1]["min_edge"] == 4
    assert rows[2]["min_edge"] == 7
