"""Central configuration: paths, seeds, chronological splits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SYNTH = ROOT / "data" / "synthetic"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS_DIR = ROOT / "models_saved"

SEED = 42
LOOKBACK = 3                      # past games per team used as features (paper: 3)
TRAIN_SEASONS = ["2012-13", "2013-14", "2014-15", "2015-16"]
VAL_SEASONS = ["2016-17"]         # used ONLY for hyperparameter tuning / early stopping
TEST_SEASONS = ["2017-18"]        # touched ONCE, at the very end
BET_ODDS = -110                   # standard sportsbook price -> break-even 52.38 %
