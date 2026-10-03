# The Bank Is Open: Predicting NBA Over/Under Totals with Machine Learning

UE24CS352A Machine Learning, mini-project. Predict the **total points scored in an NBA game** and compare the prediction with the sportsbook Over/Under line. Reference study: *The Bank Is Open: AI in Sports Gambling* (Bucquet and Sarukkai, Stanford CS229).

Models: Random Forest (baseline), Collaborative Filtering via masked SVD/ALS, fully-connected Neural Network, an LSTM written from scratch in NumPy (hand-derived back-propagation through time, checked by a gradient-check test), and an equal-weight NN + LSTM + Random Forest ensemble.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10+ recommended. No GPU and no PyTorch needed.

## 2. Get the data (real run)

1. Open the Sportsbook Reviews Online NBA odds archive:
   https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
2. Obtain the Excel file for each available season 2007-08 through 2022-23.
3. Put the files in `data/raw/` and make sure each file name contains the season, for example:
   `nba odds 2012-13.xlsx`

Each game is represented by two rows:

- `V` = visitor
- `H` = home

The expected columns are:

```
Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H
```

The loader pairs the visitor and home rows, builds the game-level table, and uses the larger of the two `Close` values as the sportsbook total line.

The repository ignores raw data files, so the source spreadsheets remain local and are not committed to Git.

## 3. Run everything

### Full real-data experiment

```bash
python run_all.py --data real
```

The full run performs validation-only hyperparameter tuning followed by the final chronological train/validation/test evaluation.

### Quick run

```bash
python run_all.py --data real --quick
```

The quick mode uses a reduced search budget and is intended for faster testing.

### Synthetic data

```bash
python run_all.py --data synthetic
```

The synthetic mode generates fake NBA-like data in the same Excel layout so that the pipeline can be tested without downloading the real data. **Synthetic results are not NBA results.**

### Output files

The pipeline writes the following outputs under `results/`:

```
results/
├── metrics.json
├── tuning_log.csv
├── test_predictions.csv
└── figures/
    ├── edge_accuracy.png
    ├── lstm_curves.png
    ├── model_comparison.png
    ├── nn_curves.png
    ├── pred_vs_actual.png
    ├── rf_importance.png
    └── tuning_overview.png
```

- `metrics.json` contains the final experiment configuration and evaluation metrics.
- `tuning_log.csv` contains the validation results from hyperparameter searches.
- `test_predictions.csv` contains predictions for the held-out test set.
- `figures/` contains evaluation and diagnostic plots.

## 4. Live demo

Run the demo for all games on a particular date:

```bash
python demo.py --data real --date 2018-03-10
```

A specific home team can also be supplied:

```bash
python demo.py --data real --date 2018-03-10 --home GoldenState
```

The demo prints:

- game date
- home team
- away team
- sportsbook total line
- actual total
- model predictions
- Over/Under picks

## 5. Tests

Run the complete test suite:

```bash
python -m pytest -q tests
```

The test suite covers:

- sportsbook-row pairing in the data loader
- prevention of look-ahead leakage in feature construction
- chronological train/validation/test split integrity
- Over/Under metric calculations
- edge-table filtering
- LSTM numerical gradient checking

The final project test suite contains 6 tests.

## 6. Method summary

| Item | Choice |
|---|---|
| Target | Total points scored in the game (home + away) |
| Data | Sportsbook Reviews Online NBA odds data |
| Seasons | 2007-08 through 2022-23 |
| Train split | 2007-08 through 2015-16 |
| Validation split | 2016-17 |
| Test split | 2017-18 |
| Models | Random Forest, CF-SVD, Neural Network, NumPy LSTM, NN + LSTM + RF ensemble |
| Main loss | Mean Squared Error (MSE) |
| Additional metrics | MAE, Over/Under accuracy, 95% confidence interval, ROI at -110 |
| Betting break-even | 52.38% |
| Feature history | Previous 3 games for each team |
| Fatigue features | Rest days, back-to-back flag, consecutive-away-games streak, games played in previous 7 days |
| Team information | Season-to-date statistics, opponent strength, home/away team one-hot features |
| Hyperparameter tuning | Validation-only tuning |
| RF search | Random search |
| NN search | Random search |
| LSTM search | Random search |
| CF search | Full grid search |
| Training-start search | Validation MSE over candidate first training seasons |

### Feature construction

Every feature for a game is computed using information available strictly before that game.

For each team, the feature pipeline uses:

- points scored in the previous 3 games
- points allowed in the previous 3 games
- whether the previous games were home or away
- rest days before previous games
- opponent season-to-date scoring statistics
- current rest days
- back-to-back indicator
- consecutive-away-games streak
- number of games played in the previous 7 days
- season-to-date points scored
- season-to-date points allowed
- team identity through one-hot encoding

The `games_last_7d` feature counts only games occurring before the current game, so the current game is never included in its own feature calculation.

## 7. Final real-data experiment

The final full experiment used:

```
Total games:           19,641
Usable feature rows:   19,592

Numeric features:      48
Team one-hot features: 60

Training games:        11,532
Validation games:       1,309
Test games:             1,311
```

The candidate first training seasons were evaluated using validation MSE:

| First training season | Validation MSE |
|---|---|
| 2007-08 | 333.87 |
| 2009-10 | 334.34 |
| 2011-12 | 339.97 |
| 2012-13 | 337.96 |
| 2013-14 | 338.22 |

The final experiment therefore used **2007-08** as the first training season.

The chronological split was:

```
Training:   2007-08 through 2015-16
Validation: 2016-17
Test:       2017-18
```

## 8. Final model results

| Model | Test MSE | Test RMSE | Test MAE | O/U Accuracy |
|---|---|---|---|---|
| Random Forest | 350.70 | 18.73 | 15.09 | 49.7% |
| CF-SVD | 403.57 | 20.09 | 16.10 | 49.9% |
| Neural Network | 348.80 | 18.68 | 14.94 | 49.2% |
| LSTM | 344.24 | 18.55 | 14.86 | 49.7% |
| NN + LSTM + RF Ensemble | 343.02 | 18.52 | 14.87 | 49.1% |
| Sportsbook line | 320.30 | — | — | — |

The sportsbook line is included as a reference baseline for the held-out test set.

## 9. Deviations from the reference paper

### Data source

The project uses the Sportsbook Reviews Online NBA odds data.

The reference paper also used Basketball Reference data containing additional statistics such as rebounds, Offensive Rating, Plus/Minus, and travel distance.

This implementation uses the available sportsbook/game data together with score-based historical features, rest information, and a consecutive-away-games travel-fatigue proxy.

### Optimisation settings

The reference paper used different optimisation settings, including a learning rate of 1e-6 and weight decay of 1 on unscaled inputs.

This implementation instead uses standardised inputs, Adam-based optimisation, and validation-based hyperparameter tuning.

### LSTM implementation

The reference paper's printed recurrent equations correspond to a GRU-style formulation.

This project implements a standard four-gate LSTM because the assigned problem requires an LSTM.

The LSTM is implemented directly in NumPy, including its forward pass and back-propagation through time. A numerical gradient-check test is included to verify the backward-pass implementation.

### Collaborative Filtering training error

The reference paper reports a much smaller training MSE for collaborative filtering.

In this implementation, training error is evaluated using the same walk-forward methodology used for evaluation, avoiding leakage from future games.

## 10. Repository layout

```
nba-overunder-ml/
│
├── run_all.py              # Full training, tuning and evaluation pipeline
├── demo.py                 # Live prediction/demo script
│
├── src/
│   ├── config.py           # Project configuration
│   ├── data_loader.py      # SBRO Excel files -> game-level data
│   ├── data_synth.py       # Synthetic NBA-like data generation
│   ├── features.py         # Leakage-free feature engineering
│   ├── tuning.py           # Validation-only hyperparameter search
│   ├── evaluate.py         # MSE, MAE, Over/Under and ROI metrics
│   ├── plots.py            # Evaluation and diagnostic plots
│   └── models/
│       ├── rf.py           # Random Forest
│       ├── cf_svd.py       # Collaborative Filtering using SVD/ALS
│       ├── nn.py           # Fully-connected Neural Network
│       └── lstm_numpy.py   # NumPy LSTM with BPTT
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_lstm_gradients.py
│
├── results/                # Final metrics, tuning logs, predictions and figures
├── report/
│   └── writeup.pdf
│
├── team.json               # Team names, SRNs and repository URL
├── requirements.txt
├── .gitignore
└── README.md
```

## 11. Reproducibility

The main experiment can be reproduced with:

```bash
python run_all.py --data real
```

The quick version can be used to verify that the complete pipeline executes:

```bash
python run_all.py --data real --quick
```

The test suite can be run independently with:

```bash
python -m pytest -q tests
```

The live demonstration can be run with:

```bash
python demo.py --data real --date 2018-03-10
```