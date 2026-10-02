# The Bank Is Open: Predicting NBA Over/Under Totals with Machine Learning

UE24CS352A Machine Learning, mini-project (Team 3). Predict the **total points scored in an NBA game** and compare the
prediction with the sportsbook Over/Under line. Reference study: *The Bank Is Open: AI in Sports Gambling* (Bucquet and Sarukkai, Stanford CS229).

Models: Random Forest (baseline), Collaborative Filtering via masked SVD/ALS, fully-connected Neural Network, and an LSTM
written from scratch in NumPy (hand-derived back-propagation through time, checked by a gradient-check test).

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ recommended. No GPU and no PyTorch needed.

## 2. Get the data (real run)

1. Open the Sportsbook Reviews Online NBA odds archive:
   `https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm`
2. Download the Excel file for every season **2007-08 ... 2017-18** (11 files).
3. Put them in `data/raw/` and make sure each file name contains the season, e.g. `nba odds 2012-13.xlsx`.

Each game is two rows (V = visitor, H = home) with columns `Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H`.
The loader pairs the rows, builds the game table and takes the larger of the two `Close` numbers as the total line.

## 3. Run everything (one command)

```bash
python run_all.py --data real            # full tuning + final evaluation   (~15-25 min on a laptop CPU)
python run_all.py --data real --quick    # smoke test with a tiny search budget (~2 min)
python scripts/build_report.py           # -> report/writeup.pdf   (numbers come from results/metrics.json)
node scripts/build_slides.js             # -> slides/deck.pptx     (needs: npm install pptxgenjs)
```

`--data synthetic` generates fake NBA-like files in the same Excel layout so the pipeline can be tested without downloads.
**Synthetic results are not NBA results**; reports/slides built from them show a "DEMO DATA" banner.

Outputs (in `results/`): `metrics.json`, `tuning_log.csv`, `test_predictions.csv`, `figures/*.png`; trained models in `models_saved/`.

## 4. Live demo

```bash
python demo.py --data real --date 2018-03-10                 # all games that day
python demo.py --data real --date 2018-03-10 --home GoldenState
```
Prints, for each game: sportsbook line, actual total, the four model predictions and each model's OVER/UNDER pick.

## 5. Tests

```bash
pytest -q tests
```
Covers: loader pairing, **no look-ahead leakage** in features, chronological split integrity, O/U metric maths, LSTM gradient check.

## 6. Method summary

| Item | Choice |
|---|---|
| Target | total points (home + away) |
| Split | train 2012-13..2015-16 (first season tuned), validation 2016-17, test 2017-18 (touched once) |
| Features | last 3 games per team (points for/against, home/away, rest, opponent season-to-date strength), rest days, back-to-back, away streak, season-to-date averages, team one-hot |
| Loss / metric | MSE; also MAE, Over/Under accuracy (+95% CI), ROI at -110 (break-even 52.38%) |
| Parameters (learned) | tree splits; U, V factors; NN weights; LSTM gate weights |
| Hyperparameters (tuned on validation only) | RF: n_estimators, max_depth, min_samples_leaf, max_features. CF: n_factors, horizon_days, reg. NN: hidden sizes, lr, L2 alpha, batch size, epochs (early stopping). LSTM: hidden size, lr, dropout, weight decay, batch size, epochs (early stopping). Data: first training season |
| Search | random search (RF, NN, LSTM), full grid (CF); all trials in `results/tuning_log.csv` |

### Deviations from the reference paper (and why)
- **Data source:** only the SBRO Excel files are needed. The paper also scraped Basketball Reference (rebounds, Offensive Rating, Plus/Minus, travel distance); that scraper is fragile, so features here are score-based plus rest/travel proxies.
- **Optimiser settings:** the paper's lr = 1e-6 with weight decay = 1 on unscaled inputs is replaced by standardised inputs, Adam and tuned values.
- **LSTM:** the gate equations printed in the paper are those of a GRU; the standard 4-gate LSTM is implemented because the problem statement asks for an LSTM.
- **CF train error:** the paper reports train MSE 2.95; here train error is computed the same walk-forward way as test (no leakage), so it is comparable.

## 7. Repository layout

```
run_all.py            full pipeline          demo.py            live predictions
src/data_loader.py    SBRO xlsx -> games     src/features.py    leakage-free features
src/models/           rf, cf_svd, nn, lstm_numpy
src/tuning.py         validation-only search src/evaluate.py    MSE / O-U / ROI
src/plots.py          figures                src/data_synth.py  synthetic demo data
scripts/              build_report.py, build_slides.js
tests/                pytest suite           results/           metrics, logs, figures
report/writeup.pdf    PDF write-up           slides/deck.pptx   review deck
team.json             names, SRNs, repo URL (edit before building report/slides)
```

