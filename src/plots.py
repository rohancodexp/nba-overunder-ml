"""Figures for the report and slides."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, GREY, TEAL = "#14213D", "#E8590C", "#8D99AE", "#2A9D8F"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def curves(hist, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(hist["train_mse"], color=NAVY, label="train"); ax.plot(hist["val_mse"], color=ORANGE, label="validation")
    ax.axvline(int(np.argmin(hist["val_mse"])), color=GREY, ls="--", lw=0.8, label="best epoch")
    ax.set(title=title, xlabel="epoch", ylabel="MSE"); ax.legend(frameon=False)
    _save(fig, path)


def cf_horizon(tuning, path):
    d = tuning[tuning.model == "cf"].copy()
    P = pd.DataFrame(d.params.apply(json.loads).tolist()); d = pd.concat([d.reset_index(drop=True), P], axis=1)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, g in d.groupby("n_factors"):
        s = g.groupby("horizon_days").val_mse.min(); ax.plot(s.index, s.values, marker="o", ms=3, label=f"k={k}")
    ax.set(title="CF/SVD: validation MSE vs. horizon", xlabel="days of league history used", ylabel="val MSE")
    ax.legend(frameon=False, title="factors"); _save(fig, path)


def tuning_overview(tuning, path):
    names = ["rf", "cf", "nn", "lstm"]; labels = ["Random Forest", "CF / SVD", "Neural Net", "LSTM"]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, n in enumerate(names):
        v = tuning[tuning.model == n].val_mse.values
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, color=GREY, s=14, alpha=.8)
        ax.scatter([i], [v.min()], color=ORANGE, s=40, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=0, fontsize=7.5)
    ax.set_yscale("log"); ax.set_yticks([250, 300, 400, 600, 1000]); ax.set_yticklabels(["250", "300", "400", "600", "1000"]); ax.minorticks_off()
    ax.set(ylabel="validation MSE (log scale)", title="Every tuning trial (orange = selected)"); _save(fig, path)


def comparison(metrics, path):
    rows = [(k, v["test"]["mse"]) for k, v in metrics["models"].items()]
    rows.append(("Sportsbook line", metrics["sportsbook"]["test_mse"]))
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    cols = [ORANGE if r[0] == "Sportsbook line" else NAVY for r in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=cols)
    for i, r in enumerate(rows): ax.text(r[1] + 2, i, f"{r[1]:.1f}", va="center", fontsize=8)
    ax.set(xlabel="test MSE (lower is better)", title="Test-season MSE"); ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    _save(fig, path)


def pred_vs_actual(y, p, name, path):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.scatter(p, y, s=4, alpha=.35, color=NAVY)
    lo, hi = min(p.min(), y.min()), max(p.max(), y.max()); ax.plot([lo, hi], [lo, hi], color=ORANGE, lw=1)
    ax.set(xlabel="predicted total", ylabel="actual total", title=f"{name}: predicted vs actual (test)"); _save(fig, path)


def edge_plot(edge_rows, name, path, be):
    e = pd.DataFrame(edge_rows).dropna()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(e.min_edge, e.accuracy * 100, marker="o", color=NAVY, label=f"{name} accuracy")
    ax.axhline(be * 100, color=ORANGE, ls="--", lw=1, label=f"break-even {be*100:.1f}%"); ax.axhline(50, color=GREY, ls=":", lw=1)
    for x, y, n in zip(e.min_edge, e.accuracy * 100, e.n_bets): ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
    ax.set(xlabel="only bet when |model - line| >= x points", ylabel="O/U accuracy (%)", title="Accuracy vs. required edge"); ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def importances(names, imp, path, top=12):
    o = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(4.2, 3.0)); ax.barh([names[i] for i in o], imp[o], color=NAVY)
    ax.set(title="Random Forest: top features", xlabel="importance"); ax.tick_params(axis="y", labelsize=6.5); _save(fig, path)
