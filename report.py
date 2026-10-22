"""report.py: run the pipeline and write the paper's tables and figures.

Week 1 builds: HAR coefficients, out-of-sample HAR vs VIX losses, and
the first figure. Run: python report.py  -> files in output/ and figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import features
import models
from data import load_all

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "output", ROOT / "figures"
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1 and 2


def har_table(tab):
    res = models.fit_har_full(tab)
    t = pd.DataFrame({"coef": res.params, "NW se": res.bse, "t": res.tvalues})
    t.attrs["r2"], t.attrs["n"] = res.rsquared, int(res.nobs)
    return t


def forecasts(df):
    """Out-of-sample HAR forecasts with GK and close-to-close features."""
    tab = features.build(df, "gk")
    fc = pd.DataFrame({"har_gk": models.walk_forward(tab),
                       "har_cc": models.walk_forward(features.build(df, "cc"))})
    fc["vix"] = tab["vix"]
    return tab, fc


def plot_rv(tab, fc, path, start="2020-01-01"):
    d = pd.concat([tab["target"], fc[["har_gk", "vix"]]], axis=1).loc[start:]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(d.index, 100 * d["target"], color=MUTED, lw=1.2, label="Realised, next 21 days")
    ax.plot(d.index, 100 * d["har_gk"], color=BLUE, lw=1.6, label="HAR forecast")
    ax.plot(d.index, 100 * d["vix"], color=ORANGE, lw=1.6, label="India VIX")
    ax.set_ylabel("Annualised volatility (%)", color=MUTED)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    ax.set_title("Nifty 50: realised volatility vs HAR forecast and India VIX",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = load_all()
    tab, fc = forecasts(df)

    ht = har_table(tab)
    print(f"HAR full sample (n={ht.attrs['n']}, R2={ht.attrs['r2']:.3f})")
    print(ht.round(4))
    ht.to_csv(OUT / "har_full_sample.csv")

    # Compare all three only where all three exist (VIX starts 2020).
    ev = models.evaluate(tab, fc)
    lo, hi, n = ev.attrs["sample"]
    print(f"\nOut-of-sample losses, {lo} to {hi}, n={n}")
    print(ev.round(4))
    ev.to_csv(OUT / "losses_har_vs_vix.csv")

    # HAR alone over its longer history.
    ev_long = models.evaluate(tab, fc[["har_gk", "har_cc"]])
    lo, hi, n = ev_long.attrs["sample"]
    print(f"\nHAR only, {lo} to {hi}, n={n}")
    print(ev_long.round(4))

    pd.concat([tab, fc[["har_gk", "har_cc"]]], axis=1).to_csv(OUT / "forecasts.csv")
    plot_rv(tab, fc, FIG / "rv_har_vix.png")
    print(f"\nwrote {OUT} and {FIG / 'rv_har_vix.png'}")


if __name__ == "__main__":
    main()
