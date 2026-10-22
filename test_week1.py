"""test_week1.py: checks that would catch the mistakes interviewers look for.

Run: python test_week1.py   (prints PASS lines, stops at the first failure)
The two bhavcopy files here are SYNTHETIC: built to the real column layouts
with prices from a known vol, so we can check the parser and IV solver.
Re-check read_bhavcopy() on one real file of each layout once you download them.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import features
import models
import options
from data import load_all

TMP = Path(__file__).resolve().parent / "data" / "test_tmp"


def check(name, ok):
    assert ok, name
    print("PASS", name)


def test_target_alignment():
    """Target at row t must use returns t+1..t+21 and nothing else."""
    close = pd.Series(np.exp(np.cumsum(np.r_[0, np.full(60, 0.01)])))
    close.iloc[25:] *= 1.5          # a jump: only day 25's return is affected
    tgt = features.forward_rv(close)
    base = tgt.iloc[2]
    # Row t covers returns t+1..t+21, so day 25 sits in rows 4..24 only.
    check("row 3 (window ends day 24) misses the jump", np.isclose(tgt.iloc[3], base))
    check("row 4 (window ends day 25) catches the jump", tgt.iloc[4] > 2 * base)
    check("row 24 (window starts day 25) catches the jump", tgt.iloc[24] > 2 * base)
    check("row 25 ignores its own day's return", np.isclose(tgt.iloc[25], base))


def test_no_lookahead():
    """Changing future targets must not change today's forecast."""
    tab = features.build(load_all()).iloc[:1200].copy()
    base = models.walk_forward(tab)
    t = 900
    poisoned = tab.copy()
    poisoned.iloc[t - features.H + 1:, poisoned.columns.get_loc("target")] = 9.99
    check("walk-forward uses no unseen targets",
          np.isclose(base.iloc[t], models.walk_forward(poisoned).iloc[t]))


def test_black76():
    F, K, T, r, s = 24000, 24100, 30 / 365, 0.065, 0.14
    c, p = options.black76(F, K, T, r, s, 1), options.black76(F, K, T, r, s, -1)
    # Put-call parity on a future: C - P = e^(-rT) (F - K)
    check("Black-76 obeys put-call parity", np.isclose(c - p, np.exp(-r * T) * (F - K)))
    check("implied vol inverts the price", np.isclose(options.implied_vol(c, F, K, T, r, 1), s))


def _fake_day(date, layout, vol=0.14, F=24030.0, r=0.065):
    exp = date + pd.Timedelta(days=29)
    T = 29 / 365
    rows = [("FUTIDX", exp, np.nan, "XX", F)]
    for K in [23900, 23950, 24000, 24050, 24100]:
        for cp, typ in [(1, "CE"), (-1, "PE")]:
            rows.append(("OPTIDX", exp, K, typ, round(options.black76(F, K, T, r, vol, cp), 2)))
    if layout == "old":
        df = pd.DataFrame(rows, columns=["INSTRUMENT", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "SETTLE_PR"])
        df["EXPIRY_DT"] = df["EXPIRY_DT"].dt.strftime("%d-%b-%Y").str.upper()
        df["TIMESTAMP"] = date.strftime("%d-%b-%Y").upper()
        df["SYMBOL"], df["CLOSE"], df["OPEN_INT"] = "NIFTY", df["SETTLE_PR"], 1000
    else:
        df = pd.DataFrame(rows, columns=["FinInstrmTp", "XpryDt", "StrkPric", "OptnTp", "SttlmPric"])
        df["FinInstrmTp"] = df["FinInstrmTp"].map({"FUTIDX": "IDF", "OPTIDX": "IDO"})
        df["OptnTp"] = df["OptnTp"].replace("XX", np.nan)
        df["XpryDt"] = df["XpryDt"].dt.strftime("%Y-%m-%d")
        df["TradDt"] = date.strftime("%Y-%m-%d")
        df["TckrSymb"], df["ClsPric"], df["OpnIntrst"] = "NIFTY", df["SttlmPric"], 1000
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"{layout}.csv.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{layout}.csv", df.to_csv(index=False))
    return path


def test_both_layouts():
    old = options.atm_series([_fake_day(pd.Timestamp("2023-03-01"), "old")])
    new = options.atm_series([_fake_day(pd.Timestamp("2025-03-03"), "new")])
    for name, res in [("old", old), ("new", new)]:
        row = res.iloc[0]
        check(f"{name} layout: ATM strike picked", row["K"] == 24050)
        check(f"{name} layout: implied vol recovered", abs(row["atm_iv"] - 0.14) < 0.002)


if __name__ == "__main__":
    test_target_alignment()
    test_no_lookahead()
    test_black76()
    test_both_layouts()
    print("all checks passed")
