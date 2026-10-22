"""options.py: NSE F&O bhavcopy -> daily ATM Nifty straddle and implied vol.

Bhavcopy = NSE's end-of-day file with one row per F&O contract.
NSE switched to a new file layout (UDiFF) from 8 Jul 2024, so this file
downloads and reads both layouts and maps them to one set of columns.

Download runs on YOUR laptop (the cloud workspace cannot reach NSE):
    python options.py download 2016-01-01 2026-10-21
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

BHAV = Path(__file__).resolve().parent / "data" / "bhavcopy"
UDIFF_START = pd.Timestamp("2024-07-08")
OLD_URL = ("https://archives.nseindia.com/content/historical/DERIVATIVES/"
           "{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip")
NEW_URL = ("https://nsearchives.nseindia.com/content/fo/"
           "BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}

# New UDiFF column -> old column name; we keep the old names everywhere.
UDIFF_MAP = {"TradDt": "TIMESTAMP", "TckrSymb": "SYMBOL", "XpryDt": "EXPIRY_DT",
             "StrkPric": "STRIKE_PR", "OptnTp": "OPTION_TYP", "OpnPric": "OPEN",
             "HghPric": "HIGH", "LwPric": "LOW", "ClsPric": "CLOSE",
             "SttlmPric": "SETTLE_PR", "OpnIntrst": "OPEN_INT",
             "TtlTradgVol": "VOLUME", "FinInstrmTp": "INSTRUMENT"}
UDIFF_TYPES = {"IDO": "OPTIDX", "IDF": "FUTIDX", "STO": "OPTSTK", "STF": "FUTSTK"}


def download(start, end, sleep=0.7):
    """Save one zip per trading day into data/bhavcopy. Skips files you have.

    Weekends are skipped; holidays return 404 and are logged, not retried.
    """
    import requests
    BHAV.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get("https://www.nseindia.com", timeout=20)    # sets the cookies NSE expects
    for d in pd.bdate_range(start, end):
        out = BHAV / f"fo_{d:%Y%m%d}.csv.zip"
        if out.exists():
            continue
        mon = d.strftime("%b").upper()
        urls = [NEW_URL, OLD_URL] if d >= UDIFF_START else [OLD_URL, NEW_URL]
        for url in urls:
            r = s.get(url.format(d=d, mon=mon), timeout=30)
            if r.status_code == 200 and r.content[:2] == b"PK":   # a real zip
                out.write_bytes(r.content)
                break
        else:
            print(f"{d.date()}: no file (holiday or not yet published)")
        time.sleep(sleep)                                # be polite to NSE


def read_bhavcopy(path):
    """Read either layout; return Nifty index options and futures only."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "TckrSymb" in df.columns:                         # new UDiFF layout
        df = df.rename(columns=UDIFF_MAP)
        df["INSTRUMENT"] = df["INSTRUMENT"].map(UDIFF_TYPES)
        fmt = {"format": "%Y-%m-%d"}                     # dates like 2024-07-08
    else:                                                # old layout
        fmt = {"format": "%d-%b-%Y"}                     # dates like 28-JAN-2016
    df = df[(df["SYMBOL"].str.strip() == "NIFTY") & df["INSTRUMENT"].isin(["OPTIDX", "FUTIDX"])].copy()
    df["date"] = pd.to_datetime(df["TIMESTAMP"].str.strip(), **fmt)
    df["expiry"] = pd.to_datetime(df["EXPIRY_DT"].str.strip(), **fmt)
    df["OPTION_TYP"] = df["OPTION_TYP"].fillna("XX").str.strip()
    keep = ["date", "INSTRUMENT", "expiry", "STRIKE_PR", "OPTION_TYP", "CLOSE", "SETTLE_PR", "OPEN_INT"]
    return df[keep].rename(columns=str.lower)


def black76(F, K, T, r, sigma, cp):
    """European option on a future (Black 1976). cp = +1 call, -1 put."""
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def implied_vol(price, F, K, T, r, cp):
    """Solve black76(sigma) = price for sigma; NaN if no solution exists."""
    f = lambda s: black76(F, K, T, r, s, cp) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:      # price below intrinsic or above the max possible
        return np.nan


def atm_straddle(day, r=0.065, target_days=30):
    """Pick the monthly expiry nearest 30 calendar days and its ATM strike.

    Using the Nifty FUTURE of that expiry as F (Black-76) means we never
    have to guess the dividend yield: the futures price already includes it.
    """
    fut = day[day["instrument"] == "FUTIDX"]
    if fut.empty:
        return None
    t0 = day["date"].iloc[0]
    fut = fut.assign(days=(fut["expiry"] - t0).dt.days)
    fut = fut[fut["days"] >= 7]                         # avoid expiry-week noise
    row = fut.iloc[(fut["days"] - target_days).abs().argmin()]
    F, exp, T = row["settle_pr"], row["expiry"], row["days"] / 365
    opts = day[(day["instrument"] == "OPTIDX") & (day["expiry"] == exp)]
    K = opts["strike_pr"].iloc[(opts["strike_pr"] - F).abs().argmin()]
    leg = opts[opts["strike_pr"] == K].set_index("option_typ")["settle_pr"]
    if not {"CE", "PE"} <= set(leg.index):
        return None
    c, p = leg["CE"], leg["PE"]
    iv = np.nanmean([implied_vol(c, F, K, T, r, 1), implied_vol(p, F, K, T, r, -1)])
    return {"date": t0, "expiry": exp, "days": row["days"], "F": F, "K": K,
            "call": c, "put": p, "straddle": c + p, "atm_iv": iv}


def atm_series(files=None):
    """One row per day: the ATM straddle and its implied vol."""
    files = files or sorted(BHAV.glob("fo_*.csv.zip"))
    rows, skipped = [], []
    for f in files:
        row = atm_straddle(read_bhavcopy(f))
        (rows if row else skipped).append(row if row else Path(f).name)
    if skipped:   # never drop days silently: say which files gave no straddle
        print(f"skipped {len(skipped)} of {len(files)} files: {skipped[:10]}")
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "download":
        download(sys.argv[2], sys.argv[3])
    else:
        print(atm_series().tail())
