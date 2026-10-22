"""data.py: load Nifty 50 and India VIX daily data into one clean table.

Output of load_all(): one row per Nifty trading day, indexed by date, with
columns open, high, low, close (index points) and vix (annualised vol, in %).
VIX is NaN on days we have no VIX for (before 2020 until the back-fill).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
NIFTY_CSV = RAW / "nifty50_daily_raw.csv"
VIX_CSV = RAW / "india_vix_daily_raw.csv"


def download_yahoo(start="2007-01-01"):
    """Fetch full Nifty (^NSEI) and India VIX (^INDIAVIX) history from Yahoo.

    Run this on your own laptop (pip install yfinance); the cloud workspace
    cannot reach Yahoo. It overwrites the two raw CSVs with longer history,
    which is how VIX gets back-filled to 2008.
    """
    import yfinance as yf  # imported here so the rest works without it

    for ticker, path in [("^NSEI", NIFTY_CSV), ("^INDIAVIX", VIX_CSV)]:
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.reset_index().to_csv(path, index=False)
        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")


def _ohlc(df):
    """Keep date + OHLC, make names lower case, sort, drop bad rows."""
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[["date", "open", "high", "low", "close"]].dropna()
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    # A valid bar has low <= open, close <= high and positive prices.
    ok = (df["low"] <= df[["open", "close"]].min(axis=1)) & \
         (df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] > 0)
    return df[ok]


def load_nifty(path=NIFTY_CSV):
    return _ohlc(pd.read_csv(path))


def load_vix(path=VIX_CSV):
    df = pd.read_csv(path, encoding="utf-8-sig")
    # The GitHub mirror names columns "Sum of Open" etc.; Yahoo uses "Open".
    df.columns = [c.replace("Sum of ", "").strip() for c in df.columns]
    return _ohlc(df)


def load_all():
    """Nifty OHLC on Nifty trading days, plus that day's VIX close."""
    nifty = load_nifty()
    vix = load_vix()["close"].rename("vix")
    # Left join: Nifty decides which days exist. A VIX print on a day
    # Nifty did not trade (rare data error) is dropped, not invented.
    return nifty.join(vix, how="left")


def quality_report(df):
    """Print the checks you should be able to quote in the paper."""
    r = np.log(df["close"]).diff()
    gaps = df.index.to_series().diff().dt.days
    print(f"rows {len(df)}, {df.index.min().date()} to {df.index.max().date()}")
    print(f"days with VIX: {df['vix'].notna().sum()} "
          f"(from {df['vix'].first_valid_index().date()})")
    print(f"calendar gaps over 5 days: {list(df.index[gaps > 5].date)}")
    big = r.abs().nlargest(3)
    print("largest daily moves:", {d.date(): round(v, 4) for d, v in big.items()})


if __name__ == "__main__":
    quality_report(load_all())
