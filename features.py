"""features.py: returns, daily variance, HAR regressors and the forward target.

Everything is in ANNUALISED VOL units (like VIX), so 0.15 means 15%.
Row t only uses information known at the close of day t, except the
target, which by design looks 21 days ahead.
"""
import numpy as np
import pandas as pd

DAYS = 252        # trading days per year, for annualising
H = 21            # forecast horizon: about one calendar month
WEEK, MONTH = 5, 22  # HAR averaging windows (Corsi 2009)


def daily_variance(df, estimator="gk"):
    """One-day variance estimate for each day (not annualised).

    cc: squared close-to-close log return. Unbiased but very noisy.
    gk: Garman-Klass using the day's range, plus the overnight gap.
        Uses open, high, low and close, so it is several times less noisy.
    """
    r = np.log(df["close"]).diff()
    if estimator == "cc":
        return (r ** 2).rename("var_cc")
    overnight = np.log(df["open"] / df["close"].shift(1))  # gap at the open
    hl = np.log(df["high"] / df["low"])                     # intraday range
    co = np.log(df["close"] / df["open"])                   # open to close
    intraday = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    return (overnight ** 2 + intraday).rename("var_gk")


def har_features(var):
    """Daily, weekly and monthly realised vol from a daily variance series.

    Average the VARIANCE over the window, then take the square root,
    so each term is a proper vol over that window.
    """
    rv_d = np.sqrt(DAYS * var)
    rv_w = np.sqrt(DAYS * var.rolling(WEEK).mean())
    rv_m = np.sqrt(DAYS * var.rolling(MONTH).mean())
    return pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})


def forward_rv(close, h=H):
    """Target: realised vol over the NEXT h days, from close-to-close returns.

    Value at row t uses returns on days t+1 ... t+h. It is what VIX at
    day t is trying to predict, so the three forecasts share one target.
    """
    r2 = np.log(close).diff() ** 2
    # rolling(h).sum() at t+h covers t+1..t+h; shift(-h) moves it back to t.
    fwd_var = r2.rolling(h).sum().shift(-h)
    return np.sqrt(DAYS / h * fwd_var).rename("target")


def build(df, estimator="gk"):
    """Model table: HAR features, VIX (as a decimal), target.

    Rows without full features are dropped. The last h rows keep a NaN
    target: they are still valid days to FORECAST from, just not to train on.
    """
    out = har_features(daily_variance(df, estimator))
    out["vix"] = df["vix"] / 100
    out["target"] = forward_rv(df["close"])
    return out.dropna(subset=["rv_d", "rv_w", "rv_m"])


if __name__ == "__main__":
    from data import load_all

    tab = build(load_all())
    print(tab.describe().T[["count", "mean", "50%", "min", "max"]].round(3))
    # Sanity check: GK and close-to-close should agree on average.
    df = load_all()
    for est in ["cc", "gk"]:
        v = daily_variance(df, est)
        print(est, "average annual vol:", round(float(np.sqrt(DAYS * v.mean())), 4))
