"""models.py: the HAR benchmark, walk-forward forecasting and loss functions.

LightGBM joins this file next week with the same walk_forward() loop,
so both models are tested in exactly the same way.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import H

HAR_X = ["rv_d", "rv_w", "rv_m"]


def fit_har_full(tab):
    """In-sample HAR fit on all rows with a target, for the paper's table.

    Targets overlap (each covers 21 days, consecutive rows share 20 of them),
    so residuals are autocorrelated and plain OLS t-stats are far too big.
    Newey-West (HAC) standard errors with H-1 lags correct for that.
    """
    d = tab.dropna(subset=HAR_X + ["target"])
    X = sm.add_constant(d[HAR_X])
    return sm.OLS(d["target"], X).fit(cov_type="HAC", cov_kwds={"maxlags": H - 1})


def _ols(X, y):
    """Least-squares coefficients with an intercept, via numpy (fast)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def walk_forward(tab, cols=HAR_X, min_train=500, fit=_ols, refit_every=1):
    """Out-of-sample forecasts with an expanding window and no look-ahead.

    At the close of day t we may only train on rows s whose target is
    already fully observed. Row s's target ends on day s+H, so s <= t-H.
    Training on rows t-H+1 .. t would use returns we have not seen yet:
    the classic look-ahead bug in overlapping-horizon forecasting.
    """
    X = tab[cols].to_numpy()
    y = tab["target"].to_numpy()
    preds = np.full(len(tab), np.nan)
    beta = None
    for t in range(min_train + H, len(tab)):
        last = t - H                       # last row whose target is known
        if beta is None or (t - min_train - H) % refit_every == 0:
            ok = ~np.isnan(y[: last + 1]) & ~np.isnan(X[: last + 1]).any(axis=1)
            beta = fit(X[: last + 1][ok], y[: last + 1][ok])
        if not np.isnan(X[t]).any():
            preds[t] = beta[0] + X[t] @ beta[1:]
    # A vol forecast can never be negative; floor it at 1% just in case.
    # (np.maximum keeps NaN as NaN, so days with no forecast stay empty.)
    return pd.Series(np.maximum(preds, 0.01), index=tab.index, name="har")


# ---------- loss functions (inputs in vol units; losses on variance) ----------

def rmse(actual, forecast):
    """Root mean squared error in vol points. Punishes big misses most."""
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def qlike(actual, forecast):
    """QLIKE loss on variances: s/h - ln(s/h) - 1, zero for a perfect forecast.

    Patton (2011): with a noisy realised-vol proxy, QLIKE (like MSE on
    variance) still ranks forecasts correctly. It also punishes
    under-prediction more than over-prediction, which matters for risk.
    """
    ratio = actual ** 2 / forecast ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def mincer_zarnowitz(actual, forecast):
    """Regress actual on forecast. Unbiased forecast: intercept 0, slope 1."""
    X = sm.add_constant(forecast.rename("forecast"))
    res = sm.OLS(actual, X).fit(cov_type="HAC", cov_kwds={"maxlags": H - 1})
    return res.params, res.bse, res.rsquared


def evaluate(tab, forecasts):
    """Loss table for several forecast columns on their common sample."""
    d = pd.concat([tab["target"], forecasts], axis=1).dropna()
    rows = {}
    for name in forecasts.columns:
        params, se, r2 = mincer_zarnowitz(d["target"], d[name])
        rows[name] = {"RMSE": rmse(d["target"], d[name]),
                      "QLIKE": qlike(d["target"], d[name]),
                      "MZ intercept": params["const"], "MZ slope": params["forecast"],
                      "MZ R2": r2, "mean forecast": d[name].mean(),
                      "mean actual": d["target"].mean()}
    out = pd.DataFrame(rows).T
    out.attrs["sample"] = (d.index.min().date(), d.index.max().date(), len(d))
    return out
