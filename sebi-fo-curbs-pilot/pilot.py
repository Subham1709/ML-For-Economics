"""Fit the pre-policy volatility baselines and report the pilot estimates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ANNOUNCEMENT = pd.Timestamp("2024-10-01")
IMPLEMENTATION = pd.Timestamp("2024-11-21")
END = pd.Timestamp("2025-01-31")
YEARS = (2022, 2023, 2024, 2025)


def read_index(name, year=None):
    suffix = "" if year is None else str(year)
    df = pd.read_csv(DATA / f"{name}{suffix}.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
    assert not df["date"].duplicated().any(), name
    return df.set_index("date").sort_index()


def read_global(name, value):
    df = pd.read_csv(DATA / f"{name}.csv", names=["date", value], parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_unique and df[value].notna().all(), name
    return df


def summarize(observed, predicted):
    gap = (observed - predicted).to_numpy()
    fit = sm.OLS(gap, np.ones(len(gap))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True}, use_t=True
    )
    return {
        "n": len(gap), "observed": observed.mean(), "predicted": predicted.mean(),
        "gap": gap.mean(), "ci_low": fit.conf_int()[0, 0], "ci_high": fit.conf_int()[0, 1],
        "relative_pct": 100 * gap.mean() / predicted.mean()
    }


nifty = pd.concat([read_index("nifty", y) for y in YEARS]).sort_index()
vix = pd.concat([read_index("indiavix", y) for y in YEARS]).sort_index()["close"]
assert nifty.index.equals(vix.index), "Series must have matching price dates"
assert (nifty["high"] >= nifty[["open", "close"]].max(axis=1)).all()
assert (nifty["low"] <= nifty[["open", "close"]].min(axis=1)).all()
overlap = read_index("niftyrecent").join(nifty, lsuffix="_check", how="inner")
crosscheck = {field: int(np.isclose(overlap[field + "_check"], overlap[field]).sum())
              for field in ["open", "high", "low", "close"]}
gk_var = 0.5 * np.log(nifty.high / nifty.low) ** 2
gk_var -= (2 * np.log(2) - 1) * np.log(nifty.close / nifty.open) ** 2
assert (gk_var >= -1e-12).all()
panel = pd.DataFrame({
    "vol": 100 * np.sqrt(gk_var.clip(lower=0)),
    "range": 100 * np.log(nifty.high / nifty.low),
    "india_vix": vix
}).reset_index()
excluded = pd.to_datetime(["2022-10-24", "2024-11-01", "2025-10-21"])
panel = panel.loc[(panel.date.dt.dayofweek < 5) & ~panel.date.isin(excluded)]
inputs = (("vixfull", "us_vix"), ("sp500", "sp_close"), ("brent", "brent_close"))
global_columns = ["us_vix"]
for name, field in inputs:
    series = read_global(name, field)
    if field.endswith("_close"):
        output = field.replace("_close", "_ret")
        series[output] = 100 * np.log(series[field]).diff()
        global_columns.append(output)
        series = series[["date", output]].dropna()
    else:
        output = field
    source_date = output + "_date"
    panel = pd.merge_asof(
        panel, series.rename(columns={"date": source_date}),
        left_on="date", right_on=source_date, direction="backward",
        allow_exact_matches=False, tolerance=pd.Timedelta(days=8)
    )
    assert panel[output].notna().all(), output
    assert (panel[source_date] < panel.date).all(), output

panel = panel.set_index("date")
weekdays = pd.get_dummies(panel.index.dayofweek, prefix="weekday", drop_first=True, dtype=float)
weekdays.index = panel.index
features = panel[global_columns].join(weekdays)


def models():
    return {
        "OLS": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=18,
                                                max_features=1.0, random_state=42, n_jobs=1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=125, learning_rate=0.03,
                                                        max_depth=2, min_samples_leaf=18,
                                                        random_state=42)
    }


def predict(model, target, mask_train, mask_predict):
    model.fit(features.loc[mask_train], panel.loc[mask_train, target])
    return pd.Series(model.predict(features.loc[mask_predict]), index=panel.index[mask_predict])


def scores(target):
    first = panel.index < "2024-01-01"
    holdout = (panel.index >= "2024-01-01") & (panel.index < ANNOUNCEMENT)
    actual = panel.loc[holdout, target]
    rmse = {"Historical mean": float(np.sqrt(np.mean((actual - panel.loc[first, target].mean())**2)))}
    biases = {}
    for name, model in models().items():
        forecast = predict(model, target, first, holdout)
        rmse[name] = float(np.sqrt(np.mean((actual - forecast)**2)))
        biases[name] = float((actual - forecast).mean())
    preferred = min(biases, key=rmse.get)
    return {"training_n": int(first.sum()), "holdout_n": int(holdout.sum()),
            "rmse": rmse, "bias": biases, "preferred": preferred}


def result_for(target):
    validation = scores(target)
    post = (panel.index >= IMPLEMENTATION) & (panel.index <= END)
    training = panel.index < ANNOUNCEMENT
    placebo = (panel.index >= "2023-11-21") & (panel.index <= "2024-01-31")
    before = panel.loc[panel.index < IMPLEMENTATION, target].tail(int(post.sum()))
    preferred = validation["preferred"]
    preferred_model = models()[preferred]
    predicted = predict(preferred_model, target, training, post)
    longer = (panel.index >= IMPLEMENTATION) & (panel.index <= "2025-12-31")
    longer_prediction = predict(models()[preferred], target, training, longer)
    same_estimator_placebo = predict(models()[preferred], target, panel.index < "2023-10-01", placebo)
    robust_to_anticipation = predict(models()[preferred], target, panel.index < IMPLEMENTATION, post)
    forecast_by_model = {name: predict(model, target, training, post) for name, model in models().items()}
    forecast_by_model["Historical mean"] = pd.Series(
        panel.loc[training, target].mean(), index=panel.index[post]
    )
    time_series = pd.DataFrame({"observed": panel.loc[post, target], "predicted": predicted})
    time_series["week"] = time_series.index.to_period("W-FRI").end_time.normalize()
    weekly = time_series.groupby("week")[["observed", "predicted"]].mean()
    return {
        "validation": validation, "refit_n": int(training.sum()),
        "pilot": summarize(panel.loc[post, target], predicted),
        "same_estimator_placebo": summarize(panel.loc[placebo, target], same_estimator_placebo),
        "through_2025": summarize(panel.loc[longer, target], longer_prediction),
        "fit_through_19_nov": summarize(panel.loc[post, target], robust_to_anticipation),
        "by_model": {name: summarize(panel.loc[post, target], prediction)
                     for name, prediction in forecast_by_model.items()},
        "naive": {"n": len(before), "mean": before.mean(), "first": str(before.index.min().date()),
                  "last": str(before.index.max().date()), "change": panel.loc[post, target].mean() - before.mean()},
        "weekly": [{"date": str(date.date()), **row.to_dict()} for date, row in weekly.iterrows()]
    }


output = {key: result_for(key) for key in ("vol", "range", "india_vix")}
output["qc"] = {"overlap": len(overlap), "crosscheck": crosscheck,
                "analysis_total_n": len(panel), "first": str(panel.index.min().date()),
                "last": str(panel.index.max().date())}
(ROOT / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
pd.DataFrame(output["vol"]["weekly"]).to_csv(
    ROOT / "weekly_forecasts.csv", index=False, float_format="%.10f", lineterminator="\n"
)
