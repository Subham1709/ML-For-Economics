"""Streamlit page for the Nifty volatility project.

Reads app/assets/forecasts.csv (written by report.py) and shows how the
HAR forecast and India VIX compare with the volatility that followed.
Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
NAMES = {"har_cc": "HAR", "vix": "India VIX"}

st.set_page_config(page_title="Nifty volatility: HAR vs India VIX", layout="wide")


@st.cache_data
def load():
    d = pd.read_csv(ASSETS / "forecasts.csv", parse_dates=["date"]).set_index("date")
    coef = pd.read_csv(ASSETS / "har_coefficients.csv", index_col=0)
    return d, coef


def rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))


def qlike(a, f):
    r = a ** 2 / f ** 2
    return float(np.mean(r - np.log(r) - 1))


df, coef = load()
full = df.dropna(subset=["target", "har_cc", "vix"])

st.title("Forecasting Nifty 50 volatility: HAR vs India VIX")
st.caption("AIL 7310 Machine Learning for Economics, IIT Jodhpur. Subham Agrawal. "
           "[Code on GitHub](https://github.com/Subham1709/ML-For-Economics)")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast vs actual", "Scores", "How HAR works", "About"])

with tab1:
    st.write("Each forecast is made at the close of a day for the volatility of the next 21 trading days. "
             "The grey line is what actually happened.")
    lo, hi = full.index.min().date(), full.index.max().date()
    start, end = st.slider("Period", lo, hi, (pd.Timestamp("2020-01-01").date(), hi))
    d = full.loc[str(start):str(end)]
    chart = (100 * d[["target", "har_cc", "vix"]]).rename(
        columns={"target": "Realised, next 21 days", "har_cc": "HAR forecast", "vix": "India VIX"})
    st.line_chart(chart, color=["#52514e", "#2a78d6", "#eb6834"], y_label="Annualised volatility (%)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average realised volatility", f"{100 * d['target'].mean():.1f}%")
    c2.metric("Average HAR forecast", f"{100 * d['har_cc'].mean():.1f}%")
    c3.metric("Average India VIX", f"{100 * d['vix'].mean():.1f}%",
              f"{100 * (d['vix'] - d['target']).mean():+.1f} points vs realised", delta_color="off")

with tab2:
    st.write("Out-of-sample errors. HAR is refitted every day using only past data. Lower is better.")
    s2, e2 = st.slider("Period for scores", lo, hi, (lo, hi), key="scores")
    d = full.loc[str(s2):str(e2)]
    rows = []
    for col, name in NAMES.items():
        rows.append({"Forecast": name,
                     "RMSE (vol points)": round(100 * rmse(d["target"], d[col]), 2),
                     "QLIKE": round(qlike(d["target"], d[col]), 3),
                     "Mean forecast (%)": round(100 * d[col].mean(), 1)})
    st.dataframe(pd.DataFrame(rows).set_index("Forecast"), width='stretch')
    st.caption(f"{len(d):,} trading days, {d.index.min().date()} to {d.index.max().date()}. "
               f"Mean realised volatility {100 * d['target'].mean():.1f}%.")
    st.write("Over the full sample India VIX has the lower error, so the market's forecast is the one "
             "the ML models have to beat. VIX is also higher on average than the volatility that follows. "
             "That gap is the variance risk premium.")
    gap = (100 * (d["vix"] - d["target"])).rename("India VIX minus realised (points)")
    st.line_chart(gap.rolling(63).mean().dropna(), y_label="63-day average gap (vol points)")

with tab3:
    st.write("HAR (Corsi, 2009) forecasts volatility from its own past over three windows:")
    st.latex(r"RV_{t+1:t+21} = \beta_0 + \beta_d\,RV^{day}_t + \beta_w\,RV^{week}_t + \beta_m\,RV^{month}_t + \varepsilon_t")
    st.write("Coefficients on the full sample (Newey-West standard errors):")
    st.dataframe(coef.round(3), width='stretch')
    st.write("Try it: set the three inputs and see the forecast.")
    a, b, c = st.columns(3)
    rd = a.slider("Volatility yesterday (%)", 5, 80, 15)
    rw = b.slider("Volatility last 5 days (%)", 5, 80, 15)
    rm = c.slider("Volatility last 22 days (%)", 5, 80, 15)
    k = coef["coef"]
    f = k["const"] + k["rv_d"] * rd / 100 + k["rv_w"] * rw / 100 + k["rv_m"] * rm / 100
    st.metric("HAR forecast for the next 21 days", f"{100 * f:.1f}%")
    st.caption("The monthly term has the largest weight: for a one-month horizon, slow information matters most.")

with tab4:
    st.markdown("""
**Question.** Can machine learning models forecast the 21-day realised volatility of the Nifty 50 better than
the HAR benchmark and better than India VIX?

**Data.** Nifty 50 daily OHLC and India VIX from Yahoo Finance. Option prices from NSE bhavcopy files (for the trading test).

**Done so far.** Data pipeline, HAR benchmark, walk-forward comparison with India VIX.

**Next.** Lasso, random forest and LightGBM on a wider feature set, a Diebold-Mariano test,
and a delta-hedged straddle backtest on NSE option prices.
""")
