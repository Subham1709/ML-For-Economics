# Did SEBI’s F&O curbs reduce daily Nifty 50 volatility?

This pilot accompanies the ML for Economics presentation. It estimates a daily volatility baseline using pre-reform relationships with global markets. The main course project studies a separate 21-day forecast and straddle strategy.

## Results

The main window covers 51 regular NSE sessions from **21 November 2024 through 31 January 2025**, before the February 2025 phase of the reforms. The outcome is daily Garman–Klass volatility from Nifty 50 OHLC prices.

| Comparison | Daily volatility or gap |
| --- | ---: |
| Observed mean | 0.664% |
| Gradient boosting forecast | 0.523% |
| Observed minus gradient boosting | +0.141 pp |
| Observed minus historical-mean forecast | +0.055 pp |
| Observed minus previous 51-session mean | +0.006 pp |
| Boosting gap in the prior-year placebo | +0.065 pp |
| Boosting gap through December 2025 | −0.012 pp |

The January–September 2024 holdout RMSE was **0.424 pp for the historical mean**, 0.441 pp for gradient boosting, 0.447 pp for random forest and 0.458 pp for OLS. Boosting underpredicted by 0.115 pp on that holdout. Subtracting that earlier average error leaves a descriptive gap of about +0.026 pp.

India VIX was 1.64 index points above its short-window forecast. Its prior-year placebo gap was 1.87 points. The pilot leaves the causal effect unresolved.

- [Results snapshot](pilot_results.json) records the key estimates, placebo results and holdout errors from the presentation run.
- [Weekly chart data](weekly_forecasts.csv) gives observed and forecast averages in daily volatility percent units. For example, 0.664 means 0.664%.

## Reproduce the run

Run these commands from this folder with Python 3.12 or 3.13:

~~~sh
python -m pip install -r requirements.txt
python prepare_data.py
python pilot.py
~~~

The downloader retrieves version-pinned public inputs. The analysis writes the full results.json and the weekly chart data in this folder. The presentation run used Python 3.13.5 and statsmodels 0.15.0.

## Design

- OLS, random forest and gradient boosting first fit January 2022–December 2023 (492 sessions) and predict January–September 2024 (184 sessions). For the short-window forecasts, each model refits on all 676 sessions through September 2024, before the announcement. The historical mean uses the same estimation dates.
- Predictors are the most recent prior-date US VIX close, S&P 500 log return and Brent futures log return, with weekday indicators. The analysis joins each global series separately by date and excludes weekend and shortened Muhurat sessions.
- The 2023 placebo uses the prior November–January window, with training through September 2023. The longer window includes additional reform phases and the NSE expiry calendar change.
- The full results include conditional 95% intervals for the mean forecast gaps, using five-lag Newey–West standard errors. These intervals cover serial dependence in the observed gaps. Forecast estimation, model bias and concurrent market changes are additional sources of uncertainty.

## Data and policy references

- [Historical files for Nifty 50, India VIX, S&P 500 and Brent futures](https://github.com/Hareeshkesavan/Stock-Market-Dataset/tree/aa60b746f089c3f54df6760bf021708c2d3ce1e8)
- [Cboe US VIX data mirror](https://github.com/datasets/finance-vix/blob/3dbea23c70cc2afbeeec8ac232c9ec72ac26ac21/data/vix-daily.csv) and [Cboe source page](https://www.cboe.com/tradable_products/vix/vix-historical-data/)
- [Independent Nifty 50 OHLC cross-check file](https://github.com/Meharjot29/NIFTY-50-Dashborad/tree/44d05656de6f2cea0e74c7d5aec2d23293426eb5) (249 matching dates across all four price fields)
- [SEBI circular](https://www.sebi.gov.in/sebi_data/attachdocs/oct-2024/1727786261636.pdf) and [NSE implementation notice](https://nsearchives.nseindia.com/content/circulars/FAOP64990.pdf)
