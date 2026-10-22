# ML For Economics: forecasting Nifty volatility and the SEBI 2024 F&O curbs

Term paper for AIL 7310 (Machine Learning for Economics), IIT Jodhpur, by Subham Agrawal.
Code is drafted with AI assistance, as the course requires; the author explains every part.

## Research questions

1. Does LightGBM forecast Nifty 50's 21-day realised volatility better than the HAR-RV benchmark and India VIX?
2. Did SEBI's November 2024 F&O curbs shrink the variance risk premium that option sellers earn?

A delta-hedged ATM straddle strategy, backtested on real NSE option prices, links the two:
it sells volatility only when VIX sits well above the model's forecast.

## First results (week 1)

![Nifty 50 realised volatility vs HAR forecast and India VIX](figures/rv_har_vix.png)

Out-of-sample, Jan 2020 to Mar 2026 (n = 1,515):

| Forecast | RMSE (vol pts) | QLIKE | Mean forecast | Mean realised |
| --- | --- | --- | --- | --- |
| HAR, Garman-Klass inputs | 8.61 | 0.369 | 16.6% | 15.2% |
| HAR, close-to-close inputs | 8.68 | 0.376 | 16.0% | 15.2% |
| India VIX | 8.54 | 0.349 | 17.3% | 15.2% |

VIX edges out HAR on both losses and runs about 2.2 vol points above realised volatility:
the variance risk premium. Significance tests (Diebold-Mariano) come with the ML model.

## Setup

```
pip install -r requirements.txt
```

## Run order

| Step | Command | What it does |
| --- | --- | --- |
| 1 | `python data.py` | Loads Nifty and VIX, prints data checks |
| 2 | `python features.py` | Builds HAR features and the 21-day target, prints a summary |
| 3 | `python report.py` | HAR fit, out-of-sample HAR vs VIX, writes `output/` and `figures/` |
| 4 | `python test_week1.py` | Checks target alignment, no look-ahead, Black-76, bhavcopy parsing |

Data downloads (need access to Yahoo Finance and nseindia.com):

| Step | Command | What it does |
| --- | --- | --- |
| A | `python -c "import data; data.download_yahoo()"` | Replaces the raw CSVs with full history (VIX back to 2008) |
| B | `python options.py download 2016-01-01 2026-10-21` | Downloads daily F&O bhavcopy zips into `data/bhavcopy/` (about an hour) |
| C | `python options.py` | Builds the daily ATM straddle and implied vol from those files |

## Files

| File | Role |
| --- | --- |
| `data.py` | Loads and cleans Nifty OHLC and India VIX into one table |
| `features.py` | Close-to-close and Garman-Klass variance, HAR terms, forward 21-day target |
| `models.py` | HAR with Newey-West errors, expanding-window walk-forward, RMSE, QLIKE, Mincer-Zarnowitz |
| `options.py` | NSE bhavcopy download and parsing (old and UDiFF layouts), Black-76, ATM implied vol |
| `report.py` | Runs the pipeline; writes tables to `output/` and figures to `figures/` |
| `test_week1.py` | Checks target alignment, no look-ahead, put-call parity and both bhavcopy layouts |

## Data

- Nifty 50 daily OHLC (Yahoo ^NSEI), 17 Sep 2007 to 13 Apr 2026, via the
  GitHub mirror `kalilurrahman/NIFTY_50_STOCK_DATA`.
- India VIX daily (Yahoo ^INDIAVIX), 2 Jan 2020 to 24 Aug 2026, via the
  GitHub mirror `tanya459/india-vix-market-data-analysis`.
- Spot checks against NSE closes: Nifty 4 Jun 2024 = 21,884.50; India VIX = 26.75.
- NSE F&O bhavcopy (not committed; regenerate with step B).

## Status

- [x] Week 1: data, features, HAR, first figure, tests
- [ ] VIX back-fill to 2008; bhavcopy download and check on real files of both layouts
- [ ] Week 2: LightGBM, Diebold-Mariano test, ATM implied vol series
- [ ] Week 3: straddle backtest on real prices, SEBI before/after test, write-up
- [ ] Week 4: VaR and ES with Kupiec test, Streamlit page, talk
