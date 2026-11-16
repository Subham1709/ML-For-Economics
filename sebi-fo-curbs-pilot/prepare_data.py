"""Download and normalize version-pinned market data for the pilot."""

import csv
import io
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

DATA = Path(__file__).resolve().parent / "data"
SOURCES = {
    "nifty2022": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "NIFTY 50-01-01-2022-to-31-12-2022.csv",
    ),
    "indiavix2022": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "hist_india_vix_-01-01-2022-to-31-12-2022.csv",
    ),
    "nifty2023": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "NIFTY 50-01-01-2023-to-31-12-2023.csv",
    ),
    "indiavix2023": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "hist_india_vix_-01-01-2023-to-31-12-2023.csv",
    ),
    "nifty2024": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "NIFTY 50-01-01-2024-to-31-12-2024.csv",
    ),
    "indiavix2024": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "hist_india_vix_-01-01-2024-to-31-12-2024.csv",
    ),
    "nifty2025": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "NIFTY 50-01-01-2025-to-31-12-2025.csv",
    ),
    "indiavix2025": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "hist_india_vix_-01-01-2025-to-31-12-2025.csv",
    ),
    "sp500": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "S&P 500 Historical Data.csv",
    ),
    "brent": (
        "Hareeshkesavan/Stock-Market-Dataset", "aa60b746f089c3f54df6760bf021708c2d3ce1e8",
        "Brent Oil Futures Historical Data.csv",
    ),
    "vixfull": (
        "datasets/finance-vix", "3dbea23c70cc2afbeeec8ac232c9ec72ac26ac21",
        "data/vix-daily.csv",
    ),
    "niftyrecent": (
        "Meharjot29/NIFTY-50-Dashborad", "44d05656de6f2cea0e74c7d5aec2d23293426eb5",
        "NIFTY 50-16-07-2024-to-16-07-2025.csv",
    ),
}


def read_source(repository, revision, filename):
    url = f"https://raw.githubusercontent.com/{repository}/{revision}/{quote(filename)}"
    with urlopen(url, timeout=45) as response:
        return response.read().decode("utf-8-sig")


def normalize(key, content):
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise ValueError(f"Missing column headers in {key}")
    reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]

    if key.startswith(("nifty", "indiavix")):
        fields = ("date", "open", "high", "low", "close")
        return fields, [tuple(row[field].strip() for field in fields) for row in reader]

    if key == "vixfull":
        start, end = date(2021, 11, 15), date(2025, 12, 31)
        rows = [(row["date"], row["close"]) for row in reader
                if start <= date.fromisoformat(row["date"]) <= end]
        return None, rows

    rows = []
    for row in reader:
        day = datetime.strptime(row["date"], "%m/%d/%Y").date()
        if 2021 <= day.year <= 2025:
            rows.append((day.isoformat(), row["price"].replace(",", "")))
    return None, rows


def main():
    DATA.mkdir(exist_ok=True)
    for key, source in SOURCES.items():
        header, rows = normalize(key, read_source(*source))
        if not rows:
            raise ValueError(f"{key} contained no data for the pilot")
        with (DATA / f"{key}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        print(f"{key}: {len(rows)} data rows")


if __name__ == "__main__":
    main()
