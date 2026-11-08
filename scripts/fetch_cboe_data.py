"""Downloads 8 free, real CBOE volatility index daily histories from CBOE's
own public CDN (no API key, no login) and writes the raw CSVs to
`data/raw/`. This is the one network-dependent step in this project; every
other script and test reads only the committed combined panel
(`data/vol_indices_combined.csv`), so a reviewer without network access can
reproduce everything downstream of this script without running it.

The 8 series, chosen to cover the SPX tenor structure, vol-of-vol, breadth
across other equity indices, and one single-name (see README "Honest
framing" and "Why these 8 indices"):
  VIX    S&P 500 30-day implied volatility (the benchmark index)
  VIX9D  S&P 500 9-day implied volatility (short end of the term structure)
  VIX3M  S&P 500 3-month implied volatility
  VIX6M  S&P 500 6-month implied volatility (long end of the term structure)
  VVIX   volatility of VIX (vol-of-vol)
  VXN    CBOE Nasdaq-100 implied volatility (breadth: a second equity index)
  RVX    CBOE Russell 2000 implied volatility (breadth: a third equity index)
  VXAPL  CBOE Apple (AAPL) implied volatility (single-name)

Usage:
    python scripts/fetch_cboe_data.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivfactor.data import SERIES_COLUMNS as SYMBOLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

URL_TEMPLATE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"


def fetch_one(symbol: str) -> Path:
    url = URL_TEMPLATE.format(symbol=symbol)
    out_path = RAW_DIR / f"{symbol}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return out_path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        path = fetch_one(symbol)
        n_lines = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"{symbol}: {path} ({n_lines - 1} rows)")
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
