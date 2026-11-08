"""Combines the 8 raw CBOE CSVs (`data/raw/`, fetched by
`scripts/fetch_cboe_data.py`, each in CBOE's own native format: four of the
eight are OHLC with the close in a `CLOSE` column, the other four are
single-value with the close in a column named after the symbol) into one
clean daily panel, inner-joined on date so every row has all 8 series with
no missing values, over the real civil-calendar window
[`START_DATE`, `END_DATE`).

The window is deliberately capped at `END_DATE` (near this project's own
Sep. 2026 "month built"), not at whatever today's date happens to be when
this script runs: a project claiming "12 years of history" dated Sep. 2026
should not be quietly measured on data from months after that date.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
COMBINED_PATH = ROOT / "data" / "vol_indices_combined.csv"

SERIES_COLUMNS = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "VXN", "RVX", "VXAPL"]
OHLC_SYMBOLS = {"VIX", "VIX9D", "VIX3M", "VIX6M", "VXN", "RVX", "VXAPL"}

START_DATE = "2014-09-01"
END_DATE = "2026-09-01"


def _load_one(con: duckdb.DuckDBPyConnection, symbol: str) -> None:
    path = RAW_DIR / f"{symbol}.csv"
    close_col = "CLOSE" if symbol in OHLC_SYMBOLS else symbol
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW t_{symbol} AS
        SELECT "DATE"::DATE AS date, {close_col}::DOUBLE AS {symbol}
        FROM read_csv_auto('{path.as_posix()}')
        WHERE "DATE"::DATE >= DATE '{START_DATE}'
          AND "DATE"::DATE < DATE '{END_DATE}'
        """
    )


def build_combined_panel() -> pd.DataFrame:
    """Rebuilds the combined panel from `data/raw/` and returns it. Does
    NOT write `COMBINED_PATH`; see `scripts/build_combined_panel.py` for
    the CLI that does, and commits the result."""
    con = duckdb.connect()
    for symbol in SERIES_COLUMNS:
        _load_one(con, symbol)

    # Every t_<symbol> view has exactly one column named "date" plus one
    # column named after the symbol, so a chain of NATURAL JOINs (matching
    # on the shared "date" column at each step) is an inner join across
    # all 8 series on date, with no row surviving unless every series has
    # a value for that date.
    join_sql = " NATURAL JOIN ".join(f"t_{s}" for s in SERIES_COLUMNS)
    query = f"SELECT * FROM {join_sql} ORDER BY date"
    df = con.execute(query).fetch_df()
    con.close()
    return df


def load_combined_panel() -> pd.DataFrame:
    """Loads the committed combined panel (no network, no `data/raw/`
    needed): this is what every other module and test calls."""
    df = pd.read_csv(COMBINED_PATH, parse_dates=["date"])
    return df
