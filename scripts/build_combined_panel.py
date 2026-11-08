"""Rebuilds `data/vol_indices_combined.csv` from `data/raw/` (run
`scripts/fetch_cboe_data.py` first). This is the one script that needs
`data/raw/`; everything else in this project reads the committed combined
CSV this script produces.

Usage:
    python scripts/build_combined_panel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivfactor.data import COMBINED_PATH, build_combined_panel


def main() -> None:
    df = build_combined_panel()
    print(f"combined panel: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.describe())
    df.to_csv(COMBINED_PATH, index=False)
    print(f"wrote {COMBINED_PATH}")


if __name__ == "__main__":
    main()
