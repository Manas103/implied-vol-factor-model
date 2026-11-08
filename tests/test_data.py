import pandas as pd
import pytest

from ivfactor.data import SERIES_COLUMNS, load_combined_panel
from ivfactor.pca import SERIES_COLUMNS as PCA_SERIES_COLUMNS


def test_combined_panel_has_expected_columns():
    panel = load_combined_panel()
    assert list(panel.columns) == ["date"] + SERIES_COLUMNS


def test_combined_panel_series_columns_match_pca_module():
    """data.py and pca.py must agree on which 8 series are in play; a
    mismatch here would silently select the wrong columns downstream."""
    assert SERIES_COLUMNS == PCA_SERIES_COLUMNS


def test_combined_panel_has_no_missing_values():
    panel = load_combined_panel()
    assert panel.isna().sum().sum() == 0


def test_combined_panel_spans_about_12_years():
    panel = load_combined_panel()
    span_years = (panel["date"].max() - panel["date"].min()).days / 365.25
    assert span_years == pytest.approx(12.0, abs=0.1)


def test_combined_panel_dates_are_sorted_and_unique():
    panel = load_combined_panel()
    assert panel["date"].is_monotonic_increasing
    assert panel["date"].is_unique


def test_combined_panel_values_are_plausible_volatility_levels():
    """Sanity check, not a tuned target: implied volatility index levels
    are positive and (for these 8, over this real 12-year window,
    including VVIX's real spike above 200 during the Mar. 2020 COVID
    crash) stay under 250."""
    panel = load_combined_panel()
    for col in SERIES_COLUMNS:
        assert (panel[col] > 0).all()
        assert (panel[col] < 250).all()
