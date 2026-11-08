import numpy as np
import pandas as pd
import pytest

from ivfactor.data import SERIES_COLUMNS
from ivfactor.features import ZSCORE_WINDOW, build_features


def _make_panel(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: 15.0 + np.cumsum(rng.normal(scale=0.5, size=n)) for col in SERIES_COLUMNS}
    df = pd.DataFrame(data)
    df.insert(0, "date", pd.date_range("2020-01-01", periods=n))
    return df


def _changes_and_scores(panel: pd.DataFrame):
    changes = panel[SERIES_COLUMNS].diff().iloc[1:].reset_index(drop=True)
    changes.insert(0, "date", panel["date"].iloc[1:].reset_index(drop=True))
    rng = np.random.default_rng(99)
    scores = rng.normal(size=(len(changes), 3))
    return changes, scores


def test_feature_count_is_exactly_38():
    panel = _make_panel()
    changes, scores = _changes_and_scores(panel)
    feats = build_features(panel, changes, scores)
    assert feats.shape[1] == 38


def test_feature_row_count_matches_changes_row_count():
    panel = _make_panel()
    changes, scores = _changes_and_scores(panel)
    feats = build_features(panel, changes, scores)
    assert len(feats) == len(changes)


def test_no_leakage_mutating_future_rows_does_not_change_past_feature_rows():
    """The structural leakage check: build features twice, once on the
    real data and once on a copy where every row from some cutoff point
    onward is replaced with unrelated random noise. If features before
    the cutoff ever used information from at or after the cutoff, they
    would differ between the two runs; they must not."""
    panel = _make_panel(n=200)
    changes, scores = _changes_and_scores(panel)
    feats_a = build_features(panel, changes, scores)

    cutoff = 150
    panel_b = panel.copy()
    rng = np.random.default_rng(777)
    for col in SERIES_COLUMNS:
        panel_b.loc[cutoff:, col] = 15.0 + np.cumsum(rng.normal(scale=5.0, size=len(panel_b) - cutoff))
    changes_b, scores_b = _changes_and_scores(panel_b)
    scores_b[cutoff - 1 :, :] = rng.normal(scale=5.0, size=(len(scores_b) - (cutoff - 1), 3))
    feats_b = build_features(panel_b, changes_b, scores_b)

    # changes/feats are indexed one behind panel; a feature row at index
    # j in changes-space is built from panel data through panel index j
    # (see features.py docstring), so rows strictly before `cutoff - 1`
    # in changes-space must be identical between the two runs.
    safe_upto = cutoff - 1
    pd.testing.assert_frame_equal(
        feats_a.iloc[:safe_upto].reset_index(drop=True),
        feats_b.iloc[:safe_upto].reset_index(drop=True),
    )


def test_zscore_feature_matches_manual_computation():
    panel = _make_panel(n=200)
    changes, scores = _changes_and_scores(panel)
    feats = build_features(panel, changes, scores)

    levels = panel[SERIES_COLUMNS].iloc[1:].reset_index(drop=True)
    shifted = levels["VIX"].shift(1)
    roll_mean = shifted.rolling(ZSCORE_WINDOW).mean()
    roll_std = shifted.rolling(ZSCORE_WINDOW).std()
    expected = (shifted - roll_mean) / roll_std

    pd.testing.assert_series_equal(
        feats["VIX_zscore60"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_lag1_change_feature_is_shifted_change():
    panel = _make_panel(n=50)
    changes, scores = _changes_and_scores(panel)
    feats = build_features(panel, changes, scores)
    expected = changes["VIX"].shift(1).reset_index(drop=True)
    pd.testing.assert_series_equal(
        feats["VIX_lag1_chg"].reset_index(drop=True), expected, check_names=False
    )


def test_warmup_rows_are_nan_then_feature_is_fully_populated():
    panel = _make_panel(n=200)
    changes, scores = _changes_and_scores(panel)
    feats = build_features(panel, changes, scores)
    assert feats.iloc[0].isna().all()
    assert feats.iloc[ZSCORE_WINDOW + 5 :].notna().all().all()
