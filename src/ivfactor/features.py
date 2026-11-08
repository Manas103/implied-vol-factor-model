"""38 lagged features built from the 8 raw level series and their PCA
factor scores, every one of them using only information available at the
END of the day before the target day, enforced structurally rather than
by convention: every rolling/lag computation below starts from
`.shift(1)`, so the feature at row `t` can only see data through day
`t-1`, before any rolling window or z-score is computed on top of it.
`tests/test_features.py` checks this mechanically.

The 38, by construction (see README "Why 38"):
  8 series x [lag1 change, lag5 change, lag20 change, 60-day level z-score] = 32
  3 lagged PCA factor scores (factor1, factor2, factor3, lag 1)            =  3
  2 term-structure slope levels (VIX3M - VIX9D, VIX6M - VIX, lag 1)        =  2
  1 rolling 20-day realized vol of VIX's own daily change, lag 1           =  1
                                                                    total  = 38
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ivfactor.pca import SERIES_COLUMNS

ZSCORE_WINDOW = 60
VOL_WINDOW = 20


def build_features(panel: pd.DataFrame, raw_changes: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """`panel`: the raw level panel (date + 8 columns). `raw_changes`: the
    8 daily-change columns (date + 8 columns), one fewer row than
    `panel`, see `pca.daily_changes`. `scores`: the PCA factor scores
    array (same row count as `raw_changes`, one column per factor; only
    the first 3 columns are used). Returns a DataFrame indexed like
    `raw_changes`, with NaNs in the warm-up period (the first
    `ZSCORE_WINDOW` rows) left in, to be dropped by the caller alongside
    the matching target rows."""
    levels = panel[SERIES_COLUMNS].reset_index(drop=True)
    changes = raw_changes[SERIES_COLUMNS].reset_index(drop=True)
    # raw_changes has one fewer row than panel; dropping panel's first row
    # before shifting aligns levels[t] with changes[t] = levels[t+1] -
    # levels[t], so "shift(1)" below means "as of yesterday relative to
    # the day changes[t] (and scores[t]) describe".
    levels = levels.iloc[1:].reset_index(drop=True)

    shifted_changes = changes.shift(1)
    shifted_levels = levels.shift(1)

    feats = {}
    for col in SERIES_COLUMNS:
        feats[f"{col}_lag1_chg"] = shifted_changes[col]
        for w in (5, 20):
            feats[f"{col}_lag{w}_chg"] = shifted_changes[col].rolling(w).sum()
        roll_mean = shifted_levels[col].rolling(ZSCORE_WINDOW).mean()
        roll_std = shifted_levels[col].rolling(ZSCORE_WINDOW).std()
        feats[f"{col}_zscore60"] = (shifted_levels[col] - roll_mean) / roll_std

    for k in range(3):
        feats[f"factor{k + 1}_lag1"] = pd.Series(scores[:, k]).shift(1)

    feats["slope_3m_9d_lag1"] = shifted_levels["VIX3M"] - shifted_levels["VIX9D"]
    feats["slope_6m_vix_lag1"] = shifted_levels["VIX6M"] - shifted_levels["VIX"]
    feats["vix_realized_vol20_lag1"] = shifted_changes["VIX"].rolling(VOL_WINDOW).std()

    feature_df = pd.DataFrame(feats)
    assert feature_df.shape[1] == 38, f"expected 38 features, built {feature_df.shape[1]}"
    return feature_df
