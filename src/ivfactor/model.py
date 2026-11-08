"""Purged walk-forward gradient boosting: an expanding training window,
stepped forward in blocks, with a gap between the end of each training
block and the start of its test block so that `ZSCORE_WINDOW`-long
rolling features computed near the training/test boundary cannot leak
test-period information backward into the last few training rows (the
purge). Out-of-sample R^2 is measured against a random-walk-no-drift
baseline (predict 0 change), the standard benchmark for a differenced
series, exactly like `recession-probability-vintage-backtest` and
`point-in-time-gas-backtest` in this portfolio measure "beat the naive
baseline" claims elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

PURGE_GAP = 5  # trading days between the end of a training block and its test block


@dataclass
class WalkForwardConfig:
    initial_train: int = 1500
    step: int = 63  # ~one quarter of trading days
    purge_gap: int = PURGE_GAP
    lgb_params: dict | None = None

    def params(self) -> dict:
        defaults = dict(
            objective="regression",
            max_depth=3,
            num_leaves=7,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=20260901,
            verbosity=-1,
        )
        if self.lgb_params:
            defaults.update(self.lgb_params)
        return defaults


def oos_r2_vs_random_walk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - SS_res(model) / SS_res(random-walk-no-drift baseline), the
    standard out-of-sample R^2 against a stated benchmark forecast
    (Welch & Goyal convention): > 0 means the model beat predicting zero
    change, < 0 means it did worse than predicting no change at all."""
    ss_res_model = np.sum((y_true - y_pred) ** 2)
    ss_res_rw = np.sum(y_true ** 2)
    return float(1.0 - ss_res_model / ss_res_rw)


def walk_forward_predict(X: np.ndarray, y: np.ndarray, config: WalkForwardConfig = WalkForwardConfig()) -> np.ndarray:
    """Returns an array the same length as `y`, with out-of-sample
    predictions for every row at or after `config.initial_train +
    config.purge_gap`, and NaN before that (never trained on, never
    scored). Each block's model is trained on all rows from 0 up to
    `train_end - purge_gap` (expanding window, purged), predicts rows
    `[test_start, test_start + step)`, then the window expands for the
    next block."""
    n = len(y)
    preds = np.full(n, np.nan)
    train_end = config.initial_train
    while train_end < n:
        test_start = train_end
        test_end = min(train_end + config.step, n)
        purged_train_end = max(0, train_end - config.purge_gap)

        X_train, y_train = X[:purged_train_end], y[:purged_train_end]
        X_test = X[test_start:test_end]

        model = lgb.LGBMRegressor(**config.params())
        model.fit(X_train, y_train)
        preds[test_start:test_end] = model.predict(X_test)

        train_end = test_end
    return preds
