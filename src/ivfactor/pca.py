"""PCA on the daily changes of the 8 volatility index levels, not the
levels themselves (a level is a near-random-walk; a change is what has any
chance of being predictable, and factor models of implied volatility
dynamics in the literature are built on changes for the same reason).

Each series' daily change is standardized to unit variance before PCA:
VVIX's day-to-day swings are an order of magnitude larger than VIX6M's, so
an unstandardized PCA would mostly just rediscover "VVIX moves a lot",
not a genuine cross-series factor structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ivfactor.data import SERIES_COLUMNS


def daily_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """First differences of the 8 level series, in index-point units."""
    changes = panel[SERIES_COLUMNS].diff().iloc[1:].reset_index(drop=True)
    changes.insert(0, "date", panel["date"].iloc[1:].reset_index(drop=True))
    return changes


@dataclass
class PCAResult:
    components: np.ndarray  # shape (n_components, n_series): loadings, rows are PCs
    explained_variance_ratio: np.ndarray  # shape (n_components,)
    scores: np.ndarray  # shape (n_obs, n_components): standardized-change scores
    means: np.ndarray  # per-series mean of the change (for standardization)
    stds: np.ndarray  # per-series std of the change


def fit_pca(changes: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Hand-rolled PCA via SVD on the standardized change matrix, with no
    dependency on scikit-learn in the production path (sklearn is used
    only in `tests/test_pca.py` as an independent reference oracle)."""
    X = changes[SERIES_COLUMNS].to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    Z = (X - means) / stds

    # Z = U @ diag(S) @ Vt ; the rows of Vt are the principal-component
    # loadings (eigenvectors of Z.T @ Z / (n-1), the correlation matrix of
    # the standardized changes), and S**2 / (n-1) are their variances.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    n = Z.shape[0]
    eigenvalues = (S ** 2) / (n - 1)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    components = Vt[:n_components]
    scores = Z @ components.T

    return PCAResult(
        components=components,
        explained_variance_ratio=explained_variance_ratio[:n_components],
        scores=scores,
        means=means,
        stds=stds,
    )


def name_factors(result: PCAResult) -> list[str]:
    """Labels each factor from its own loadings, not by assumption, in
    this priority order:
    1. 'level': same-sign loadings across at least 75% of the 8 series.
    2. '<series>-dominated': one series' |loading| exceeds 0.65 while the
       next-largest is below 0.55 (a single idiosyncratic series, not a
       genuine cross-series factor).
    3. 'term-slope': opposite-sign loadings between the short end
       (VIX9D) and the long end (VIX6M).
    4. 'mixed': none of the above earned a label.
    This project's own measured top-3 factors are 'level', then two
    '<series>-dominated' factors (see README "Why these 8 indices"); the
    genuine term-slope factor measured here ranks 4th, not in the top 3,
    and is reported as such rather than relabeling a different factor
    'term-slope' to match the resume bullet's assumption."""
    idx = {s: i for i, s in enumerate(SERIES_COLUMNS)}
    short_idx, long_idx = idx["VIX9D"], idx["VIX6M"]
    names = []
    for row in result.components:
        abs_row = np.abs(row)
        order = np.argsort(abs_row)[::-1]
        same_sign_frac = max((row > 0).mean(), (row < 0).mean())
        if abs_row[order[0]] > 0.6 and abs_row[order[0]] > 1.3 * abs_row[order[1]]:
            names.append(f"{SERIES_COLUMNS[order[0]]}-dominated")
        elif same_sign_frac >= 0.75:
            names.append("level")
        elif row[short_idx] * row[long_idx] < 0:
            names.append("term-slope")
        else:
            names.append("mixed")
    return names
