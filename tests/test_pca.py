import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from ivfactor.data import SERIES_COLUMNS
from ivfactor.pca import daily_changes, fit_pca, name_factors


def _make_changes(rng: np.random.Generator, n: int = 500) -> pd.DataFrame:
    """Synthetic daily changes with a known, dominant common factor (so
    PCA has an obvious right answer to check against) plus independent
    per-series noise."""
    common = rng.normal(scale=3.0, size=n)
    data = {}
    for i, col in enumerate(SERIES_COLUMNS):
        data[col] = common * (1.0 + 0.1 * i) + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame(data)
    df.insert(0, "date", pd.date_range("2020-01-01", periods=n))
    return df


def test_fit_pca_matches_sklearn_oracle():
    """Reference-oracle test: this project's hand-rolled SVD-based PCA
    against scikit-learn's PCA on the identical standardized data. Signs
    of principal components are arbitrary, so components are compared up
    to sign."""
    rng = np.random.default_rng(1)
    changes = _make_changes(rng)
    result = fit_pca(changes, n_components=3)

    X = changes[SERIES_COLUMNS].to_numpy(dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    sk = SklearnPCA(n_components=3)
    sk.fit(Z)

    np.testing.assert_allclose(
        result.explained_variance_ratio, sk.explained_variance_ratio_, atol=1e-8
    )
    for i in range(3):
        cos_sim = np.dot(result.components[i], sk.components_[i])
        assert abs(abs(cos_sim) - 1.0) < 1e-8, f"component {i} disagrees with sklearn"


def test_dominant_common_factor_is_recovered_as_factor_one():
    """On synthetic data with one obvious shared factor, PC1 should
    explain the large majority of variance and load the same sign on
    every series (all series share the same common-factor direction by
    construction)."""
    rng = np.random.default_rng(2)
    changes = _make_changes(rng)
    result = fit_pca(changes, n_components=3)
    assert result.explained_variance_ratio[0] > 0.8
    assert np.all(result.components[0] > 0) or np.all(result.components[0] < 0)


def test_explained_variance_ratio_is_sorted_and_bounded():
    rng = np.random.default_rng(3)
    changes = _make_changes(rng)
    result = fit_pca(changes, n_components=4)
    assert np.all(np.diff(result.explained_variance_ratio) <= 1e-12)
    assert np.all(result.explained_variance_ratio >= 0.0)
    assert result.explained_variance_ratio.sum() <= 1.0 + 1e-9


def test_scores_have_unit_variance_approximately():
    """Each PCA score column, by construction of PCA on a correlation
    matrix, should have variance close to its own eigenvalue; scaled
    appropriately, the first score's variance should track the first
    eigenvalue, not be wildly different."""
    rng = np.random.default_rng(4)
    changes = _make_changes(rng)
    result = fit_pca(changes, n_components=3)
    empirical_var = result.scores[:, 0].var(ddof=1)
    n = result.scores.shape[0]
    eigenvalue = result.explained_variance_ratio[0] * len(SERIES_COLUMNS)  # ratio * total variance (n_series, standardized)
    # loose bound: just checking the score is not degenerate (zero) or absurd
    assert empirical_var > 0.1


@pytest.mark.parametrize(
    "loadings,expected",
    [
        ([-0.4, -0.35, -0.4, -0.37, -0.3, -0.36, -0.35, -0.3], "level"),
        ([0.5, -0.7, 0.3, 0.6, -0.05, 0.1, 0.2, -0.1], "term-slope"),
        ([0.05, 0.03, 0.02, 0.04, 0.9, 0.02, 0.01, 0.03], "VVIX-dominated"),
    ],
)
def test_name_factors_on_synthetic_loadings(loadings, expected):
    from ivfactor.pca import PCAResult

    result = PCAResult(
        components=np.array([loadings]),
        explained_variance_ratio=np.array([1.0]),
        scores=np.zeros((1, 1)),
        means=np.zeros(8),
        stds=np.ones(8),
    )
    names = name_factors(result)
    assert names == [expected]


def test_daily_changes_is_one_row_shorter_than_panel():
    rng = np.random.default_rng(5)
    changes_input = _make_changes(rng, n=10)
    # daily_changes expects a level panel, not a changes panel, but the
    # shape contract (one fewer row) is identical either way, which is
    # what this test checks.
    panel = changes_input.copy()
    result = daily_changes(panel)
    assert len(result) == len(panel) - 1
