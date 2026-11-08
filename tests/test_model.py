import numpy as np
import pytest

from ivfactor.model import WalkForwardConfig, oos_r2_vs_random_walk, walk_forward_predict


def test_oos_r2_is_one_for_perfect_predictions():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    assert oos_r2_vs_random_walk(y, y) == pytest.approx(1.0)


def test_oos_r2_is_zero_for_always_predict_zero():
    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    preds = np.zeros_like(y)
    assert oos_r2_vs_random_walk(y, preds) == pytest.approx(0.0)


def test_oos_r2_is_negative_when_predictions_are_worse_than_zero():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    preds = -3.0 * y  # systematically wrong-signed and oversized
    assert oos_r2_vs_random_walk(y, preds) < 0.0


def test_walk_forward_predict_leaves_initial_window_as_nan():
    rng = np.random.default_rng(4)
    n = 300
    X = rng.normal(size=(n, 5))
    y = rng.normal(size=n)
    config = WalkForwardConfig(initial_train=200, step=50, lgb_params=dict(n_estimators=10))
    preds = walk_forward_predict(X, y, config)
    assert np.all(np.isnan(preds[:200]))
    assert np.all(~np.isnan(preds[200:]))


def test_walk_forward_predict_output_length_matches_input():
    rng = np.random.default_rng(5)
    n = 250
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    config = WalkForwardConfig(initial_train=150, step=40, lgb_params=dict(n_estimators=10))
    preds = walk_forward_predict(X, y, config)
    assert len(preds) == n


def test_walk_forward_predict_does_not_leak_future_targets():
    """Structural leakage check: mutating y strictly AFTER the end of a
    training block must not change that block's predictions, since the
    purge gap means the model backing those predictions was never fit on
    data past `train_end - purge_gap`."""
    rng = np.random.default_rng(6)
    n = 400
    X = rng.normal(size=(n, 6))
    y = rng.normal(size=n)
    config = WalkForwardConfig(initial_train=200, step=50, purge_gap=5, lgb_params=dict(n_estimators=10, random_state=1))

    preds_a = walk_forward_predict(X, y, config)

    y_b = y.copy()
    y_b[250:] = rng.normal(scale=50.0, size=n - 250)  # blow up everything from block 2 onward
    preds_b = walk_forward_predict(X, y_b, config)

    # Predictions for the first test block (rows 200:250) are trained on
    # rows [0, 200 - purge_gap), entirely before the mutation at row 250,
    # so they must be identical between the two runs.
    np.testing.assert_allclose(preds_a[200:250], preds_b[200:250])


def test_walk_forward_predict_recovers_a_strong_linear_signal():
    """Sanity check that the pipeline can detect an obvious, strong
    signal when one is actually present (not a claim about the real,
    noisy financial data in run_factor_model.py, just a check that the
    machinery itself works)."""
    rng = np.random.default_rng(7)
    n = 2000
    X = rng.normal(size=(n, 3))
    y = 5.0 * X[:, 0] + rng.normal(scale=0.1, size=n)
    config = WalkForwardConfig(initial_train=1000, step=200, lgb_params=dict(n_estimators=50))
    preds = walk_forward_predict(X, y, config)
    mask = ~np.isnan(preds)
    r2 = oos_r2_vs_random_walk(y[mask], preds[mask])
    assert r2 > 0.8
