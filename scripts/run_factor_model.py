"""End-to-end run: loads the committed 12-year combined panel, fits PCA on
daily changes, builds the 38 lagged features, and walk-forward evaluates
LightGBM against a random-walk-no-drift baseline for each of the top 3
factors (plus, for disclosure, the 4th component, which is where this
project's genuine term-slope factor actually ranks; see README "Why these
8 indices").

Usage:
    python scripts/run_factor_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ivfactor.data import load_combined_panel
from ivfactor.features import build_features
from ivfactor.model import WalkForwardConfig, oos_r2_vs_random_walk, walk_forward_predict
from ivfactor.pca import SERIES_COLUMNS, daily_changes, fit_pca, name_factors


def main() -> None:
    panel = load_combined_panel()
    print(f"panel: {len(panel)} rows, {panel['date'].min().date()} to {panel['date'].max().date()}, "
          f"{len(SERIES_COLUMNS)} series: {SERIES_COLUMNS}")

    changes = daily_changes(panel)
    n_components = 4  # top 3 for the headline claims, 4th kept for disclosure
    result = fit_pca(changes, n_components=n_components)
    names = name_factors(result)
    print("\nPCA on standardized daily changes:")
    print(f"  explained variance ratio: {[round(x, 4) for x in result.explained_variance_ratio.tolist()]}")
    print(f"  cumulative, top 3: {result.explained_variance_ratio[:3].sum():.4f}")
    print(f"  factor names (from loadings, see pca.name_factors): {names}")
    loadings = pd.DataFrame(result.components, columns=SERIES_COLUMNS, index=[f"factor{i+1}" for i in range(n_components)])
    print(loadings.round(3).to_string())

    feats = build_features(panel, changes, result.scores[:, :3])
    valid_mask = feats.notna().all(axis=1).to_numpy()
    X = feats[valid_mask].to_numpy()
    n_valid = valid_mask.sum()
    print(f"\nfeatures: {feats.shape[1]} columns, {n_valid} / {len(feats)} rows valid after warm-up")

    config = WalkForwardConfig()
    print(f"walk-forward: initial_train={config.initial_train}, step={config.step}, "
          f"purge_gap={config.purge_gap}, lgb_params={config.params()}")

    results = {}
    t0 = time.time()
    for k in range(n_components):
        y = result.scores[valid_mask, k]
        preds = walk_forward_predict(X, y, config)
        oos_mask = ~np.isnan(preds)
        r2 = oos_r2_vs_random_walk(y[oos_mask], preds[oos_mask])
        label = names[k] if k < len(names) else f"factor{k+1}"
        results[f"factor{k+1}_{label}"] = {"oos_r2": r2, "n_oos": int(oos_mask.sum())}
        print(f"  factor{k+1} ({label}): oos_r2={r2:.4f}  n_oos={int(oos_mask.sum())}")
    t1 = time.time()
    print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    print("SUMMARY (resume claims vs measured):")
    print(
        json.dumps(
            {
                "claim_years_of_history": 12,
                "measured_years_of_history": round((panel["date"].max() - panel["date"].min()).days / 365.25, 2),
                "claim_n_indices": 8,
                "measured_n_indices": len(SERIES_COLUMNS),
                "claim_n_factors": 3,
                "claim_variance_held_pct": 91,
                "measured_top3_variance_pct": round(100 * result.explained_variance_ratio[:3].sum(), 2),
                "claim_n_features": 38,
                "measured_n_features": feats.shape[1],
                "claim_term_slope_r2": 0.038,
                "claim_level_r2": -0.004,
                "measured_results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
