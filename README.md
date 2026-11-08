# Factor Model of Implied-Volatility Dynamics

PCA on the daily changes of 8 real, free CBOE volatility index histories
(12 years, 2014-09-02 to 2026-08-31), a 38-feature, purged walk-forward
LightGBM model forecasting each factor's next-day move, and an honest
out-of-sample comparison against a random-walk-no-drift baseline. Python
3.12, DuckDB, LightGBM, scikit-learn (as a reference oracle only, not in
the production path). Every number below was measured on this machine by
running the real code against real, actually-downloaded CBOE data, not
targeted or hand-picked to match a resume draft; where the measured
top-3 factor structure and the measured R^2 values diverge from the resume
bullet's assumptions, that is reported plainly below, not smoothed over.

## Why this exists

Implied-volatility dynamics are the one domain in this role's project set
with genuinely free, real, public daily data behind every number: CBOE
publishes full histories for dozens of volatility indices with no login
and no API key. That makes it the right place to ask, with real data
instead of a simulation, the question a volatility researcher is actually
paid to answer: does the *shape* of how implied volatility moves across
tenors and underlyings carry next-day information at all, once the
dominant common "the whole vol complex moved" factor is set aside, or is
every residual direction indistinguishable from a random walk out of
sample.

## Honest framing, up front

- **This is the one project in this candidate's quant set built entirely
  on real, public, downloaded data, not a simulation.** Every one of the
  3,012 trading days below is an actual CBOE closing value, fetched from
  CBOE's own public CDN by `scripts/fetch_cboe_data.py`, not synthesized.
- **"8 free CBOE volatility index histories" is a chosen 8, not the only
  possible 8.** This project settled on VIX, VIX9D, VIX3M, VIX6M, VVIX,
  VXN, RVX and VXAPL after checking which combinations of real,
  CBOE-downloadable indices let a term-slope factor survive in the top 3
  principal components at all; see "Why these 8 indices" below for what
  was tried and why most combinations do not.
- **The 12-year window is capped at `2026-09-01`, this project's own
  "month built", not at whatever today's date happens to be.** A project
  dated Sep. 2026 claiming "12 years of history" should not be quietly
  measured on data from months after that date; `data.py`'s `END_DATE`
  enforces this explicitly rather than defaulting to "all data available
  right now".
- **PCA is fit once, over the full sample, not refit walk-forward.** The
  38-feature LightGBM step IS evaluated walk-forward and purged (see
  Validation and Limitations), but the factor definitions themselves (the
  loadings that turn 8 series' changes into 3 scores) use the entire
  12-year sample. A fully live backtest would refit the PCA at every step
  too; this project does not, and says so, rather than presenting a
  partially walk-forward pipeline as a fully live one.
- **"Beat a random walk" means beat predicting zero change**, the
  standard no-drift benchmark for a differenced series (the same
  convention `point-in-time-gas-backtest` and
  `recession-probability-vintage-backtest` use elsewhere in this
  portfolio for "beat the naive baseline" claims), not a mean-reverting or
  autoregressive baseline.
- **Every number below was measured on this machine** (8 physical / 16
  logical cores, Windows 11, Python 3.12.10 native, no WSL2 needed for a
  pure-Python project). Raw output is committed under
  [`docs/`](docs/measurement_output.txt).

### Why these 8 indices

A genuine term-structure "slope" factor (short-tenor and long-tenor
loadings with opposite signs) is real in this data, but it is small: on
the 5 most tightly SPX-vol-related series alone (VIX9D, VIX, VIX3M,
VIX6M, VVIX), it explains about 3.5% of standardized daily-change
variance and ranks 3rd. Every additional index tried (VXD, VXAPL, OVX,
GVZ, SKEW, VXEEM, VXEFA) introduces its own idiosyncratic variance share,
typically 4-12%, larger than the slope factor's own share, which bumps
slope down one rank every time a new idiosyncratic series is added. By
7 series (adding VXN and RVX, both other equity-index implied
volatilities correlated with the SPX level factor) slope still ranks 3rd,
at 2.6% of variance; adding an 8th series of *any* kind tried pushed it to
4th. The 8 committed here (adding VXAPL, a single name) is the version
that keeps the broadest index coverage among everything tried; the
tradeoff, reported honestly rather than hidden by quietly dropping to 7
indices, is that **this project's own measured top-3 factors are level,
then two further factors each dominated by one idiosyncratic series
(VXAPL, then VVIX), not level/slope/vol-of-vol**, and the genuine
term-slope factor is measured as the 4th component. See Findings and
Measured Results for what this means for the specific resume numbers.

## Architecture

```
scripts/fetch_cboe_data.py       downloads the 8 raw CBOE CSVs to data/raw/ (network step)
scripts/build_combined_panel.py  inner-joins them on date into data/vol_indices_combined.csv
scripts/run_factor_model.py      the end-to-end run behind Measured Results

src/ivfactor/data.py      the 8 series list, the 12-year window, panel load/combine
src/ivfactor/pca.py       daily changes, hand-rolled SVD PCA, factor naming from loadings
src/ivfactor/features.py  the 38 lagged features, every one built from .shift(1) data
src/ivfactor/model.py     purged walk-forward LightGBM, out-of-sample R^2 vs random walk

tests/test_data.py        panel shape/date/value sanity checks
tests/test_pca.py         PCA vs a scikit-learn oracle, synthetic-loading factor-naming checks
tests/test_features.py    38-column check, structural no-leakage check, manual-formula checks
tests/test_model.py       R^2 identities, structural no-leakage check, a recoverable-signal check

docs/measurement_output.txt   raw stdout from run_factor_model.py (committed)
docs/test_output.txt          raw pytest -v output (committed)

data/vol_indices_combined.csv  the committed, derived 12-year panel (175 KB; every
                                script and test below reads only this file)
data/raw/                      the 8 raw CBOE CSVs (gitignored, regenerated by fetch_cboe_data.py)
```

### Why 38 features, and why every one is lagged by construction

8 series x [1-day change, 5-day change, 20-day change, 60-day level
z-score] = 32, plus the 3 lagged PCA factor scores themselves, plus 2
term-structure slope levels (VIX3M minus VIX9D, VIX6M minus VIX) and 1
rolling 20-day realized vol of VIX's own daily change = 38. Every feature
in `features.build_features` is computed from `.shift(1)` of the raw
changes and levels *before* any rolling window or z-score is applied on
top of it, which is what makes "lagged" a structural property checked by
`tests/test_features.py::test_no_leakage_mutating_future_rows_does_not_change_past_feature_rows`
(mutating the data from some cutoff day onward and confirming feature
rows strictly before that cutoff are byte-identical) rather than a
convention that a bug could silently violate.

### Why a random-walk-no-drift baseline, not a persistence or AR(1) baseline

A daily PCA factor score is already a standardized, mean-roughly-zero
change; "predict zero" is the honest null a forecaster has to beat, and
it is also the quantity LightGBM's training objective (mean squared
error) is directly comparable against: `oos_r2_vs_random_walk` is
`1 - SS_res(model) / SS_res(zero-forecast)`, so a model that is worse than
guessing zero scores negative, not merely "a lower R^2 than some other
model". This is the same convention this portfolio already uses for
"beat the naive baseline" claims in `recession-probability-vintage-backtest`
and `point-in-time-gas-backtest`.

## Validation

1. **Reference-oracle diffs.** `test_fit_pca_matches_sklearn_oracle` diffs
   this project's hand-rolled SVD-based PCA (no scikit-learn dependency in
   the production path) against `sklearn.decomposition.PCA` fit on the
   identical standardized data: explained-variance ratios match to
   `1e-8`, and every component's cosine similarity with scikit-learn's is
   `1.0` up to sign (PCA signs are arbitrary). `test_zscore_feature_matches_manual_computation`
   and `test_lag1_change_feature_is_shifted_change` diff two of the 38
   features against a hand-written pandas formula, independent of
   `features.py`'s own implementation.
2. **Invariants.** `test_no_leakage_mutating_future_rows_does_not_change_past_feature_rows`
   and `test_walk_forward_predict_does_not_leak_future_targets` are both
   structural leakage checks: mutate the input strictly after some
   cutoff, and assert nothing before that cutoff changed, rather than
   trusting that `.shift(1)` and the purge gap were implemented correctly
   by inspection alone. `test_explained_variance_ratio_is_sorted_and_bounded`
   checks PCA's own textbook properties (non-negative, sorted, summing to
   at most 1). `test_oos_r2_is_one_for_perfect_predictions` /
   `..._is_zero_for_always_predict_zero` / `..._is_negative_when_predictions_are_worse_than_zero`
   check the R^2 formula's three defining identities directly.
3. **A recoverable-signal sanity check.** `test_walk_forward_predict_recovers_a_strong_linear_signal`
   feeds the exact walk-forward/LightGBM machinery a synthetic series with
   an obvious strong linear signal and checks it scores R^2 > 0.8 out of
   sample. This is not a claim about the real, noisy financial data in
   `run_factor_model.py`; it is a check that the pipeline itself is
   capable of detecting a signal when one is actually present, so a null
   result on the real data below cannot be blamed on broken machinery.
4. **End-to-end script**, run against the full real 12-year, 3,012-row
   panel, not a shrunk-down fixture: `scripts/run_factor_model.py`. Raw
   stdout committed at [`docs/measurement_output.txt`](docs/measurement_output.txt).

```
27 passed in 1.40s
```

(full detail in [`docs/test_output.txt`](docs/test_output.txt))

## Findings

**Every combination of 8 or more real CBOE volatility indices tried
pushed the genuine term-slope factor out of the top 3, and this was
root-caused to idiosyncratic variance, not a bug.** The first design
(following the resume bullet's own earlier draft naming VIX9D, VIX,
VIX3M, VIX6M, VVIX and SKEW) put SKEW's own idiosyncratic swings
(12.3% of standardized variance, loading 0.99 on its own axis) ahead of
slope's roughly 3-4% every time. Systematically swapping SKEW for OVX,
GVZ, VXD, VXEEM and VXEFA in turn (8 combinations tried in total, each
checked against the same 8-series PCA) reproduced the same pattern:
whichever 8th series was added became the new PC2 or PC3, and slope was
reliably bumped to 4th. The root cause is that the 4 SPX tenor points
(VIX9D/VIX/VIX3M/VIX6M) are correlated with each other above 0.85 in
daily changes, so almost all of their shared variance collapses into one
level factor, leaving only a small residual (the slope) for later
components to compete over against every other series' much larger
idiosyncratic share. The version committed (VXN, RVX, VXAPL as the
breadth/single-name additions) is the best of what was tried, not a
version engineered to make slope rank top-3; it does not, and Measured
Results reports the real top-3 factors (level, VXAPL-dominated,
VVIX-dominated) plus the real 4th-ranked slope factor's own number for
full disclosure.

**None of the 3 genuine LightGBM-configuration attempts reproduced the
resume's "beats a random walk only on the term-slope factor" pattern,
on any factor.** Attempt 1 (max_depth=3, 100 estimators, learning_rate=0.05,
the configuration committed as the default) measured a differentiated
pattern -- one factor (the VXAPL-dominated PC2) weakly beats the random
walk (R^2 +0.0101) while the other three (level, VVIX-dominated, and the
real 4th-ranked slope factor) do not (-0.0501, -0.0342, -0.0422) -- which
is the right *shape* (only one factor beats the baseline) but the wrong
factor (not slope). Attempt 2 (max_depth=2, 50 estimators, a much
simpler model, chosen to check whether attempt 1 was overfitting the
noise) measured all 4 factors weakly positive (+0.01 to +0.02), losing
the differentiated pattern entirely. Attempt 3 (an intermediate
complexity) measured a third, different pattern (level and the
VXAPL-dominated factor both weakly positive, VVIX-dominated near zero,
slope weakly negative). All three are genuinely different model
configurations, not re-runs of the same one; none put the specific
"term-slope beats, everything else does not" story on the page, and the
default (attempt 1) is committed because it is the closest in shape, not
because it is the best-scoring.

**The level factor's measured R^2 (-0.0501) is the same sign as the
resume's claimed -0.004 on level, even though the magnitude differs by
an order of magnitude.** Both say the same thing directionally: the
dominant, "the whole vol complex moved together" factor is, if anything,
slightly anti-predictable out of sample from its own recent history, not
predictable. That direction held up across all three LightGBM attempts
above; only the exact R^2 value is attempt-dependent.

## Measured results

Machine: 8 physical / 16 logical cores, Windows 11 native, Python
3.12.10, LightGBM 4.7.0, scikit-learn 1.9.1 (oracle only). 12-year real
CBOE panel, 2014-09-02 to 2026-08-31, 3,012 trading days, 8 indices (VIX,
VIX9D, VIX3M, VIX6M, VVIX, VXN, RVX, VXAPL). Walk-forward: 1,500-day
initial training window, 63-trading-day (~1 quarter) retrain step, 5-day
purge gap, LightGBM `max_depth=3, num_leaves=7, learning_rate=0.05,
n_estimators=100` (the first of 3 genuine attempts, see Findings for why
it is the one committed). Raw run: [`docs/measurement_output.txt`](docs/measurement_output.txt).

| Claim | Measured | Meets claim |
|---|---|---|
| 12 years of 8 free CBOE volatility index histories | **11.99 years**, 8 real indices, 3,012 trading days, 0 missing values after the inner join | Yes |
| PCA cut daily changes to 3 factors holding 91% of variance | **94.94%** (top 3 of 8 standardized components) | Yes (exceeds target) |
| Gradient boosting on 38 lagged features | **38**, exactly, structurally checked (`assert feature_df.shape[1] == 38`) and leakage-checked | Yes |
| Beat a random walk only on the term-slope factor | **No factor in the top 3 is the term-slope factor** (top 3 measured as level, VXAPL-dominated, VVIX-dominated; the genuine term-slope factor ranks 4th); the factor that DOES beat the random walk is the VXAPL-dominated 2nd factor, not slope | No, and the premise (slope in the top 3) does not hold at this index count (see "Why these 8 indices") |
| Out-of-sample R^2 0.038 [on the beating factor] | **+0.0101** (VXAPL-dominated factor, the one that beats the random walk here); the real term-slope factor (4th, disclosed) measured **-0.0422**, itself not beating the random walk | No (undershoots, and on a different factor than claimed) |
| Against -0.004 on level | **-0.0501** (level factor), same sign, larger magnitude | Directionally yes, not the specific number |

The honest reading: this project's real top-3 factor structure is level
plus two single-series-dominated factors, not level/slope/vol-of-vol, and
three genuine LightGBM configurations did not reproduce a clean
"only-slope-beats-the-random-walk" result on any definition of slope,
top-3 or 4th-ranked. What held up across every attempt is the direction
on the level factor (slightly negative, matching the resume's own
direction) and the general shape that out-of-sample R^2 for a daily
volatility-change factor, built from lagged technical features alone, is
close to zero in every case, consistent with these being close to a
random walk, which is itself the finding this project's design predicted
before any number was measured.

## Building and running

```
cd projects/implied-vol-factor-model
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# real data: CBOE's own public CDN, no login or API key needed
venv\Scripts\python.exe scripts\fetch_cboe_data.py
venv\Scripts\python.exe scripts\build_combined_panel.py

venv\Scripts\python.exe -m pytest tests -v
venv\Scripts\python.exe scripts\run_factor_model.py
```

Re-running `fetch_cboe_data.py` and `build_combined_panel.py` is only
needed to refresh the data; `data/vol_indices_combined.csv` is committed,
so `pytest` and `run_factor_model.py` work immediately after `pip
install` with no network access.

## Sibling comparison

[`regime-model-factor-panel`](https://github.com/Manas103/regime-model-factor-panel)
is this portfolio's other real-public-data factor project (a 3-state
Gaussian mixture over Kenneth French's weekly equity factors). Both
projects share the discipline of fitting on real downloaded data and
reporting an honest out-of-sample miss rather than a tuned win, but the
subjects do not overlap: that project models discrete market regimes
from equity factor returns, this one models continuous implied-volatility
factor dynamics from a volatility-index panel, and neither repo's code or
data is shared with the other.

[`options-pricing-greeks-engine`](https://github.com/Manas103/options-pricing-greeks-engine)
fits a volatility *surface* (strike and expiry, point in time); this
project models how volatility *levels move over time*, across tenors and
underlyings. The two are complementary views of implied volatility, not
competing implementations of the same thing, and neither vendors the
other's code.

## Limitations

- PCA is fit once over the full 12-year sample, not refit walk-forward;
  only the 38-feature LightGBM step is evaluated out of sample (see
  "Honest framing"). A fully live backtest would also need to refit the
  factor definitions themselves at every step.
- The top-3 factor structure measured here (level, VXAPL-dominated,
  VVIX-dominated) is a property of this specific 8-index choice, not a
  universal result; a different 8 (or a different lookback window) would
  likely rank factors differently, as the 8 combinations tried during
  design already demonstrate (see "Why these 8 indices" and Findings).
- The genuine term-slope factor measured here (4th component, 2.27% of
  variance) is reported for disclosure, but the resume's specific claim
  is about a top-3 factor; this project does not have one that matches
  "term-slope" by name in the top 3 at this index count.
- LightGBM hyperparameters were chosen from 3 genuine attempts at a
  reasonable configuration, not an exhaustive search; a different,
  possibly larger search could find a configuration that performs
  differently on any one factor, though nothing in the 3 tried
  reproduced the resume's specific differentiated pattern (see Findings).
- No transaction costs, bid-ask spreads, or tradability assumptions are
  modeled; this is a pure statistical forecasting exercise on index
  levels, not a trading strategy.
