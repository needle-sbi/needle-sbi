# FAIR Universe Inference Debugging

Date: 2026-05-05
Model: Assisted by GPT 5.5. Extra High

## Goal

Debug local inference for already-trained networks using a small representative subset of
`/Users/levievans/Dev/needle/fair-universe-data/data`, then rebuild the histogram and Neyman-style artifacts after
pipeline consistency fixes.

Initial debugging checklist from Levi:

- Check for inconsistent nuisance-parameter ordering between dataset creation and evaluation.
- Verify that `predict()` does not pass explicit nuisance values and then overwrite them with `useRand=True`.
- Ensure Neyman inference loads the classifier in eval mode. The classifier uses `BatchNorm1d`, so train-vs-eval mode can
  change score distributions; `HistogramTask` already evaluates the classifier in eval mode.
- Confirm which nuisances the likelihood can model. It currently morphs TES/JES templates only, so soft MET or
  background-normalization shifts may be absorbed by TES/JES and drive boundary behavior.
- Clarify histogram binning: `np.linspace(0, 1, num=200)` creates 199 bins, not 200.
- Check dtype consistency across inference and fitting.
- Run closure tests by removing nuisances and then adding them back one at a time.

## Working Hypothesis

The spline boundary warnings are not yet reliable evidence that the TES/JES grid is too narrow. The stronger evidence
points to inconsistency between generated evaluation data, classifier/NF inference mode, and the templates used by the
likelihood.

## Bugs Fixed

- `EvalTask` now passes nuisance parameters to `createJetData` in the expected order:
  `[ttbar_scale, diboson_scale, bkg_scale, tes, jes, soft_met]`.
- `predict()` now uses `useRand=False` when explicit nuisance parameters are supplied.
- Classifier and NF models are forced to eval-mode float32 inference in histogram, Neyman, eval, `return1j2j()`, and
  `compute_signal_fraction()`.
- Stale absolute checkpoint paths in snapshots are rebased onto the local snapshot run directory when possible.
- The FAIR Universe data loader accepts the local Codabench split layout and assembles `detailed_labels`, `labels`, and
  `weights` sidecars.
- The README data path example was corrected.

## Local Debug Script

Use this to build fresh local artifacts without retraining:

```bash
FAIR_UNIVERSE_DATA="/Users/levievans/Dev/needle/fair-universe-data/data" \
uv run python examples/fair_universe_demo/scripts/debug_subset_inference.py \
  --subset-size 20000 \
  --max-source-rows 200000 \
  --weight-scale 100 \
  --signal-weight-scale 10000 \
  --grid-size 5 \
  --neyman-samples 3 \
  --mu-values 0.5 1.0 1.5 \
  --output-dir runs/fair_universe_demo_debug_subset
```

Outputs:

- `runs/fair_universe_demo_debug_subset/hist.json`
- `runs/fair_universe_demo_debug_subset/neyman.json`
- `runs/fair_universe_demo_debug_subset/summary.json`

## Evidence Log

- Focused synthetic fitter tests passed: `6 passed`.
- Data path smoke test resolved the local Codabench root, train/data directory, and `data.parquet` file to the same
  parquet.
- Existing stale artifacts ran end-to-end after consistency fixes, but nominal `predict()` still hit the lower
  signal-fraction bound. This suggests the old artifacts need rebuilding before interpreting the fit.
- The first-5k first-row subset was too brittle for template rebuilding; the debug script now samples from a bounded
  source window with a fixed random seed.
- Local Codabench sidecar weights are fractional and do not come with the bundled sample metadata rescaling. On small
  subsets, signal weights are especially tiny, so the debug script applies explicit non-signal and signal scales before
  pseudo-experiment bootstrapping.

## 2026-05-05 Subset Run

Command:

```bash
FAIR_UNIVERSE_DATA="/Users/levievans/Dev/needle/fair-universe-data/data" \
uv run python examples/fair_universe_demo/scripts/debug_subset_inference.py \
  --subset-size 20000 \
  --max-source-rows 200000 \
  --weight-scale 100 \
  --signal-weight-scale 10000 \
  --grid-size 5 \
  --neyman-samples 3 \
  --mu-values 0.5 1.0 1.5 \
  --output-dir runs/fair_universe_demo_debug_subset
```

Artifacts:

- `runs/fair_universe_demo_debug_subset/hist.json`
- `runs/fair_universe_demo_debug_subset/neyman.json`
- `runs/fair_universe_demo_debug_subset/summary.json`
- `runs/fair_universe_demo_debug_subset/plots/signal_fraction_closure.png`
- `runs/fair_universe_demo_debug_subset/plots/neyman_calibration.png`
- `runs/fair_universe_demo_debug_subset/plots/event_counts.png`
- `runs/fair_universe_demo_debug_subset/plots/nuisance_scan.png`
- `runs/fair_universe_demo_debug_subset/plots/profile_likelihood_scan.png`

Closure summary:

| `mu_true` | mean true signal fraction | mean fitted signal fraction | mean Neyman ratio |
| --- | ---: | ---: | ---: |
| 0.5 | 0.0962 | 0.0991 | 0.5917 |
| 1.0 | 0.1733 | 0.1719 | 1.0263 |
| 1.5 | 0.2391 | 0.2356 | 1.4065 |

Fresh-artifact nominal `predict()` check before calibration fix:

- Fit returned `f_s = 0.16001`, `nu1 = 1.00063`, `nu2 = 1.00788`.
- This is a good sign for template/data consistency: the nuisance fit is near nominal and not railing.
- The final `mu_hat` from `get_confidence_interval()` was still invalid because `predict()` passes raw `f_s_hat` into a
  Neyman calibration that stores `f_s_hat / f_s_nominal` ratios. This is now the next pipeline consistency bug to fix.

Fresh-artifact nominal `predict()` check after calibration fix:

- Fit returned `f_s = 0.16001`, `f_s_nominal = 0.16754`, and calibrated observable `real_mu = 0.95506`.
- Final calibrated result: `mu_hat = 0.93486`, `p16 = 0.91292`, `p84 = 0.95496`.
- Nuisance fit remained near nominal: `nu1 = 1.00063`, `nu2 = 1.00788`.
- This confirms `predict()` now uses the same observable as Neyman: `f_s_hat / f_s_nominal`.

Profile-likelihood scan:

- Produced from the nominal debug pseudo-experiment with the rebuilt TES/JES templates.
- Best fit: observed `mu = 0.95`, `f_s = 0.15916`, `nu1 = 1.00081`, `nu2 = 1.00804`.
- The true label-derived signal fraction for this pseudo-experiment is `0.17344`, so the profile minimum is consistent
  with the same small low bias seen in the nominal `predict()` check.
- Plot: `runs/fair_universe_demo_debug_subset/plots/profile_likelihood_scan.png` now follows the example scan style with
  68%/95% CL guides and a local Gaussian approximation.

Nuisance scan summary:

| scan | mean observed `mu` | mean `nu1` | mean `nu2` | any bound hit |
| --- | ---: | ---: | ---: | --- |
| nominal | 1.0103 | 0.9993 | 1.0064 | false |
| TES low | 1.0192 | 0.9780 | 0.9975 | false |
| TES high | 0.9963 | 1.0545 | 1.0023 | false |
| JES low | 1.0790 | 0.9950 | 0.9550 | false |
| JES high | 0.9898 | 1.0054 | 1.0534 | false |
| soft MET | 1.0180 | 1.0031 | 1.0079 | false |
| ttbar up | 1.0077 | 0.9984 | 1.0064 | false |
| diboson up | 1.0097 | 0.9989 | 1.0059 | false |
| background up | 0.9730 | 1.0002 | 1.0074 | false |

Interpretation:

- TES/JES shifted closure behaves as expected: the corresponding fitted nuisance moves in the right direction and does
  not rail at the spline boundary.
- Soft MET and background-normalization shifts do not cause TES/JES railing on this debug subset.
- The remaining visible issue is calibration/statistics rather than a hard template-boundary failure.
- Extra plots: `runs/fair_universe_demo_debug_subset/plots/neyman_calibration_residuals.png` and
  `runs/fair_universe_demo_debug_subset/plots/nuisance_impact_ranking.png`.

## Wider Local Validation

To check that the conclusion is not an artifact of the smaller debug subset, a wider run was produced in
`runs/fair_universe_demo_debug_subset_wide`:

```bash
FAIR_UNIVERSE_DATA="/Users/levievans/Dev/needle/fair-universe-data/data" \
uv run python examples/fair_universe_demo/scripts/debugging/debug_subset_inference.py \
  --subset-size 50000 \
  --max-source-rows 1000000 \
  --weight-scale 100 \
  --signal-weight-scale 10000 \
  --grid-size 5 \
  --neyman-samples 10 \
  --mu-values 0.1 0.5 1.0 1.5 2.0 2.5 3.0 \
  --output-dir runs/fair_universe_demo_debug_subset_wide
```

Closure summary:

| `mu_true` | mean true signal fraction | mean fitted signal fraction | mean Neyman ratio | Neyman ratio std |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 0.0192 | 0.0210 | 0.1230 | 0.0213 |
| 0.5 | 0.0928 | 0.0951 | 0.5581 | 0.0292 |
| 1.0 | 0.1680 | 0.1681 | 0.9866 | 0.0348 |
| 1.5 | 0.2307 | 0.2271 | 1.3334 | 0.0260 |
| 2.0 | 0.2874 | 0.2821 | 1.6560 | 0.0203 |
| 2.5 | 0.3362 | 0.3261 | 1.9146 | 0.0290 |
| 3.0 | 0.3770 | 0.3666 | 2.1520 | 0.0220 |

Wide nuisance scan:

| scan | mean observed `mu` | mean `nu1` | mean `nu2` | any bound hit |
| --- | ---: | ---: | ---: | --- |
| nominal | 0.9708 | 1.0036 | 0.9965 | false |
| TES low | 0.9425 | 0.9720 | 0.9997 | false |
| TES high | 0.9885 | 1.0256 | 1.0051 | false |
| JES low | 0.9433 | 0.9924 | 0.9651 | false |
| JES high | 0.9239 | 1.0011 | 1.0543 | false |
| soft MET | 0.9609 | 1.0058 | 0.9982 | false |
| ttbar up | 0.9497 | 1.0063 | 0.9963 | false |
| diboson up | 0.9686 | 1.0038 | 0.9966 | false |
| background up | 0.9539 | 1.0043 | 1.0028 | false |

Wide profile scan:

- Best fit: observed `mu = 1.05`, `f_s = 0.17885`, `nu1 = 0.99877`, `nu2 = 0.99840`.
- True label-derived signal fraction for that pseudo-experiment: `0.17213`.
- Plot: `runs/fair_universe_demo_debug_subset_wide/plots/profile_likelihood_scan.png`.
- Extra plots: `runs/fair_universe_demo_debug_subset_wide/plots/neyman_calibration_residuals.png` and
  `runs/fair_universe_demo_debug_subset_wide/plots/nuisance_impact_ranking.png`.

Interpretation:

- The wider closure still recovers the fitted signal fraction close to the label-derived truth across the full tested
  `mu` range.
- No tested nuisance scenario drives TES/JES to the spline boundary.
- The observable `f_s / f_s_nominal` is clearly nonlinear as a function of true `mu` at high signal strength, so the
  remaining work is calibration: the final Neyman correction should be built with enough statistics and should not rely
  on an over-simple linear approximation if full-statistics calibration shows the same curvature.

## Closed and Open Questions

- Closed for the local debug subset: nominal closure recovers fitted signal fraction near the label-derived truth.
- Closed for the local debug subset: TES-only and JES-only scans do not rail at `[0.9, 1.1]`.
- Closed for the local debug subset: soft MET and tested background-normalization shifts do not drive TES/JES to the
  spline boundary.
- Closed for local validation: increasing from the 5k first-row subset to 20k and then 50k randomly selected bounded
  subsets stabilizes the templates enough for debugging.
- Still open for production: build the final full-stat Neyman calibration and use a calibration model flexible enough to
  capture the observed nonlinearity between `f_s / f_s_nominal` and true `mu`.
