# Downstream Tasks

All downstream tasks are plain `luigi.Task` subclasses. They are run after training is complete
(via `SnapshotTask`) and are invoked through `DownstreamTask` with the `--downstream` flag.

The pipeline runs in order: `histogram → neyman → eval → score → plot`. Validation tasks
(`validation_nf`, `validation_classifier`) run independently and are required by `plot`.

---

## `HistogramTask`

**Source:** `fair_universe_demo/tasks/histogram.py`
**Run with:** `law run DownstreamTask --downstream histogram`
**Output:** `{results_path_downstream}/hist.json`

### What it does

Builds a lookup table of classifier score histograms as a function of the two main systematic
parameters (JES and TES). This lookup table is the bridge between the trained models and the
statistical inference step.

The histogram covers a 10×10 grid: JES ∈ [0.9, 1.1] and TES ∈ [0.9, 1.1], with 10 equally
spaced points each → 100 (JES, TES) combinations.

For each combination:
1. Generate a pseudo-experimental dataset from the test partition with `createJetData(jet_num="all", ...)`.
2. Compute NF scores for 1-jet and 2-jet events using `return1j2j()`.
3. Score all events with the classifier.
4. Histogram the scores into 200 bins in [0, 1], separately for signal and background events.

The result is a JSON file:
```json
{
  "(0.9, 0.9)": { "sig": [0.1, 0.3, ...], "bg": [0.8, 0.5, ...] },
  "(0.9, 1.0)": { "sig": [...], "bg": [...] },
  ...
}
```

The histograms are density-normalised (sum × bin_width = 1 for each class).

### Parallelisation

The 100 histogram entries are computed in parallel using `concurrent.futures.ProcessPoolExecutor`
with 5 workers. Each worker loads its own copy of the model checkpoints from the snapshot and
processes its assigned (JES, TES) pair independently.

The initial dataset is loaded in the main process and shared with workers via the `init_worker`
mechanism (using a process-local global variable `shared_data`). This avoids re-reading the
~1.4 GB Parquet file in each worker.

> **⚠ Memory note:** With 5 workers, each holding its own copy of the model in memory, this
> step is relatively memory-hungry. If you run out of memory, reduce the number of workers in
> `ProcessPoolExecutor(max_workers=5, ...)`.

### Parsing the snapshot

`HistogramTask.parse_snapshot(filepath)` is a static method that reads the `dag_snapshot.json`
and extracts checkpoint paths:
- NF models: keys starting with `est=nf*` → collected as `{model_name: checkpoint_path}` dict
- Classifier: key starting with `est=classifier` → single checkpoint path

This parsing is also used by `NeymanTask`, `EvalTask`, and the validation tasks — it is the
standard way all downstream tasks load trained models.

---

## `NeymanTask`

**Source:** `fair_universe_demo/tasks/neyman.py`
**Run with:** `law run DownstreamTask --downstream neyman`
**Output:** `{results_path_downstream}/neyman.json` + calibration plots

### What it does in an NSBI context

The Neyman construction (or "toy-based calibration") studies how well the estimator recovers
the true μ across a range of injected μ values. It answers: "If the true signal strength were
μ=2, what would our estimator return on average? And what is the spread?"

This is a crucial validation step for any inference method in HEP: an unbiased estimator should
satisfy ⟨μ̂⟩ = μ_true for all μ_true values, and the spread σ(μ̂) gives the expected uncertainty.

### Algorithm

1. **Load histograms:** Read `hist.json` and reorganise as `{(JES, TES): (S_hist, B_hist)}`.

2. **Fit 2D splines bin-by-bin (`fit_2D_splines_bin_by_bin_from_dict`):**
   For each of the 200 histogram bins, fit a `SmoothBivariateSpline` to the 100 (JES, TES)
   grid points. This gives a smooth interpolant that can predict the histogram at any (JES, TES)
   value, not just the 10×10 grid points.

3. **Scan μ from 0.1 to 3.2 (10 points), 10 pseudo-experiments each:**
   For each (μ_true, seed) pair:
   - Generate a pseudo-experimental dataset with `createJetData(set_mu=μ_true)`.
   - Compute classifier scores with `return1j2j()`.
   - Call `compute_signal_fraction()` to extract the MLE of μ.

4. **Save results:** `{μ_true: [μ̂_seed0, μ̂_seed1, ...]}` as JSON.

5. **Plot calibration curve** (μ_true vs ⟨μ̂⟩) and signal yield curve.

### Maximum likelihood estimation (`compute_signal_fraction`)

**Source:** `fair_universe_demo/utils/stats.py`

The MLE of μ is indirect — the code actually estimates the **signal fraction** f_s (the fraction
of all events that are signal), then converts:

```python
μ̂ = f_s_hat / f_s_nominal
```

where `f_s_nominal` is the signal fraction at μ=1 (computed once at the start of `NeymanTask`
and cached as a `cached_property`).

The signal fraction is extracted by minimising a negative log-likelihood over the observed score
histogram:

```
params = (f_s, ν_1, ν_2)    # signal fraction + two nuisance parameters (JES, TES)

E(bin) = N_total × [f_s × S(bin; ν_1, ν_2) + (1 - f_s) × B(bin; ν_1, ν_2)] × bin_width

NLL = Σ_bin [E(bin) - n_obs(bin) × log(E(bin))]
    + Gaussian priors on ν_1, ν_2 (σ = 0.01)
```

where `S(bin; ν_1, ν_2)` and `B(bin; ν_1, ν_2)` are the signal and background histogram bin
contents at parameter values (ν_1, ν_2), evaluated by interpolating the fitted 2D splines.

The optimisation uses `scipy.optimize.minimize` with the L-BFGS-B method and bounds:
- f_s ∈ [0, 1]
- ν_1, ν_2 ∈ [-3, 3]

> **⚠ Known issues and open questions:**
>
> 1. **Sample size for calibration:** Only 10 pseudo-experiments per μ_true value is very few
>    for a robust calibration. The spread of μ̂ estimates will be large, making it hard to
>    characterise the resolution and check for bias. In production, 100-1000 toys per μ_true
>    point is typical. The current setup is a compromise for speed.
>
> 2. **Spline extrapolation:** The spline is fitted on [0.9, 1.1] for both JES and TES. If the
>    MLE optimizer queries outside this range (ν ∈ [-3, 3] in units where σ=0.01 means the
>    physical range is [0.97, 1.03]), there is a mismatch in parameterisation. The prior
>    σ=0.01 should keep ν near 0, but this coupling between nuisance parameter units and
>    physical units is implicit and fragile.
>
> 3. **Fixed μ=1 normalisation:** `f_s_nominal` is computed with a single seed (seed=0). This
>    introduces a small statistical uncertainty on the normalisation itself.

---

## `EvalTask`

**Source:** `fair_universe_demo/tasks/eval.py`
**Run with:** `law run DownstreamTask --downstream eval`
**Output:** `{results_path_downstream}/eval.json`

### What it does

Runs the full evaluation as specified by the FAIR Universe challenge: for each pseudo-experiment
defined in the test settings file, it calls the inference pipeline and collects the model's
predictions of μ̂, δμ̂, and the 68% confidence interval [p16, p84].

### Test settings

The test settings JSON (`conf/test_settings.json` or `conf/test_settings_nominal.json`) controls:
- `ground_truth_mus`: list of μ values for the test sets (e.g. [0.5, 1.0, 1.5, ...])
- `num_pseudo_experiments`: how many pseudo-experiments per μ value
- `num_of_sets`: how many distinct μ sets
- `systematics`: which systematics to activate (tes, jes, soft_met, ttbar_scale, etc.)

The `nominal` flag in `custom_settings` controls which test settings file is used:
- `nominal=true` → `test_settings_nominal.json` (only JES and TES active, simpler)
- `nominal=false` → `test_settings.json` (all systematics active)

### Inference per pseudo-experiment

For each (set_index, test_set_index) combination:

1. Compute seed = `set_index × num_pseudo_experiments + test_set_index + initial_seed`
2. Sample systematic parameters from their prior distributions (clipped Gaussians/lognormal)
3. Call `predict(mu, hist_path, neyman_path, snapshot_path, data, nuisance_parameters)`
4. Record `{mu_hat, delta_mu_hat, p16, p84, mu_true, test_set_index}`

The `predict()` function (`utils/eval.py`) generates a pseudo-experimental dataset with the
sampled systematics, runs the classifier, and uses the Neyman construction results (spline
fits + bias correction from `neyman.json`) to compute the confidence interval.

### Confidence interval calculation (`utils/stats.py`)

The confidence interval uses a **posterior-based approach**:
1. Load the bias function from `neyman.json` (linear fit: `μ̂ = a × μ_true + b`).
2. Compute the corrected uncertainty σ(μ_true) by interpolating the Neyman construction spread
   and dividing by the slope `a` of the bias function.
3. Build a likelihood function: `L(μ_true | μ̂_obs) ∝ exp(-½((μ̂_obs_corrected - μ_true)/σ(μ_true))²)`.
4. Numerically integrate to get the CDF, then find the 16th and 84th percentiles.

> **⚠ Approximation:** This is a Gaussian approximation to the posterior. It assumes the
> uncertainty is well-described by a Gaussian and that the bias correction is linear. For
> very small or large μ (where the Poisson statistics become important), this approximation
> may not hold. A likelihood-based confidence interval using the full profile likelihood would
> be more rigorous.

---

## `ScoreTask`

**Source:** `fair_universe_demo/tasks/score.py`
**Run with:** `law run DownstreamTask --downstream score`
**Output:** `{results_path_downstream}/scores/`

### What it does

Computes the official FAIR Universe challenge scoring metric using the predictions from
`EvalTask`. The scoring evaluates:
- **Coverage probability:** What fraction of confidence intervals [p16, p84] contain the true μ?
  (Should be ~68% for correctly calibrated 1σ intervals.)
- **Expected uncertainty:** The average width of the confidence intervals.
- **Bias:** Systematic shift in the μ̂ estimates.

The scoring code is adapted from the official FAIR Universe challenge repository.

---

## `PlottingTask`

**Source:** `fair_universe_demo/tasks/plot_results.py`
**Run with:** `law run DownstreamTask --downstream plot`
**Output:** `{results_path_downstream}/plots/`

### What it does

Aggregates all results and produces final visualisation plots:
- μ̂ vs μ_true calibration curve (from Neyman construction)
- Confidence interval coverage plots
- Score distributions for signal and background
- Summary metrics

Requires all of: `score`, `validation_nf`, `validation_classifier`.

---

## `NormalizingFlowValidationTask`

**Source:** `fair_universe_demo/tasks/plot_nf.py`
**Run with:** `law run DownstreamTask --downstream validation_nf`
**Output:** `{results_path_downstream}/plots/`

Validates each of the 8 NF model variants by plotting:
- Log-probability distributions for signal vs background test events
- Feature-by-feature density comparisons (NF samples vs real data)
- Loss curves

Run separately for each model via the `model_name` expansion in the config.

---

## `ClassifierValidationTask`

**Source:** `fair_universe_demo/tasks/plot_classifier.py`
**Run with:** `law run DownstreamTask --downstream validation_classifier`
**Output:** `{results_path_downstream}/plots/classifier/`

Validates the classifier for each jet category (1-jet and 2-jet) by plotting:
- ROC curves and AUC
- Score distributions
- Loss curves

---

## Debugging tips

**Task is stuck / not progressing:**
Run `law run DownstreamTask --downstream <name> --print-status 0` to see which tasks are
complete and which are pending without running anything.

**Rerunning a failed task:**
```bash
law run DownstreamTask --downstream histogram \
    --config-file conf/config.yaml \
    --remove-output 0,a,y
```
The `--remove-output 0,a,y` flag deletes this task's output files so LAW re-runs it.
Use with care — it only removes the target task's outputs, not upstream ones.

**Checking histogram quality:**
Open `hist.json` and inspect a few entries. The signal and background arrays should each have
non-negative values that sum to approximately 1/bin_width (since they are density-normalised).
Completely flat histograms or all-zero entries indicate the model is not discriminating.

**Numerical issues in MLE:**
If `compute_signal_fraction` fails to converge (`opt_result.success = False`), check:
- Whether the histogram has bins with zero counts (adds -∞ to the NLL; handled by clipping to 1e-10).
- Whether the spline extrapolation returns negative values (handled by `np.clip(..., 0, None)`).
- The initial parameter guess `initial_params = [0.001, 1, 1]` assumes small signal fraction and
  nuisance parameters at their nominal values. For pseudo-experiments with very large or small μ,
  this starting point may not be in the right basin.
