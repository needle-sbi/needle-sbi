# FAIR Universe Demo

NEEDLE ships with one reference example that demonstrates a complete end-to-end machine learning
pipeline for a typical HEP inference problem.

**Location:** `examples/fair_universe_demo/`

This example implements an NSBI pipeline for the
[FAIR Universe HiggsML Challenge](https://github.com/FAIR-Universe/HEP-Challenge). It trains
conditional normalizing flows and a combined classifier on Higgs boson decay events, then
performs statistical inference to estimate the signal strength parameter μ.

## Quick start

```bash
export FAIR_UNIVERSE_DATA=/path/to/fair_universe_data
source .venv/bin/activate
source setup.sh

# Train everything and run the full analysis
law run DownstreamTask \
    --downstream plot \
    --config-file examples/fair_universe_demo/conf/config.yaml

# Test with the bundled ~1000-event dataset
law run DownstreamTask \
    --downstream eval \
    --config-file examples/fair_universe_demo/conf/config.yaml \
    --hydra-overrides "custom_settings.use_test_data=True"
```

## What the pipeline produces

Running the full demo trains 9 models (4 NFs × 2 systematic variants + 1 classifier) and then:

1. Generates a 10×10 grid of classifier score histograms across JES and TES variations.
2. Fits 2D splines to interpolate between histogram bins.
3. Performs a Neyman construction: scans μ from 0.1 to 3.2 and estimates the MLE for each
   pseudo-experiment.
4. Runs the official FAIR Universe scoring to get a coverage probability and expected uncertainty.
5. Produces validation plots for each model and final result plots.

Output lands in `runs/fair_universe_demo_fixed_normalization/stat_only_histogram_mu_one/`.

## Contents

- [Overview](overview.md) — A bit more context on the FAIR Universe HiggsML Challenge
- [Models](models.md) — conditional normalizing flows and the combined classifier: architecture,
  hyperparameters, training objectives, and known limitations.
- [Downstream Tasks](tasks.md) — each analysis step explained in detail, including the physics
  reasoning behind the histogram morphing and Neyman construction, and flagged issues.

```{toctree}
:hidden:
:maxdepth: 2

overview
models
tasks
```
