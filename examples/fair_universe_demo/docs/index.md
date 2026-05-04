# FAIR Universe Demo

NEEDLE ships with one reference example that demonstrates a complete end-to-end machine learning
pipeline for a High Energy Physics inference problem.

**Location:** `examples/fair_universe_demo/`

This example implements a Neural Simulation-Based Inference (NSBI) pipeline for the
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

# Smoke test with the bundled ~1000-event test dataset
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

- [Overview and Physics Context](overview.md) — what NSBI is, the HiggsML challenge, and how the
  pipeline solves it. Start here if you are new to the project.
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

## At a glance

```
conf/config.yaml
 ├── estimators:
 │    ├── nf_signal_1jet    (CNF, c=0.5 and c=2.0 variants)
 │    ├── nf_signal_2jet    (CNF, c=0.5 and c=2.0 variants)
 │    ├── nf_background_1jet  (CNF, c=0.5 and c=2.0 variants)
 │    ├── nf_background_2jet  (CNF, c=0.5 and c=2.0 variants)
 │    └── classifier          (requires all four NFs above)
 └── downstream_tasks:
      ├── validation_nf       → validate each NF model
      ├── validation_classifier → validate classifier
      ├── histogram           → build JES/TES score histogram grid
      ├── neyman              → Neyman construction (μ calibration)
      ├── eval                → run pseudo-experiments, get μ̂ ± δμ̂
      ├── score               → official FAIR Universe scoring
      └── plot                → final result plots
```
