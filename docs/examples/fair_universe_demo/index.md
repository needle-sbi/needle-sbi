# FAIR Universe Demo

NEEDLE ships with one reference example that demonstrates a complete end-to-end machine learning
pipeline for a typical HEP inference problem.

**Location:** `examples/fair_universe_demo/`

This example implements an NSBI pipeline for the
[FAIR Universe HiggsML Challenge](https://github.com/FAIR-Universe/HEP-Challenge). It trains
conditional normalizing flows and a combined classifier on Higgs boson decay events, then
performs statistical inference to estimate the signal strength parameter μ.

## Quick start

Install the code with `git clone` to access the source code for the examples.

From the repository root, install NEEDLE and the demo. `uv sync` installs both, since the demo is a
uv workspace member; with imperative uv use `uv pip install .` in the root and again in
`examples/fair_universe_demo`; with pip use `pip install -e .` and
`pip install -e examples/fair_universe_demo`.

```bash
uv sync
source .venv/bin/activate
source setup.sh   # sets LAW_* and puts the repository root on PYTHONPATH

export FAIR_UNIVERSE_DATA=/path/to/fair_universe_data
cd examples/fair_universe_demo

# Train everything and run the full analysis
law run DownstreamTask \
    --downstream plot \
    --config-file conf/config.yaml

# Test with the bundled ~1000-event dataset
law run DownstreamTask \
    --downstream eval \
    --config-file conf/config.yaml \
    --hydra-overrides "custom_settings.use_test_data=True"
```

See [Usage](../../setup/usage.md), [LAW Tasks](../../concepts/law_tasks.md) and
[b2luigi Tasks](../../concepts/b2luigi_tasks.md) for the full per-backend CLI reference.

Running with b2luigi is simply

```bash
b2luigi run DownstreamTask \
    --param downstream=plot \
    --param config-file conf/config.yaml

```

When using `b2luigi`, install the `fair_universe_demo` package locally so the Tasks can be
registered: `pip install -e examples/fair_universe_demo` from the repository root. `source setup.sh`
(or `export PYTHONPATH=$PWD`) is still required for the `examples.*` targets. Also keep the `--param`
change in mind.

## What the pipeline produces

Running the full demo trains 9 models (4 NFs × 2 systematic variants + 1 classifier) and then:

1. Generates a 10×10 grid of classifier score histograms across JES and TES variations.
2. Fits 2D splines to interpolate between histogram bins.
3. Performs a Neyman construction: scans μ from 0.1 to 3.2 and estimates the MLE for each
   pseudo-experiment.
4. Runs the official FAIR Universe scoring to get a coverage probability and expected uncertainty.
5. Produces validation plots for each model and final result plots.

Output lands in `runs/fair_universe_demo_fixed_normalization/stat_only_histogram_mu_one/`.

## Credit

The physics analysis code in `fair_universe_demo/` (models, systematics, statistics and
downstream tasks) is adapted from the original winning solution with [Contrastive Normalizing Flows](https://github.com/ibrahimEls/CNFParameterEstimation) by Ibrahim Elsharkawy for the
[FAIR Universe HiggsML Challenge](https://github.com/FAIR-Universe/HEP-Challenge).

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
