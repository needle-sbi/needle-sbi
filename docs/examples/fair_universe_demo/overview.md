# Overview

## What this demo does

This demo implements a complete **Neural Simulation-Based Inference (NSBI)** pipeline for the
[FAIR Universe HiggsML Challenge](https://github.com/FAIR-Universe/HEP-Challenge). The central
question is: given a dataset of collision events, estimate the signal strength parameter μ
(how many Higgs bosons were produced relative to the Standard Model expectation) while accounting
for detector systematic uncertainties.

The pipeline:
1. Trains four **conditional normalizing flow** models to learn the density of jet features for
   signal (H→ττ) and background events, separately for 1-jet and 2-jet topologies, for a total of eight.
2. Trains a **combined classifier** that takes the raw jet features plus the NF log-likelihood
   scores as inputs and distinguishes signal from background.
3. Builds **classifier score histograms** across a grid of detector systematic parameter values
   (JES and TES).
4. Performs **statistical inference** using histogram morphing and maximum likelihood estimation
   to recover μ from pseudo-experimental datasets.

## Directory structure of this example

```
fair_universe_demo/
├── conf/                    # Hydra config files for this example
│   ├── config.yaml          # Main config: estimators and downstream tasks
│   ├── models/cnf.yaml      # Normalizing flow hyperparameters
│   ├── models/classifier.yaml
│   ├── datamodules/cnf.yaml
│   ├── datamodules/classifier.yaml
│   ├── trainers/cnf.yaml
│   └── trainers/classifier.yaml
├── fair_universe_demo/
│   ├── models/              # LightningModules and DataModules
│   ├── tasks/               # Downstream Luigi tasks
│   └── utils/               # Data loading, systematics, statistics
├── test_data/               # ~1000 events for smoke testing
└── docs/                    # This documentation
```
