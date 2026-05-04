# Overview and Physics Context

## What this demo does

This demo implements a complete **Neural Simulation-Based Inference (NSBI)** pipeline for the
[FAIR Universe HiggsML Challenge](https://github.com/FAIR-Universe/HEP-Challenge). The central
question is: given a dataset of collision events, estimate the signal strength parameter μ
(how many Higgs bosons were produced relative to the Standard Model expectation) while accounting
for detector systematic uncertainties.

The pipeline:
1. Trains four **conditional normalizing flow** models to learn the density of jet features for
   signal (H→ττ) and background events, separately for 1-jet and 2-jet topologies.
2. Trains a **combined classifier** that takes the raw jet features plus the NF log-likelihood
   scores as inputs and distinguishes signal from background.
3. Builds **classifier score histograms** across a grid of detector systematic parameter values
   (JES and TES).
4. Performs **statistical inference** using histogram morphing and maximum likelihood estimation
   to recover μ from pseudo-experimental datasets.

## The NSBI framework in brief

Traditional cut-and-count analyses reduce the event data to a few hand-crafted observables before
doing statistics. NSBI replaces the hand-crafted step with trained models that learn summary
statistics directly from the full feature space. The key idea is:

> Train a model to approximate the likelihood ratio p(x|signal) / p(x|background), then use that
> ratio as a test statistic for hypothesis testing.

For a binary classification problem, the optimal classifier output *is* a monotone transformation
of the likelihood ratio (by the Neyman-Pearson lemma). So training a good signal-vs-background
classifier is equivalent to approximating the likelihood ratio.

The complication in real experiments is **systematic uncertainties**: the detector response is not
perfectly known, so the expected distributions of observables depend on nuisance parameters θ
(like the jet energy scale). NSBI handles this by training on data with varied θ, or by building
template histograms as a function of θ and then marginalising over them during inference.

## The HiggsML dataset

The dataset comes from the ATLAS experiment's search for H→ττ (Higgs decaying to two tau leptons).
Events are classified by how many jets are reconstructed:

| Category | Signal? | Background processes |
|---|---|---|
| 1-jet | H→ττ (htautau) | Z→ττ (ztautau), tt̄ (ttbar), diboson |
| 2-jet | H→ττ (htautau) | Z→ττ (ztautau), tt̄ (ttbar), diboson |

The features are a mix of **primary** features (directly measured by the detector) prefixed
`PRI_` and **derived** features computed from combinations of primaries, prefixed `DER_`:

| Feature group | Examples | Physical meaning |
|---|---|---|
| Lepton kinematics | `PRI_lep_pt`, `PRI_lep_eta`, `PRI_lep_phi` | Transverse momentum, pseudorapidity, azimuth of the reconstructed lepton |
| Hadronic tau | `PRI_had_pt`, `PRI_had_eta`, `PRI_had_phi` | Same for the hadronic tau candidate |
| Missing transverse energy | `PRI_met`, `PRI_met_phi` | Inferred from momentum imbalance (neutrinos escape unseen) |
| Jet kinematics | `PRI_jet_leading_pt`, `PRI_jet_subleading_pt`, etc. | Momenta of reconstructed jets |
| Derived masses and angles | `DER_mass_transverse_met_lep`, `DER_deltaeta_jet_jet`, etc. | Physics-motivated combinations (e.g. transverse mass, ΔR) |

After filtering by jet count and dropping zero-variance columns, the model sees:
- **1-jet events**: 20 features
- **2-jet events**: 27 features

## Detector systematics modelled

Two main kinematic systematics are varied during histogram generation and inference:

**Jet Energy Scale (JES):** A global rescaling of all jet transverse momenta.
- Physical motivation: the detector's energy calibration for jets has an uncertainty of ~1-2%.
- In the dataset: `PRI_jet_leading_pt *= jes`, `PRI_jet_subleading_pt *= jes`, etc.
- Effect: shifts events across the `PRI_n_jets` threshold, so some 2-jet events become 1-jet
  events (and vice versa) and the MET changes because jets contribute to the MET calculation.
- Range varied: JES ∈ [0.9, 1.1], sampled on a 10×10 grid with TES.

**Tau Energy Scale (TES):** A global rescaling of the hadronic tau momentum.
- Physical motivation: tau reconstruction efficiency and energy scale have separate calibrations.
- In the dataset: `PRI_had_pt *= tes`, with the MET updated to conserve momentum.
- Range varied: TES ∈ [0.9, 1.1], sampled simultaneously with JES.

Additional normalisation systematics are handled as weight adjustments (ttbar_scale,
diboson_scale, bkg_scale) and are modelled stochastically in pseudo-experiments during evaluation.

## Signal strength parameter μ

The signal strength μ is the ratio of the measured Higgs production cross-section to the Standard
Model prediction. μ=1 means "SM Higgs observed as expected", μ=0 means "no Higgs signal". The
inference task is to estimate μ and its 68% confidence interval [p16, p84] from a dataset.

In pseudo-experiments, μ controls how many signal events are injected:
- Signal events are scaled by μ.
- Background events are scaled by their normalisations (which may also be varied by systematics).
- The resulting mixture is passed through the trained classifier to get a score distribution.
- The MLE of μ is extracted by fitting the observed score histogram to a linear combination of
  signal and background templates.

## Pipeline overview diagram

```
 Training phase (NEEDLE MainTask → SnapshotTask)
 ─────────────────────────────────────────────────────────────────────
   Data → NF(signal,1j,c=0.5) ─┐
   Data → NF(signal,1j,c=2.0) ─┼──┐
   Data → NF(background,1j)   ─┘  │  NF scores as features
   Data → NF(signal,2j)       ─┐  ↓
   Data → NF(background,2j)   ─┼→ Classifier(1j+2j)
                                ↓
                          dag_snapshot.json
                          (checkpoint paths)

 Inference phase (DownstreamTask chain)
 ─────────────────────────────────────────────────────────────────────
   snapshot → HistogramTask → hist.json        (JES/TES grid)
   snapshot → NeymanTask    → neyman.json      (μ scan + MLE)
   snapshot → EvalTask      → eval.json        (pseudo-experiments)
   eval.json → ScoreTask    → scores/          (official metric)
   everything → PlottingTask → plots/          (validation figures)
```

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
