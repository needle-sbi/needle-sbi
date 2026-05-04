# FAIR Universe Demo Documentation

This is the documentation for the FAIR Universe HiggsML demonstration project. It shows how
to use NEEDLE to implement a complete Neural Simulation-Based Inference pipeline for a realistic
High Energy Physics problem.

## Contents

- [Overview and Physics Context](overview.md) — what NSBI is, the HiggsML challenge, and how the
  pipeline solves it. Start here if you are new to the project.
- [Models](models.md) — conditional normalizing flows and the combined classifier: architecture,
  hyperparameters, training objectives, and known limitations.
- [Data and Systematics](data_and_systematics.md) — dataset structure, feature engineering, jet
  selection, and how detector systematic variations are applied.
- [Downstream Tasks](tasks.md) — each analysis step explained in detail, including the physics
  reasoning behind the histogram morphing and Neyman construction, and flagged issues.

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
