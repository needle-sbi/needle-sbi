# FAIR Universe Demo within NEEDLE

## Repo Structure

This example trains conditional normalizing flow (CNF) models on the FAIR Universe HiggsML dataset and
 then trains a classifier from the NF outputs.

### Models

- `nf_signal_1jet`, `nf_signal_2jet`: signal CNF models for 1-jet and 2-jet events.
- `nf_background_1jet`, `nf_background_2jet`: background CNF models for 1-jet and 2-jet events.
- `classifier`: combined classifier that uses features built from the four NF models.

### Data ingestion

- `models/nf_datamodule.py` loads jet data with `createJetData` from `utils/selection.py`.
- `models/classifier_datamodule.py` loads the parquet dataset from `test_data/FAIR_Universe_HiggsML_data.parquet`
 together with `FAIR_Universe_HiggsML_data_metadata.json`.
- The classifier datamodule uses `createMultiJetMultiNuanData` to build paired 1-jet and 2-jet tensors
 and then appends NF feature scores from the pretrained NF models.

### Directory layout

- `conf/`: Config (in hydra style)
- `models/`: Models, layers and datamodules (all compatible with pytorch Lightning)
- `test_data/`
  - `FAIR_Universe_HiggsML_data.parquet`: raw feature data.
  - `FAIR_Universe_HiggsML_data_metadata.json`: metadata for the parquet dataset.
- `utils/` Helper functions for dataset selection, derived quantities, and systematics.

## Luigi Tasks

We expect the training to happen within the NEEDLE Framework, but upstream Tasks can still be executed
 using luigi. You just have to manually where the input / output files should be.

 ### Histogramming

 ```bash
 luigi \
  --module examples.fair_universe_demo.stats.histogram CreateHistogramTask \
  --root-dir examples/fair_universe_demo/test_data \
  --snapshot-path runs/fair_universe_demo_test/dag_snapshot.json \
  --json-save-path runs/fair_universe_demo_test/ \
  --local-scheduler \
  --lock-pid-dir examples/fair_universe_demo/.luigi_locks
```

This example assumes that you run this Task from the root NEEDLE directory. If you want to run it from this directory instead, adjust the paths. A bit more info about each arg is provided in the argparser
of the Task (can be printed using `--help` in the CLI). The last two luigi-specific args indicate to use the local scheduler of the machine (even if a central luigi scheduler is provided) and where to keep
the lock file for luigi Tasks. These can be written into a config file if needed.
