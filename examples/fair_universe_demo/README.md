# fair_universe_demo

This example trains conditional normalizing flow (CNF) models on the FAIR Universe HiggsML dataset and then trains a classifier from the NF outputs.

## Models

- `nf_signal_1jet`, `nf_signal_2jet`: signal CNF models for 1-jet and 2-jet events.
- `nf_background_1jet`, `nf_background_2jet`: background CNF models for 1-jet and 2-jet events.
- `classifier`: combined classifier that uses features built from the four NF models.

## Data ingestion

- `models/nf_datamodule.py` loads jet data with `createJetData` from `utils/selection.py`.
- `models/classifier_datamodule.py` loads the parquet dataset from `test_data/FAIR_Universe_HiggsML_data.parquet` together with `FAIR_Universe_HiggsML_data_metadata.json`.
- The classifier datamodule uses `createMultiJetMultiNuanData` to build paired 1-jet and 2-jet tensors and then appends NF feature scores from the pretrained NF models.

## Directory layout

- `conf/`: Config (in hydra style)
- `models/`: Models, layers and datamodules (all compatible with pytorch Lightning)
- `test_data/`
  - `FAIR_Universe_HiggsML_data.parquet`: raw feature data.
  - `FAIR_Universe_HiggsML_data_metadata.json`: metadata for the parquet dataset.
- `utils/` Helper functions for dataset selection, derived quantities, and systematics.
