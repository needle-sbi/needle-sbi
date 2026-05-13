# FAIR Universe Demo within NEEDLE

A demonstration of training conditional normalizing flow (CNF) models on the FAIR Universe HiggsML dataset,
followed by training a combined classifier using the flow outputs as features.

## Setup

### Using `uv` (Recommended)

```bash
# Install project dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### Using `pip`

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Docs

Refer to the documentation of NEEDLE under the section `examples/fair_universe_demo`

## Directory Structure

```
fair_universe_demo/
├── conf/                         # Hydra configuration files
│   ├── config.yaml               # Main config file to point to from NEEDLE repo
│   ├── datamodules/              # DataModule configs
│   ├── models/                   # Model architecture configs
│   └── trainers/                 # PyTorch Lightning trainer configs
├── fair_universe_demo/
│   ├── models/                   # PyTorch Lightning modules & datamodules
│   │   ├── nf_model.py           # Conditional Normalizing Flow
│   │   ├── classifier.py         # Combined classifier model
│   │   ├── nf_layers.py          # Flow building blocks
│   │   ├── nf_datamodule.py      # DataModule for normalizing flows
│   │   └── classifier_datamodule.py  # DataModule for classifier
│   ├── tasks/                    # Luigi/NEEDLE workflow tasks
│   │   ├── eval.py               # Model evaluation tasks
│   │   ├── histogram.py          # Histogram binning tasks
│   │   ├── plot.py               # Visualization tasks (after all statistics are performed)
│   │   ├── score.py              # Scoring tasks (based on FAIR Universe scoring tool)
│   │   ├── validation.py         # Validation tasks (for NF model)
│   │   └── neyman.py             # Statistical analysis (NLRE and confidence interval)
│   └── utils/                    # Helper utilities
│       ├── datasets.py           # Data loading utilities
│       ├── derived_quantities.py # Feature engineering (based on FAIR Universe code)
│       ├── selection.py          # Event selection cuts and data ingestion
│       ├── stats.py              # Statistical functions
│       ├── systematics.py        # Systematic uncertainties and bootstrap data creation
│       └── eval.py               # Evaluation utilities
└── test_data/                    # Sample data for testing (around 1000events)
    ├── FAIR_Universe_HiggsML_data.parquet
    └── FAIR_Universe_HiggsML_data_metadata.json
```

## Models

### Conditional Normalizing Flow (CNF)
- **File**: `models/nf_model.py`
- **Purpose**: Learns density of jet features independently for signal and background events
- **Architecture**: Quad flow with configurable number of layers
- **Configured for**: 1-jet and 2-jet event types
- **Output**: Log-likelihood scores used as discriminative features

### Combined Classifier
- **File**: `models/classifier.py`
- **Purpose**: Binary classification using normalizing flow scores as features
- **Architecture**: Separate input towers for 1-jet and 2-jet tensors, shared trunk with classification heads
- **Input**: Features from pretrained CNF models + raw event features
- **Output**: Classification scores for signal vs background

## Data Flow

1. **CNF Training**: Normalizing flows learn unconditional jet feature distributions
   - Loads jet data via `utils/selection.py` using `createJetData()`
   - Separate models for 1-jet/2-jet, signal/background

2. **Classifier Training**: Uses pretrained CNF outputs as features
   - Loads FAIR Universe parquet data via `utils/datasets.py`
   - Computes CNF scores for all events
   - Trains classifier on combined feature set

3. **Downstream Tasks**: Evaluate and analyze model predictions
   - Compute histograms with systematic variations
   - Generate plots and statistical tests
   - Validate model performance

## Downstream Tasks

The example includes Luigi-compatible downstream tasks for evaluation and analysis. These can be run
directly using Luigi or integrated into the NEEDLE workflow via `DownstreamTask`.

### Running Tasks from NEEDLE

To use these tasks as downstream tasks in NEEDLE:

1. Add to your config under `downstream_tasks`:
   ```yaml
   downstream_tasks:
     histogram:
       args:
         _target_: fair_universe_demo.tasks.histogram.HistogramTask
         root_dir: ./test_data  # point to FAIR Universe Data
         json_save_path: "${results_path}/histograms"
   ```

2. Run from NEEDLE:
   ```bash
   law run DownstreamTask --downstream histogram
   ```
