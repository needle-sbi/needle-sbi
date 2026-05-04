# Data Pipeline and Systematics

## Dataset

The demo uses the **FAIR Universe HiggsML dataset**, stored as a single Parquet file:

```
{FAIR_UNIVERSE_DATA}/
├── FAIR_Universe_HiggsML_data.parquet       # Event data (~1.4 GB)
└── FAIR_Universe_HiggsML_data_metadata.json # Croissant-format metadata
```

A bundled test dataset with ~1000 events lives in `test_data/` for smoke testing.

### Loading: `Data` class (`utils/dataset.py`)

The `Data` class wraps the Parquet file and handles train/test splitting. The split is fixed
at construction time (default 70% test, 30% train) using a stratified split on event class.

```python
data = Data(
    input_dir=root_dir,
    parquet_filename="FAIR_Universe_HiggsML_data.parquet",
    metadata_filename="FAIR_Universe_HiggsML_data_metadata.json",
    test_size=0.3,
)
data.load_train_set()
data.load_test_set()
```

The loaded train/test sets are dicts with keys:
- `data`: pandas DataFrame of features
- `weights`: event weights (reflecting the relative cross-sections of each process)
- `labels`: binary labels (1=signal, 0=background)
- `detailed_labels`: process names (`"htautau"`, `"ztautau"`, `"ttbar"`, `"diboson"`)

**Why keep train and test split separate?**
During NF training, models see only the *training partition*. During inference (histogram
generation, Neyman construction, evaluation), pseudo-experimental datasets are built from the
*test partition* — the models have never seen these events. This is a strict train/test separation
that mimics the real experimental condition where you train on simulation and apply to data.

### Shared loading: `dataset_sharing.py`

Loading the ~1.4 GB Parquet file is slow. When multiple downstream tasks need the same data,
`fetch_dataset()` uses a process-level cache so the file is only read once per process:

```python
from utils.dataset_sharing import fetch_dataset
data = fetch_dataset(root_dir)  # cached after first call
```

## Feature engineering: `derived_quantities.py`

After applying systematics (see below), derived features are recomputed by `DER_data()`. These
include physics-motivated quantities:

| Derived feature | Physical meaning |
|---|---|
| `DER_mass_transverse_met_lep` | Transverse mass of MET + lepton system |
| `DER_mass_vis` | Visible invariant mass of lepton + hadronic tau |
| `DER_pt_h` | Estimated Higgs pT (vector sum of visible + MET) |
| `DER_deltaeta_jet_jet` | Pseudorapidity gap between leading jets (VBF discriminant) |
| `DER_mass_jet_jet` | Invariant mass of dijet system (VBF discriminant) |
| `DER_prodeta_jet_jet` | Product of jet η values (another VBF discriminant) |
| `DER_deltar_had_lep` | ΔR between hadronic tau and lepton |
| `DER_pt_tot` | Scalar pT sum of all visible objects + MET |
| `DER_sum_pt` | Scalar sum of all object pTs |
| `DER_pt_ratio_lep_tau` | Ratio of lepton pT to tau pT |
| `DER_met_phi_centrality` | Azimuthal centrality of MET relative to lepton and tau |
| `DER_lep_eta_centrality` | Pseudorapidity centrality of lepton between the two jets |

The `DER_data` function is applied *after* systematics so the derived quantities are consistent
with the shifted primary kinematics.

## Jet selection and feature preparation: `selection.py`

### `filterbyjet(jet_num, data_vis)`

Splits the dataset by jet count and drops irrelevant columns:

| jet_num | Kept events | Dropped columns |
|---|---|---|
| 1 | `PRI_n_jets == 1` | `PRI_n_jets`, all `subleading` jet columns, zero-variance columns |
| 2 | `PRI_n_jets >= 2` | `PRI_n_jets`, zero-variance columns |
| 0 | `PRI_n_jets == 0` | `PRI_n_jets`, all jet columns, zero-variance columns |

After filtering, the value `-25` is used as a sentinel for missing kinematic variables (e.g.
subleading jet eta when there is no subleading jet). Events with any `-25` value are removed
before passing data to the neural network.

### Log-transform

For positively skewed kinematic variables (transverse momenta, masses), a log-transform is applied
before training. The column indices that receive the transform are **hardcoded by jet category**:

```python
# 1-jet:  columns 0, 3, 6, 9, 10, 13, 14, 16, 17
# 2-jet:  columns 0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23
```

> **⚠ Fragility warning:** These indices correspond to specific features only if the column order
> in the filtered DataFrame is stable. The column order depends on which columns are dropped by
> `filterbyjet` and the original Parquet column order. If the dataset schema changes (different
> column order or additional features), these hard-coded indices will silently apply the log to
> the wrong features. A more robust approach would be to log-transform by column name.

### `createJetData(jet_num, useTestData, ...)`

The main data-loading function. It combines bootstrapping, systematic application, and jet
filtering into one call:

```python
features, labels, weights, feature_names = createJetData(
    jet_num=1,
    useTestData=False,
    root_dir="/path/to/data",
    set_mu=1.0,      # μ for pseudo-experiment (scales signal)
    seed=42,
    n_param=[1, 1, 1, 1, 1, 0],  # [ttbar, diboson, bkg, TES, JES, soft_met]
)
```

The `n_param` list controls all systematic variations in one call:

| Index | Systematic | Nominal value | Typical range |
|---|---|---|---|
| 0 | ttbar_scale | 1.0 | [0.8, 1.2] |
| 1 | diboson_scale | 1.0 | [0.0, 2.0] |
| 2 | bkg_scale | 1.0 | [0.99, 1.01] |
| 3 | TES | 1.0 | [0.9, 1.1] |
| 4 | JES | 1.0 | [0.9, 1.1] |
| 5 | soft_met | 0.0 | [0.0, 5.0] |

When `useTestData=False` (training mode), it loads from the train partition and rebalances signal
and background to match the class fractions seen in the test bootstrapped sample.

When `useTestData=True` (inference mode), it creates a pseudo-experiment from the test partition
using Poisson bootstrapping.

### `return1j2j(alljet_data, models, device)`

Used during inference (histogram generation, Neyman construction) to:
1. Split `alljet_data` (the output of `createJetData(..., jet_num="all")`) into 1-jet and 2-jet.
2. Apply the log-transform.
3. Run all 8 NF models to get log-probability scores.
4. Append the 4 NF scores per jet category to the feature tensor.

Returns `(data_2j, data_1j, labels_2j, labels_1j)` where each data tensor has raw features +
NF scores concatenated.

## Systematics: `systematics.py`

### The `V4` class

A simple 4-vector class for particle physics calculations, operating on numpy arrays. It supports:

```python
v = V4(px, py, pz, e)
v.pt()     # transverse momentum √(px² + py²)
v.eta()    # pseudorapidity arcsinh(pz/pt)
v.phi()    # azimuthal angle arctan2(py, px)
v.m()      # invariant mass √(E² - p²)
v.deltaR(other)   # angular distance √(Δφ² + Δη²)
```

The class is used to propagate systematic shifts through the MET calculation (when you rescale a
jet's energy, the MET changes because jets and MET are correlated by momentum conservation).

### `mom4_manipulate(data, tes, jes, soft_met, seed)`

Applies kinematic systematics to a DataFrame of events:

1. **TES (Tau Energy Scale):** Scales `PRI_had_pt` by `tes`. The hadronic tau's 4-vector changes,
   which induces a change in MET (momentum must be conserved: if the tau gains energy, the MET
   decreases by the same vectorial amount).

2. **JES (Jet Energy Scale):** Scales `PRI_jet_leading_pt` and `PRI_jet_subleading_pt` (and
   `PRI_jet_all_pt`) by `jes`. Similarly updates MET from the vectorial change in jet momenta.

3. **Soft MET:** Adds Gaussian noise to MET components. This models soft hadronic activity and
   pile-up fluctuations that are not captured by reconstructed objects.

The MET propagation is done correctly via 4-vector arithmetic: the delta 4-vector of the rescaled
object is computed and added to the MET 4-vector.

### `postprocess(data)`

After systematics, applies hard kinematic thresholds (matching detector trigger requirements):
- `PRI_had_pt > 26 GeV` (if below, event is removed)
- `PRI_lep_pt > 20 GeV`
- `PRI_jet_leading_pt > 26 GeV` (if below, the jet is "removed" by setting its values to -25
  and decrementing `PRI_n_jets`)
- `PRI_jet_subleading_pt > 26 GeV` (same treatment)

This is important for JES variations: scaling jets down can push them below threshold, changing
the jet multiplicity of the event. A 2-jet event with JES=0.9 might become a 1-jet event.

> **⚠ Note:** The assumption in the postprocessing comment ("systematics transformation is
> monotonous in pt so that leading and subleading jet should never be swapped") is generally
> true for a uniform scaling but would break for more complex JES systematics.

### `get_bootstrapped_dataset(test_set, mu, seed, ...)`

Creates a Poisson-fluctuated pseudo-experiment from the test set. For each process class:
- The expected number of events is `mu × N_signal` (for signal) or `scale × N_background`.
- A Poisson draw determines the actual number of events (integer weights).
- Events are repeated by their integer weight and shuffled.

This mimics the statistical fluctuations you would see in real data: even if μ=1, each
pseudo-experiment will have a slightly different number of signal events.

## NF DataModule (`NormalizingFlowDatamodule`)

**Source:** `fair_universe_demo/models/nf_datamodule.py`

Wraps the data pipeline for NF training:

```python
datamodule = NormalizingFlowDatamodule(
    train_on_signal=True,       # True → train on H→ττ events; False → train on background
    num_jets=1,                 # 1 or 2
    root_dir="/path/to/data",
    batch_size=1000,
    train_test_split=0.8,       # 80% of training partition for training, 20% for validation
    fold_index=0,               # k-fold CV fold index
    n_folds=1,                  # total number of folds (1 = no CV)
)
```

During `setup()`:
1. Calls `createJetData(useTestData=False, ...)` to get training data.
2. Balances signal and background (keeps all signal, undersamples background to match signal count).
3. Splits into train and validation using `train_test_split`.
4. Computes `X_mean` and `X_std` from the training half (not the full dataset, to avoid leakage).

The mean and std are exposed as public attributes so `ConditionalNormalizingFlowModule.on_train_start()`
can copy them into the model's buffers.
