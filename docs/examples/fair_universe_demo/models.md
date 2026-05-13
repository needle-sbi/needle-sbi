# Models

The FAIR Universe demo uses two neural network architectures trained sequentially:

1. **Conditional Normalizing Flows (CNF)** — density estimators for jet feature distributions.
2. **Combined Classifier** — signal-vs-background discriminant that takes NF scores as additional
   features.

## Conditional Normalizing Flow

**Source:** `fair_universe_demo/models/nf_model.py`

### What it learns

A normalizing flow learns an invertible transformation between a simple base distribution (here,
a standard Normal) and the target data distribution. Given event features x, the flow computes:

```
x  →  [invertible layers]  →  z ~ N(0, I)
                                    ↕  (track log|det J|)
log p(x) = log p_N(z) + log|det(∂z/∂x)|
```

Because the transformation is invertible and the Jacobian determinant is tracked, the log-
likelihood of any input x is exact (no variational bound). This makes normalizing flows suitable
as likelihood ratio estimators.

In the context of NSBI: if you train one flow on signal events and another on background events,
the difference in their log-likelihoods approximates the log likelihood ratio log[p(x|S)/p(x|B)],
which is the optimal test statistic.

### Architecture

The flow uses a **Quadratic Rational Spline (NormalizingQuadFlow)** architecture defined in
`nf_layers.py`. It consists of `num_layers` coupling layers, each of which:
- Splits the input features into two halves.
- Uses a small neural network (conditioner) to predict spline parameters for one half.
- Applies the spline transformation to the other half, leaving the first unchanged.
- Alternates which half is transformed each layer.

This is a standard Neural Spline Flow (NSF) design. The spline parameterisation gives more
expressive transformations than affine coupling layers (RealNVP) while remaining stable to train.

### Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `num_jets` | — (required) | Input dimension: 1→20 features, 2→27 features |
| `num_layers` | 10 | Number of coupling layers. More layers = more expressive but slower. |
| `lr` | 1e-3 | Adam learning rate. |
| `c` | 1.0 | Adversarial loss weight (see below). |
| `clamp_val` | -10.0 | Minimum log-prob for the adversarial term (prevents -∞ gradient). |

### Training objective

The standard training objective for a normalizing flow is maximum likelihood: minimize -log p(x).

This demo uses a modified **adversarial objective** when both signal and background data are
available in the batch:

```python
loss = -c * log_prob_signal.mean() + log_prob_background.mean()
```

**Interpretation:** The first term pushes the flow to assign high probability to signal events
(maximise the signal log-likelihood). The second term pushes the flow to assign *low* probability
to background events (minimise the background log-likelihood), effectively making the flow a
likelihood ratio estimator rather than just a density estimator.

The `c` hyperparameter controls the trade-off:
- `c = 1.0`: equal weight on signal fit quality and background rejection.
- `c = 0.5`: less weight on signal → the flow is more conservative about boosting signal scores.
- `c = 2.0`: more weight on signal → the flow aggressively maximises signal scores.

The different `c` values are exposed as **systematic variants** in the config (`c_0.5`, `c_2.0`)
so their effect on the final result can be studied.

> **⚠ Open question:** Whether the adversarial objective genuinely improves likelihood ratio
> estimation or simply shifts the flow's focus is not yet fully validated. The nominal case
> (no adversarial term) would use `c=1` with only signal data in the batch. The current setup
> with mixed batches and adversarial loss may need further theoretical justification.

### Training setup

```yaml
# conf/trainers/cnf.yaml
max_epochs: 500
callbacks:
  - EarlyStopping(monitor="val_loss", patience=100)
  - ModelCheckpoint(monitor="val_loss", save_top_k=3)
```

Early stopping with patience=100 means training continues until the validation loss stops
improving for 100 consecutive epochs. For the full dataset this typically converges well before
epoch 500; on the test dataset (1000 events) it may stop very early.

---

## Combined Classifier

**Source:** `fair_universe_demo/models/classifier.py`

### What it learns

The classifier is trained to distinguish signal (H→ττ) from background events. It uses binary
cross-entropy loss and outputs a logit (pre-sigmoid score) for each event. At inference time,
`torch.sigmoid(logit)` gives a score in [0, 1] where 1 = signal-like.

The reason for having a *classifier* after the *density estimators* is that the NF log-likelihoods
provide useful features that capture different aspects of the signal/background separation than
the raw kinematic features. By concatenating them, the classifier has access to both learned
density information and the raw discriminating power of the kinematic variables.

### Architecture

The classifier has **separate input towers** for 1-jet and 2-jet events, because the two
topologies have different numbers of features (different jets present). Both branches share the
same trunk and have category-specific heads:

```
1-jet branch:  Linear(24) → BN → GELU → Linear(128) × 2 → hidden
               hidden → shared_trunk(128 → 128 × 4 → 256) → GELU
               256 → Linear(256) → GELU → Linear(1)   [1j logit]

2-jet branch:  Linear(31) → BN → GELU → Linear(128) × 2 → hidden
               hidden → shared_trunk (same weights)
               256 → Linear(256) → GELU → Linear(1)   [2j logit]
```

**Input dimensions:**
- 1-jet: 20 raw features + 4 NF scores = 24
- 2-jet: 27 raw features + 4 NF scores = 31

The NF scores are computed by `return1j2j()` in `utils/selection.py` during data loading. Each
score is the sigmoid of the NF log-likelihood, which maps it to [0, 1].

### Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `hidden_dim` | 128 | Width of all hidden layers in input towers and trunk. |
| `latent_dim` | 256 | Width of the trunk's output and classification head. |
| `lr` | 1e-3 | Adam learning rate. |

### Training

The classifier is trained jointly on both 1-jet and 2-jet events simultaneously:

```python
loss = BCE(logits_2j, labels_2j) + BCE(logits_1j, labels_1j)
```

Both categories contribute equally to the loss. The batch format from `ClassifierDatamodule`
provides both categories in every batch (as a dict with keys `x_2j`, `x_1j`, `l_2j`, `l_1j`).

**Dependency on NF training:** The classifier requires all four NF models to be trained first
(specified via `requires:` in `config.yaml`). The NF checkpoints are loaded inside the
`ClassifierDatamodule.setup()` call, which computes NF features on the fly for each batch.

### DataModule for the classifier (`ClassifierDatamodule`)

**Source:** `fair_universe_demo/models/classifier_datamodule.py`

The ClassifierDatamodule:
1. Loads the FAIR Universe parquet data.
2. Loads all four NF models from the snapshot (checkpoint paths passed as a parameter).
3. For each event, computes the 4 NF log-probability scores and appends them to the raw features.
4. Splits by jet count into 1-jet and 2-jet tensors.
5. Balances signal and background classes (undersamples the majority class).
6. Wraps in a `Dataset1j2j` which returns dicts with both jet categories in each item.

The `Dataset1j2j` format is required because the classifier processes both jet categories in a
single training step — the two losses must be combined before calling `optimizer.step()`.

> **⚠ Known limitation:** The class balancing in the datamodule is done by matching the per-
> background-process fractions observed in the test set bootstrapping. This is a heuristic that
> may not be optimal for training the classifier, and the relationship between training class
> balance and inference performance is not studied in detail.
