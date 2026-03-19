"""
Simple evaluation script using NEEDLE API
"""
import needle as nd
import torch

# Load config (uses initialize_hydra_config internally)
cfg = nd.config("../conf/config.yaml")

# The config is now fully resolved with *_override fields populated
print(f"Config resolved: {cfg.config._resolved}")
print(f"Datamodule override exists: {cfg.config.estimators.model_A.datamodule_override is not None}")

# Load test dataset
test_data = nd.dataset(cfg, split="test", estimator="model_A")
X, y = test_data.get_tensor(max_samples=1000)

# Load model
model = nd.model("../runs/dag_snapshot.json")

# Get predictions
likelihoods = model(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {likelihoods.shape}")
print(f"Target shape: {y.shape}")

# Calculate MSE
mse = torch.nn.functional.mse_loss(likelihoods, y)
print(f"Test MSE: {mse:.4f}")