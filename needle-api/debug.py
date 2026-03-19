"""
Debug script to see actual config structure after Hydra composition
"""
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

config_path = Path("../conf/config.yaml").resolve()
config_dir = str(config_path.parent)
config_name = config_path.stem

with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name=config_name)
    
    print("=" * 80)
    print("Full config structure:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print()
    
    print("=" * 80)
    print("Estimator model_A keys:")
    print("=" * 80)
    for key in cfg.estimators.model_A.keys():
        print(f"  - {key}")
    print()
    
    print("=" * 80)
    print("Estimator model_A structure:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg.estimators.model_A))