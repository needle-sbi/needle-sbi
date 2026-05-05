#!/usr/bin/env python3
"""
Sanity check script for PseudoModel Benchmark Framework

Verifies that all components are properly installed and configured.
"""

import sys
from pathlib import Path
import yaml

def check_file(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_executable(filepath, description):
    """Check if a file exists and is executable."""
    path = Path(filepath)
    if path.exists():
        if path.stat().st_mode & 0o111:
            print(f"✓ {description} (executable): {filepath}")
            return True
        else:
            print(f"⚠ {description} exists but not executable: {filepath}")
            return False
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists."""
    path = Path(dirpath)
    if path.exists() and path.is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"⚠ {description} does not exist (will be created): {dirpath}")
        return False

def check_python_imports():
    """Check if required Python modules can be imported."""
    print("\n" + "=" * 60)
    print("Checking Python Dependencies")
    print("=" * 60)
    
    required = ['torch', 'numpy', 'yaml', 'matplotlib', 'pandas']
    optional = ['seaborn']
    
    all_ok = True
    
    for module in required:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} NOT FOUND (REQUIRED)")
            all_ok = False
    
    for module in optional:
        try:
            __import__(module)
            print(f"✓ {module} (optional)")
        except ImportError:
            print(f"⚠ {module} not found (optional, will use fallback)")
    
    return all_ok

def check_grid_config():
    """Check grid_config.yaml is valid."""
    print("\n" + "=" * 60)
    print("Checking Grid Configuration")
    print("=" * 60)
    
    config_path = Path("benchmarks/grid_config.yaml")
    
    if not config_path.exists():
        print(f"✗ Grid config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['n_networks', 'batch_size', 'network_size', 'benchmark', 'output']
        for key in required_keys:
            if key in config:
                print(f"✓ Config section: {key}")
            else:
                print(f"✗ Missing config section: {key}")
                return False
        
        # Print parameter ranges
        n_ensembles = len(config['n_networks']['ensembles'])
        n_folds = len(config['n_networks']['folds'])
        n_batches = len(config['batch_size'])
        n_sizes = len(config['network_size']['configs'])
        total_jobs = n_ensembles * n_folds * n_batches * n_sizes
        
        print(f"\n  Parameter space:")
        print(f"    Ensembles: {n_ensembles} values")
        print(f"    Folds: {n_folds} values")
        print(f"    Batch sizes: {n_batches} values")
        print(f"    Network sizes: {n_sizes} configs")
        print(f"    Total jobs: {total_jobs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading grid config: {e}")
        return False

def check_model_config():
    """Check that MockTransformer config has required parameters."""
    print("\n" + "=" * 60)
    print("Checking Model Configuration")
    print("=" * 60)
    
    config_path = Path("conf/models/mock_transformer.yaml")
    
    if not config_path.exists():
        print(f"✗ Model config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for new parameters
        if 'hidden_dim' in config and 'n_hidden' in config:
            print(f"✓ MockTransformer has configurable architecture")
            print(f"    hidden_dim: {config['hidden_dim']}")
            print(f"    n_hidden: {config['n_hidden']}")
            return True
        else:
            print(f"✗ Missing hidden_dim or n_hidden in model config")
            return False
        
    except Exception as e:
        print(f"✗ Error reading model config: {e}")
        return False

def main():
    print("=" * 60)
    print("PseudoModel Benchmark Framework - Sanity Check")
    print("=" * 60)
    
    all_ok = True
    
    # Check framework files
    print("\n" + "=" * 60)
    print("Checking Framework Files")
    print("=" * 60)
    
    files = [
        ("benchmarks/grid_config.yaml", "Grid configuration"),
        ("benchmarks/README.md", "Documentation"),
        ("benchmarks/IMPLEMENTATION_SUMMARY.md", "Implementation summary"),
        ("benchmarks/requirements.txt", "Python requirements"),
    ]
    
    for filepath, desc in files:
        all_ok &= check_file(filepath, desc)
    
    # Check executable scripts
    print("\n" + "=" * 60)
    print("Checking Executable Scripts")
    print("=" * 60)
    
    scripts = [
        ("benchmarks/orchestrator.py", "Job orchestrator"),
        ("benchmarks/benchmark_runner.py", "Benchmark runner"),
        ("benchmarks/plot_results.py", "Plotting utilities"),
        ("benchmarks/monitor_jobs.py", "Job monitor"),
        ("benchmarks/quick_start.sh", "Quick start guide"),
    ]
    
    for filepath, desc in scripts:
        all_ok &= check_executable(filepath, desc)
    
    # Check output directories (may not exist yet)
    print("\n" + "=" * 60)
    print("Checking Output Directories")
    print("=" * 60)
    
    directories = [
        ("/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/jobs", "Jobs directory"),
        ("/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/output", "Results directory"),
        ("/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/plots", "Plots directory"),
    ]
    
    for dirpath, desc in directories:
        check_directory(dirpath, desc)
    
    # Check Python dependencies
    all_ok &= check_python_imports()
    
    # Check grid configuration
    all_ok &= check_grid_config()
    
    # Check model configuration
    all_ok &= check_model_config()
    
    # Final summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Framework Ready!")
    else:
        print("⚠ SOME CHECKS FAILED - Review errors above")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Review: benchmarks/README.md")
    print("  2. Configure: benchmarks/grid_config.yaml")
    print("  3. Test: python benchmarks/orchestrator.py --dry-run --max-jobs 3")
    print("  4. Submit: python benchmarks/orchestrator.py --max-jobs 5")
    print("  5. Monitor: python benchmarks/monitor_jobs.py")
    print("  6. Analyze: python benchmarks/plot_results.py --results-dir ... --output-dir ... --plot-type all")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
