#!/usr/bin/env python3
"""
Grid Search Orchestrator for PseudoModel Benchmark Analysis

This script:
1. Generates all parameter combinations from grid_config.yaml
2. Creates modified YAML configs for each combination
3. Generates HTCondor submission files
4. Submits jobs to HTCondor
5. Tracks job status and collects results
"""

import argparse
import itertools
import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import yaml


class GridSearchOrchestrator:
    """Orchestrates grid search for PseudoModel benchmarking."""
    
    def __init__(self, grid_config_path: str, workspace_root: str):
        """
        Initialize the orchestrator.
        
        Args:
            grid_config_path: Path to grid_config.yaml
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = Path(workspace_root)
        self.grid_config_path = Path(grid_config_path)
        
        # Load grid configuration
        with open(self.grid_config_path, 'r') as f:
            self.grid_config = yaml.safe_load(f)
        
        # Setup output directories
        base_dir = Path(self.grid_config['output']['base_dir'])
        self.jobs_dir = base_dir / self.grid_config['output']['jobs_dir']
        self.results_dir = base_dir / self.grid_config['output']['results_dir']
        self.plots_dir = base_dir / self.grid_config['output']['plots_dir']
        
        # Create directories
        for dir_path in [self.jobs_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.job_metadata = []
        
    def calculate_n_params(self, hidden_dim: int, n_hidden: int, num_features: int = 1) -> int:
        """
        Calculate total number of parameters in MockTransformer.
        
        Args:
            hidden_dim: Hidden layer dimension
            n_hidden: Number of hidden layers
            num_features: Number of input features
            
        Returns:
            Total number of parameters
        """
        # Input layer: (num_features * hidden_dim) + hidden_dim
        input_params = (num_features * hidden_dim) + hidden_dim
        
        # Hidden layers: n_hidden * ((hidden_dim * hidden_dim) + hidden_dim)
        hidden_params = n_hidden * ((hidden_dim * hidden_dim) + hidden_dim)
        
        # Output layer: (hidden_dim * 1) + 1
        output_params = hidden_dim + 1
        
        return input_params + hidden_params + output_params
    
    def generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid config.
        
        Returns:
            List of parameter dictionaries
        """
        param_grid = []
        
        # Extract dimensions
        ensembles_list = self.grid_config['n_networks']['ensembles']
        folds_list = self.grid_config['n_networks']['folds']
        batch_sizes = self.grid_config['batch_size']
        network_configs = self.grid_config['network_size']['configs']
        
        # Generate all combinations
        for ensemble, fold, batch_size, net_config in itertools.product(
            ensembles_list, folds_list, batch_sizes, network_configs
        ):
            n_networks = ensemble * fold
            n_params = self.calculate_n_params(
                net_config['hidden_dim'], 
                net_config['n_hidden']
            )
            
            param_set = {
                'n_networks': n_networks,
                'num_ensembles': ensemble,
                'folds': fold,
                'batch_size': batch_size,
                'n_network_params': n_params,
                'network_size_name': net_config['name'],
                'hidden_dim': net_config['hidden_dim'],
                'n_hidden': net_config['n_hidden'],
            }
            param_grid.append(param_set)
        
        return param_grid
    
    def create_config_files(self, param_set: Dict[str, Any], job_dir: Path) -> Dict[str, Path]:
        """
        Create modified YAML config files for a specific parameter set.
        
        Args:
            param_set: Parameter dictionary
            job_dir: Directory to save config files
            
        Returns:
            Dictionary mapping config types to file paths
        """
        config_paths = {}
        
        # 1. Copy and modify main config.yaml
        main_config_src = self.workspace_root / "conf" / "config.yaml"
        main_config_dst = job_dir / "config.yaml"
        
        with open(main_config_src, 'r') as f:
            main_config = yaml.safe_load(f)
        
        # Update ensemble and fold counts
        for estimator_name in main_config['estimators']:
            main_config['estimators'][estimator_name]['expands']['ensembles']['num_ensembles'] = param_set['num_ensembles']
            main_config['estimators'][estimator_name]['expands']['folds'] = param_set['folds']
        
        with open(main_config_dst, 'w') as f:
            yaml.dump(main_config, f, default_flow_style=False)
        
        config_paths['main'] = main_config_dst
        
        # 2. Copy and modify datamodules/padded.yaml (maintain directory structure)
        datamodule_src = self.workspace_root / "conf" / "datamodules" / "padded.yaml"
        datamodule_dir = job_dir / "datamodules"
        datamodule_dir.mkdir(exist_ok=True)
        datamodule_dst = datamodule_dir / "padded.yaml"
        
        with open(datamodule_src, 'r') as f:
            datamodule_config = yaml.safe_load(f)
        
        datamodule_config['batch_size'] = param_set['batch_size']
        
        with open(datamodule_dst, 'w') as f:
            yaml.dump(datamodule_config, f, default_flow_style=False)
        
        config_paths['datamodule'] = datamodule_dst
        
        # 3. Copy and modify models/mock_transformer.yaml (maintain directory structure)
        model_src = self.workspace_root / "conf" / "models" / "mock_transformer.yaml"
        model_dir = job_dir / "models"
        model_dir.mkdir(exist_ok=True)
        model_dst = model_dir / "mock_transformer.yaml"
        
        with open(model_src, 'r') as f:
            model_config = yaml.safe_load(f)
        
        model_config['hidden_dim'] = param_set['hidden_dim']
        model_config['n_hidden'] = param_set['n_hidden']
        
        with open(model_dst, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        
        config_paths['model'] = model_dst
        
        # 4. Copy dataset config unchanged (maintain directory structure)
        dataset_src = self.workspace_root / "conf" / "datasets" / "fair_universe.yaml"
        dataset_dir = job_dir / "datasets"
        dataset_dir.mkdir(exist_ok=True)
        dataset_dst = dataset_dir / "fair_universe.yaml"
        shutil.copy(dataset_src, dataset_dst)
        config_paths['dataset'] = dataset_dst
        
        # 5. Copy trainer config unchanged (maintain directory structure)
        trainer_src = self.workspace_root / "conf" / "trainers" / "default.yaml"
        trainer_dir = job_dir / "trainers"
        trainer_dir.mkdir(exist_ok=True)
        trainer_dst = trainer_dir / "default.yaml"
        shutil.copy(trainer_src, trainer_dst)
        config_paths['trainer'] = trainer_dst
        
        return config_paths
    
    def create_htcondor_submission(
        self, 
        param_set: Dict[str, Any], 
        job_dir: Path,
        job_id: int
    ) -> Tuple[Path, Path]:
        """
        Create HTCondor submission files for a job.
        
        Args:
            param_set: Parameter dictionary
            job_dir: Directory for this job
            job_id: Unique job identifier
            
        Returns:
            Tuple of (submission_file_path, run_script_path)
        """
        # Create run script
        run_script_path = job_dir / "Run.sh"
        
        run_script_content = f"""#!/bin/bash
# HTCondor run script for PseudoModel benchmark
# Job ID: {job_id}
# Parameter set: {param_set['network_size_name']}_e{param_set['num_ensembles']}_f{param_set['folds']}_b{param_set['batch_size']}

set -e  # Exit on error

echo "=== Job Start ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Job ID: {job_id}"
echo "Working directory: $(pwd)"
echo ""

# Setup environment
export NEEDLE_WORKSPACE={self.workspace_root}
export PYTHONUNBUFFERED=1  # Force unbuffered output for real-time logs
export FAIR_UNIVERSE_DATA=/data/dust/group/atlas/needle/FAIRUnv/UncertaintyChallenge_2024/ProcessedData_v1_2025-10-03/CombData-part0.parquet
cd $NEEDLE_WORKSPACE

# Source setup script if it exists
if [ -f setup.sh ]; then
    source setup.sh
fi

echo "=== Training Phase ==="
echo "[$(date +%T)] Starting LAW..."
START_TIME=$(date +%s)

# Create job-specific output directory
JOB_OUTPUT_DIR="runs/{param_set['network_size_name']}_e{param_set['num_ensembles']}_f{param_set['folds']}_b{param_set['batch_size']}"
mkdir -p "$JOB_OUTPUT_DIR"

# Run LAW training task with custom config and job-specific output
law run MainTask \\
    --config-file {job_dir / 'config.yaml'} \\
    --SnapshotTask-rel-results-path "$JOB_OUTPUT_DIR" \\
    --workers 4 \\
    --log-level INFO

END_TIME=$(date +%s)
TRAIN_DURATION=$((END_TIME - START_TIME))
echo "[$(date +%T)] LAW completed in ${{TRAIN_DURATION}}s"

echo ""
echo "=== Training Complete ==="

# Wait for outputs to settle
sleep 5

# Find the snapshot in job-specific output directory
SNAPSHOT_JSON="$JOB_OUTPUT_DIR/dag_snapshot.json"
if [ ! -f "$SNAPSHOT_JSON" ]; then
    echo "ERROR: Snapshot JSON not found: $SNAPSHOT_JSON"
    exit 1
fi

echo "Using snapshot: $SNAPSHOT_JSON"

echo ""
echo "=== Benchmark Phase ==="
# Run benchmark script
python benchmarks/benchmark_runner.py \\
    --snapshot "$SNAPSHOT_JSON" \\
    --output {self.results_dir / f'benchmark_job_{job_id}.json'} \\
    --n-tests {self.grid_config['benchmark']['n_tests']} \\
    --warmup {self.grid_config['benchmark']['warmup_iterations']} \\
    --test-size {self.grid_config['benchmark']['test_data_size']} \\
    --param-set '{json.dumps(param_set)}'

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"
"""
        
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        
        # Make executable
        run_script_path.chmod(0o755)
        
        # Create .sub file
        sub_file_path = job_dir / "job.sub"
        
        sub_content = f"""# HTCondor submission file
# Job ID: {job_id}
# Generated: {datetime.now().isoformat()}

executable              = {run_script_path}
arguments               = $(ClusterId)$(ProcId)
output                  = {job_dir}/job.$(ClusterId).$(ProcId).out
error                   = {job_dir}/job.$(ClusterId).$(ProcId).err
log                     = {job_dir}/job.$(ClusterId).log

# Resource requests
request_runtime         = 14000
request_memory          = 16000
request_GPUs             = 0
request_CPUs             = 2

# Environment
getenv                  = True
universe                = vanilla

# Queue single job
queue 1
"""
        
        with open(sub_file_path, 'w') as f:
            f.write(sub_content)
        
        return sub_file_path, run_script_path
    
    def submit_job(self, sub_file_path: Path, dry_run: bool = False) -> str:
        """
        Submit a job to HTCondor.
        
        Args:
            sub_file_path: Path to .sub file
            dry_run: If True, don't actually submit
            
        Returns:
            Cluster ID from HTCondor (or "DRY_RUN")
        """
        if dry_run:
            print(f"[DRY RUN] Would submit: {sub_file_path}")
            return "DRY_RUN"
        
        result = subprocess.run(
            ['condor_submit', str(sub_file_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"condor_submit failed: {result.stderr}")
        
        # Parse cluster ID from output
        # Output format: "1 job(s) submitted to cluster 12345."
        for line in result.stdout.split('\n'):
            if 'submitted to cluster' in line:
                cluster_id = line.split()[-1].rstrip('.')
                return cluster_id
        
        raise RuntimeError(f"Could not parse cluster ID from: {result.stdout}")
    
    def run_grid_search(self, dry_run: bool = False, max_jobs: int = None):
        """
        Execute the full grid search.
        
        Args:
            dry_run: If True, generate files but don't submit jobs
            max_jobs: Maximum number of jobs to submit (for testing)
        """
        print("=" * 80)
        print("PseudoModel Benchmark Grid Search")
        print("=" * 80)
        
        # Generate parameter grid
        param_grid = self.generate_parameter_grid()
        total_jobs = len(param_grid)
        
        if max_jobs is not None:
            param_grid = param_grid[:max_jobs]
        
        print(f"\nTotal parameter combinations: {total_jobs}")
        print(f"Jobs to submit: {len(param_grid)}")
        print(f"Dry run: {dry_run}")
        print(f"\nOutput directories:")
        print(f"  Jobs: {self.jobs_dir}")
        print(f"  Results: {self.results_dir}")
        print(f"  Plots: {self.plots_dir}")
        print("\n" + "=" * 80)
        
        # Submit jobs
        submitted_jobs = []
        
        for job_id, param_set in enumerate(param_grid):
            job_name = (
                f"job_{job_id:04d}_"
                f"{param_set['network_size_name']}_"
                f"e{param_set['num_ensembles']}_"
                f"f{param_set['folds']}_"
                f"b{param_set['batch_size']}"
            )
            
            print(f"\n[{job_id + 1}/{len(param_grid)}] Preparing {job_name}")
            print(f"  n_networks={param_set['n_networks']}, "
                  f"batch_size={param_set['batch_size']}, "
                  f"n_params={param_set['n_network_params']:,}")
            
            # Create job directory
            job_dir = self.jobs_dir / job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Save parameter set
            param_file = job_dir / "param_set.json"
            with open(param_file, 'w') as f:
                json.dump(param_set, f, indent=2)
            
            # Create config files
            config_paths = self.create_config_files(param_set, job_dir)
            print(f"  ✓ Created config files")
            
            # Create submission files
            sub_file, run_script = self.create_htcondor_submission(
                param_set, job_dir, job_id
            )
            print(f"  ✓ Created submission files")
            
            # Submit job
            try:
                cluster_id = self.submit_job(sub_file, dry_run=dry_run)
                print(f"  ✓ Submitted to cluster: {cluster_id}")
                
                # Track job metadata
                job_metadata = {
                    'job_id': job_id,
                    'job_name': job_name,
                    'cluster_id': cluster_id,
                    'param_set': param_set,
                    'job_dir': str(job_dir),
                    'submission_time': datetime.now().isoformat(),
                }
                self.job_metadata.append(job_metadata)
                submitted_jobs.append(job_name)
                
            except Exception as e:
                print(f"  ✗ Submission failed: {e}")
        
        # Save job metadata
        metadata_file = self.jobs_dir / "job_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.job_metadata, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"Grid search setup complete!")
        print(f"Submitted {len(submitted_jobs)} jobs")
        print(f"Job metadata saved to: {metadata_file}")
        print("=" * 80)
        
        return submitted_jobs
    
    def run_grid_search_with_law_remote(self, dry_run: bool = False, max_jobs: int = None):
        """
        Execute grid search using LAW's remote HTCondor workflow.
        
        This runs LAW locally, which submits FoldTasks directly to HTCondor.
        Much cleaner than nested job submission!
        
        Args:
            dry_run: If True, generate files but don't run LAW
            max_jobs: Maximum number of jobs to submit (for testing)
        """
        print("=" * 80)
        print("PseudoModel Benchmark Grid Search (LAW Remote Execution)")
        print("=" * 80)
        
        # Generate parameter grid
        param_grid = self.generate_parameter_grid()
        total_jobs = len(param_grid)
        
        if max_jobs is not None:
            param_grid = param_grid[:max_jobs]
        
        print(f"\nTotal parameter combinations: {total_jobs}")
        print(f"Jobs to run: {len(param_grid)}")
        print(f"Dry run: {dry_run}")
        print(f"\nExecution mode: LAW HTCondor Workflow")
        print(f"  - {param_grid[0]['num_ensembles'] * param_grid[0]['folds']} FoldTasks per job")
        print(f"  - Total FoldTasks: {len(param_grid) * param_grid[0]['num_ensembles'] * param_grid[0]['folds']}")
        print(f"\nOutput directories:")
        print(f"  Jobs: {self.jobs_dir}")
        print(f"  Results: {self.results_dir}")
        print(f"  Plots: {self.plots_dir}")
        print("\n" + "=" * 80)
        
        successful_runs = []
        
        for job_id, param_set in enumerate(param_grid):
            job_name = (
                f"job_{job_id:04d}_"
                f"{param_set['network_size_name']}_"
                f"e{param_set['num_ensembles']}_"
                f"f{param_set['folds']}_"
                f"b{param_set['batch_size']}"
            )
            
            print(f"\n[{job_id + 1}/{len(param_grid)}] Setting up {job_name}")
            print(f"  n_networks={param_set['n_networks']}, "
                  f"batch_size={param_set['batch_size']}, "
                  f"n_params={param_set['n_network_params']:,}")
            
            # Create job directory
            job_dir = self.jobs_dir / job_name
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Save parameter set
            param_file = job_dir / "param_set.json"
            with open(param_file, 'w') as f:
                json.dump(param_set, f, indent=2)
            
            # Create config files
            config_paths = self.create_config_files(param_set, job_dir)
            print(f"  ✓ Created config files")
            
            # Setup output directory
            output_dir = f"runs/{param_set['network_size_name']}_e{param_set['num_ensembles']}_f{param_set['folds']}_b{param_set['batch_size']}"
            
            # Build LAW command
            law_cmd = [
                "law", "run", "MainTask",
                "--config-file", str(config_paths['main']),
                "--SnapshotTask-rel-results-path", output_dir,
                "--workflow", "htcondor",  # Enable remote execution
                "--FoldTask-htcondor-request-cpus", "2",
                "--FoldTask-htcondor-request-memory", "32GB",
                "--FoldTask-max-runtime", "120",
                "--poll-interval", "30",
                "--parallel-jobs", "50",  # Submit up to 50 FoldTasks at once
                "--log-level", "INFO",
            ]
            
            print(f"  LAW command: {' '.join(law_cmd)}")
            
            if not dry_run:
                try:
                    # Run LAW (it will submit FoldTasks to HTCondor)
                    print(f"  🚀 Running LAW workflow (this submits FoldTasks to HTCondor)...")
                    result = subprocess.run(
                        law_cmd,
                        cwd=self.workspace_root,
                        capture_output=True,
                        text=True,
                        timeout=3600,  # 1 hour timeout for workflow setup
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✓ LAW workflow completed successfully")
                        
                        # Run benchmark
                        snapshot_path = self.workspace_root / output_dir / "dag_snapshot.json"
                        if snapshot_path.exists():
                            print(f"  🔬 Running benchmark on snapshot...")
                            self._run_benchmark(snapshot_path, param_set, job_id)
                        else:
                            print(f"  ⚠ Warning: Snapshot not found at {snapshot_path}")
                        
                        successful_runs.append(job_name)
                    else:
                        print(f"  ✗ LAW workflow failed!")
                        print(f"  Error: {result.stderr[:500]}")
                        
                except subprocess.TimeoutExpired:
                    print(f"  ✗ LAW workflow timed out after 1 hour")
                except Exception as e:
                    print(f"  ✗ Error running LAW: {e}")
            else:
                print(f"  [DRY RUN] Would run: {' '.join(law_cmd)}")
                successful_runs.append(job_name)
            
            # Track metadata
            job_metadata = {
                'job_id': job_id,
                'job_name': job_name,
                'param_set': param_set,
                'job_dir': str(job_dir),
                'output_dir': output_dir,
                'law_command': ' '.join(law_cmd),
                'execution_time': datetime.now().isoformat(),
            }
            self.job_metadata.append(job_metadata)
        
        # Save metadata
        metadata_file = self.jobs_dir / 'grid_search_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.job_metadata, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"Grid search complete!")
        print(f"Successful runs: {len(successful_runs)}/{len(param_grid)}")
        print(f"Metadata saved to: {metadata_file}")
        print("=" * 80)
        
        return successful_runs
    
    def _run_benchmark(self, snapshot_path: Path, param_set: Dict, job_id: int):
        """Helper to run benchmark on completed snapshot"""
        benchmark_cmd = [
            "python", "benchmarks/benchmark_runner.py",
            "--snapshot", str(snapshot_path),
            "--output", str(self.results_dir / f"benchmark_job_{job_id}.json"),
            "--n-tests", str(self.grid_config['benchmark']['n_tests']),
            "--warmup", str(self.grid_config['benchmark']['warmup_iterations']),
            "--test-size", str(self.grid_config['benchmark']['test_data_size']),
            "--param-set", json.dumps(param_set),
        ]
        
        try:
            result = subprocess.run(
                benchmark_cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"  ✓ Benchmark completed")
            else:
                print(f"  ✗ Benchmark failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ✗ Benchmark error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate grid search for PseudoModel benchmarking"
    )
    parser.add_argument(
        '--grid-config',
        type=str,
        default='benchmarks/grid_config.yaml',
        help='Path to grid configuration file'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        default='.',
        help='Workspace root directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate files but do not submit jobs'
    )
    parser.add_argument(
        '--max-jobs',
        type=int,
        default=None,
        help='Maximum number of jobs to submit (for testing)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['wrapper', 'law-remote'],
        default='law-remote',
        help='Execution mode: wrapper=HTCondor jobs with local DAG, law-remote=LAW submits FoldTasks to HTCondor (recommended)'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = GridSearchOrchestrator(
        grid_config_path=args.grid_config,
        workspace_root=args.workspace
    )
    
    # Run grid search with selected mode
    if args.mode == 'law-remote':
        print("Using LAW remote execution mode (recommended)")
        print("  - Runs LAW locally on submission node")
        print("  - LAW submits FoldTasks directly to HTCondor")
        print("  - No nested job submission\n")
        orchestrator.run_grid_search_with_law_remote(
            dry_run=args.dry_run,
            max_jobs=args.max_jobs
        )
    else:
        print("Using wrapper script mode")
        print("  - Submits wrapper HTCondor jobs")
        print("  - Each job runs entire DAG locally\n")
        orchestrator.run_grid_search(
            dry_run=args.dry_run,
            max_jobs=args.max_jobs
        )


if __name__ == '__main__':
    main()
