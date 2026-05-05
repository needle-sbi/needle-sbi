# PseudoModel Benchmark Analysis Framework

A comprehensive framework for analyzing PseudoModel performance across three dimensions: network count, batch size, and network architecture size.

## Overview

This framework performs a systematic grid search to understand how PseudoModel inference performance varies with:

1. **Network Count** (`n_networks`): Number of models in the DAG (ensembles × folds)
2. **Batch Size** (`batch_size`): Input batch size for inference
3. **Network Size** (`n_network_params`): Total parameters per model

Three execution strategies are benchmarked:
- **Sequential**: Models executed one at a time
- **Parallel**: Level-wise parallel execution with thread pool/CUDA streams
- **Vectorized**: Batch execution using `torch.vmap`

## Directory Structure

```
benchmarks/
├── grid_config.yaml          # Grid search parameter configuration
├── orchestrator.py           # HTCondor job submission orchestrator
├── benchmark_runner.py       # Per-job benchmark execution script
├── plot_results.py          # Visualization and analysis tools
└── README.md                # This file

Output directories (configured in grid_config.yaml):
/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/
├── jobs/                    # Job configurations and submission files
│   ├── job_0000_small_e2_f2_b32/
│   │   ├── param_set.json   # Parameter configuration
│   │   ├── config.yaml      # Main LAW config
│   │   ├── padded.yaml      # Datamodule config
│   │   ├── mock_transformer.yaml  # Model config
│   │   ├── job.sub          # HTCondor submission file
│   │   ├── Run.sh           # Execution script
│   │   └── job.*.{out,err,log}  # HTCondor logs
│   └── job_metadata.json    # All job metadata
├── output/                  # Benchmark results
│   └── benchmark_job_*.json # Individual benchmark results
└── plots/                   # Generated visualizations
    ├── sequential_3d.png
    ├── sequential_2d_projections.png
    ├── sequential_1d_slices.png
    ├── parallel_*.png
    ├── vectorized_*.png
    ├── strategy_comparison.png
    └── summary_statistics.csv
```

## Prerequisites

### Software Requirements
- Python 3.8+
- PyTorch
- LAW (Luigi Analysis Workflow)
- HTCondor access
- Standard scientific Python stack (numpy, matplotlib, pandas, seaborn)

### Model Updates
The framework requires MockTransformer to support configurable architecture:

```yaml
# conf/models/mock_transformer.yaml
_target_: ml.lightning.models.mock_transformer.MockTransformerModule
factor: 0.1
patience: 10
init_lr: 0.001
hidden_dim: 512    # Now configurable!
n_hidden: 30       # Now configurable!
```

## Usage

### 1. Configure Grid Search

Edit `benchmarks/grid_config.yaml` to define your parameter grid:

```yaml
# Dimension 1: Network count
n_networks:
  ensembles: [2, 5, 10]
  folds: [2, 5, 10]

# Dimension 2: Batch size
batch_size: [32, 64, 128, 256, 512, 1024]

# Dimension 3: Network size
network_size:
  configs:
    - name: "small"
      hidden_dim: 64
      n_hidden: 5
    - name: "medium"
      hidden_dim: 128
      n_hidden: 10
    - name: "large"
      hidden_dim: 256
      n_hidden: 15
    - name: "xlarge"
      hidden_dim: 512
      n_hidden: 30

# Benchmark settings
benchmark:
  n_tests: 10          # Repetitions per parameter point
  warmup_iterations: 3  # Warmup before timing
  test_data_size: 1000  # Samples for inference
```

**Total jobs**: `len(ensembles) × len(folds) × len(batch_sizes) × len(network_sizes)`
- Example: 3 × 3 × 6 × 4 = **216 jobs**

### 2. Generate and Submit Jobs

#### Dry Run (recommended first)
```bash
cd /home/sjiggins/Documents/Work_Directory_DESY/NEEDLE/NEEDLE-software/NEEDLE/2026-01-29/orchestrator

python benchmarks/orchestrator.py \
    --grid-config benchmarks/grid_config.yaml \
    --workspace . \
    --dry-run
```

This will:
- Generate all job directories
- Create config files for each parameter combination
- Create HTCondor submission files
- **NOT** actually submit jobs

#### Test Run (submit first 5 jobs)
```bash
python benchmarks/orchestrator.py \
    --grid-config benchmarks/grid_config.yaml \
    --workspace . \
    --max-jobs 5
```

#### Full Submission
```bash
python benchmarks/orchestrator.py \
    --grid-config benchmarks/grid_config.yaml \
    --workspace .
```

### 3. Monitor Jobs

```bash
# Check job status
condor_q

# Check specific user's jobs
condor_q sjiggins

# Watch job progress
watch -n 5 condor_q

# Check job history
condor_history -limit 10

# Analyze why a job failed
condor_q -analyze <cluster_id>
```

### 4. Collect and Analyze Results

#### Generate All Visualizations
```bash
python benchmarks/plot_results.py \
    --results-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/output \
    --output-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/plots \
    --plot-type all
```

This creates:
- **3D scatter plots**: One per strategy showing all three dimensions
- **2D projections**: Three plots per strategy (n_networks vs batch_size, etc.)
- **1D slices**: Mean ± std along each dimension
- **Strategy comparison**: Side-by-side performance across all strategies
- **Summary statistics**: CSV with mean, median, std, min, max per strategy

#### Generate Specific Plots

```bash
# Just 3D plot for sequential strategy
python benchmarks/plot_results.py \
    --results-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/output \
    --output-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/plots \
    --plot-type 3d \
    --strategy sequential

# Just summary statistics
python benchmarks/plot_results.py \
    --results-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/output \
    --output-dir /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/plots \
    --plot-type stats
```

## Workflow Details

### Job Execution Flow

Each submitted job performs the following steps:

1. **Environment Setup**
   - Source `setup.sh` if available
   - Set environment variables

2. **Training Phase**
   - Execute `law run MainTask --config-file <custom_config.yaml>`
   - Trains models with specified parameters (ensembles, folds, network size)
   - Generates DAG snapshot in `runs/snapshot_*/dag_snapshot.json`

3. **Benchmark Phase**
   - Load snapshot with three strategies: Sequential, Parallel, Vectorized
   - Run warmup iterations (default: 3)
   - Execute timed tests (default: 10 repetitions)
   - Measure execution time with high-precision timers
   - Save results to JSON: `benchmark_job_<id>.json`

4. **Output Artifacts**
   - Trained model checkpoints
   - DAG snapshot JSON
   - Benchmark results JSON
   - HTCondor logs (.out, .err, .log)

### Data Format

#### Benchmark Results JSON
```json
{
  "metadata": {
    "snapshot_path": "runs/snapshot_xyz/dag_snapshot.json",
    "test_size": 1000,
    "warmup_iterations": 3,
    "test_iterations": 10,
    "param_set": {
      "n_networks": 10,
      "num_ensembles": 5,
      "folds": 2,
      "batch_size": 128,
      "n_network_params": 263168,
      "network_size_name": "medium",
      "hidden_dim": 128,
      "n_hidden": 10
    }
  },
  "results": [
    {
      "strategy": "sequential",
      "total_time": 4.52,
      "breakdown": {
        "forward_pass": 4.52
      },
      "param_set": { ... }
    },
    {
      "strategy": "parallel",
      "total_time": 6.23,
      "breakdown": {
        "forward_pass": 6.23,
        "total": 6.23
      },
      "param_set": { ... }
    },
    {
      "strategy": "vectorized",
      "total_time": 18.45,
      "breakdown": {
        "forward_pass": 18.45,
        "total": 18.45
      },
      "param_set": { ... }
    }
  ]
}
```

## Expected Results

Based on initial testing with 5 models on CPU:

| Strategy | Typical Time | Best Use Case |
|----------|-------------|---------------|
| Sequential | 0.9 - 5 ms | < 10 models, CPU, small models |
| Parallel | 2 - 10 ms | GPU, large models, 20+ models |
| Vectorized | 5 - 20 ms | GPU, inference servers, very large batches |

**Key Insights:**
- On CPU with small models: Sequential wins (lowest overhead)
- On GPU with many models: Parallel/Vectorized can be 2-5× faster
- Batch size has logarithmic impact on performance
- Network size scales linearly with execution time

## Troubleshooting

### Jobs Not Starting
```bash
# Check submission file syntax
condor_submit -dry-run <job.sub>

# Check job requirements
condor_q -analyze <cluster_id>

# Check held jobs
condor_q -hold
```

### Jobs Failing During Training
```bash
# Check output logs
tail -n 50 /data/dust/user/sjiggins/.../jobs/job_XXXX/job.*.err

# Check LAW task status
law run MainTask --print-status 3

# Verify config files
cat /data/dust/user/sjiggins/.../jobs/job_XXXX/config.yaml
```

### Missing Benchmark Results
```bash
# Find jobs that completed training but didn't benchmark
cd /data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark
ls jobs/*/param_set.json | wc -l  # Total jobs
ls output/benchmark_job_*.json | wc -l  # Completed benchmarks

# Re-run benchmark manually for a specific snapshot
python benchmarks/benchmark_runner.py \
    --snapshot runs/snapshot_xyz/dag_snapshot.json \
    --output output/benchmark_job_XXXX.json \
    --n-tests 10 \
    --warmup 3 \
    --test-size 1000 \
    --param-set "$(cat jobs/job_XXXX/param_set.json)"
```

### Plotting Errors
```bash
# Check data loading
python -c "
from benchmarks.plot_results import BenchmarkAnalyzer
analyzer = BenchmarkAnalyzer('/data/dust/.../output')
analyzer.load_results()
"

# Generate plots one at a time
python benchmarks/plot_results.py \
    --results-dir /data/dust/.../output \
    --output-dir /data/dust/.../plots \
    --plot-type stats  # Start with simple stats
```

## Performance Optimization Tips

### For Large Grid Searches
1. **Use job priorities**: Add to .sub file
   ```
   priority = 10
   ```

2. **Batch submissions**: Use `--max-jobs` to submit in batches
   ```bash
   for i in {0..200..20}; do
       python benchmarks/orchestrator.py --max-jobs 20 ...
       sleep 60
   done
   ```

3. **Parallel training**: Update trainer config for multi-GPU if available

### For Analysis
1. **Filter results**: Only analyze specific parameter ranges
   ```python
   # In plot_results.py, modify load_results()
   self.data = self.data[self.data['n_networks'] >= 10]
   ```

2. **Downsample plots**: Use fewer points for quick visualization
   ```python
   # Sample every 10th point
   data_subset = data_subset.iloc[::10]
   ```

## Advanced Usage

### Custom Network Architectures

Add new configurations to `grid_config.yaml`:

```yaml
network_size:
  configs:
    - name: "custom_wide"
      hidden_dim: 1024
      n_hidden: 5
    - name: "custom_deep"
      hidden_dim: 128
      n_hidden: 50
```

### Custom Aggregation Methods

Edit `conf/config.yaml` to test different aggregation strategies:

```yaml
aggregation:
  fold_method: "weighted_mean"
  fold_weights: [0.1, 0.2, 0.3, 0.2, 0.2]  # Must sum to 1
```

### GPU vs CPU Comparison

Modify .sub file to request GPUs:

```
request_GPUs = 1
request_GPUs_Capability = "7.0"  # Minimum CUDA capability
```

## References

- **LAW Documentation**: [luigi-law.readthedocs.io](https://luigi-law.readthedocs.io)
- **HTCondor Manual**: [htcondor.readthedocs.io](https://htcondor.readthedocs.io)
- **PseudoModel Strategies**: `ml/lightning/models/PARALLELIZATION_STRATEGIES.md`

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{needle_benchmark_framework,
  title={PseudoModel Benchmark Analysis Framework},
  author={Jiggins, Stephen},
  year={2026},
  organization={DESY}
}
```

## Contact

For questions or issues:
- Stephen Jiggins: sjiggins@desy.de
- NEEDLE Project: [github.com/NEEDLE-software](https://github.com/NEEDLE-software)
