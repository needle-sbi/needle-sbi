#!/usr/bin/env python3
"""
Benchmark Runner for PseudoModel Analysis

This script runs detailed benchmarks on trained models and saves results
in a structured format for analysis and plotting.
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import sys

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import NEEDLE modules (these will work when run from workspace root)
try:
    from ml.lightning.models.pseudo_model import PseudoModel
    from ml.lightning.models.pseudo_model_parallel import PseudoModelParallel
    from ml.lightning.models.pseudo_model_vectorized import PseudoModelVectorized
except ImportError as e:
    print(f"Warning: Could not import PseudoModel modules: {e}")
    print("This is expected if running outside the NEEDLE environment.")
    print("The script will work correctly when run via HTCondor jobs.")
    # Define dummy classes for testing
    class PseudoModel: pass
    class PseudoModelParallel: pass
    class PseudoModelVectorized: pass


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    strategy: str  # 'sequential', 'parallel', 'vectorized'
    total_time: float  # Total execution time (ms)
    breakdown: Dict[str, float]  # Component times (ms)
    param_set: Dict[str, Any]  # Parameter configuration
    
    def to_dict(self):
        return asdict(self)


@contextmanager
def timer():
    """Context manager for precise timing."""
    start = time.perf_counter()
    timings = {}
    
    def record(name: str, t: float):
        timings[name] = t * 1000  # Convert to ms
    
    yield record
    
    end = time.perf_counter()
    timings['total'] = (end - start) * 1000


def profile_sequential(
    model: PseudoModel,
    test_data: torch.Tensor,
    warmup: int = 3,
    n_tests: int = 10
) -> Dict[str, float]:
    """
    Profile sequential execution with detailed breakdown.
    
    Returns:
        Dictionary of timing components (in ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = model(test_data)
    
    timings = []
    
    for _ in range(n_tests):
        with timer() as record:
            # Forward pass
            start = time.perf_counter()
            output = model(test_data)
            end = time.perf_counter()
            
            record('forward_pass', end - start)
        
        timings.append({
            'forward_pass': (end - start) * 1000,
        })
    
    # Calculate statistics
    avg_timings = {
        key: np.mean([t[key] for t in timings])
        for key in timings[0].keys()
    }
    
    return avg_timings


def profile_parallel(
    model: PseudoModelParallel,
    test_data: torch.Tensor,
    warmup: int = 3,
    n_tests: int = 10
) -> Dict[str, float]:
    """
    Profile parallel execution with detailed breakdown.
    
    Returns:
        Dictionary of timing components (in ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = model(test_data)
    
    timings = []
    
    for _ in range(n_tests):
        start_total = time.perf_counter()
        
        # Forward pass
        start_forward = time.perf_counter()
        output = model(test_data)
        end_forward = time.perf_counter()
        
        end_total = time.perf_counter()
        
        timings.append({
            'forward_pass': (end_forward - start_forward) * 1000,
            'total': (end_total - start_total) * 1000,
        })
    
    # Calculate statistics
    avg_timings = {
        key: np.mean([t[key] for t in timings])
        for key in timings[0].keys()
    }
    
    return avg_timings


def profile_vectorized(
    model: PseudoModelVectorized,
    test_data: torch.Tensor,
    warmup: int = 3,
    n_tests: int = 10
) -> Dict[str, float]:
    """
    Profile vectorized execution with detailed breakdown.
    
    Returns:
        Dictionary of timing components (in ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = model(test_data)
    
    timings = []
    
    for _ in range(n_tests):
        start_total = time.perf_counter()
        
        # Forward pass
        start_forward = time.perf_counter()
        output = model(test_data)
        end_forward = time.perf_counter()
        
        end_total = time.perf_counter()
        
        timings.append({
            'forward_pass': (end_forward - start_forward) * 1000,
            'total': (end_total - start_total) * 1000,
        })
    
    # Calculate statistics
    avg_timings = {
        key: np.mean([t[key] for t in timings])
        for key in timings[0].keys()
    }
    
    return avg_timings


def run_benchmarks(
    snapshot_path: str,
    test_data: torch.Tensor,
    warmup: int = 3,
    n_tests: int = 10,
    param_set: Dict[str, Any] = None
) -> List[BenchmarkResult]:
    """
    Run all benchmark strategies on a trained model.
    
    Args:
        snapshot_path: Path to dag_snapshot.json
        test_data: Test input data
        warmup: Number of warmup iterations
        n_tests: Number of test iterations
        param_set: Parameter configuration dictionary
        
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    
    print("Loading models...")
    
    # 1. Sequential
    print("\nBenchmarking Sequential Strategy...")
    try:
        model_seq = PseudoModel(snapshot_path)
        model_seq.eval()
        
        with torch.no_grad():
            timings_seq = profile_sequential(model_seq, test_data, warmup, n_tests)
        
        result_seq = BenchmarkResult(
            strategy='sequential',
            total_time=timings_seq['forward_pass'],
            breakdown=timings_seq,
            param_set=param_set or {}
        )
        results.append(result_seq)
        
        print(f"  Average time: {timings_seq['forward_pass']:.2f} ms")
        
    except Exception as e:
        print(f"  Sequential benchmark failed: {e}")
    
    # 2. Parallel
    print("\nBenchmarking Parallel Strategy...")
    try:
        model_par = PseudoModelParallel(snapshot_path)
        model_par.eval()
        
        with torch.no_grad():
            timings_par = profile_parallel(model_par, test_data, warmup, n_tests)
        
        result_par = BenchmarkResult(
            strategy='parallel',
            total_time=timings_par['forward_pass'],
            breakdown=timings_par,
            param_set=param_set or {}
        )
        results.append(result_par)
        
        print(f"  Average time: {timings_par['forward_pass']:.2f} ms")
        
    except Exception as e:
        print(f"  Parallel benchmark failed: {e}")
    
    # 3. Vectorized
    print("\nBenchmarking Vectorized Strategy...")
    try:
        model_vec = PseudoModelVectorized(snapshot_path)
        model_vec.eval()
        
        with torch.no_grad():
            timings_vec = profile_vectorized(model_vec, test_data, warmup, n_tests)
        
        result_vec = BenchmarkResult(
            strategy='vectorized',
            total_time=timings_vec['forward_pass'],
            breakdown=timings_vec,
            param_set=param_set or {}
        )
        results.append(result_vec)
        
        print(f"  Average time: {timings_vec['forward_pass']:.2f} ms")
        
    except Exception as e:
        print(f"  Vectorized benchmark failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run PseudoModel benchmarks and save results"
    )
    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to dag_snapshot.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--n-tests',
        type=int,
        default=10,
        help='Number of test iterations'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=1000,
        help='Number of samples in test data'
    )
    parser.add_argument(
        '--param-set',
        type=str,
        default=None,
        help='JSON string of parameter set configuration'
    )
    
    args = parser.parse_args()
    
    # Parse param_set
    param_set = None
    if args.param_set:
        param_set = json.loads(args.param_set)
    
    print("=" * 80)
    print("PseudoModel Benchmark Runner")
    print("=" * 80)
    print(f"\nSnapshot: {args.snapshot}")
    print(f"Output: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Test iterations: {args.n_tests}")
    
    if param_set:
        print(f"\nParameter Set:")
        for key, value in param_set.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    
    # Generate test data
    # Note: Shape needs to match model input - adjust as needed
    print("\nGenerating test data...")
    test_data = torch.randn(args.test_size, 1, 1)
    
    # Run benchmarks
    results = run_benchmarks(
        snapshot_path=args.snapshot,
        test_data=test_data,
        warmup=args.warmup,
        n_tests=args.n_tests,
        param_set=param_set
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'snapshot_path': args.snapshot,
            'test_size': args.test_size,
            'warmup_iterations': args.warmup,
            'test_iterations': args.n_tests,
            'param_set': param_set,
        },
        'results': [r.to_dict() for r in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    # Print summary
    print("\nBenchmark Summary:")
    for result in results:
        print(f"\n{result.strategy.upper()}:")
        print(f"  Total time: {result.total_time:.2f} ms")
        print(f"  Breakdown:")
        for component, time_ms in result.breakdown.items():
            print(f"    {component}: {time_ms:.2f} ms")


if __name__ == '__main__':
    main()
