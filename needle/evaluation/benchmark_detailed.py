"""
Detailed profiling benchmark showing overhead breakdown for each strategy.
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from needle.evaluation.pseudo_model import PseudoModel
from needle.evaluation.pseudo_model_parallel import PseudoModelParallel
from needle.evaluation.pseudo_model_vectorized import PseudoModelVectorized


@contextmanager
def timer(name: str, timings: Dict[str, float]):
    """Context manager to time code blocks."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    timings[name] = elapsed


def profile_sequential(snapshot_path: str, x: torch.Tensor, device: str) -> Dict[str, float]:
    """Profile Sequential implementation with detailed breakdown."""
    timings = {}

    # Model initialization
    with timer("init", timings):
        model = PseudoModel(snapshot_path, device=device)

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Profile forward pass components
    with timer("total_forward", timings):
        # Execute all leaf models
        with timer("leaf_models", timings):
            outputs_cache = {}
            for node_id, m in model.models.items():
                outputs_cache[node_id] = m(x)

        # Aggregation
        with timer("aggregation", timings):
            for edge in model.execution_order:
                source_outputs = [outputs_cache[node] for node in edge.source_nodes]
                aggregated, variance = model._aggregate(source_outputs, edge)
                outputs_cache[edge.target_node] = aggregated

    # Calculate per-model time
    timings["per_model"] = timings["leaf_models"] / len(model.models)
    timings["num_models"] = len(model.models)

    return timings


def profile_parallel(snapshot_path: str, x: torch.Tensor, device: str) -> Dict[str, float]:
    """Profile Parallel implementation with detailed breakdown."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    timings = {}

    # Model initialization
    with timer("init", timings):
        model = PseudoModelParallel(snapshot_path, device=device, num_workers=4)

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Profile forward pass components
    with timer("total_forward", timings):
        outputs_cache = {}

        # Thread pool setup + execution
        with timer("parallel_execution", timings):
            if device.startswith("cuda"):
                # GPU path
                outputs_cache = model._parallel_folds_gpu(x)
            else:
                # CPU path - measure thread pool overhead
                with timer("thread_pool_setup", timings):
                    executor = ThreadPoolExecutor(max_workers=model.num_workers)

                with timer("thread_execution", timings):

                    def run_model(node_id: str, m, x):
                        return node_id, m(x)

                    futures = {
                        executor.submit(run_model, node_id, m, x.clone()): node_id
                        for node_id, m in model.models.items()
                    }

                    for future in as_completed(futures):
                        node_id, output = future.result()
                        outputs_cache[node_id] = output

                with timer("thread_cleanup", timings):
                    executor.shutdown(wait=True)

        # Aggregation
        with timer("aggregation", timings):
            for level_edges in model.execution_levels:
                level_results = model._execute_level_parallel(level_edges, outputs_cache)
                outputs_cache.update(level_results)

    timings["num_models"] = len(model.models)
    timings["per_model"] = (
        timings.get("thread_execution", 0) / len(model.models) if "thread_execution" in timings else 0
    )

    return timings


def profile_vectorized(snapshot_path: str, x: torch.Tensor, device: str) -> Dict[str, float]:
    """Profile Vectorized implementation with detailed breakdown."""
    timings = {}

    # Model initialization (includes parameter stacking)
    with timer("init", timings):
        model = PseudoModelVectorized(snapshot_path, device=device)

    # Break down init time
    temp_model = PseudoModel(snapshot_path, device=device)

    with timer("parameter_stacking", timings):
        # Simulate parameter stacking overhead
        from needle.evaluation.pseudo_model_vectorized import VectorizedEnsemble

        for arch_name, node_ids in model.model_groups.items():
            models = [model.models[nid] for nid in node_ids]
            if len(models) > 1:
                _ = VectorizedEnsemble(models)

    del temp_model

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Profile forward pass components
    with timer("total_forward", timings):
        outputs_cache = {}

        # Vectorized execution
        with timer("vectorized_execution", timings):
            for arch_name, vectorized_ensemble in model.vectorized_ensembles.items():
                node_ids = model.model_groups[arch_name]

                with timer("vmap_call", timings):
                    batched_outputs = vectorized_ensemble(x)

                # Unpack
                for i, node_id in enumerate(node_ids):
                    outputs_cache[node_id] = batched_outputs[i]

        # Aggregation
        with timer("aggregation", timings):
            for level_edges in model.execution_levels:
                for edge in level_edges:
                    source_outputs = [outputs_cache[node] for node in edge.source_nodes]
                    aggregated, variance = model._aggregate_vectorized(source_outputs, edge)
                    outputs_cache[edge.target_node] = aggregated

    timings["num_models"] = len(model.models)

    return timings


def print_breakdown(name: str, timings: Dict[str, float], total_time: float):
    """Print timing breakdown in tree format."""
    print(f"\n{name}: {total_time:.2f}ms total")

    if name == "Sequential":
        print(f"├─ Model loading: {timings.get('init', 0):.2f}ms")
        num_models = int(timings.get("num_models", 1))
        per_model = timings.get("per_model", 0)
        print(f"├─ {num_models}× Forward passes: ~{per_model:.2f}ms each = {timings.get('leaf_models', 0):.2f}ms")
        print(f"└─ Aggregation: {timings.get('aggregation', 0):.2f}ms")

    elif name == "Parallel":
        print(f"├─ Model loading: {timings.get('init', 0):.2f}ms")
        if "thread_pool_setup" in timings:
            print(f"├─ Thread pool setup: {timings['thread_pool_setup']:.2f}ms OVERHEAD")
        num_models = int(timings.get("num_models", 1))
        per_model = timings.get("per_model", 0)
        exec_time = timings.get("thread_execution", timings.get("parallel_execution", 0))
        print(f"├─ {num_models}× Forward passes (parallel): ~{per_model:.2f}ms each = {exec_time:.2f}ms")
        if "thread_cleanup" in timings:
            print(f"├─ Thread synchronization: {timings['thread_cleanup']:.2f}ms OVERHEAD")
        print(f"└─ Aggregation: {timings.get('aggregation', 0):.2f}ms")

    elif name == "Vectorized":
        print(f"├─ Model loading: {timings.get('init', 0):.2f}ms")
        print(f"├─ Parameter stacking: {timings.get('parameter_stacking', 0):.2f}ms OVERHEAD")
        print(f"├─ vmap + functional_call: {timings.get('vmap_call', 0):.2f}ms OVERHEAD")
        exec_time = timings.get("vectorized_execution", 0)
        print(f"├─ Batched forward: {exec_time:.2f}ms")
        print(f"└─ Aggregation: {timings.get('aggregation', 0):.2f}ms")


def main():
    snapshot_path = "/home/sjiggins/Documents/Work_Directory_DESY/NEEDLE/NEEDLE-software/NEEDLE/2026-01-29/orchestrator/runs/dag_snapshot.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"DETAILED PERFORMANCE BREAKDOWN")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Snapshot: {Path(snapshot_path).name}")

    # Test input
    batch_size = 1000
    num_features = 1
    x = torch.randn(batch_size, num_features, 1, device=device)

    results = {}

    # 1. Sequential
    print("\n" + "=" * 70)
    print("[1/3] Profiling Sequential Implementation...")
    print("=" * 70)
    timings_seq = profile_sequential(snapshot_path, x, device)
    results["Sequential"] = timings_seq
    print_breakdown("Sequential", timings_seq, timings_seq["total_forward"])

    # 2. Parallel
    print("\n" + "=" * 70)
    print("[2/3] Profiling Parallel Implementation...")
    print("=" * 70)
    timings_par = profile_parallel(snapshot_path, x, device)
    results["Parallel"] = timings_par
    print_breakdown("Parallel", timings_par, timings_par["total_forward"])

    # 3. Vectorized
    print("\n" + "=" * 70)
    print("[3/3] Profiling Vectorized Implementation...")
    print("=" * 70)
    try:
        timings_vec = profile_vectorized(snapshot_path, x, device)
        results["Vectorized"] = timings_vec
        print_breakdown("Vectorized", timings_vec, timings_vec["total_forward"])
    except Exception as e:
        print(f" Vectorized profiling failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("OVERHEAD ANALYSIS")
    print("=" * 70)

    for name, timings in results.items():
        total = timings["total_forward"]
        if name == "Sequential":
            compute = timings.get("leaf_models", 0) + timings.get("aggregation", 0)
            overhead = total - compute
        elif name == "Parallel":
            compute = timings.get("thread_execution", 0) + timings.get("aggregation", 0)
            overhead = total - compute
        elif name == "Vectorized":
            compute = timings.get("aggregation", 0)
            overhead = total - compute
        else:
            compute = total
            overhead = 0

        overhead_pct = (overhead / total * 100) if total > 0 else 0

        print(f"\n{name}:")
        print(f"  Total time:    {total:6.2f}ms")
        print(f"  Computation:   {compute:6.2f}ms ({100-overhead_pct:5.1f}%)")
        print(f"  Overhead:      {overhead:6.2f}ms ({overhead_pct:5.1f}%)")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    fastest = min(results.items(), key=lambda x: x[1]["total_forward"])
    print(f"\n Fastest strategy: {fastest[0]} ({fastest[1]['total_forward']:.2f}ms)")

    if device == "cpu" and timings_seq["num_models"] < 10:
        print(f"\n Analysis:")
        print(f"   Your setup has {int(timings_seq['num_models'])} models on CPU.")
        print(f"   Sequential is optimal because:")
        print(f"   • Models are small ({timings_seq['per_model']:.2f}ms each)")
        print(f"   • Thread/vmap overhead > computation time")
        print(f"   • No benefit from CPU parallelization")
        print(f"\n   Parallel/Vectorized would help if:")
        print(f"   • Running on GPU (CUDA streams efficient)")
        print(f"   • Models were larger (50+ layers)")
        print(f"   • Ensemble was larger (20+ models)")


if __name__ == "__main__":
    main()
