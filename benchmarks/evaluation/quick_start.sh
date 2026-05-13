#!/bin/bash
# Quick Start Guide for PseudoModel Benchmark Framework
#
# This script provides example commands to get started quickly

set -e

WORKSPACE_ROOT="/home/sjiggins/Documents/Work_Directory_DESY/NEEDLE/NEEDLE-software/NEEDLE/2026-01-29/orchestrator"
OUTPUT_BASE="/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark"

echo "=================================="
echo "PseudoModel Benchmark Quick Start"
echo "=================================="
echo ""

# Check if in correct directory
if [ ! -f "benchmarks/orchestrator.py" ]; then
    echo "ERROR: Please run this script from the orchestrator directory:"
    echo "  cd $WORKSPACE_ROOT"
    exit 1
fi

echo "Step 1: Configure Grid Search"
echo "-----------------------------"
echo "Edit benchmarks/grid_config.yaml to define your parameter grid."
echo "Current configuration:"
cat benchmarks/grid_config.yaml | grep -A 2 "ensembles:\|batch_size:\|configs:" | head -20
echo ""
read -p "Press Enter to continue or Ctrl+C to edit config first..."
echo ""

echo "Step 2: Dry Run (Generate Files Only)"
echo "--------------------------------------"
echo "Running dry run to generate job files without submitting..."
python benchmarks/orchestrator.py \
    --grid-config benchmarks/grid_config.yaml \
    --workspace . \
    --dry-run \
    --max-jobs 3
echo ""
read -p "Press Enter to continue..."
echo ""

echo "Step 3: Test Submission (5 jobs)"
echo "--------------------------------"
read -p "Submit 5 test jobs to HTCondor? (y/n): " answer
if [ "$answer" = "y" ]; then
    python benchmarks/orchestrator.py \
        --grid-config benchmarks/grid_config.yaml \
        --workspace . \
        --max-jobs 5
    echo ""
    echo "Jobs submitted! Check status with:"
    echo "  condor_q"
    echo ""
else
    echo "Skipped test submission"
fi
echo ""

echo "Step 4: Monitor Jobs"
echo "-------------------"
echo "Use these commands to monitor job progress:"
echo ""
echo "  # Check all your jobs"
echo "  condor_q"
echo ""
echo "  # Watch jobs update every 5 seconds"
echo "  watch -n 5 condor_q"
echo ""
echo "  # Check specific job details"
echo "  condor_q -analyze <cluster_id>"
echo ""
echo "  # Check completed jobs"
echo "  condor_history -limit 10"
echo ""
read -p "Press Enter to continue..."
echo ""

echo "Step 5: Check Results"
echo "--------------------"
echo "Results are saved to:"
echo "  $OUTPUT_BASE/output/benchmark_job_*.json"
echo ""
echo "Check how many results are ready:"
echo "  ls $OUTPUT_BASE/output/benchmark_job_*.json 2>/dev/null | wc -l"
echo ""
NUM_RESULTS=$(ls $OUTPUT_BASE/output/benchmark_job_*.json 2>/dev/null | wc -l || echo "0")
echo "Current results: $NUM_RESULTS"
echo ""
read -p "Press Enter to continue..."
echo ""

if [ "$NUM_RESULTS" -gt 0 ]; then
    echo "Step 6: Generate Plots"
    echo "---------------------"
    read -p "Generate plots from available results? (y/n): " answer
    if [ "$answer" = "y" ]; then
        python benchmarks/plot_results.py \
            --results-dir $OUTPUT_BASE/output \
            --output-dir $OUTPUT_BASE/plots \
            --plot-type all
        echo ""
        echo "Plots saved to: $OUTPUT_BASE/plots"
        echo ""
        echo "View summary statistics:"
        if [ -f "$OUTPUT_BASE/plots/summary_statistics.csv" ]; then
            cat $OUTPUT_BASE/plots/summary_statistics.csv
        fi
    else
        echo "Skipped plotting"
    fi
else
    echo "Step 6: Generate Plots"
    echo "---------------------"
    echo "No results available yet. Wait for jobs to complete."
    echo ""
    echo "When results are ready, run:"
    echo "  python benchmarks/plot_results.py \\"
    echo "      --results-dir $OUTPUT_BASE/output \\"
    echo "      --output-dir $OUTPUT_BASE/plots \\"
    echo "      --plot-type all"
fi
echo ""

echo "Step 7: Full Grid Search (Optional)"
echo "-----------------------------------"
echo "When ready to run the complete grid search, execute:"
echo ""
echo "  python benchmarks/orchestrator.py \\"
echo "      --grid-config benchmarks/grid_config.yaml \\"
echo "      --workspace ."
echo ""
echo "This will submit ALL parameter combinations."
echo "Check grid_config.yaml for total job count."
echo ""

echo "=================================="
echo "Quick Start Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Monitor jobs: condor_q"
echo "  2. Check logs: $OUTPUT_BASE/jobs/job_*/job.*.err"
echo "  3. Analyze results: python benchmarks/plot_results.py"
echo ""
echo "For detailed documentation, see: benchmarks/README.md"
