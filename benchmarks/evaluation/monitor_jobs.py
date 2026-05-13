#!/usr/bin/env python3
"""
Job Status Monitor for PseudoModel Benchmark Grid Search

Tracks progress of submitted HTCondor jobs and benchmark completion.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


class JobMonitor:
    """Monitor HTCondor job status and benchmark completion."""

    def __init__(self, jobs_dir: str, results_dir: str):
        """
        Initialize monitor.

        Args:
            jobs_dir: Directory containing job metadata
            results_dir: Directory containing benchmark results
        """
        self.jobs_dir = Path(jobs_dir)
        self.results_dir = Path(results_dir)
        self.metadata_file = self.jobs_dir / "job_metadata.json"

    def load_job_metadata(self) -> List[Dict]:
        """Load job metadata from orchestrator."""
        if not self.metadata_file.exists():
            print(f"Error: Metadata file not found: {self.metadata_file}")
            print("Have you run the orchestrator yet?")
            return []

        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def get_condor_status(self, cluster_id: str) -> str:
        """
        Get HTCondor job status.

        Args:
            cluster_id: HTCondor cluster ID

        Returns:
            Status string: 'running', 'idle', 'held', 'completed', 'unknown'
        """
        if cluster_id == "DRY_RUN":
            return "dry_run"

        try:
            # Check active queue
            result = subprocess.run(
                ["condor_q", cluster_id, "-format", "%s", "JobStatus"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                status_code = result.stdout.strip()
                status_map = {
                    "1": "idle",
                    "2": "running",
                    "3": "removed",
                    "4": "completed",
                    "5": "held",
                    "6": "transferring",
                }
                return status_map.get(status_code, "unknown")

            # Check history if not in queue
            result = subprocess.run(
                ["condor_history", cluster_id, "-limit", "1", "-format", "%s", "JobStatus"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                return "completed"

            return "not_found"

        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception as e:
            return f"error: {e}"

    def check_benchmark_completion(self, job_id: int) -> bool:
        """
        Check if benchmark result exists for a job.

        Args:
            job_id: Job ID number

        Returns:
            True if benchmark result exists
        """
        result_file = self.results_dir / f"benchmark_job_{job_id}.json"
        return result_file.exists()

    def get_job_summary(self) -> Dict:
        """
        Generate summary of all jobs.

        Returns:
            Dictionary with status counts and lists
        """
        metadata = self.load_job_metadata()

        if not metadata:
            return {}

        summary = {
            "total_jobs": len(metadata),
            "condor_status": {
                "idle": 0,
                "running": 0,
                "held": 0,
                "completed": 0,
                "not_found": 0,
                "dry_run": 0,
                "other": 0,
            },
            "benchmark_status": {
                "completed": 0,
                "pending": 0,
            },
            "jobs": [],
        }

        for job in metadata:
            job_id = job["job_id"]
            cluster_id = job["cluster_id"]

            # Check HTCondor status
            condor_status = self.get_condor_status(cluster_id)

            # Check benchmark completion
            benchmark_complete = self.check_benchmark_completion(job_id)

            # Update counts
            if condor_status in summary["condor_status"]:
                summary["condor_status"][condor_status] += 1
            else:
                summary["condor_status"]["other"] += 1

            if benchmark_complete:
                summary["benchmark_status"]["completed"] += 1
            else:
                summary["benchmark_status"]["pending"] += 1

            # Store job info
            job_info = {
                "job_id": job_id,
                "job_name": job["job_name"],
                "cluster_id": cluster_id,
                "condor_status": condor_status,
                "benchmark_complete": benchmark_complete,
                "submission_time": job["submission_time"],
                "param_set": job["param_set"],
            }
            summary["jobs"].append(job_info)

        return summary

    def print_summary(self, summary: Dict):
        """Print formatted job summary."""
        if not summary:
            print("No jobs found!")
            return

        print("=" * 80)
        print("PseudoModel Benchmark Job Status")
        print("=" * 80)
        print()

        # Overall stats
        print(f"Total Jobs: {summary['total_jobs']}")
        print()

        # HTCondor status
        print("HTCondor Status:")
        print("-" * 40)
        for status, count in summary["condor_status"].items():
            if count > 0:
                pct = (count / summary["total_jobs"]) * 100
                print(f"  {status:15s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Benchmark status
        print("Benchmark Completion:")
        print("-" * 40)
        for status, count in summary["benchmark_status"].items():
            pct = (count / summary["total_jobs"]) * 100
            print(f"  {status:15s}: {count:4d} ({pct:5.1f}%)")
        print()

        # Progress bar
        completed = summary["benchmark_status"]["completed"]
        total = summary["total_jobs"]
        pct = (completed / total) * 100 if total > 0 else 0
        bar_width = 50
        filled = int(bar_width * completed / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"Progress: [{bar}] {completed}/{total} ({pct:.1f}%)")
        print()

        print("=" * 80)

    def print_detailed_status(self, summary: Dict, filter_status: str = None):
        """
        Print detailed job-by-job status.

        Args:
            summary: Job summary dictionary
            filter_status: Optional filter by status (e.g., 'running', 'held')
        """
        if not summary or not summary["jobs"]:
            print("No jobs found!")
            return

        jobs = summary["jobs"]

        # Filter if requested
        if filter_status:
            jobs = [j for j in jobs if j["condor_status"] == filter_status]
            if not jobs:
                print(f"No jobs with status: {filter_status}")
                return

        print()
        print("=" * 80)
        print(f"Detailed Job Status{f' (filtered: {filter_status})' if filter_status else ''}")
        print("=" * 80)
        print()

        for job in jobs:
            status_symbol = "✓" if job["benchmark_complete"] else "○"
            print(f"{status_symbol} Job {job['job_id']:04d}: {job['job_name']}")
            print(f"   Cluster ID: {job['cluster_id']}")
            print(f"   HTCondor: {job['condor_status']}")
            print(f"   Benchmark: {'Complete' if job['benchmark_complete'] else 'Pending'}")
            print(f"   Submitted: {job['submission_time']}")
            print(
                f"   Parameters: n_networks={job['param_set']['n_networks']}, "
                f"batch_size={job['param_set']['batch_size']}, "
                f"size={job['param_set']['network_size_name']}"
            )
            print()

    def check_failed_jobs(self, summary: Dict) -> List[Dict]:
        """
        Identify jobs that likely failed.

        Args:
            summary: Job summary dictionary

        Returns:
            List of potentially failed jobs
        """
        failed = []

        for job in summary["jobs"]:
            # Job is failed if HTCondor completed but no benchmark result
            if job["condor_status"] == "completed" and not job["benchmark_complete"]:
                failed.append(job)
            # Or if job is held
            elif job["condor_status"] == "held":
                failed.append(job)

        return failed

    def print_failed_jobs(self, summary: Dict):
        """Print information about failed jobs."""
        failed = self.check_failed_jobs(summary)

        if not failed:
            print("\n✓ No failed jobs detected!")
            return

        print()
        print("=" * 80)
        print(f"Potentially Failed Jobs: {len(failed)}")
        print("=" * 80)
        print()

        for job in failed:
            print(f"✗ Job {job['job_id']:04d}: {job['job_name']}")
            print(f"   Status: {job['condor_status']}")
            print(f"   Cluster: {job['cluster_id']}")
            print(f"   Check logs: {self.jobs_dir / job['job_name'] / 'job.*.err'}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Monitor PseudoModel benchmark job progress")
    parser.add_argument(
        "--jobs-dir",
        type=str,
        default="/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/jobs",
        help="Directory containing job metadata",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/data/dust/user/sjiggins/NEEDLE/NEEDLE/2026-01-29/PseudoModel-Benchmark/output",
        help="Directory containing benchmark results",
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed job-by-job status")
    parser.add_argument(
        "--filter-status",
        type=str,
        choices=["idle", "running", "held", "completed", "not_found"],
        help="Filter detailed view by status",
    )
    parser.add_argument("--check-failed", action="store_true", help="Only show potentially failed jobs")

    args = parser.parse_args()

    # Create monitor
    monitor = JobMonitor(args.jobs_dir, args.results_dir)

    # Get summary
    summary = monitor.get_job_summary()

    # Print requested information
    if args.check_failed:
        monitor.print_failed_jobs(summary)
    elif args.detailed:
        monitor.print_summary(summary)
        monitor.print_detailed_status(summary, args.filter_status)
    else:
        monitor.print_summary(summary)

        # Also show failed jobs by default
        failed = monitor.check_failed_jobs(summary)
        if failed:
            print()
            print(f"⚠ Warning: {len(failed)} potentially failed jobs detected")
            print("   Run with --check-failed for details")


if __name__ == "__main__":
    main()
