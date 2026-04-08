"""
Original authors: FAIR-Universe HiggsML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""

import json
import os
from typing import Any, Dict, List, Optional

import luigi
import matplotlib.pyplot as plt
import numpy as np

from .eval import PredictResult


class PlottingTask(luigi.Task):
    test_settings_path: str = luigi.Parameter(description="Path to the test settings file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(
        description="Path to the directory containing the FAIR Universe Data",
    )  # type: ignore
    ingestion_results_path: str = luigi.Parameter(
        description="Path to the result file from the 'EvalTask' (aka. Ingestion)",
    )  # type: ignore
    plot_save_dir: str = luigi.Parameter(
        description="Path to the directory where to save the plots resulting from this Task",
    )  # type: ignore

    @property
    def test_settings(self) -> Dict[str, Any]:
        with open(self.test_settings_path, "r") as f:
            _test_settings = json.load(f)

        return _test_settings

    @property
    def ingestion_results(self) -> Dict[int, PredictResult]:
        with open(self.ingestion_results_path, "r") as f:
            _ingestion_results = json.load(f)

        return _ingestion_results

    @staticmethod
    def visualize_scatter(
        ingestion_result_dict: Dict[int, PredictResult],
        ground_truth_mu: Dict[int, List[float]],
        savepath: Optional[str] = None,
    ) -> None:
        """
        Plots a scatter Plot of ground truth vs. predicted mu values.

        Args:
            ingestion_result_dict (dict): A dictionary containing the ingestion results.
            ground_truth_mu (dict): A dictionary of ground truth mu values.
            savepath (str): Where to save the resulting plot. If None (default), show the plot instead.
        """
        plt.figure(figsize=(6, 4))

        for test_set_index, ingestion_result in ingestion_result_dict.items():
            mu_hat = np.mean(ingestion_result["mu_hat"])
            mu = ground_truth_mu[int(test_set_index)]
            plt.scatter(mu, mu_hat, c="b", marker="o")

        plt.xlabel("Ground Truth $\\mu$")
        plt.ylabel("Predicted $\\mu$ (averaged over test sets)")
        plt.title("Ground Truth vs. Predicted $\\mu$ Values")

        if savepath:
            plt.savefig(os.path.join(savepath, "ground_truth_vs_predicted_mu"))
        else:
            plt.show()

    def run(self) -> None:
        os.makedirs(os.path.abspath(self.plot_save_dir), exist_ok=True)

        self.visualize_scatter(
            ingestion_result_dict=self.ingestion_results,
            ground_truth_mu=self.test_settings["ground_truth_mus"],
            savepath=self.plot_save_dir,
        )
