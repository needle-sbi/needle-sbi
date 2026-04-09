"""
Original authors: FAIR-Universe HiggsML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""

import json
import os
from functools import cached_property
from typing import Any, Dict, List, Optional

import luigi
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .eval import PredictResult


class PlottingTask(luigi.Task):
    test_settings_path: str = luigi.Parameter(description="Path to the test settings file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(
        description="Path to the directory containing the FAIR Universe Data",
    )  # type: ignore
    ingestion_results_path: str = luigi.Parameter(
        description="Path to the result file from the 'EvalTask' (aka. Ingestion)",
    )  # type: ignore
    score_path: str = luigi.Parameter(
        description="Path to the score file from the 'ScoreTask'",
    )  # type: ignore
    plot_save_dir: str = luigi.Parameter(
        description="Path to the directory where to save the plots resulting from this Task",
    )  # type: ignore

    @cached_property
    def test_settings(self) -> Dict[str, Any]:
        with open(self.test_settings_path, "r") as f:
            _test_settings = json.load(f)

        return _test_settings

    @cached_property
    def ingestion_results(self) -> Dict[int, PredictResult]:
        with open(self.ingestion_results_path, "r") as f:
            _ingestion_results = json.load(f)

        return _ingestion_results

    @cached_property
    def scores(self):
        with open(self.score_path) as f:
            _scores = json.load(f)

        return _scores

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

    @staticmethod
    def roc_curve_wrapper(
        score: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        savepath: str,
        *,
        plot_label: str = "model",
        color="b",
        lw: int = 2,
    ) -> None:
        """
        Plots the ROC curve.

        Args:
            * score (ndarray): The score.
            * labels (ndarray): The labels.
            * weights (ndarray): The weights.
            * plot_label (str, optional): The plot label. Defaults to "model".
            * color (str, optional): The color. Defaults to "b".
            * lw (int, optional): The line width. Defaults to 2.
        """
        plt.figure(figsize=(8, 7))

        auc = roc_auc_score(y_true=labels, y_score=score, sample_weight=weights)
        fpr, tpr, _ = roc_curve(y_true=labels, y_score=score, sample_weight=weights)

        plt.plot(fpr, tpr, color=color, lw=lw, label=plot_label + " AUC :" + f"{auc:.3f}")
        plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")

        if savepath:
            plt.savefig(os.path.join(savepath, "roc_curve"))
        else:
            plt.show()

    def run(self) -> None:
        os.makedirs(os.path.abspath(self.plot_save_dir), exist_ok=True)

        self.visualize_scatter(
            ingestion_result_dict=self.ingestion_results,
            ground_truth_mu=self.test_settings["ground_truth_mus"],
            savepath=self.plot_save_dir,
        )
