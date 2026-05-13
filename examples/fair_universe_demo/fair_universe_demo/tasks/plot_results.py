"""
Original authors: FAIR-Universe HiggsML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""
import json
import os
from functools import cached_property
from typing import Any, Dict, List

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from .eval import PredictResult
from .plotting_mixin import PlottingMixin


class PlottingTask(PlottingMixin):
    cached_test_settings_file: str = luigi.Parameter(
        description="Path to the cache test_settings.json file. Should be in the results directory",
    )  # type: ignore
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
        with open(self.cached_test_settings_file, "r") as f:
            _test_settings: Dict[str, Any] = json.load(f)

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

    @PlottingMixin.plot(name="ground_truth_vs_predicted_mu")
    def visualize_scatter(
        self,
        ingestion_result_dict: Dict[int, PredictResult],
        ground_truth_mu: Dict[int, List[float]],
    ) -> Figure:
        """
        Plots a scatter Plot of ground truth vs. predicted mu values.

        Args:
            ingestion_result_dict (dict): A dictionary containing the ingestion results.
            ground_truth_mu (dict): A dictionary of ground truth mu values.
            savepath (str): Where to save the resulting plot. If None (default), show the plot instead.
        """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        xlims = (0, 3)

        for test_set_index, ingestion_result in ingestion_result_dict.items():
            mu_hat = np.mean(ingestion_result["mu_hat"])
            mu = ground_truth_mu[int(test_set_index)]
            plt.scatter(mu, mu_hat, c="k", marker="+")

        x = np.linspace(*xlims)
        ax.plot(x, x, linestyle="--", label=r"$y=x$")

        ax.legend(loc="upper left", title=r"$H \to \tau \tau$ (MC) averaged")

        ax.set_xlim(*xlims)
        ax.set_ylim(-4, 4)

        ax.set_xlabel(r"$\mu_{\text{true}}$", loc="right")
        ax.set_ylabel(r"$\mu_{\text{predicted}}$)", loc="top")
        return fig

    @PlottingMixin.plot(name="ground_truth_vs_predicted_mu_errorbars")
    def visualize_errorbars(
        self,
        ingestion_result_dict: Dict[int, PredictResult],
        ground_truth_mu: Dict[int, List[float]],
    ) -> Figure:
        """
        Plots a scatter plot of ground truth vs. predicted mu values with error bars in y direction.

        Args:
            ingestion_result_dict (dict): A dictionary containing the ingestion results.
            ground_truth_mu (dict): A dictionary of ground truth mu values.
        """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        xlims = (0, 3)

        for test_set_index, ingestion_result in ingestion_result_dict.items():
            mu_hat_values = ingestion_result["mu_hat"]
            mu_hat = np.mean(mu_hat_values)
            mu_hat_err = np.std(mu_hat_values)
            mu = ground_truth_mu[int(test_set_index)]
            ax.errorbar(mu, mu_hat, yerr=mu_hat_err, fmt="k+", capsize=5)

        x = np.linspace(*xlims)
        ax.plot(x, x, linestyle="--", label=r"$y=x$")

        ax.legend(loc="upper left", title=r"$H \to \tau \tau$ (MC) averaged")

        ax.set_xlim(*xlims)
        ax.set_ylim(*xlims)

        ax.set_xlabel(r"$\mu_{\text{true}}$", loc="right")
        ax.set_ylabel(r"$\mu_{\text{predicted}}$", loc="top")
        return fig

    def roc_curve_wrapper(
        self,
        score: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        *,
        plot_label: str = "model",
        color="b",
        lw: int = 2,
    ) -> Figure:
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
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

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
        return fig

    def run(self) -> None:
        self.visualize_scatter(
            ingestion_result_dict=self.ingestion_results,
            ground_truth_mu=self.test_settings["ground_truth_mus"],
        )
        self.visualize_errorbars(
            ingestion_result_dict=self.ingestion_results,
            ground_truth_mu=self.test_settings["ground_truth_mus"],
        )

        self.upload_plots_to_webpage(os.getenv("NEEDLE_WEB_DIR"))
