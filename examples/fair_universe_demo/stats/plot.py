"""
Original authors: FAIR-Universe HiggsML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""

import inspect
import json
import os
from functools import cached_property, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
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

    def output(self) -> Dict[str, luigi.LocalTarget]:  # type: ignore
        return {
            plot_name: luigi.LocalTarget(Path(os.path.join(self.plot_save_dir, plot_name)).absolute())
            for plot_name in self.registered_plots.keys()
        }

    @staticmethod
    def plot(
        _func: Callable | None = None,
        *,
        name: str = None,
    ) -> Callable[[Callable[..., Figure]], Callable[..., Figure]]:
        """This decorator does two things:
            1. Register the given function as "to-be-plotted" which means it is registered automatically
                as a law Target, no need to explicitly write the output file.
            2. When the function is actually run, the resulting plot is automatically saved to file

        Args:
            _func (Callable | None, optional): The function to register. Defaults to None.
            name (str, optional): Name of the plot. Defaults to None, in which case the name of the
                function is used instead.
        """

        def decorator(func: Callable[..., Figure]) -> Callable[..., Figure]:
            """Register the function as "to-be-plotted"

            Args:
                func (Callable[..., Figure]): The function to register

            Returns:
                Callable[..., Figure]: Registered function
            """
            setattr(func, "_plot_name", name or func.__name__)

            @wraps(func)
            def wrapper(self: luigi.Task, *args, **kwargs) -> Figure:
                """Wraps the call signature of the function so that the plot is automatically saved.

                Raises:
                    TypeError: TypeError: If the function does not return a Figure object

                Returns:
                    Figure: Non-rendered Figure object for debugging
                """
                fig = func(self, *args, **kwargs)

                if not isinstance(fig, Figure):
                    raise TypeError(f"Function {func.__name__} must return matplotlib.figure.Figure")

                save_path = self.output()[getattr(func, "_plot_name")].path
                fig.savefig(save_path)
                plt.close(fig)

                return fig

            return wrapper

        if _func is not None:
            return decorator(_func)

        return decorator

    @property
    def registered_plots(self) -> Dict[str, Callable[..., Figure]]:
        plots = {}

        for name, member in inspect.getmembers(type(self), predicate=callable):
            if hasattr(member, "_plot_name"):
                plot_name = getattr(member, "_plot_name")

                if plot_name in plots:
                    raise ValueError(f"Duplicate plot names: {plot_name} for other function {member}")

                plots[plot_name] = member

        return plots

    @plot(name="ground_truth_vs_predicted_mu")
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
        fig = plt.figure(figsize=(6, 4))

        for test_set_index, ingestion_result in ingestion_result_dict.items():
            mu_hat = np.mean(ingestion_result["mu_hat"])
            mu = ground_truth_mu[int(test_set_index)]
            plt.scatter(mu, mu_hat, c="b", marker="o")

        plt.xlabel("Ground Truth $\\mu$")
        plt.ylabel("Predicted $\\mu$ (averaged over test sets)")
        plt.title("Mu Prediction")
        return fig

    @plot(name="roc_curve")
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
        fig = plt.figure(figsize=(8, 7))

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
        os.makedirs(os.path.abspath(self.plot_save_dir), exist_ok=True)

        self.visualize_scatter(
            ingestion_result_dict=self.ingestion_results,
            ground_truth_mu=self.test_settings["ground_truth_mus"],
        )
