"""Plot metrics for Classifier model
Disclaimer: Generated with the help of Claude 4.5
"""

from logging import Logger

import luigi
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.dataset_sharing import fetch_dataset
from ..utils.selection import createJetData, return1j2j
from .histogram import HistogramTask
from .plotting_mixin import PlottingMixin

logger = Logger("validation")


class ClassifierValidationTask(PlottingMixin):
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(
        description="Path to the directory containing the FAIR Universe Data"
    )  # type: ignore

    default_fig_size = (5, 4)
    num_jets: int = luigi.Parameter(description="Number of jets to select from the dataset. Either 1 or 2")  # type: ignore

    @property
    def plot_save_dir_override(self):
        return f"{self.plot_save_dir}/num_jets_{self.num_jets}"

    def prepare(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(self.snapshot_path)
        self.nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts)
        self.classifier = CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])
        self.classifier = self.classifier.to(self.device).eval().to(torch.float32)

        alljet_data, _ = createJetData(
            jet_num="all",
            useTestData=True,
            set_mu=1,
            seed=0,
            useRand=False,
            loaded_data=fetch_dataset(self.root_dir),
        )
        (
            self.logits_2j,
            self.logits_1j,
            label_2j,
            label_1j,
        ) = return1j2j(
            alljet_data=alljet_data,
            models=self.nf_models,
            device=self.device,
        )
        self.label_1j = label_1j.cpu().numpy()
        self.label_2j = label_2j.cpu().numpy()

        with torch.no_grad():
            self.scores_2j = torch.sigmoid(self.classifier(self.logits_2j, 2)).cpu().numpy()
            self.scores_1j = torch.sigmoid(self.classifier(self.logits_1j, 1)).cpu().numpy()

    def get_labels(self, num_jets: int) -> np.ndarray:
        return {
            1: self.label_1j,
            2: self.label_2j,
        }[num_jets]

    def get_scores(self, num_jets: int) -> np.ndarray:
        return {
            1: self.scores_1j,
            2: self.scores_2j,
        }[num_jets]

    @PlottingMixin.plot(name="roc_curve")
    def plot_roc_curve(self, num_jets: int) -> Figure:
        labels = self.get_labels(num_jets)
        probs = self.get_scores(num_jets)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
        fig = self.text(f"Number of jets: {num_jets}", fig, ax)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig

    @PlottingMixin.plot(name="recall")
    def plot_recall(self, num_jets: int) -> Figure:
        fig, ax = plt.subplots(figsize=(5, 4))
        precision, recall, _ = precision_recall_curve(
            self.get_labels(num_jets),
            self.get_scores(num_jets),
        )
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.3f})")
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig = self.text(f"Number of jets: {num_jets}", fig, ax)
        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="confusion_matrix", add_needle_plot_style=False)
    def plot_confusion_matrix(
        self,
        num_jets: int,
        threshold: float = 0.5,
    ) -> Figure:
        predictions = (self.get_scores(num_jets) > threshold).astype(int)
        confusion = confusion_matrix(self.get_labels(num_jets), predictions)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(confusion, cmap="Blues", aspect="auto")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    confusion[i, j],
                    ha="center",
                    va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                    fontsize=14,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(f"Number of jets: {num_jets}")
        ax.set_xticklabels(["Predicted Background", "Predicted Signal"])
        ax.set_yticklabels(["True Background", "True Signal"])
        fig.colorbar(im, ax=ax)
        fig = self.set_needle_plot_style(fig, ax)
        fig.tight_layout()
        return fig

    @PlottingMixin.plot(name="score_distribution")
    def plot_score_distribution(
        self,
        num_jets: int,
        bins: int = 30,
    ) -> Figure:
        scores = self.get_scores(num_jets)
        labels = self.get_labels(num_jets)
        signal_scores = scores[labels == 1]
        bg_scores = scores[labels == 0]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(signal_scores, bins=bins, label="Signal")
        ax.hist(bg_scores, bins=bins, label="Background")
        ax.set_xlabel("Model Score", fontsize=11)
        ax.set_ylabel("Normalized Count", fontsize=11)
        ax.legend()
        fig.tight_layout()
        return fig

    def run(self):
        self.prepare()
        logger.info(f"Size of dataset for {self.num_jets=}: {len(self.get_labels(self.num_jets))}")
        self.plot_roc_curve(self.num_jets)
        self.plot_recall(self.num_jets)
        self.plot_confusion_matrix(self.num_jets)
        self.plot_score_distribution(self.num_jets)
