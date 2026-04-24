import os
from typing import Dict, List

import luigi
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.selection import createJetData
from .histogram import HistogramTask
from .plotting_mixin import PlottingMixin


class ValidationTask(PlottingMixin):
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(
        description="Path to the directory containing the FAIR Universe Data"
    )  # type: ignore
    model_name: str = luigi.Parameter(
        description="Name of the model to validate. Has to match the names in the snapshot.json file"
    )  # type: ignore

    @PlottingMixin.plot(name="log_prob_distribution")
    def plot_log_prob_distributions(
        self,
        signal_logprobs: np.ndarray,
        bg_logprobs: np.ndarray,
    ) -> plt.Figure:
        """Compare signal vs background log-probability distributions.

        Args:
            signal_logprobs: Shape (n_sig,)
            bg_logprobs: Shape (n_bg,)
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram comparison
        axes[0].hist(signal_logprobs, bins=50, alpha=0.6, label="Signal", density=True)
        axes[0].hist(bg_logprobs, bins=50, alpha=0.6, label="Background", density=True)
        axes[0].set_xlabel("Log-Probability")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Quantile comparison
        q_sig = np.quantile(signal_logprobs, np.linspace(0, 1, 100))
        q_bg = np.quantile(bg_logprobs, np.linspace(0, 1, 100))
        axes[1].plot(q_sig, label="Signal", linewidth=2)
        axes[1].plot(q_bg, label="Background", linewidth=2)
        axes[1].set_xlabel("Quantile Index")
        axes[1].set_ylabel("Log-Probability Value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="log_prob_statistics")
    def plot_log_prob_statistics(
        self,
        signal_logprobs: np.ndarray,
        bg_logprobs: np.ndarray,
    ) -> plt.Figure:
        """Box plot and summary statistics for log-probabilities.

        Args:
            signal_logprobs: Shape (n_sig,)
            bg_logprobs: Shape (n_bg,)
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plot
        data_to_plot = [signal_logprobs, bg_logprobs]
        bp = axes[0].boxplot(data_to_plot, labels=["Signal", "Background"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["lightblue", "lightcoral"]):
            patch.set_facecolor(color)
        axes[0].set_ylabel("Log-Probability")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Statistics table
        stats = {
            "Signal": {
                "Mean": np.mean(signal_logprobs),
                "Std": np.std(signal_logprobs),
                "Median": np.median(signal_logprobs),
                "Min": np.min(signal_logprobs),
                "Max": np.max(signal_logprobs),
            },
            "Background": {
                "Mean": np.mean(bg_logprobs),
                "Std": np.std(bg_logprobs),
                "Median": np.median(bg_logprobs),
                "Min": np.min(bg_logprobs),
                "Max": np.max(bg_logprobs),
            },
        }

        axes[1].axis("off")
        table_data = [["Metric", "Signal", "Background"]]
        for metric in ["Mean", "Std", "Median", "Min", "Max"]:
            table_data.append(
                [
                    metric,
                    f"{stats['Signal'][metric]:.4f}",
                    f"{stats['Background'][metric]:.4f}",
                ]
            )

        table = axes[1].table(
            table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.35, 0.35],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="log_prob_vs_feature")
    def plot_log_prob_vs_feature(
        self,
        signal_data: torch.Tensor,
        signal_logprobs: np.ndarray,
        bg_data: torch.Tensor,
        bg_logprobs: np.ndarray,
        num_features: int = 4,
    ) -> plt.Figure:
        """Scatter plots of log-prob vs individual features.

        Args:
            signal_data: Shape (n_sig, n_features)
            signal_logprobs: Shape (n_sig,)
            bg_data: Shape (n_bg, n_features)
            bg_logprobs: Shape (n_bg,)
            num_features: Number of features to plot

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        sig_data_np = signal_data.cpu().numpy()
        bg_data_np = bg_data.cpu().numpy()

        for i in range(min(num_features, 4)):
            axes[i].scatter(sig_data_np[:, i], signal_logprobs, alpha=0.3, s=5, label="Signal")
            axes[i].scatter(bg_data_np[:, i], bg_logprobs, alpha=0.3, s=5, label="Background")
            axes[i].set_xlabel(f"Feature {i}")
            axes[i].set_ylabel("Log-Probability")
            axes[i].set_title(f"Log-Prob vs Feature {i}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="training_curves")
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training Curves",
    ) -> plt.Figure:
        """Plot training and validation loss curves.

        Args:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = np.arange(len(train_losses))
        ax.plot(epochs, train_losses, label="Training Loss", linewidth=2)
        ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="calibration_curve")
    def plot_calibration_curve(
        self,
        signal_logprobs: np.ndarray,
        bg_logprobs: np.ndarray,
        num_bins: int = 20,
    ) -> plt.Figure:
        """Plot ROC-like calibration: signal efficiency vs background rejection.

        Args:
            signal_logprobs: Shape (n_sig,)
            bg_logprobs: Shape (n_bg,)
            num_bins: Number of threshold bins
            title: Plot title

        Returns:
            matplotlib Figure
        """
        thresholds = np.linspace(
            min(signal_logprobs.min(), bg_logprobs.min()),
            max(signal_logprobs.max(), bg_logprobs.max()),
            num_bins,
        )

        sig_eff = []  # Fraction of signal above threshold (true positive rate)
        bg_acc = []  # Fraction of background above threshold (false positive rate)

        for thresh in thresholds:
            sig_eff.append(np.mean(signal_logprobs > thresh))
            bg_acc.append(np.mean(bg_logprobs > thresh))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            bg_acc,
            sig_eff,
            linewidth=2,
            marker="o",
            markersize=5,
            label="Contrastive Normalizing Flow",
        )
        ax.set_xlabel("Background Acceptance (false positive rate)")
        ax.set_ylabel("Signal Efficiency (true positive rate)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*[0, 1])
        ax.set_ylim(*[0, 1])

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random classifier")
        ax.legend()

        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="feature_distribution")
    def plot_feature_distributions(
        self,
        signal_data: torch.Tensor,
        bg_data: torch.Tensor,
        num_features: int = 4,
    ) -> plt.Figure:
        """Compare feature distributions between signal and background.

        Args:
            signal_data: Shape (n_sig, n_features)
            bg_data: Shape (n_bg, n_features)
            num_features: Number of features to plot

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        sig_data_np = signal_data.cpu().numpy()
        bg_data_np = bg_data.cpu().numpy()

        for i in range(min(num_features, 4)):
            axes[i].hist(sig_data_np[:, i], bins=50, alpha=0.6, label="Signal", density=True)
            axes[i].hist(bg_data_np[:, i], bins=50, alpha=0.6, label="Background", density=True)
            axes[i].set_xlabel(f"Feature {i}")
            axes[i].set_ylabel("Density")
            axes[i].set_title(f"Feature {i} Distribution")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @property
    def nf_ckpts(self) -> Dict[str, str]:
        return HistogramTask.parse_snapshot(self.snapshot_path)[0]

    @property
    def nf_models(self) -> torch.nn.ModuleDict:
        return ClassifierDatamodule.load_nf_models(self.nf_ckpts)

    @property
    def num_jets(self) -> int:
        name_parts = self.model_name.split("&")[0]  # "nf_signal_1jet" or "nf_background_2jet"
        if "1jet" in name_parts:
            _num_jets = 1
        elif "2jet" in name_parts:
            _num_jets = 2
        else:
            raise ValueError(f"Could not extract jet count from model name: {self.model_name}")

        return _num_jets

    @property
    def plot_save_dir_override(self) -> str:
        return os.path.join(self.plot_save_dir, self.model_name)

    def run(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nf_model = self.nf_models[self.model_name].to(device).eval()

        data, labels, _, _ = createJetData(
            jet_num=self.num_jets,
            useTestData=False,
            seed=78,
            root_dir=self.root_dir,
        )
        signal_mask = labels == 1
        bg_mask = labels == 0

        signal_data = data[signal_mask]
        bg_data = data[bg_mask]

        with torch.no_grad():
            signal_logprobs = nf_model(signal_data.to(device)).cpu().numpy()
            bg_logprobs = nf_model(bg_data.to(device)).cpu().numpy()

        self.plot_log_prob_distributions(
            signal_logprobs,
            bg_logprobs,
        )

        self.plot_log_prob_statistics(
            signal_logprobs,
            bg_logprobs,
        )

        self.plot_log_prob_vs_feature(
            signal_data,
            signal_logprobs,
            bg_data,
            bg_logprobs,
            num_features=4,
        )

        self.plot_calibration_curve(
            signal_logprobs,
            bg_logprobs,
        )

        self.plot_feature_distributions(
            signal_data,
            bg_data,
            num_features=4,
        )
