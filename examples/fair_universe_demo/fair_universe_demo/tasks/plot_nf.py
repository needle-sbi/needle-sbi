import os
from logging import Logger
from typing import Dict

import luigi
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.dataset_sharing import fetch_dataset
from ..utils.selection import createJetData
from .histogram import HistogramTask
from .plotting_mixin import PlottingMixin

logger = Logger("validation")


class NormalizingFlowValidationTask(PlottingMixin):
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(
        description="Path to the directory containing the FAIR Universe Data"
    )  # type: ignore
    model_name: str = luigi.Parameter(
        description="Name of the model to validate. Has to match the names in the snapshot.json file"
    )  # type: ignore

    default_fig_size = (5, 4)

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
        fig, axes = plt.subplots(1, 2, figsize=(2 * self.default_fig_size[0], self.default_fig_size[1]))

        # Histogram comparison
        bins = np.linspace(-40, 20, 60 + 1)
        axes[0].hist(signal_logprobs, bins=bins, label="Signal", histtype="step")
        axes[0].hist(bg_logprobs, bins=bins, label="Background", histtype="step")
        axes[0].set_xlabel("Log-Probability")
        axes[0].set_ylabel(f"Density / [{(bins[1] - bins[0]):.0f}]")
        axes[0].set_yscale("log")
        axes[0].legend(loc="upper left")

        # Quantile comparison
        bins = np.linspace(0, 1, 100)
        q_sig = np.quantile(signal_logprobs, bins)
        q_bg = np.quantile(bg_logprobs, bins)
        axes[1].plot(q_sig, label="Signal", linewidth=2)
        axes[1].plot(q_bg, label="Background", linewidth=2)
        axes[1].set_xlabel("Quantile Index")
        axes[1].set_ylabel("Log-Probability")
        axes[1].set_yscale("symlog")
        axes[1].legend(loc="upper left")
        axes[1].set_xlim(0, 100)

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

    @PlottingMixin.plot(name="log_prob_vs_feature", add_needle_plot_style=False)
    def plot_log_prob_vs_feature(
        self,
        signal_data: np.ndarray,
        signal_logprobs: np.ndarray,
        bg_data: np.ndarray,
        bg_logprobs: np.ndarray,
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
        fig, axes = plt.subplots(4, 2, figsize=(10, 16))

        y_min, y_max = (-100, 20)

        sig_mask = (signal_logprobs >= y_min) & (signal_logprobs <= y_max)
        bg_mask = (bg_logprobs >= y_min) & (bg_logprobs <= y_max)

        if sig_mask.any() and bg_mask.any():
            sig_data_filtered = signal_data[sig_mask]
            sig_logprobs_filtered = signal_logprobs[sig_mask]
            bg_data_filtered = bg_data[bg_mask]
            bg_logprobs_filtered = bg_logprobs[bg_mask]
        else:
            sig_data_filtered = signal_data
            sig_logprobs_filtered = signal_logprobs
            bg_data_filtered = bg_data
            bg_logprobs_filtered = bg_logprobs
            y_min = min(sig_logprobs_filtered.min(), bg_logprobs_filtered.min())
            y_max = max(sig_logprobs_filtered.max(), bg_logprobs_filtered.max())

        vmin = 1  # Avoid log(0)
        vmax_sig = 0
        vmax_bg = 0

        for i in range(4):
            h_sig, _, _ = np.histogram2d(
                sig_data_filtered[:, i],
                sig_logprobs_filtered,
                bins=50,
                range=[[sig_data_filtered[:, i].min(), sig_data_filtered[:, i].max()], [y_min, y_max]],
            )
            h_bg, _, _ = np.histogram2d(
                bg_data_filtered[:, i],
                bg_logprobs_filtered,
                bins=50,
                range=[[bg_data_filtered[:, i].min(), bg_data_filtered[:, i].max()], [y_min, y_max]],
            )
            vmax_sig = max(vmax_sig, h_sig.max())
            vmax_bg = max(vmax_bg, h_bg.max())

        for i in range(4):
            # Signal heatmap (left)
            signal_axis = axes[i, 0]
            x_min_sig, x_max_sig = sig_data_filtered[:, i].min(), sig_data_filtered[:, i].max()
            h_sig, xedges, yedges = np.histogram2d(
                sig_data_filtered[:, i], sig_logprobs_filtered, bins=50, range=[[x_min_sig, x_max_sig], [y_min, y_max]]
            )
            im_sig = signal_axis.imshow(
                h_sig.T,
                origin="lower",
                aspect="auto",
                cmap="Blues",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                norm=LogNorm(vmin=vmin, vmax=vmax_sig),
                label="Signal",
            )
            signal_axis.set_xlabel(f"Feature {i}", fontsize=10)
            signal_axis.set_ylabel("Log-Probability", fontsize=10)
            signal_axis.set_ylim(y_min, y_max)
            cbar_sig = plt.colorbar(im_sig, ax=signal_axis)
            cbar_sig.set_label("Count (log scale)", fontsize=9)
            signal_axis.legend(loc="lower right")

            # Background heatmap (right)
            background_axis = axes[i, 1]
            x_min_bg, x_max_bg = bg_data_filtered[:, i].min(), bg_data_filtered[:, i].max()
            h_bg, xedges, yedges = np.histogram2d(
                bg_data_filtered[:, i], bg_logprobs_filtered, bins=50, range=[[x_min_bg, x_max_bg], [y_min, y_max]]
            )
            im_bg = background_axis.imshow(
                h_bg.T,
                origin="lower",
                aspect="auto",
                cmap="Reds",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                norm=LogNorm(vmin=vmin, vmax=vmax_bg),
                label="Background",
            )
            background_axis.set_xlabel(f"Feature {i}", fontsize=10)
            background_axis.set_ylabel("Log-Probability", fontsize=10)
            background_axis.set_ylim(y_min, y_max)
            cbar_bg = plt.colorbar(im_bg, ax=background_axis)
            cbar_bg.set_label("Count (log scale)", fontsize=9)
            background_axis.legend(loc="lower right")

        fig = PlottingMixin.set_needle_plot_style(fig, axes=axes.flatten().tolist())
        return fig

    @PlottingMixin.plot(name="calibration_curve")
    def plot_calibration_curve(
        self,
        signal_logprobs: np.ndarray,
        bg_logprobs: np.ndarray,
        num_bins: int = 100,
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
            -50,
            15,
            num_bins,
        )

        sig_eff = []  # Fraction of signal above threshold (true positive rate)
        bg_acc = []  # Fraction of background above threshold (false positive rate)

        for thresh in thresholds:
            sig_eff.append(np.mean(signal_logprobs > thresh))
            bg_acc.append(np.mean(bg_logprobs > thresh))

        fig, ax = plt.subplots(figsize=self.default_fig_size)
        ax.plot(
            bg_acc,
            sig_eff,
            linewidth=2,
            marker="o",
            markersize=5,
            label="CNF",
        )
        ax.set_xlabel("Background Acceptance (false positive rate)")
        ax.set_ylabel("Signal Efficiency (true positive rate)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*[0, 1])
        ax.set_ylim(*[0, 1])

        ax.text(
            0.5,
            0.20,
            f"Bins: {num_bins}\nThreshold: ({thresholds.min()}, {thresholds.max()})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

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
        fig, axes = plt.subplots(2, 2, figsize=(2 * self.default_fig_size[0], 2 * self.default_fig_size[1]))
        axes = axes.flatten()

        sig_data_np = signal_data.cpu().numpy()
        bg_data_np = bg_data.cpu().numpy()

        for i in range(min(num_features, 4)):
            axes[i].hist(
                [sig_data_np[:, i], bg_data_np[:, i]], bins=50, label=["Signal", "Background"], histtype="step"
            )
            axes[i].set_xlabel(f"Feature {i}")
            axes[i].set_ylabel("Counts")
            axes[i].legend(loc="lower right")

        plt.tight_layout()
        return fig

    @property
    def nf_ckpts(self) -> Dict[str, str]:
        return HistogramTask.parse_snapshot(self.snapshot_path)[0]

    @property
    def classifier(self) -> CombinedClassifier:
        ckpt = HistogramTask.parse_snapshot(self.snapshot_path)[1]
        return CombinedClassifier.load_from_checkpoint(ckpt["classifier"])

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
            loaded_data=fetch_dataset(self.root_dir),
            set_mu=1,
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
            signal_data.cpu().numpy(),
            signal_logprobs,
            bg_data.cpu().numpy(),
            bg_logprobs,
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

        self.upload_plots_to_webpage(os.getenv("NEEDLE_WEB_DIR"))
