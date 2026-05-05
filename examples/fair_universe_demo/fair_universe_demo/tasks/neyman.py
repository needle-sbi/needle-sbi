"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List

import luigi
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from ml.utils.epoch_timer import timing
from tqdm import tqdm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.dataset_sharing import fetch_dataset
from ..utils.selection import createJetData, return1j2j
from ..utils.stats import (
    compute_signal_fraction,
    fit_2D_splines_bin_by_bin_from_dict,
    string_to_tuple_str,
)
from .histogram import HistogramTask
from .plot_results import PlottingMixin

try:
    from ml.utils.epoch_timer import timing
except ModuleNotFoundError:

    def timing(func):
        return func


class NeymanTask(PlottingMixin):
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    hist_path: str = luigi.Parameter(description="Path to the histogram file (.json)")  # type: ignore
    output_path: str = luigi.Parameter(description="Path to the output file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.hist_path.endswith(".json") and not os.path.exists(Path(self.hist_path).parent):
            raise FileNotFoundError(f"Argument `hist_path`='{self.hist_path}' must point to a valid .json file")

    def output(self):  # type: ignore
        plots = super().output()
        plots.update({"neyman": luigi.LocalTarget(Path(self.output_path))})
        return plots

    @cached_property
    def f_s_nominal(self) -> float:
        """Return the true signal fraction for the nominal mu=1 case.

        uses the label-derived ratio with index 1, not the fitted f_s_hat (index 0)
        
        normalising predicted mu by true nominal makes calibration robust to
        fit bias at mu=1 and to single-seed statistical fluctuation.
        """
        return self._get_sig_to_bkg_ratio(seed=0, mu_true=1)[1]

    def _get_sig_to_bkg_ratio(
        self,
        seed: int,
        mu_true: float,
    ) -> tuple[float, float]:
        alljet_data, _ = createJetData(
            jet_num="all",
            useTestData=True,
            loaded_data=self.loaded_data,
            set_mu=mu_true,
            seed=seed,
            useRand=False,
        )
        data_2j, data_1j, labels2j, labels1j = return1j2j(
            alljet_data,
            models=self.nf_models,
            device=self.device,
        )
        f_s_hat = compute_signal_fraction(
            test_data_2j=data_2j,
            test_data_1j=data_1j,
            dnn_model=self.classifier,
            bin_splines_S=self.bin_splines_S_class,
            bin_splines_BG=self.bin_splines_BG_class,
            eval_device=self.device,
        )
        labels = np.concatenate([labels2j.cpu().numpy(), labels1j.cpu().numpy()])
        num_signal = len(labels[labels == 1])
        signal_fraction_true = float(num_signal / len(labels))
        return f_s_hat, signal_fraction_true

    def _compute_neyman_entry(self, seed: int, mu_true: float):
        f_s_hat, signal_fraction_true = self._get_sig_to_bkg_ratio(seed, mu_true)
        mu_predicted = f_s_hat / self.f_s_nominal
        return float(mu_predicted), signal_fraction_true

    def create_neyman_construction(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(self.snapshot_path)
        self.nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts)
        self.classifier = CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])

        self.loaded_data = fetch_dataset(root_dir=self.root_dir)

        with open(self.hist_path, "r") as f:
            serializable_dict: Dict = json.load(f)

        if not serializable_dict:
            raise ValueError("Histogram dict is empty")

        hist_dict = {key: (np.array(v["sig"]), np.array(v["bg"])) for key, v in serializable_dict.items()}

        # Create dictionaries mapping parameter tuples to signal and background arrays.
        S_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][0] for i in hist_dict.keys()}
        B_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][1] for i in hist_dict.keys()}

        for parameter_mapping in (S_templates_2d_2j, B_templates_2d_2j):
            if not any(parameter_mapping.keys()):
                raise ValueError(f"Parameter mapping dict is fully malformed {parameter_mapping.keys()=}")

        # Fit 2D splines bin-by-bin using the dictionaries.
        self.bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
        self.bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)

        # Loop over a range of "mu" values and compute MLE ratios.
        MLE_ratio_arr: Dict[float, List[float]] = {}
        num_signal_dict: Dict[float, List[float]] = {}
        mu_true_array = np.linspace(0.1, 3.2, 10)
        N_sample = 10

        for mu_true in tqdm(mu_true_array, "Mu", position=0, leave=True):
            MLE_ratio_arr[float(mu_true)] = []
            num_signal_dict[float(mu_true)] = []  # TODO
            seed_array = np.random.randint(100_000, size=N_sample)

            for seed in tqdm(seed_array, "Seed", position=1, leave=False):
                mu, num_signal_2j = self._compute_neyman_entry(seed=seed, mu_true=mu_true)
                MLE_ratio_arr[mu_true].append(mu)
                num_signal_dict[mu_true].append(num_signal_2j)
                tqdm.write(f"Estimated mu: {mu:5f}, with mu_true {mu_true:5f}")

        with open(self.output_path, "w") as f:
            json.dump(MLE_ratio_arr, f)

        self.plot_mu_calibration(MLE_ratio_arr)
        self.plot_signal_yield(num_signal_dict)

    @PlottingMixin.plot(name="mu_calibration")
    def plot_mu_calibration(
        self,
        mle_ratio_dict: Dict[str, List[float]],
    ) -> Figure:
        x = np.array([float(k) for k in mle_ratio_dict.keys()])
        # need to avg over seeds instead of taking only first seed
        y = np.array([np.mean(v) for v in mle_ratio_dict.values()])
        yerr = np.array([np.std(v) for v in mle_ratio_dict.values()])

        fig, ax = plt.subplots(figsize=(5, 4))
        # to see uncertainty from the seeds
        plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
        plt.plot([x.min(), x.max()], [x.min(), x.max()], "k--", label="Ideal")

        plt.xlabel(r"$\mu_\text{true}$")
        plt.ylabel(r"$\mu_\text{estimated}$")
        plt.legend()
        plt.tight_layout()
        return fig

    @PlottingMixin.plot(name="signal_yield")
    def plot_signal_yield(
        self,
        num_signal_dict: Dict[str, List[float]],
    ) -> Figure:
        x = np.array([float(k) for k in num_signal_dict.keys()])
        y = np.array([np.mean(v) for v in num_signal_dict.values()])
        yerr = np.array([np.std(v) for v in num_signal_dict.values()])

        fig, ax = plt.subplots(figsize=(5, 4))
        # to see uncertainty from the seeds
        plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)

        plt.xlabel(r"$\mu_\text{true}$")
        plt.ylabel(r"$N_\text{signal}$ (2j)")
        plt.tight_layout()
        return fig

    @timing
    def run(self) -> None:
        self.create_neyman_construction()
