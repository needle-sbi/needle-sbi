"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
import os
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
from ..utils.selection import createJetData, load_train_set_data, return1j2j
from ..utils.stats import (
    compute_mu_nuan_2NP_class,
    fit_2D_splines_bin_by_bin_from_dict,
    string_to_tuple_str,
)
from .histogram import HistogramTask
from .plot import PlottingMixin


class NeymanTask(PlottingMixin):
    plot_save_dir: str = luigi.Parameter(description="Path to the directory where to save the validation plots")  # type: ignore
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

    def _compute_neyman_entry(
        self,
        seed: int,
        frac: float,  # alias for "mu"
    ) -> Any:
        def _save_objects(tmp_dir: str = "/tmp/needle") -> None:
            alljet_data["weights"].to_csv(f"{tmp_dir}/weights.csv")
            alljet_data["detailed_labels"].to_csv(f"{tmp_dir}/detailed_labels.csv")
            alljet_data["labels"].to_csv(f"{tmp_dir}/labels.csv")
            alljet_data["data"].to_csv(f"{tmp_dir}/data.csv")
            torch.save(data_1j, f"{tmp_dir}/data_1j")
            torch.save(data_2j, f"{tmp_dir}/data_2j")

        alljet_data, _ = createJetData(  # type: ignore
            jet_num="all",
            useTestData=True,
            loaded_data=self.loaded_data,
            set_mu=frac,
            seed=seed,
            n_param=[1, 1, 1, 1, 1, 0],
            useRand=True,
            root_dir=self.root_dir,
        )
        data_2j, data_1j, _, _ = return1j2j(
            alljet_data,
            models=self.nf_models,
            device=self.device,
        )  # type: ignore
        mu = compute_mu_nuan_2NP_class(
            test_data_2j=data_2j,
            test_data_1j=data_1j,
            dnn_model=self.classifier,
            bin_splines_S=self.bin_splines_S_class,
            bin_splines_BG=self.bin_splines_BG_class,
            eval_device=self.device,
        )
        return mu

    def create_neyman_construction(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(self.snapshot_path)
        self.nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts)
        self.classifier = CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])

        self.loaded_data = load_train_set_data(root_dir=self.root_dir)

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

        self.plot_validate_s_templates_2d_2j()

        # Loop over a range of "mu" values and compute MLE ratios.
        MLE_ratio_arr: Dict[str, List[float]] = {}
        frac_array = np.linspace(0.1, 3.2, 10)
        N_sample = 50

        for frac in tqdm(frac_array, "Mu", position=0, leave=False):
            MLE_ratio_arr[frac] = []
            seed_array = np.random.randint(100_000, size=N_sample)

            for seed in tqdm(seed_array, "Seed", position=1, leave=False):
                mu = self._compute_neyman_entry(seed=seed, frac=frac)

                MLE_ratio_arr[frac].append(mu)
                tqdm.write(f"Estimated mu: {mu}, with mu_true {frac}")

        with open(self.output_path, "w") as f:
            json.dump(MLE_ratio_arr, f)

    def plot_validate_s_templates_2d_2j(self) -> Figure:
        """TODO Currently not implemented"""
        fig, ax = plt.subplots(figsize=(5, 4), dpi=400)

        return fig

    @timing
    def run(self) -> None:
        self.create_neyman_construction()
