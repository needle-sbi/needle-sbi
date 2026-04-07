"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
import os
from pathlib import Path
from typing import Dict

import luigi
import numpy as np
import torch
from tqdm import tqdm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.selection import createJetData, return1j2j
from ..utils.stats import (
    compute_mu_nuan_2NP_class,
    fit_2D_splines_bin_by_bin_from_dict,
    string_to_tuple_str,
)
from .histogram import HistogramTask


class NeymanTask(luigi.Task):
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    hist_path: str = luigi.Parameter(description="Path to the histogram file (.json)")  # type: ignore
    output_path: str = luigi.Parameter(description="Path to the output file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.hist_path.endswith(".json") and not os.path.exists(
            Path(self.hist_path).parent
        ):
            raise FileNotFoundError(
                f"Argument `hist_path`='{self.hist_path}' must point to a valid .json file"
            )

    def create_neyman_construction(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(self.snapshot_path)
        nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device)
        class_model_load = (
            CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])
            .to(device)
            .eval()
            .to(torch.float32)
        )

        with open(self.hist_path, "r") as f:
            serializable_dict: Dict = json.load(f)

        if not serializable_dict:
            raise ValueError("Histogram dict is empty")

        hist_dict = {
            key: (np.array(v["sig"]), np.array(v["bg"]))
            for key, v in serializable_dict.items()
        }

        # Create dictionaries mapping parameter tuples to signal and background arrays.
        S_templates_2d_2j = {
            string_to_tuple_str(i): hist_dict[i][0] for i in hist_dict.keys()
        }
        B_templates_2d_2j = {
            string_to_tuple_str(i): hist_dict[i][1] for i in hist_dict.keys()
        }

        for parameter_mapping in (S_templates_2d_2j, B_templates_2d_2j):
            if not any(parameter_mapping.keys()):
                raise ValueError(
                    f"Parameter mapping dict is fully malformed {parameter_mapping.keys()=}"
                )

        # Fit 2D splines bin-by-bin using the dictionaries.
        bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
        bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)

        # Loop over a range of "mu" values and compute MLE ratios.
        MLE_ratio_arr = {}
        frac_array = np.linspace(0.1, 3.2, 10)
        N_sample = 50

        for frac in tqdm(frac_array, "Mu", position=0):
            MLE_ratio_arr[frac] = []
            # Generate a set of random seeds.
            seed_array = np.random.randint(100_000, size=N_sample)

            for seed in tqdm(seed_array, "Seed", position=1, leave=False):
                # Create jet data. The 'createJetData' function is assumed to use the
                # provided data object to return the full set of jets.
                alljet_data, _ = createJetData(  # type: ignore
                    "all",
                    True,
                    set_mu=frac,
                    seed=seed,
                    n_param=[1, 1, 1, 1, 1, 0],
                    useRand=True,
                    root_dir=self.root_dir,
                )
                # Split the data into 2-jet and 1-jet subsets.
                data_2j, data_1j, _, _ = return1j2j(
                    alljet_data, models=nf_models, device=device
                )

                # Compute the MLE mu using the provided classifier and fitted splines.
                mu = compute_mu_nuan_2NP_class(
                    data_2j,
                    data_1j,
                    class_model_load,
                    bin_splines_S_class,
                    bin_splines_BG_class,
                )

                MLE_ratio_arr[frac].append(mu)
                print(f"Estimated mu: {mu}")

        output_filename = os.path.join(self.output_path)

        with open(output_filename, "w") as f:
            json.dump(MLE_ratio_arr, f)
