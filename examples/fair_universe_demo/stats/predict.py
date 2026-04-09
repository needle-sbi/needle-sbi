"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
from logging import Logger
from typing import Any, Dict, List

import luigi
import numpy as np
import torch

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..stats.histogram import HistogramTask
from ..utils.selection import createJetData, return1j2j
from ..utils.stats import (
    compute_mu_nuan_2NP_class,
    fit_2D_splines_bin_by_bin_from_dict,
    get_confidence_interval,
    load_bias_data,
    string_to_tuple_str,
)

logger = Logger("predict")


class PredictTask(luigi.Task):
    hist_path: str = luigi.Parameter(description="Path to the histogram file (.json).")  # type: ignore
    output_path: str = luigi.Parameter(description="Path to save the result file (.json).")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json).")  # type: ignore
    neyman_path: str = luigi.Parameter(description="Path to the Neyman construction file (.json)")  # type: ignore
    mu: float = luigi.FloatParameter(description="Hyperparameter 'mu'", default=1.0)  # type: ignore
    predict_on_test = luigi.BoolParameter(
        description="Whether to test on a test dataset", default=True
    )
    predict_num_events: int = luigi.IntParameter(
        description="Number of events to test if predict_mu_test is False",
        default=0,
    )  # type: ignore

    @staticmethod
    def predict(
        mu: float,
        hist_path: str,
        neyman_path: str,
        snapshot_path: str,
        root_dir: str,
        device: str = None,
        predict_num_events: int = 0,
        nuissance_parameters: List[float | None] = [1, 1, 1, 1, 1, 0],
    ) -> Dict:
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Running pipeline on test dataset with {mu=}")

        with open(hist_path, "r") as f:
            serializable_dict: Dict = json.load(f)

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

        # Fit 2D splines bin-by-bin using the dictionaries.
        bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
        bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)

        # loading Neyman data
        std_corrected_interp, a, b = load_bias_data(neyman_path)

        nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(snapshot_path)
        models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device)
        class_model_load = (
            CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])
            .to(device)
            .eval()
            .to(torch.float32)
        )

        alljet_data, _ = createJetData(  # type: ignore
            "all",
            True,
            set_mu=mu,
            seed=31245,
            n_param=nuissance_parameters,
            useRand=True,
            root_dir=root_dir,
        )

        results: Dict[str, Any] = {"mu": mu}

        if not predict_num_events:
            logger.info("Running in prediction mode")
            data_2j, data_1j, label_2j, label_1j = return1j2j(
                alljet_data, models, device=device
            )

            # Compute the MLE mu using the provided classifier and fitted splines.
            mu = compute_mu_nuan_2NP_class(
                data_2j,
                data_1j,
                class_model_load,
                bin_splines_S_class,
                bin_splines_BG_class,
            )
            mu_MLE, mu_lower, mu_upper = get_confidence_interval(
                mu, std_corrected_interp, a, b
            )

            results.update(
                {
                    "real_mu": mu,
                    "mu_hat": mu_MLE,
                    "p16": mu_lower,
                    "p84": mu_upper,
                    "delta_mu_hat": abs(mu_upper - mu_lower) / 2,
                }
            )

        else:
            logger.info(
                f"Running classifier (not as 'mu' estimator) for {predict_num_events} events"
            )
            data_2j, data_1j, label_2j, label_1j = return1j2j(
                alljet_data,
                models,
                cut=True,
                nevents=predict_num_events,
                device=device,
            )

            with torch.no_grad():
                scores_2j = torch.sigmoid(class_model_load(data_2j, 2)).cpu().numpy()
                scores_1j = torch.sigmoid(class_model_load(data_1j, 1)).cpu().numpy()

            results.update(
                {
                    "scores_2j": scores_2j.tolist(),
                    "labels_2j": label_2j.tolist(),
                    "scores_1j": scores_1j.tolist(),
                    "labels_1j": label_1j.tolist(),
                }
            )

        return results

    def run(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = self.predict(
            mu=self.mu,
            hist_path=self.hist_path,
            snapshot_path=self.snapshot_path,
            neyman_path=self.neyman_path,
            root_dir=self.root_dir,
            device=device,
            predict_num_events=0,
        )

        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=4)
