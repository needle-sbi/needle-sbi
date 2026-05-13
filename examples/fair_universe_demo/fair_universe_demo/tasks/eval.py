"""
Original Authors: FAIR Universe Higgs ML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""

# flake8: noqa: E704

import json
import os
from functools import cached_property
from itertools import product
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import luigi
import numpy as np
from tqdm import tqdm

from ..utils.dataset_sharing import fetch_dataset
from ..utils.eval import predict

logger = Logger("eval")


class ModelResult(TypedDict):
    mu_hat: float
    delta_mu_hat: float
    p16: float
    p84: float


class PredictResult(TypedDict):
    mu_hat: List[float]
    delta_mu_hat: List[float]
    p16: List[float]
    p84: List[float]


class EvalTask(luigi.Task):
    hist_path: str = luigi.Parameter(description="Path to the histogram file (.json).")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore
    output_path: str = luigi.Parameter(description="Path to save the result file (.json).")  # type: ignore
    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json).")  # type: ignore
    neyman_path: str = luigi.Parameter(description="Path to the Neyman construction file (.json)")  # type: ignore
    test_settings_path: str = luigi.Parameter(description="Path to the test settings file (.json)")  # type: ignore

    DEFAULT_INGESTION_SEED = 31415

    @cached_property
    def test_settings(self) -> Dict[str, Any]:
        cached_test_settings_file = Path(self.output()["test_settings"].path)

        if cached_test_settings_file.exists():
            with open(cached_test_settings_file, "r") as f:
                _test_settings: Dict[str, Any] = json.load(f)

            return _test_settings

        with open(self.test_settings_path, "r") as f:
            _test_settings: Dict[str, Any] = json.load(f)

        with open(self.output()["test_settings"].path, "w") as f:
            json.dump(_test_settings, f)

        return _test_settings

    def output(self) -> Dict[str, luigi.LocalTarget]:  # type: ignore
        output_dir = Path(self.output_path).parent
        test_settings_filename = Path(self.test_settings_path).name
        return {
            "eval": luigi.LocalTarget(self.output_path),
            "test_settings": luigi.LocalTarget(os.path.join(output_dir, test_settings_filename)),
        }

    def prepare(self) -> None:
        self.data = fetch_dataset(self.root_dir)

    def predict_submission(self, initial_seed: int = DEFAULT_INGESTION_SEED):
        logger.info(f"Calling predict method of submitted model with seed: {initial_seed}")

        dict_systematics = self.test_settings["systematics"]
        num_pseudo_experiments = self.test_settings["num_pseudo_experiments"]
        num_of_sets = self.test_settings["num_of_sets"]

        set_indices = np.arange(0, num_of_sets, dtype=int)
        test_set_indices = np.arange(0, num_pseudo_experiments, dtype=int)

        # create a product of set and test set indices all combinations of tuples
        all_combinations = list(product(set_indices, test_set_indices))

        # randomly shuffle all combinations of indices
        random_state_initial = np.random.RandomState(initial_seed)
        random_state_initial.shuffle(all_combinations)

        self.results_dict: PredictResult = {}  # type: ignore

        for set_index, test_set_index in tqdm(all_combinations, leave=None):
            seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

            # get mu value of set from test settings
            set_mu = self.test_settings["ground_truth_mus"][set_index]

            random_state = np.random.RandomState(seed)

            tes = (
                np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
                if dict_systematics["tes"]
                else 1.0
            )
            jes = (
                np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
                if dict_systematics["jes"]
                else 1.0
            )
            soft_met = (
                np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)
                if dict_systematics["soft_met"]
                else 0.0
            )
            ttbar_scale = (
                np.clip(random_state.normal(loc=1.0, scale=0.02), a_min=0.8, a_max=1.2)
                if dict_systematics["ttbar_scale"]
                else None
            )
            diboson_scale = (
                np.clip(random_state.normal(loc=1.0, scale=0.25), a_min=0.0, a_max=2.0)
                if dict_systematics["diboson_scale"]
                else None
            )
            bkg_scale = (
                np.clip(random_state.normal(loc=1.0, scale=0.001), a_min=0.99, a_max=1.01)
                if dict_systematics["bkg_scale"]
                else None
            )

            logger.debug(f"Set_index: {set_index} - test_set_index: {test_set_index} - seed: {seed}")

            model_prediction = predict(
                mu=set_mu,
                hist_path=self.hist_path,
                neyman_path=self.neyman_path,
                snapshot_path=self.snapshot_path,
                data=self.data,
                nuissance_parameters=[
                    ttbar_scale,
                    diboson_scale,
                    bkg_scale,
                    tes,
                    jes,
                    soft_met,
                ],
                predict_num_events=0,
            )
            predicted_dict = {}
            predicted_dict.update(model_prediction)
            predicted_dict["mu_true"] = set_mu
            predicted_dict["test_set_index"] = float(test_set_index)

            logger.info(f"Predicted: {predicted_dict}")

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []

            self.results_dict[set_index].append(predicted_dict)

    def save_result(self):
        # Aggregate results from list-of-dicts to dict-of-lists format
        aggregated_results = {}
        for set_index, predictions_list in self.results_dict.items():
            aggregated_results[int(set_index)] = {
                "mu_hat": [p["mu_hat"] for p in predictions_list],
                "delta_mu_hat": [p["delta_mu_hat"] for p in predictions_list],
                "p16": [p["p16"] for p in predictions_list],
                "p84": [p["p84"] for p in predictions_list],
            }

        result_path = self.output()["eval"].path
        dump = json.dumps(aggregated_results, indent=4)

        with open(result_path, "w") as f:
            f.write(dump)

        logger.info(f"Saved evaluation result to '{result_path}'")

    @timing
    def run(self) -> None:
        self.prepare()
        self.predict_submission()
        self.save_result()
