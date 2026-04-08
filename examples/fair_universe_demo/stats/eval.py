"""
Original Authors: FAIR Universe Higgs ML Challenge
Repository: https://github.com/FAIR-Universe/HEP-Challenge
Adapted by: K. Schmidt
"""

# flake8: noqa: E704

import json
from itertools import product
from logging import Logger
from typing import Any, Dict, List, TypedDict

import luigi
import numpy as np
import pandas as pd

from ..utils.selection import load_train_set_data
from ..utils.systematics import get_bootstrapped_dataset, get_systematics_dataset

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
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore
    output_path: str = luigi.Parameter(description="Path to save the result file (.json).")  # type: ignore
    test_settings_path: str = luigi.Parameter(description="Path to the test settings file (.json)")  # type: ignore
    predict_path: str = luigi.Parameter(description="Path to the prediction generated from the 'PredictTask'")  # type: ignore

    DEFAULT_INGESTION_SEED = 31415

    def __init__(self, test_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_settings(self) -> Dict[str, Any]:
        with open(self.test_settings_path, "r") as f:
            _test_settings = json.load(f)

        return _test_settings

    @property
    def model_prediction(self) -> ModelResult:
        with open(self.predict_path, "r") as f:
            _prediction = json.load(f)

        return _prediction

    @staticmethod
    def _generate_pseudo_exp_data(
        data: Dict | pd.DataFrame,
        set_mu: float = 1.0,
        tes: float = 1.0,
        jes: float = 1.0,
        soft_met: float = 0.0,
        ttbar_scale=None,
        diboson_scale=None,
        bkg_scale=None,
        seed: int = 0,
    ) -> Dict | pd.DataFrame:
        pesudo_exp_data = get_bootstrapped_dataset(
            data,
            mu=set_mu,
            ttbar_scale=ttbar_scale,
            diboson_scale=diboson_scale,
            bkg_scale=bkg_scale,
            seed=seed,
        )
        test_set = get_systematics_dataset(
            pesudo_exp_data,
            tes=tes,
            jes=jes,
            soft_met=soft_met,
        )
        return test_set

    def prepare(self) -> None:
        self.data = load_train_set_data(self.root_dir)

    def predict_submission(self, initial_seed: int = DEFAULT_INGESTION_SEED):
        logger.info("Calling predict method of submitted model with seed: %s", initial_seed)

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

        full_test_set = self.data.get_test_set()

        self.results_dict: PredictResult = {}  # type: ignore

        for set_index, test_set_index in all_combinations:
            seed = (set_index * num_pseudo_experiments) + test_set_index + initial_seed

            # get mu value of set from test settings
            set_mu = self.test_settings["ground_truth_mus"][set_index]

            random_state = np.random.RandomState(seed)

            if dict_systematics["tes"]:
                tes = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            else:
                tes = 1.0
            if dict_systematics["jes"]:
                jes = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            else:
                jes = 1.0
            if dict_systematics["soft_met"]:
                soft_met = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)
            else:
                soft_met = 0.0

            if dict_systematics["ttbar_scale"]:
                ttbar_scale = np.clip(random_state.normal(loc=1.0, scale=0.02), a_min=0.8, a_max=1.2)
            else:
                ttbar_scale = None

            if dict_systematics["diboson_scale"]:
                diboson_scale = np.clip(random_state.normal(loc=1.0, scale=0.25), a_min=0.0, a_max=2.0)
            else:
                diboson_scale = None

            if dict_systematics["bkg_scale"]:
                bkg_scale = np.clip(random_state.normal(loc=1.0, scale=0.001), a_min=0.99, a_max=1.01)
            else:
                bkg_scale = None

            test_set = self._generate_pseudo_exp_data(
                data=full_test_set,
                set_mu=set_mu,
                tes=tes,
                jes=jes,
                soft_met=soft_met,
                ttbar_scale=ttbar_scale,
                diboson_scale=diboson_scale,
                bkg_scale=bkg_scale,
                seed=seed,
            )

            logger.debug(f"set_index: {set_index} - test_set_index: {test_set_index} - seed: {seed}")

            predicted_dict = {}
            predicted_dict.update(self.model_prediction)
            predicted_dict["test_set_index"] = test_set_index

            logger.debug(f"Predicted: {predicted_dict}")

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []

            self.results_dict[set_index].append(predicted_dict)

    def compute_result(self):
        for key in self.results_dict.keys():
            set_result = self.results_dict[key]
            set_result.sort(key=lambda x: x["test_set_index"])
            mu_hats, delta_mu_hats, p16, p84 = [], [], [], []

            for test_set_dict in set_result:
                mu_hats.append(test_set_dict["mu_hat"])
                delta_mu_hats.append(test_set_dict["delta_mu_hat"])
                p16.append(test_set_dict["p16"])
                p84.append(test_set_dict["p84"])

            ingestion_result_dict: PredictResult = {
                "mu_hat": mu_hats,
                "delta_mu_hat": delta_mu_hats,
                "p16": p16,
                "p84": p84,
            }
            self.results_dict[key] = ingestion_result_dict

    def save_result(self):
        results_dict_serializable = {int(key): val for key, val in self.results_dict.items()}

        with open(self.output_path, "w") as f:
            f.write(json.dumps(results_dict_serializable, indent=4))

    def run(self) -> None:
        self.prepare()
        self.predict_submission()
        self.compute_result()
        self.save_result()
