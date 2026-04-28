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
from ml.utils.epoch_timer import timing
from tqdm import tqdm

from ..utils.eval import predict
from ..utils.selection import load_train_set_data

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

        self.results_dict: PredictResult = {}  # type: ignore

        for set_index, test_set_index in tqdm(all_combinations, leave=None):
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

            logger.debug(f"set_index: {set_index} - test_set_index: {test_set_index} - seed: {seed}")

            model_prediction = predict(
                mu=set_mu,
                hist_path=self.hist_path,
                neyman_path=self.neyman_path,
                snapshot_path=self.snapshot_path,
                data=self.data,
                nuissance_parameters=[
                    tes,
                    jes,
                    soft_met,
                    ttbar_scale,
                    diboson_scale,
                    bkg_scale,
                ],
                predict_num_events=0,
            )
            predicted_dict = {}
            predicted_dict.update(model_prediction)
            predicted_dict["mu_true"] = set_mu
            predicted_dict["test_set_index"] = test_set_index

            logger.debug(f"Predicted: {predicted_dict}")

            if set_index not in self.results_dict:
                self.results_dict[set_index] = []

            self.results_dict[set_index].append(predicted_dict)

    def save_result(self):
        results_dict_serializable = {int(key): val for key, val in self.results_dict.items()}
        result_path = self.output()["eval"].path

        with open(result_path, "w") as f:
            f.write(json.dumps(results_dict_serializable, indent=4))

        logger.info(f"Saved evaluation result to '{result_path}'")

    @timing
    def run(self) -> None:
        self.prepare()
        self.predict_submission()
        self.save_result()
