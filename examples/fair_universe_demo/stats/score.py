import json
from functools import cached_property
from logging import Logger
from typing import Any, Dict

import luigi

from ..utils.score import Scoring

logger = Logger("score")


class ScoreTask(luigi.Task):
    predict_path: str = luigi.Parameter(description="Path to the prediction results from PredictTask (.json)")  # type: ignore
    output_dir: str = luigi.Parameter(description="Path to the file with the scores produced by this Task (.json)")  # type: ignore
    test_settings_path: str = luigi.Parameter(description="Path to the test settings file (.json)")  # type: ignore

    @cached_property
    def test_settings(self) -> Dict[str, Any]:
        with open(self.test_settings_path, "r") as f:
            _test_settings = json.load(f)

        return _test_settings

    def run(self) -> None:
        scoring = Scoring()

        scoring.start_timer()
        scoring.load_ingestion_results(
            self.predict_path, self.output_dir
        )  # TODO Account for the fact that we store everything in the same .json

        num_samples = len(self.test_settings["ground_truth_mus"])

        scoring.compute_scores(self.test_settings)
        scoring.compute_bootstrapped_scores(n_bootstraps=1000, sample_size=num_samples)
        scoring.stop_timer()
        scoring.write_scores()
        logger.info(f"Scoring duration: {scoring.get_duration()}")
