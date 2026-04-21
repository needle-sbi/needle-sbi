import json
import os
from functools import cached_property
from logging import Logger
from pathlib import Path
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
        output_dir = Path(self.output_dir).parent
        test_settings_filename = Path(self.test_settings_path).name
        return {
            "scores": luigi.LocalTarget(os.path.join(self.output_dir, "scores.json")),
            "detailed_scores": luigi.LocalTarget(os.path.join(self.output_dir, "detailed_results.html")),
            "test_settings": luigi.LocalTarget(os.path.join(output_dir, test_settings_filename)),
        }

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        scoring = Scoring()

        scoring.start_timer()
        scoring.load_ingestion_results(
            self.predict_path,
            self.output_dir,
        )  # TODO Account for the fact that we store everything in the same .json

        num_samples = len(self.test_settings["ground_truth_mus"])

        scoring.compute_scores(self.test_settings)
        scoring.compute_bootstrapped_scores(n_bootstraps=1000, sample_size=num_samples)
        scoring.stop_timer()
        scoring.write_scores()
        logger.info(f"Scoring duration: {scoring.get_duration()}")
