import os
from pathlib import Path
from typing import List

import law
from law_tasks.estimator import EstimatorTask
from law_tasks.mixins import HydraMixin
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class MainTask(HydraMixin, law.WrapperTask):
    """Root Task that is the main entry point for the Task Graph

    Will run all EstimatorTasks listed in the config.
    """

    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        default="runs",
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        if self.results_path != "runs":
            if self.config.results_path:
                logger.warning(
                    f"Conflicting value for arg `--results-path`. Config indicates '{self.config.results_path}' "
                    f"while CLI arg is '{self.results_path}'. The CLI value takes precedence. You can also "
                    f"set this parameter using `--hydra-overrides='results_path={self.results_path}'`."
                )
                return Path(self.config.results_path)

        return Path(os.path.abspath(self.results_path))

    def requires(self) -> List[EstimatorTask]:
        cache_config_file = os.path.join(self.results_path, "config.yaml")
        self.config.to_yaml(cache_config_file)

        return [
            EstimatorTask(
                config_file=cache_config_file,
                hydra_overrides=self.hydra_overrides,
                estimator=estimator_key,
                results_path=self.abs_results_path,
            )
            for estimator_key in self.config.estimators.keys()
        ]
