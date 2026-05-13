import os
from pathlib import Path
from typing import List

import law
from omegaconf import OmegaConf

from law_tasks.estimator import EstimatorTask
from law_tasks.mixins import HydraMixin
from needle.utils.logging import ColorFormatter, LogOnce

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
        """Get the absolute path to the results directory.

        Resolves potential conflicts between config-specified and CLI-specified paths.
        The CLI value (--results-path) takes precedence if both are provided.

        Returns:
            Path: Absolute path to results directory.
        """
        if self.results_path != "runs":
            if self.config.results_path:
                LogOnce(logger).warn_once(
                    f"Conflicting value for arg `--results-path`. Config indicates '{self.config.results_path}' "
                    f"while CLI arg is '{self.results_path}'. The CLI value takes precedence. You can also "
                    f"set this parameter using `--hydra-overrides='results_path={self.results_path}'`."
                )
                return Path(self.config.results_path)

        return Path(os.path.abspath(self.results_path))

    def requires(self) -> List[EstimatorTask]:
        """Create EstimatorTask instances for all estimators in the config.

        Also caches the resolved config to ensure consistency across all dependent tasks.

        Returns:
            List[EstimatorTask]: Tasks for each estimator key in the config.
        """
        cache_config_file = os.path.join(self.abs_results_path, "config.yaml")
        self.config._resolved = True

        with open(cache_config_file, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(self.config), resolve=True))

        return [
            EstimatorTask(
                config_file=cache_config_file,
                hydra_overrides=self.hydra_overrides,
                estimator=estimator_key,
                results_path=self.abs_results_path,
            )
            for estimator_key in self.config.estimators.keys()
        ]
