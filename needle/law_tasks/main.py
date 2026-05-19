import os
from enum import Enum
from pathlib import Path
from typing import List

import law
from omegaconf import OmegaConf

from needle.law_tasks.estimator import EstimatorTask
from needle.law_tasks.mixins import HydraMixin
from needle.utils.config_utils import compare_configs, initialize_hydra_config
from needle.utils.logging import ColorFormatter, LogOnce

logger = ColorFormatter.get_logger("dag")


class ConfigStrictness(Enum):
    IGNORE = "IGNORE"
    WARN = "WARN"
    RAISE = "RAISE"


class MainTask(HydraMixin, law.WrapperTask):
    """This Task serves as the main entry point for all the trainings.

    It is responsible for:

    - Loading and resolving Hydra configuration
    - Creating EstimatorTask instances for all estimators in the config
    - Managing the complete training DAG

    The Task resolves configuration conflicts and manages results paths, then propagates all the settings
    down the Task tree.
    """

    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        default="runs",
        significant=False,
    )  # type: ignore
    strict_config: str = law.Parameter(
        description=(
            "Level of strictness used to enforce a unique config for each run. Either one of "
            f"these options: {ConfigStrictness._member_map_}. The cached config will be updated "
            "and therefore prevent this check during the next run. Using lower cases is possible"
        ),
        default=ConfigStrictness.WARN.value,
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        """Get the absolute path to the results directory.

        Resolves potential conflicts between config-specified and CLI-specified paths.
        The CLI value (`--results-path`) takes precedence if both are provided. The third way of overriding
        the value (from the CLI) with `hydra_override="results_path=..."` is recommended in case you
        want to that information to be stored as it will be directly injected into the hydra config.

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

        Also caches the resolved config to ensure consistency across all dependent tasks to `<results_path>/config.yaml`

        Returns:
            List[EstimatorTask]: Tasks for each estimator key in the config.
        """
        os.makedirs(self.abs_results_path, exist_ok=True)
        cache_config_filepath = Path(os.path.join(self.abs_results_path, "config.yaml"))
        self.config._resolved = True

        if cache_config_filepath.exists():
            cached_config = initialize_hydra_config(
                cache_config_filepath.parent._str,
                cache_config_filepath.stem,
            )
            config_diff = compare_configs(self.config, cached_config)

            if config_diff:
                msg = (
                    "The cached version of your config does not match the new instance. Training results "
                    "might differ based on the changes lines. Use `--remove-output` to delete the cached files "
                    f"from the previous run if you want a fresh run. Offending entries are (new, old):\n{config_diff}"
                )
                match self.strict_config.upper():
                    case ConfigStrictness.WARN.value:
                        logger.warning(msg)
                    case ConfigStrictness.RAISE.value:
                        raise RuntimeError(msg)
                    case ConfigStrictness.IGNORE.value:
                        pass
                    case _:
                        raise ValueError(
                            f"Unknown value {self.strict_config} for Parameter 'strict_config'. Must "
                            f"be one of {ConfigStrictness._member_names_}"
                        )

        with open(cache_config_filepath, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(self.config), resolve=True))

        return [
            EstimatorTask(
                config_file=cache_config_filepath,
                hydra_overrides=self.hydra_overrides,
                estimator=estimator_key,
                results_path=self.abs_results_path,
            )
            for estimator_key in self.config.estimators.keys()
        ]
