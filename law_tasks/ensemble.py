"""
Task for a single ensemble training, includes multiple folds.
"""
import json
import law
import luigi

from preprocessor.utils import ColorFormatter
from law_tasks.fold import FoldTask

logger = ColorFormatter.get_logger("ensemble")


class EnsembleTask(law.Task):
    
    num_folds = luigi.IntParameter(
        description="Number of folds in the ensemble.",
        default=10,
    )
    config_yaml = law.Parameter(
        description="Path to the YAML configuration file for the training.",
        default="config.yaml"
    )
    results_dir = law.Parameter(
        description="Directory where the ensemble training results will be saved.",
        default="ensemble_runs",
        significant=False,
    )

    def requires(self):
        return [
            FoldTask.req(
                self,
                fold=i,
                config_yaml=self.config_yaml,
                results_dir=f"{self.results_dir}/fold_{i}",
            )
            for i in range(self.num_folds)  # type: ignore
        ]

    def output(self):
        return {
            "outputs": law.LocalFileTarget(f"{self.results_dir}/ensemble_done.txt")
        }

    def run(self):
        with open(self.output()["outputs"].path, "w") as f:
            json.dump("done", f, indent=4)
