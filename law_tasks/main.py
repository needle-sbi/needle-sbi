import law

from law_tasks.mixins import HydraMixin
from law_tasks.estimator import EstimatorTask


class MainTask(law.WrapperTask, HydraMixin):
    def requires(self):
        return [
            EstimatorTask.req(
                self,
                config_file=self.config_file,
                estimator=estimator_key,
            )
            for estimator_key in self.config.estimators.keys()
        ]
