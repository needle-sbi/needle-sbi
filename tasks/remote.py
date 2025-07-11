"""
Mixin classes for remote task executing. Currently supports:
 - HTCondor
"""
import law

law.contrib.load("htcondor")

# Plan:
# Just a simple mixin class that add HTCondor and maybe adjusts the outputs
# Request a GPU from NAF
# Add a htcondor config for NAF


class HTCondorMixin(law.htcondor.HTCondorWorkflow):

    max_runtime = law.DurationParameter(
        default=60.0,
        unit="min",
        significant=False,
        description="Maximum runtime of the task in minutes.",
    )

    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.local_path())
