"""
Mixin classes for remote task executing. Currently supports:
 - HTCondor
"""

import law

law.contrib.load("htcondor")  # type: ignore


class HTCondorMixin(law.htcondor.HTCondorWorkflow):  # type: ignore
    """
    Mixin to submit LAW tasks as individual HTCondor jobs.
    
    Usage:
        class MyTask(HTCondorMixin, law.Task):
            ...
        
        # Submit with HTCondor workflow
        law run MyTask --workflow htcondor
    """
    
    max_runtime = law.DurationParameter(
        default=120.0,
        unit="min",
        significant=False,
        description="Maximum runtime of the task in minutes.",
    )
    
    htcondor_request_cpus = law.IntParameter(
        default=2,
        significant=False,
        description="Number of CPUs to request per job",
    )
    
    htcondor_request_memory = law.BytesParameter(
        default=32.0,
        unit="GB",
        significant=False,
        description="Memory to request per job",
    )

    def htcondor_output_directory(self):
        """Directory where HTCondor job logs are stored"""
        return law.LocalDirectoryTarget(self.local_path())
    
    def htcondor_bootstrap_file(self):
        """Bootstrap script to setup environment on remote node"""
        return law.util.rel_path(__file__, "../../setup.sh")
    
    def htcondor_job_config(self, config, job_num, branches):
        """Configure HTCondor job submission parameters"""
        
        # Resource requests
        config.custom_content.append(("request_cpus", self.htcondor_request_cpus))
        config.custom_content.append(("request_memory", f"{self.htcondor_request_memory}MB"))
        config.custom_content.append(("+Request_Runtime", int(self.max_runtime * 60)))  # seconds
        config.custom_content.append(("request_GPUs", "0"))
        
        # Environment
        config.custom_content.append(("getenv", "True"))
        config.custom_content.append(("universe", "vanilla"))
        
        # Export FAIR_UNIVERSE_DATA path
        config.custom_content.append((
            "+Environment",
            '"FAIR_UNIVERSE_DATA=/data/dust/group/atlas/needle/FAIRUnv/UncertaintyChallenge_2024/ProcessedData_v1_2025-10-03/CombData-part0.parquet"'
        ))
        
        return config
