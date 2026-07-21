from needle.tasks.mixins.hydra import HydraParamsMixin

# Re-export so that law_tasks and user code importing from here continue to work.
# law.Parameter IS luigi.Parameter (LAW re-exports it unchanged), so this is a no-op change.
__all__ = ["HydraParamsMixin"]

# Backwards-compatible alias used throughout law_tasks
HydraMixin = HydraParamsMixin
