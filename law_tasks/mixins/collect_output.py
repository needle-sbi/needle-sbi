import law

from orchestrator.luigi_utils import collect_output_paths
from preprocessor.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("orchestrator")


class CollectOutputMixin(law.Task):
    collect_output_paths: int = law.CSVParameter(
        default=(),
        significant=False,
        description="Print all the output paths up to the provided depth, with -1 being fully recursive",
    )  # type: ignore

    interactive_params = law.Task.interactive_params + ["collect_output_paths"]

    def _collect_output_paths(self, args) -> bool:
        depth = int(args[0]) if args else -1
        logger.info(f"Collected paths up to depth {depth}")

        for output_path in collect_output_paths(self, current_depth=depth):
            print(output_path)

        return False
