import law

from needle.utils.logging import ColorFormatter
from needle.utils.luigi_utils import collect_output_paths

logger = ColorFormatter.get_logger("dag")


class CollectOutputMixin(law.Task):
    """Mixin providing interactive output path collection for debugging and exploration.

    Adds the --collect-output-paths parameter to Law tasks, allowing users to interactively
    traverse and print all output paths from a task and its dependencies.
    """

    collect_output_paths: int = law.CSVParameter(
        default=(),
        significant=False,
        description="Print all the output paths up to the provided depth, with -1 being fully recursive",
    )  # type: ignore

    interactive_params = law.Task.interactive_params + ["collect_output_paths"]

    def _collect_output_paths(self, args) -> bool:
        """Collect and print output paths up to a specified depth.

        Interactive method called when --collect-output-paths parameter is used.
        Recursively traverses task dependencies and prints all output target paths.

        Args:
            args: Arguments containing the depth level (-1 for full recursion).

        Returns:
            bool: False to indicate task should not run after collecting paths.
        """
        depth = int(args[0]) if args else -1
        logger.info(f"Collected paths up to depth {depth}")

        for output_path in collect_output_paths(self, current_depth=depth):
            print(output_path)

        return False
