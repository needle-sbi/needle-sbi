import json
import os
from functools import wraps
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, List

import gallery
import luigi
import matplotlib.pyplot as plt
import mplhep
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = Logger("PlottingMixin")


class PlottingMixin(luigi.Task):
    """Mixin Task for automatic saving of plots using luigi

    Adds the `plot_save_dir` as a `luigi.Parameter` that determines where to save all the plots. This
    parameter can be overridden using the `plot_save_dir_override` property to interject custom sub-
    directory structure.

    This class provides the `@plot(name=...)` decorator that marks a given method as to-be-plotted.
    Once executed (for example during `run()`), the plot will automatically get the NEEDLE style and
    be saved to the correct path `plot_save_dir` (or `plot_save_dir_override` if overridden).

    The format is 'pdf' by default but can be changed by expanding the `self.PlottingSettings.formats`
    list.

    The final plot file name is constructed as `plot_save_dir/<func_name>` where func_name is either:
     - The value entered when applying the `@plot(name=...)` decorator
     - The full name of the function itself. For example

        ```python
        PlottingSettings.formats = ["pdf"]
        plot_save_dir = "/path/to/plots/"  # hardcoded in this case

        @plot
        def my_custom_plot_func(self) -> Figure:
            ...
            return fig
        ```

        Will be saved to `/path/to/plots/my_custom_plot_func.pdf`.

    Raises:
        TypeError: If the functions registered as to-be-plotted to not return `matplotlib.figure.Figure`.
        ValueError: If a name was duplicated for two different functions
    """

    plot_save_dir: str = luigi.Parameter(
        description="Path to the directory where to save the plots resulting from this Task",
    )  # type: ignore

    class PlottingSettings:
        formats: List[str] = ["pdf", "png"]

    @property
    def plot_save_dir_override(self) -> str:
        """Property used to override the output directory where to save the plots. Useful if you want
        to have an individual directory for each family of plots. In the default implementation, this
        method directly returns `plot_save_dir`.

        Returns:
            str: The plot used by the Mixin Task
        """
        return self.plot_save_dir

    def output(self) -> Dict[str, luigi.LocalTarget]:  # type: ignore
        os.makedirs(os.path.abspath(self.plot_save_dir_override), exist_ok=True)

        return {
            f"{plot_name}.{fmt}": luigi.LocalTarget(
                Path(os.path.join(self.plot_save_dir_override, f"{plot_name}.{fmt}")).absolute()
            )
            for plot_name in self.registered_plots.keys()
            for fmt in self.PlottingSettings.formats
        }

    @staticmethod
    def set_needle_plot_style(fig: Figure, axes: Axes | List[Axes] | None = None) -> Figure:
        if axes is None:
            axes = fig.axes  # type: ignore

        if not isinstance(axes, List):
            axes = [axes]

        for ax in axes:
            mplhep.label.exp_label(
                loc=0,
                exp="NEEDLE",
                ax=ax,
                rlabel="FAIR Universe HiggsML",
            )
        plt.tight_layout()
        return fig

    @staticmethod
    def text(text: str, fig: Figure, axes: Axes | List[Axes] | None = None) -> Figure:
        if axes is None:
            axes = fig.axes

        if not isinstance(axes, List):
            axes = [axes]

        for axis in axes:
            axis.text(
                0.05,
                0.95,
                text,
                transform=axis.transAxes,
                ha="left",
                va="top",
            )
        return fig

    @staticmethod
    def plot(
        *,
        name: str = None,
        add_needle_plot_style: bool = True,
    ) -> Callable[[Callable[..., Figure]], Callable[..., Figure]]:
        """This decorator does two things:
            1. Register the given function as "to-be-plotted" which means it is registered automatically
                as a law Target, no need to explicitly write the output file.
            2. When the function is actually run, the resulting plot is automatically saved to file

        Args:
            name (str, optional): Name of the plot. Defaults to None, in which case the name of the
                function is used instead.
        """

        def decorator(func: Callable[..., Figure]) -> Callable[..., Figure]:
            """Register the function as "to-be-plotted"

            Args:
                func (Callable[..., Figure]): The function to register

            Returns:
                Callable[..., Figure]: Registered function
            """
            setattr(func, "_plot_name", name or func.__name__)

            @wraps(func)
            def wrapper(self: "PlottingMixin", *args, **kwargs) -> Figure:
                """Wraps the call signature of the function so that the plot is automatically saved.

                Raises:
                    TypeError: TypeError: If the function does not return a Figure object

                Returns:
                    Figure: Non-rendered Figure object for debugging
                """
                fig = func(self, *args, **kwargs)

                if not isinstance(fig, Figure):
                    raise TypeError(f"Function {func.__name__} must return matplotlib.figure.Figure")

                fig = self.set_needle_plot_style(fig) if add_needle_plot_style else fig
                name = getattr(func, "_plot_name")

                for fmt in self.PlottingSettings.formats:
                    save_path = self.output()[f"{name}.{fmt}"].path
                    fig.savefig(save_path)
                    print(f"Saved plot '{name}' to '{save_path}'.")

                plt.close(fig)
                return fig

            return wrapper

        return decorator

    @property
    def registered_plots(self) -> Dict[str, Callable[..., Figure]]:
        """Registers a function as to-be-plotted by adding the private `_plot_name` attribute.

        Raises:
            ValueError: If two functions share the same name.

        Returns:
            Dict[str, Callable[..., Figure]]: A dict of name-func pairs
        """
        plots = {}
        daughter = self.__class__.mro()[0]

        for name, member in daughter.__dict__.items():
            if hasattr(member, "_plot_name"):
                plot_name = getattr(member, "_plot_name")

                if plot_name in plots:
                    raise ValueError(f"Duplicate plot names: {plot_name} for other function {member}")

                plots[plot_name] = getattr(self, name)

        return plots

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        daughter_run = cls.run

        @wraps(daughter_run)
        def wrap_daughter_run(self):
            daughter_run(self)
            self._validate_plot_exist_after_run()

        cls.run = wrap_daughter_run

    def _validate_plot_exist_after_run(self) -> None:
        """Method that checks that all registered plotting functions were ran and successfully produced
        the corresponding output file.

        Raises:
            RuntimeError: If any output is missing. A detailed error message informs the user which plots
                were not produced
        """
        missing_plots = set()

        for name, target in self.output().items():
            if not target.exists():
                missing_plots.add(Path(name).stem)

        if missing_plots:
            missing_plot_functions = {name: self.registered_plots[name].__name__ for name in missing_plots}
            raise RuntimeError(
                f"PlottingMixin Task '{self.__class__.__name__}' registered {len(self.registered_plots)} "
                f"plots but only generated {len(self.registered_plots) - len(missing_plots)}. "
                f"Missing plots were (name, func):\n{json.dumps(missing_plot_functions, indent=4)}\nEvery plotting "
                "function has to be called inside the `run()` method of your luigi Task."
            )

    def upload_plots_to_webpage(
        self,
        web_dir: str | None,
    ) -> None:
        if not web_dir:
            return None

        name = Path(self.plot_save_dir).stem
        source = gallery.GallerySource(
            name,
            path=self.plot_save_dir,
        )
        gallery.generate(
            web_folder=web_dir,
            sources=[source],
        )
