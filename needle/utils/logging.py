import functools
import logging
import time
import warnings


class TipFilter(logging.Filter):
    """Suppress advertisement for litmodels checkpointing. Will eventually be fixed by the Lightning
    team.

    Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/21294
    """

    def filter(self, record):
        return "💡 Tip" not in record.getMessage()


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(TipFilter())
logging.getLogger("lightning.pytorch.trainer.connectors.accelerator_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.trainer.connectors.signal_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.loggers.mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.WARNING)
warnings.filterwarnings("once", message="The '*' does not have many workers*")


class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages.

    Credit: @meekamunz (via GitHub)
    Source: https://github.com/meekamunz/Now-Playing-Traktor/blob/main/logger_config.py
    """

    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

    @classmethod
    def get_logger(cls, name, level: int | str = logging.INFO) -> logging.Logger:
        """
        Create a new logger with the specified name.

        Format:
            "%(levelname)s: NEEDLE-%(name)s (%(asctime)s) - %(message)s"
        With the time formatter
        """
        new_logger = logging.getLogger(name)

        if not new_logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = cls(
                "%(levelname)s: NEEDLE-%(name)s (%(asctime)s) - %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            new_logger.addHandler(handler)

        new_logger.setLevel(level)
        new_logger.propagate = False
        return new_logger


global _needle_logging_cache
_needle_logging_cache = set()


class LogOnce:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def warn_once(self, message: str) -> None:
        if message not in _needle_logging_cache:
            self.logger.warning(message)
            _needle_logging_cache.add(message)

    def info_once(self, message: str) -> None:
        if message not in _needle_logging_cache:
            self.logger.info(message)
            _needle_logging_cache.add(message)


def timing(func):
    """
    Decorator to time a function's execution. Uses the 'ml' logger to print.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"Function '{func.__module__}.{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper
