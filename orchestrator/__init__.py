import logging


class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages.
    Credit: https://github.com/meekamunz/Now-Playing-Traktor/blob/c34923df35a6cb5d7fa4ea40606a67973951713c/logger_config.py
    """
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

    @classmethod
    def get_new_logger(cls, name, level: int = logging.INFO) -> logging.Logger:
        """
        Create a new logger with the specified name.

        Format:
            "%(levelname)s: %(name)s (%(asctime)s) - %(message)s"
        With the time formatter
        """
        new_logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = cls(
            "%(levelname)s: %(name)s (%(asctime)s) - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)
        new_logger.setLevel(level)
        return new_logger


logger = ColorFormatter.get_new_logger("orchestrator")
