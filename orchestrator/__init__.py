import logging


logger = logging.getLogger("orchestrator")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(levelname)s: needle-orchestrator (%(asctime)s) - %(message)s",
    datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
