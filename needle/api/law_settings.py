from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union


def configure_law(
    law_home: Optional[Union[str, Path]] = None,
    law_config_file: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Set ``LAW_HOME``/``LAW_CONFIG_FILE`` from Python instead of ``source setup.sh``.

    This does NOT eliminate ``law.cfg`` as a required file — law's own CLI still
    reads it from ``LAW_CONFIG_FILE``. It only removes the need to source
    ``setup.sh`` in order to point law at that file.

    Args:
        law_home: Project root to use as ``LAW_HOME``. Defaults to the current
            working directory.
        law_config_file: Path to ``law.cfg``. Defaults to ``law.cfg`` under `law_home`.

    Returns:
        Dict[str, str]: the ``LAW_HOME``/``LAW_CONFIG_FILE`` values that were set.

    Raises:
        FileNotFoundError: if the resolved `law.cfg` does not exist.
    """
    root = Path(law_home).resolve() if law_home else Path.cwd()
    cfg = Path(law_config_file).resolve() if law_config_file else root / "law.cfg"

    if not cfg.exists():
        raise FileNotFoundError(f"law.cfg not found at {cfg}; run `needle init` first")

    os.environ["LAW_HOME"] = str(root)
    os.environ["LAW_CONFIG_FILE"] = str(cfg)
    return {"LAW_HOME": str(root), "LAW_CONFIG_FILE": str(cfg)}
