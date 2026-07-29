"""b2luigi workflow configuration helpers.

Provides ``configure_b2luigi()`` as the programmatic alternative to a ``settings.json``
file.  Users can call this before ``b2luigi.process()`` to set batch system, result
directory, and environment script.

Alternatively, create ``settings.json`` at the project root and b2luigi will read it
automatically:

.. code-block:: json

    {
      "batch_system": "htcondor",
      "htcondor_settings": {
        "request_memory": "2048MB",
        "request_cpus": 2,
        "+RequestRuntime": 600
      }
    }

Supported batch systems: ``"local"``, ``"htcondor"``, ``"slurm"``, ``"lsf"``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_project_root() -> str:
    """Return the project root directory.

    Uses ``$SCRIPT_DIR`` when set (exported by ``setup.sh``), falling back to ``cwd``.
    """
    script_dir = os.getenv("SCRIPT_DIR")
    return script_dir if script_dir else str(Path.cwd())


def configure_b2luigi(
    batch_system: str = "local",
    env_script: str | None = None,
    **kwargs: Any,
) -> None:
    """Configure the b2luigi runtime settings programmatically.

    Equivalent to placing the same values in ``settings.json`` at the project root.
    Call this before ``b2luigi.process()``. The ``results_dir```global b2luigi parameter is unused.

    Args:
        batch_system: One of ``"local"``, ``"htcondor"``, ``"slurm"``, ``"lsf"``.
        env_script: Path to the environment setup script (default: ``setup.sh`` in
            the project root). Passed to batch workers so the Python environment is
            available when running remote jobs.
        **kwargs: Additional b2luigi settings forwarded verbatim to
            ``b2luigi.set_setting()``.  Useful for ``htcondor_settings``,
            ``slurm_settings``, etc.
    """
    import b2luigi

    project_root = get_project_root()

    b2luigi.set_setting("batch_system", batch_system)
    b2luigi.set_setting("working_dir", project_root)

    resolved_env_script = env_script or str(Path(project_root) / "setup.sh")
    b2luigi.set_setting("env_script", resolved_env_script)

    for key, value in kwargs.items():
        b2luigi.set_setting(key, value)
