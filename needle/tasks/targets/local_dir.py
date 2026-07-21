from __future__ import annotations

import os

import luigi


class LocalDirectoryTarget:
    """Pure-Python drop-in for ``law.LocalDirectoryTarget`` on shared filesystems.

    Provides the ``child()`` / ``exists()`` / ``makedirs()`` interface used by the
    task ``output()`` methods, without any dependency on the LAW library.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = str(path)

    def child(self, name: str, type: str = "f") -> luigi.LocalTarget:  # noqa: A002
        return luigi.LocalTarget(os.path.join(self.path, name))

    def exists(self) -> bool:
        return os.path.isdir(self.path)

    def makedirs(self) -> None:
        os.makedirs(self.path, exist_ok=True)

    def __repr__(self) -> str:
        return f"LocalDirectoryTarget({self.path!r})"
