""" Based on https://github.com/KIT-CMS/ETPlaw/blob/main/tasks/htcondor/bundle_files.py
"""
import os

import law
import luigi
from law.contrib.git import BundleGitRepository
from law.contrib.tasks import TransferLocalFile
from law.decorator import safe_output
from law.util import human_bytes


class BundleRepoTask(
    BundleGitRepository,
    TransferLocalFile,
):
    replicas = luigi.IntParameter(
        default=1,
        description="number of replicas to generate; default: 1 ",
    )

    exclude_files = ["tmp", "*~", "*.pyc", ".vscode/"]
    task_namespace = None

    @property
    def script_dir(self):
        _path = os.getenv("SCRIPT_DIR")

        if not _path:
            raise ValueError("Base directory '$SCRIPT_DIR' could not be found")

        return _path

    def get_repo_path(self) -> str:  # type: ignore
        """
        Path to the git repository that should be bundled.
        This is required by the BundleGitRepository task.
        """
        return self.script_dir

    def single_output(self) -> law.LocalFileTarget:  # type: ignore
        repo_base = os.path.basename(self.get_repo_path())
        path = os.path.join(self.script_dir, "bundles", f"{repo_base}.{self.checksum}.tgz")
        return law.LocalFileTarget(path, tmp_dir=True)

    def output(self):  # type: ignore
        return TransferLocalFile.output(self)

    @safe_output  # type: ignore
    def run(self) -> None:
        bundle = law.LocalFileTarget(path=os.path.join(self.script_dir, "bundle.tgz"), is_tmp=False)
        self.bundle(bundle)

        self.publish_message(f"Bundled repository archive of size {human_bytes(bundle.stat().st_size, fmt=True)}")

        self.transfer(bundle)
