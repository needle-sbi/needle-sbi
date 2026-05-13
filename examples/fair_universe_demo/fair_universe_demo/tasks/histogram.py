"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cached_property
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs

import luigi
import numpy as np
import torch
from tqdm import tqdm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.selection import Data, createJetData, load_train_set_data, return1j2j

logger = Logger("histogram")
device = "cuda" if torch.cuda.is_available() else "cpu"


def init_worker(data: Data):
    global shared_data
    shared_data = data


def _compute_histogram_entry(args):
    """Worker function for a single (jes, tes) combination."""
    tqdm.disable = True
    i, j, bins, snapshot_path = args
    n_params = [1, 1, 1, j, i, 0]

    nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(snapshot_path)
    nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device).eval().to(torch.float32)
    classifier = CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])
    classifier = classifier.to(device).eval().to(torch.float32)

    alljet_data, _ = createJetData(
        jet_num="all",
        useTestData=True,
        set_mu=1,
        seed=0,
        n_param=n_params,
        useRand=False,
        loaded_data=shared_data,
    )
    data_2j, data_1j, label_2j, label_1j = return1j2j(
        alljet_data=alljet_data,
        models=nf_models,
        device=device,
    )
    with torch.no_grad():
        scores_2j = torch.sigmoid(classifier(data_2j, 2)).cpu().numpy()
        scores_1j = torch.sigmoid(classifier(data_1j, 1)).cpu().numpy()

    total_score = np.concatenate([scores_2j, scores_1j])
    total_label = np.concatenate([label_2j.cpu().numpy(), label_1j.cpu().numpy()])

    S_hist, _ = np.histogram(total_score[total_label == 1], bins=bins, density=True)
    BG_hist, _ = np.histogram(total_score[total_label == 0], bins=bins, density=True)
    return (i, j), S_hist, BG_hist


class HistogramTask(luigi.Task):
    """Luigi task for generating classifier score histograms from FAIR Universe snapshot data.

    This task loads a saved snapshot that contains trained normalizing flow models and a classifier
    checkpoint, generates synthetic jet data using the FAIR Universe dataset, evaluates the classifier
    on 1-jet and 2-jet events for a grid of JES and TES variations, and writes signal/background
    histograms to a JSON file.

    Args:
        snapshot_path: Path to the snapshot JSON file containing model checkpoint locations.
        json_save_path: Path to the output histogram JSON file.
        root_dir: Root directory containing the FAIR Universe data for data generation.
    """

    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    json_save_path: str = luigi.Parameter(description="Path to the output histogram file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore
    bins: int = luigi.IntParameter(description="Number of entries to use per dimension", default=200)  # type: ignore

    @cached_property
    def loaded_data(self):
        return load_train_set_data(self.root_dir)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.json_save_path.endswith(".json") and not os.path.exists(Path(self.json_save_path).parent):
            raise FileNotFoundError(
                f"Argument `json_save_path`='{self.json_save_path}' must point to a valid .json file"
            )

    def output(self) -> luigi.LocalTarget:  # type: ignore
        """Return the Luigi target for the generated histogram file."""
        return luigi.LocalTarget(self.json_save_path)

    @staticmethod
    def parse_snapshot(filepath: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Parse a snapshot JSON file and extract model checkpoint paths.

        The snapshot is expected to contain nodes with query-string-style names. Normalizing flow
        checkpoints are collected separately from the classifier checkpoint.

        Args:
            filepath: Path to the snapshot JSON file.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: A tuple containing the normalizing flow checkpoint
            mapping and a dictionary with the classifier checkpoint path.
        """
        snapshot_path = Path(filepath)

        def resolve_checkpoint_path(checkpoint_path: str) -> str:
            path = Path(checkpoint_path)
            if path.exists():
                return str(path)

            # Snapshots can be moved between machines while preserving the run
            # directory layout. Rebase stale absolute paths onto this snapshot.
            parts = path.parts
            if snapshot_path.parent.name in parts:
                run_dir_idx = parts.index(snapshot_path.parent.name)
                candidate = snapshot_path.parent.joinpath(*parts[run_dir_idx + 1 :])
                if candidate.exists():
                    return str(candidate)

            return checkpoint_path

        with open(snapshot_path, "r") as f:
            snapshot = json.load(f)

        nodes: Dict[str, Any] = snapshot["nodes"]

        if not nodes:
            raise ValueError(f"Snapshot file does not contain any nodes: {nodes}")

        nf_nodes: Dict[str, Any] = {}
        classifier_node: Dict[str, Any] = {"classifier": None}

        for name, node in nodes.items():
            name_dict: Dict[str, Any] = parse_qs(name)
            estimator_name: str = name_dict["est"][0]

            if estimator_name.startswith("nf"):
                nf_nodes[name] = resolve_checkpoint_path(node["checkpoint_path"])
            elif estimator_name.startswith("classifier"):
                if not classifier_node.get("classifier"):
                    classifier_node["classifier"] = resolve_checkpoint_path(node["checkpoint_path"])
                else:
                    raise ValueError(
                        f"More than one classifier found in snapshot: existing are {list(classifier_node.keys())} and"
                        f" new would be '{estimator_name}'"
                    )
            else:
                logger.warning(f"Unaccounted estimator found in snapshot: {name}")

        return nf_nodes, classifier_node

    def create_histogram(self):
        jes_arr = np.linspace(0.9, 1.1, 10)
        tes_arr = np.linspace(0.9, 1.1, 10)

        bins = np.linspace(0, 1, num=self.bins)
        data = load_train_set_data(root_dir=self.root_dir)
        args_list = [(i, j, bins, self.snapshot_path) for j in tes_arr for i in jes_arr]

        hist_dict_class = {}
        futures = []
        progress_bar = tqdm(total=len(args_list), desc="Histogram entries")

        with ProcessPoolExecutor(
            max_workers=5,
            initializer=init_worker,
            initargs=(data,),
        ) as executor:
            for args in args_list:
                futures.append(executor.submit(_compute_histogram_entry, args))
                progress_bar.update(1)

            for future in as_completed(futures):
                key, S_hist, BG_hist = future.result()
                hist_dict_class[key] = [S_hist, BG_hist]

        serializable_dict = {
            str(key): {"sig": val[0].tolist(), "bg": val[1].tolist()} for key, val in hist_dict_class.items()
        }
        with open(self.output().path, "w") as f:
            json.dump(serializable_dict, f)

    @timing
    def run(self) -> None:
        self.create_histogram()
