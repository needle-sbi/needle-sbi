"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import argparse
import json
from logging import Logger
from typing import Any, Dict, NamedTuple, Tuple
from urllib.parse import parse_qs

import luigi
import numpy as np
import torch
from tqdm import tqdm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.selection import createJetData, return1j2j

logger = Logger("histogram")


class CreateHistogramTask(luigi.Task):
    """Luigi task for generating classifier score histograms from FAIR Universe snapshot data.

    This task loads a saved snapshot that contains trained normalizing flow models and a classifier
    checkpoint, generates synthetic jet data using the FAIR Universe dataset, evaluates the classifier
    on 1-jet and 2-jet events for a grid of JES and TES variations, and writes signal/background
    histograms to a JSON file.

    Args:
        snapshot_path: Path to the snapshot JSON file containing model checkpoint locations.
        json_save_path: Path prefix to the output histogram JSON file.
        root_dir: Root directory containing the FAIR Universe data for data generation.
    """

    snapshot_path: str = luigi.Parameter(description="Path to the snapshot file (.json)")  # type: ignore
    json_save_path: str = luigi.Parameter(description="Path to the output histogram file (.json)")  # type: ignore
    root_dir: str = luigi.Parameter(description="Path to the directory containing the FAIR Universe Data")  # type: ignore

    def output(self) -> luigi.LocalTarget:  # type: ignore
        """Return the Luigi target for the generated histogram file."""
        return luigi.LocalTarget("hist.json")

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
        with open(filepath, "r") as f:
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
                nf_nodes[name] = node["checkpoint_path"]
            elif estimator_name.startswith("classifier"):
                if not classifier_node.get("classifier"):
                    classifier_node["classifier"] = node["checkpoint_path"]
                else:
                    raise ValueError(
                        f"More than one classifier found in snapshot: existing are {list(classifier_node.keys())} and"
                        f" new would be '{estimator_name}'"
                    )
            else:
                logger.warning(f"Unaccounted estimator found in snapshot: {name}")

        return nf_nodes, classifier_node

    def create_histogram(self):
        """Generate histograms from classifier scores and save them to a JSON file.

        This method loads model checkpoints from the snapshot, generates synthetic jet data
        across a grid of JES and TES variations, evaluates classifier scores for 1-jet and 2-jet
        events, computes signal and background histograms, and writes the results to a
        JSON file at the configured output path. Will always run on CPU.
        """
        device = "cpu"

        nf_ckpts, classifier_ckpt = self.parse_snapshot(self.snapshot_path)
        nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device=device)
        classifier = (
            CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"])
            .to(device)
            .eval()
            .to(torch.float32)
        )

        # Define the parameter arrays for jet energy scale (jes_arr) and testing scale (tes_arr).
        jes_arr = np.linspace(0.9, 1.1, 10)
        tes_arr = np.linspace(0.9, 1.1, 10)
        # Define histogram parameters.
        nbins = 200
        bins = np.linspace(0, 1, num=nbins)

        hist_dict_class = {}
        for j in tqdm(tes_arr, "TES", position=0):
            for i in tqdm(jes_arr, "JES", position=1, leave=False):
                # Define parameter list for data generation.
                n_params = [1, 1, 1, j, i, 0]

                # Create jet data using the provided root directory.
                alljet_data, _ = createJetData(  # type: ignore
                    "all",
                    True,
                    set_mu=1000,
                    seed=0,
                    n_param=n_params,
                    useRand=False,
                    root_dir=self.root_dir,
                )
                # Split the data into 2-jet and 1-jet sets and obtain corresponding labels.
                data_2j, data_1j, label_2j, label_1j = return1j2j(
                    alljet_data=alljet_data,
                    models=nf_models,
                    device=device,
                )

                # Obtain classifier scores for each jet type without computing gradients.
                with torch.no_grad():
                    scores_2j = torch.sigmoid(classifier(data_2j, 2)).cpu().numpy()
                    scores_1j = torch.sigmoid(classifier(data_1j, 1)).cpu().numpy()

                total_score = np.concatenate([scores_2j, scores_1j])
                total_label = np.concatenate([label_2j.numpy(), label_1j.numpy()])

                # Compute histograms for signal (label==1) and background (label==0) separately.
                S_hist_class, _ = np.histogram(
                    total_score[total_label == 1], bins=bins, density=True
                )
                BG_hist_class, _ = np.histogram(
                    total_score[total_label == 0], bins=bins, density=True
                )

                hist_dict_class[(i, j)] = [S_hist_class, BG_hist_class]

        serializable_dict = {
            str(key): {"sig": val[0].tolist(), "bg": val[1].tolist()}
            for key, val in hist_dict_class.items()
        }
        with open(self.json_save_path + "hist.json", "w") as f:
            json.dump(serializable_dict, f)
