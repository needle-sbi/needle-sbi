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

import numpy as np
import torch
from tqdm import tqdm

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..utils.selection import createJetData, return1j2j

logger = Logger("histogram")


class Args(NamedTuple):
    snapshot_path: str
    json_save_path: str
    root_dir: str


def parse_snapshot(filepath: str) -> Tuple[Dict[str, str], Dict[str, str]]:
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


def create_histogram(
    snapshot_path: str,
    json_save_path: str,
    root_dir: str,
):
    """
    Main function to process jet data, generate histograms using a classifier model,
    and save the results to a JSON file.
    """
    device = "cpu"

    nf_ckpts, classifier_ckpt = parse_snapshot(snapshot_path)
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
    # Loop over combinations of test and jet energy scale parameters.
    for j in tqdm(tes_arr, "TES", position=0):
        for i in tqdm(jes_arr, "JES", position=1, leave=False):
            # Define parameter list for data generation.
            n_params = [1, 1, 1, j, i, 0]

            # Create jet data using the provided root directory.
            alljet_data, _ = createJetData(
                "all",
                True,
                set_mu=1000,
                seed=0,
                n_param=n_params,
                useRand=False,
                root_dir=root_dir,
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
    with open(json_save_path + "hist.json", "w") as f:
        json.dump(serializable_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process jet data and generate histograms using the classifier model."
    )
    parser.add_argument(
        "--root-dir", type=str, help="Path to the directory with the FAIR Universe data"
    )
    parser.add_argument(
        "--snapshot-path", type=str, help="Path to the file with the JSON snapshot file"
    )
    parser.add_argument(
        "--json-save-path",
        type=str,
        help="Path to directory where to save the resulting JSON file.",
    )
    args: Args = parser.parse_args()  # type: ignore
    create_histogram(
        args.snapshot_path,
        args.json_save_path,
        args.root_dir,
    )
