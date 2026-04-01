"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from typing import Dict, List, Tuple, TypedDict

import lightning as L
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.selection import filterbyjet
from .nf_model import ConditionalNormalizingFlowModule


class ClassifierSamplesTensorDict(TypedDict):
    x_2j: torch.Tensor
    x_1j: torch.Tensor
    l_2j: torch.Tensor
    l_1j: torch.Tensor


class Dataset1j2j(Dataset):
    """
    Custom Dataset to hold paired 1-jet and 2-jet data samples.

    Each sample is a dictionary containing:
        - 'x_2j': Data for 2-jet events.
        - 'x_1j': Data for 1-jet events.
        - 'l_2j': Labels for 2-jet events.
        - 'l_1j': Labels for 1-jet events.
    """

    def __init__(
        self,
        data_sys_list_2j: List[torch.Tensor],
        data_sys_list_1j: List[torch.Tensor],
        label_list_2j: List[torch.Tensor],
        label_list_1j: List[torch.Tensor],
    ) -> None:
        self.samples: List[ClassifierSamplesTensorDict] = []

        for i in range(len(data_sys_list_2j)):
            self.samples.append(
                {
                    "x_2j": data_sys_list_2j[i],
                    "x_1j": data_sys_list_1j[i],
                    "l_2j": label_list_2j[i],
                    "l_1j": label_list_1j[i],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> ClassifierSamplesTensorDict:
        return self.samples[idx]


def return1j2j(
    alljet_data: Dict,
    models: List[torch.nn.Module],
    cut: bool = False,
    nevents: int = 10,
    device: str = "cpu",
) -> Tuple[torch.Tensor, ...]:
    """
    Process the input data for 1-jet and 2-jet events, apply feature transforms,
    and append normalizing flow (NF) features computed from the given models.

    Parameters:
        alljet_data (dict): Dictionary containing the combined jet data.
        models (list): List of pre-trained models for feature extraction.

    Returns:
        tuple: Data tensors and label tensors for 2-jet and 1-jet events.
    """
    # Process 2-jet events
    filtered_data, filtered_det_labels, _filtered_weights, _feature_names = filterbyjet(2, alljet_data)
    temp_labels = filtered_det_labels.values == "htautau"
    temp_labels = torch.tensor([int(val) for val in temp_labels])
    data_2j = torch.tensor(filtered_data.values)
    label_2j = temp_labels.clone().detach()

    mask = torch.any(data_2j == -25, dim=1)
    data_2j = data_2j[~mask]
    label_2j = label_2j[~mask]

    # Log-transform specified columns for 2-jet events
    log_indices_2j = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]
    for col_idx in range(data_2j.shape[1]):
        if col_idx in log_indices_2j:
            data_2j[:, col_idx] = torch.log(data_2j[:, col_idx])

    # Process 1-jet events
    filtered_data, filtered_det_labels, _filtered_weights, _feature_names = filterbyjet(1, alljet_data)
    temp_labels = filtered_det_labels.values == "htautau"
    temp_labels = torch.tensor([int(val) for val in temp_labels])
    data_1j = torch.tensor(filtered_data.values)
    label_1j = temp_labels.clone().detach()

    mask = torch.any(data_1j == -25, dim=1)
    data_1j = data_1j[~mask]
    label_1j = label_1j[~mask]

    # Log-transform specified columns for 1-jet events
    log_indices_1j = [0, 3, 6, 9, 10, 13, 14, 16, 17]
    for col_idx in range(data_1j.shape[1]):
        if col_idx in log_indices_1j:
            data_1j[:, col_idx] = torch.log(data_1j[:, col_idx])

    if cut:
        data_1j = data_1j[:nevents]
        data_2j = data_2j[:nevents]
        label_2j = label_2j[:nevents]
        label_1j = label_1j[:nevents]

    data_1j = data_1j.to(device)
    data_2j = data_2j.to(device)
    label_1j = label_1j.to(device)
    label_2j = label_2j.to(device)

    # Compute NF features from the provided models
    with torch.no_grad():
        NF_feat_s1j = torch.sigmoid(models[3](data_1j)).to(device).unsqueeze(1)
        NF_feat_b1j = torch.sigmoid(models[0](data_1j)).to(device).unsqueeze(1)
        NF_feat_s1j_3 = torch.sigmoid(models[2](data_1j)).to(device).unsqueeze(1)
        NF_feat_b1j_3 = torch.sigmoid(models[1](data_1j)).to(device).unsqueeze(1)

        NF_feat_s2j = torch.sigmoid(models[7](data_2j)).to(device).unsqueeze(1)
        NF_feat_b2j = torch.sigmoid(models[4](data_2j)).to(device).unsqueeze(1)
        NF_feat_s2j_3 = torch.sigmoid(models[6](data_2j)).to(device).unsqueeze(1)
        NF_feat_b2j_3 = torch.sigmoid(models[5](data_2j)).to(device).unsqueeze(1)

        # Append the NF features to the original data
        data_2j = torch.cat([data_2j, NF_feat_s2j, NF_feat_s2j_3, NF_feat_b2j, NF_feat_b2j_3], dim=1)
        data_1j = torch.cat([data_1j, NF_feat_s1j, NF_feat_s1j_3, NF_feat_b1j, NF_feat_b1j_3], dim=1)

    return data_2j, data_1j, label_2j, label_1j


def load_nf_models(models_paths: str, device: str):
    """
    Load NormalizingFlowModel models from a directory structure.

    Returns:
        A list of loaded models in order (first the 1_jet models, then the 2_jet models).
    """

    models: List[torch.nn.Module] = []

    for ckpt_path in tqdm(checkpoints):
        models.append(
            ConditionalNormalizingFlowModule.load_from_checkpoint(ckpt_path).to(device).eval().to(torch.float32)
        )

    return models


class ClassifierDatamodule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
