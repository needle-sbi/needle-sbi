"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from typing import Dict, List, Optional, TypedDict
from urllib.parse import parse_qs

import lightning as L
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.selection import createMultiJetMultiNuanData
from .nf_model import ConditionalNormalizingFlowModule


class ClassifierSamplesTensorDict(TypedDict):
    x_2j: torch.Tensor
    x_1j: torch.Tensor
    l_2j: torch.Tensor
    l_1j: torch.Tensor


class Dataset1j2j(Dataset):
    """Custom Dataset to hold paired 1-jet and 2-jet data samples."""

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


class ClassifierDatamodule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        input_models: Dict[str, str],
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.input_models_dict = input_models

    def setup(self, stage: Optional[str]) -> None:
        self.input_models = self.load_nf_models(self.input_models_dict)

        if len(self.input_models) != 8:
            raise ValueError(f"Expected to load exactly eight models but found {len(self.input_models)}")

        j2_data, j2_detlabel, _, _ = createMultiJetMultiNuanData(
            2,
            False,
            seed=0,
            root_dir=self.root_dir,
        )
        j1_data, j1_detlabel, _, _ = createMultiJetMultiNuanData(
            1,
            False,
            seed=0,
            root_dir=self.root_dir,
        )
        j2_data = j2_data.cpu()
        j2_detlabel = j2_detlabel.cpu()
        j1_data = j1_data.cpu()
        j1_detlabel = j1_detlabel.cpu()

        # Extract features from the loaded models. For 1-jet models, indices 0-3 are used.
        # For 2-jet models, indices 4-7 are used.
        raise RuntimeError(f"DEBUG {self.input_models.keys()=}")  # TODO
        with torch.no_grad():
            # Process 1-jet data.
            NF_feat_s1j = torch.sigmoid(self.input_models["nf_signal_1jet"](j1_data)).cpu().unsqueeze(1)
            NF_feat_b1j = torch.sigmoid(self.input_models["nf_background_1jet"](j1_data)).cpu().unsqueeze(1)
            NF_feat_s1j_3 = torch.sigmoid(self.input_models[""](j1_data)).cpu().unsqueeze(1)
            NF_feat_b1j_3 = torch.sigmoid(self.input_models[3](j1_data)).cpu().unsqueeze(1)

            # Process 2-jet data.
            NF_feat_s2j = torch.sigmoid(self.input_models[4](j2_data)).cpu().unsqueeze(1)
            NF_feat_b2j = torch.sigmoid(self.input_models[5](j2_data)).cpu().unsqueeze(1)
            NF_feat_s2j_3 = torch.sigmoid(self.input_models[6](j2_data)).cpu().unsqueeze(1)
            NF_feat_b2j_3 = torch.sigmoid(self.input_models[7](j2_data)).cpu().unsqueeze(1)

            # Append the Normalizing Flow features to the original data.
            j1_data = torch.cat([j1_data, NF_feat_s1j, NF_feat_s1j_3, NF_feat_b1j, NF_feat_b1j_3], dim=1)
            j2_data = torch.cat([j2_data, NF_feat_s2j, NF_feat_s2j_3, NF_feat_b2j, NF_feat_b2j_3], dim=1)

    @staticmethod
    def load_nf_models(input_models: Dict[str, str]):
        """
        Load ConditionalNormalizingFlowModule models from the input_models Dict

        Returns:
            ModuleDict
        """

        models = torch.nn.ModuleDict()

        for name, ckpt_path in tqdm(input_models.items()):
            key = str(parse_qs(name)["est"])
            model = ConditionalNormalizingFlowModule.load_from_checkpoint(ckpt_path)
            models[key] = model

        models = models.eval().to(torch.float32)
        return models
