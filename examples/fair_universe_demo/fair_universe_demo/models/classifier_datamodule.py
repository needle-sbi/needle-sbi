"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

from typing import Dict, List, Optional, TypedDict
from urllib.parse import parse_qs

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
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
        data_sys_list_2j: torch.Tensor,
        data_sys_list_1j: torch.Tensor,
        label_list_2j: torch.Tensor,
        label_list_1j: torch.Tensor,
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
    """Datamodule for classifier training using NF-derived features.

    Loads the parquet dataset, applies NF feature extraction with pretrained
    normalizing flow models, and constructs train/validation splits.
    """

    def __init__(
        self,
        root_dir: str,
        input_models: Dict[str, str],
        n_folds: int = -1,
        fold_index: int = -1,
        parquet_filename: str = "FAIR_Universe_HiggsML_data.parquet",
        metadata_filename: str = "FAIR_Universe_HiggsML_data_metadata.json",
        batch_size: int = 1000,
        device: str = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.input_models_dict = input_models
        self.n_folds = n_folds
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.parquet_filename = parquet_filename
        self.metadata_filename = metadata_filename
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self, stage: Optional[str]) -> None:
        """Load NF models and prepare classifier training and validation data.

        Args:
            stage (Optional[str]): Optional stage name, not used.

        Side effects:
            Sets `self.input_models`, `self.train_dataset`, and `self.val_dataset`.

        Raises:
            ValueError: When the expected eight NF models are not found.
            KeyError: When a required model key is missing during feature extraction.
        """
        self.input_models = self.load_nf_models(self.input_models_dict)

        if len(self.input_models) != 8:
            raise ValueError(f"Expected to load exactly eight models but found {len(self.input_models)}")

        j2_data, j2_detlabel, _, _ = createMultiJetMultiNuanData(
            jet_num=2,
            useTestData=False,
            seed=0,
            root_dir=self.root_dir,
            parquet_filename=self.parquet_filename,
            metadata_filename=self.metadata_filename,
        )
        j1_data, j1_detlabel, _, _ = createMultiJetMultiNuanData(
            jet_num=1,
            useTestData=False,
            seed=0,
            root_dir=self.root_dir,
            parquet_filename=self.parquet_filename,
            metadata_filename=self.metadata_filename,
        )
        j2_data = j2_data.to(self.device)
        j2_detlabel = j2_detlabel.to(self.device)
        j1_data = j1_data.to(self.device)
        j1_detlabel = j1_detlabel.to(self.device)
        self.input_models.to(self.device)

        # Extract features from the loaded models. For 1-jet models, indices 0-3 are used.
        # For 2-jet models, indices 4-7 are used.
        with torch.no_grad():
            try:
                # fmt: off
                NF_s1j_0p5 = torch.sigmoid(self.input_models["nf_signal_1jet&c_0p5"](j1_data)).unsqueeze(1)
                NF_b1j_0p5 = torch.sigmoid(self.input_models["nf_background_1jet&c_0p5"](j1_data)).unsqueeze(1)
                NF_s1j_2p0 = torch.sigmoid(self.input_models["nf_signal_1jet&c_2p0"](j1_data)).unsqueeze(1)
                NF_b1j_2p0 = torch.sigmoid(self.input_models["nf_background_1jet&c_2p0"](j1_data)).unsqueeze(1)
                NF_s2j_0p5 = torch.sigmoid(self.input_models["nf_signal_2jet&c_0p5"](j2_data)).unsqueeze(1)
                NF_b2j_0p5 = torch.sigmoid(self.input_models["nf_background_2jet&c_0p5"](j2_data)).unsqueeze(1)
                NF_s2j_2p0 = torch.sigmoid(self.input_models["nf_signal_2jet&c_2p0"](j2_data)).unsqueeze(1)
                NF_b2j_2p0 = torch.sigmoid(self.input_models["nf_background_2jet&c_2p0"](j2_data)).unsqueeze(1)
                # fmt: on
            except KeyError as e:
                raise KeyError(f"No key `{e}` found in model Dict. Available keys are {self.input_models.keys()}")

            # Append the Normalizing Flow features to the original data.
            j1_data = torch.cat([j1_data, NF_s1j_0p5, NF_s1j_2p0, NF_b1j_0p5, NF_b1j_2p0], dim=1)
            j2_data = torch.cat([j2_data, NF_s2j_0p5, NF_s2j_2p0, NF_b2j_0p5, NF_b2j_2p0], dim=1)

        max_shape = min(len(j1_data), len(j2_data))
        print(f"Number of data points used: {max_shape}")
        j1_data = j1_data[:max_shape]
        j2_data = j2_data[:max_shape]
        j1_detlabel = j1_detlabel[:max_shape]
        j2_detlabel = j2_detlabel[:max_shape]

        all_jet_dataset = Dataset1j2j(j2_data, j1_data, j2_detlabel, j1_detlabel)

        # Split the dataset into training and validation sets.
        n_val = int(0.1 * len(all_jet_dataset))
        n_train = len(all_jet_dataset) - n_val
        self.train_dataset, self.val_dataset = random_split(all_jet_dataset, [n_train, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    @staticmethod
    def load_nf_models(input_models: Dict[str, str]) -> torch.nn.ModuleDict:
        """Load ConditionalNormalizingFlowModule checkpoints from provided paths.

        Args:
            input_models (Dict[str, str]): Mapping from model identifiers to checkpoint paths.

        Returns:
            torch.nn.ModuleDict: Loaded NF models keyed by `est&syst` strings.

        Important:
            The way this is done here implies that the dict keys are directly tied to the value of the hyperparameter c.
            This should be changed so that arbitrary values of c are valid. However, the ordering is important, so you
            cannot rely on the list of input models to be properly sorted.
        """

        models = torch.nn.ModuleDict()

        for name, ckpt_path in tqdm(input_models.items(), desc="Loading NF models", leave=False):
            name_dict = parse_qs(name)
            prefix = name_dict["est"][0]
            suffix = name_dict["syst"][0].replace(".", "p")
            key = f"{prefix}&{suffix}"
            model = ConditionalNormalizingFlowModule.load_from_checkpoint(ckpt_path)
            models[key] = model

        if not list(models.keys()):
            raise ValueError(f"No valid models found in the input Dict: {input_models}")

        models = models.eval().to(torch.float32)
        return models
