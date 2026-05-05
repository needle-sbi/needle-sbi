import logging
from typing import Any, Dict, List, Literal, Tuple, overload

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import Data
from .systematics import get_bootstrapped_dataset, systematics

logger = logging.getLogger(__name__)


def filterbyjet(jet_num: int, data_vis: Dict[str, Any] | pd.DataFrame):
    """
    Filter the dataset based on the number of jets.

    Parameters:
        jet_num (int): The jet count to filter on (0, 1, or 2).
        data_vis (dict): Dictionary with keys "data", "detailed_labels", "weights", "labels".

    Returns:
        tuple: (filtered_data, filtered_det_labels, filtered_weights, feature_names)
    """
    if jet_num == 2:
        # Filter rows with PRI_n_jets >= 2
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] >= jet_num]  # Unused in this branch

        # Drop columns containing 'PRI_n_jets' and those with zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)

    elif jet_num == 1:
        # Filter rows with exactly 1 jet
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] == jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] == jet_num]
        logger.debug(f"{filtered_det_labels.shape}")
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] == jet_num]  # Unused variable
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] == jet_num]

        # Drop columns with 'PRI_n_jets', 'subleading', or zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "subleading" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)

    elif jet_num == 0:
        # Filter rows with exactly 0 jets
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] == jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] == jet_num]
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] == jet_num]  # Unused variable
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] == jet_num]

        # Drop columns with 'PRI_n_jets', 'jet', 'subleading', or zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "jet" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "subleading" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)
    else:
        raise ValueError(f"Variable `jet_num`={jet_num} is out of bounds (accepted are 0, 1 and 2)")

    return filtered_data, filtered_det_labels, filtered_weights, feature_names


def load_train_set_data(
    root_dir: str,
    parquet_filename: str = "FAIR_Universe_HiggsML_data.parquet",
    metadata_filename: str = "FAIR_Universe_HiggsML_data_metadata.json",
) -> Data:
    """Eagerly load the dataset from disk

    Args:
        root_dir (str): _description_
        parquet_filename (str, optional): Name of the data file. Defaults to "FAIR_Universe_HiggsML_data.parquet".
        metadata_filename (str, optional): Name of the metadata file. Defaults to "FAIR_Universe_HiggsML_data_metadata.json".

    Returns:
        Data: The dataset with train and test partitions loaded into memory
    """
    data = Data(
        input_dir=root_dir,
        parquet_filename=parquet_filename,
        metadata_filename=metadata_filename,
    )
    data.load_train_set()
    data.load_test_set()
    data.print_dataset_info()
    return data


@overload
def createJetData(
    jet_num: str,  # Case where 'jet_num' == "all"
    useTestData: bool,
    *,
    root_dir: str = None,
    loaded_data: Data = None,
    set_mu: float = 3,
    seed: int = 0,
    n_param: List[float | None] = None,
    useRand: bool = False,
) -> Tuple[pd.DataFrame | Dict, pd.Series]:
    ...


@overload
def createJetData(
    jet_num: int,
    useTestData: bool,
    *,
    root_dir: str = None,
    loaded_data: Data = None,
    set_mu: float = 3,
    seed: int = 0,
    n_param: List[float | None] = None,
    useRand: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List]:
    ...


def createJetData(
    jet_num: int | str,
    useTestData: bool,
    *,
    root_dir: str = None,
    loaded_data: Data = None,
    set_mu: float = 3,
    seed: int = 0,
    n_param: List[float | None] = None,
    useRand: bool = False,
) -> Tuple[pd.DataFrame | Dict, pd.Series] | Tuple[torch.Tensor, torch.Tensor, np.ndarray, List]:
    """
    Create jet data with optional systematic variations and data processing.

    Parameters:
        jet_num (int or str): Jet number to filter (or "all" to return full dataset).
        useTestData (bool): Whether to use test data.
        root_dir (str, optional): Path to the FAIR Universe Data directory.
        loaded_data (Data): Data with train and test set loaded for more efficient dataloading if using
            this function in a loop.
        set_mu (float, optional): Mu parameter for bootstrapping. Defaults to 3.
        seed (int, optional): Random seed. Defaults to 0.
        n_param (list, optional): List of systematic parameters. Defaults to [1,1,1,1,1,0]. Order is
            [ttbar_scale, diboson_scale, bkg_scale, TES, JES, soft MET]
        useRand (bool, optional): Whether to apply random systematic shifts. Defaults to False.

    Returns:
        Returns a Tensor for the first item in the Tuple if 'useTestData' is False, otherwise return
        a Dict compatible with 'return1j2j' function call.
    """
    if not n_param:
        n_param = [1, 1, 1, 1, 1, 0]

    if loaded_data:
        data = loaded_data
    else:
        if not root_dir:
            raise ValueError(f"Arg `root_dir` is empty: {root_dir}")

        data = load_train_set_data(root_dir=root_dir)

    # Optionally apply random systematic shifts
    if useRand:
        random_state = np.random.RandomState(seed)
        n_param[-3] = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        n_param[-2] = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        n_param[-1] = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)
        logger.debug(f"Number of parameters: {n_param}")

    # Get the test set (assumed to be defined in a global 'data' object)
    test_set = data.get_test_set()

    # Create a pseudo-experimental dataset with bootstrapping
    pseudo_exp_data = get_bootstrapped_dataset(
        test_set,
        mu=set_mu,
        ttbar_scale=n_param[0],
        diboson_scale=n_param[1],
        bkg_scale=n_param[2],
        seed=seed,
    )
    # Prepare weights and detailed labels
    weights = np.ones(pseudo_exp_data.shape[0])
    detailed_labels = pseudo_exp_data["Label"]
    pseudo_exp_data.drop(columns="Label", inplace=True)
    labels = detailed_labels[detailed_labels == "htautau"]

    # Apply systematics to the pseudo-experimental data
    data_vis = systematics(
        data_set={
            "data": pseudo_exp_data,
            "weights": weights,
            "detailed_labels": detailed_labels,
            "labels": labels,
        },
        tes=n_param[3],
        jes=n_param[4],
        soft_met=n_param[5],
    )

    # If jet_num is not "all", filter by jet number
    if jet_num != "all":
        (
            filtered_data,
            filtered_det_labels,
            filtered_weights,
            feature_names,
        ) = filterbyjet(jet_num, data_vis)
        temp_labels = filtered_det_labels.values == "htautau"
        temp_labels = torch.tensor([int(val) for val in temp_labels])
    else:
        return data_vis, detailed_labels

    if not useTestData:
        # Compute background ratios relative to non-signal events
        ratio_ztt = len(filtered_data[filtered_det_labels == "ztautau"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )
        ratio_ttbar = len(filtered_data[filtered_det_labels == "ttbar"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )
        ratio_diboson = len(filtered_data[filtered_det_labels == "diboson"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )

        # Get the training set and limit the number of events
        data_vis_train = data.get_train_set()
        MAX_NUM_EVENTS = 5000000
        for key in data_vis_train.keys():
            if key != "settings":
                try:
                    subset = data_vis_train[key]
                    subset = subset.iloc[:MAX_NUM_EVENTS].reset_index(drop=True)
                    data_vis_train[key] = subset
                except KeyError:
                    data_vis_train[key] = data_vis_train[key][:MAX_NUM_EVENTS]

        # Apply systematics to the training data
        data_vis = systematics(
            data_set={
                "data": data_vis_train,
                "weights": data_vis_train["weights"],
                "detailed_labels": data_vis_train["detailed_labels"],
                "labels": data_vis_train["labels"],
            },
            tes=n_param[3],
            jes=n_param[4],
            soft_met=n_param[5],
        )

        (
            filtered_data,
            filtered_det_labels,
            filtered_weights,
            feature_names,
        ) = filterbyjet(jet_num, data_vis)

        # Determine event counts based on computed ratios
        count_ztt = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ztt)
        count_ttbar = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ttbar)
        count_diboson = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_diboson)

        # Create balanced datasets for signal and backgrounds
        temp_labels = []
        signal_data = filtered_data[filtered_det_labels == "htautau"]
        temp_labels.extend([1] * len(signal_data))
        ztt_data = filtered_data[filtered_det_labels == "ztautau"][:count_ztt]
        temp_labels.extend([0] * len(ztt_data))
        ttbar_data = filtered_data[filtered_det_labels == "ttbar"][:count_ttbar]
        temp_labels.extend([0] * len(ttbar_data))
        diboson_data = filtered_data[filtered_det_labels == "diboson"][:count_diboson]
        temp_labels.extend([0] * len(diboson_data))

        # Concatenate the subsets
        filtered_data = pd.concat((signal_data, ztt_data, ttbar_data, diboson_data), ignore_index=True)

    # Convert to torch tensors
    filtered_data = torch.tensor(filtered_data.values)
    filtered_det_labels = torch.tensor(temp_labels)

    # Remove any rows with a value of -25
    mask = torch.any(filtered_data == -25, dim=1)
    filtered_data = filtered_data[~mask]
    filtered_det_labels = filtered_det_labels[~mask]

    # Determine columns on which to apply logarithm transform based on jet type
    if jet_num == 1:
        log_columns = [0, 3, 6, 9, 10, 13, 14, 16, 17]
    elif jet_num == 0:
        log_columns = [0, 3, 6, 9, 10, 12, 13]
    else:
        log_columns = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]

    for col_idx in range(filtered_data.shape[1]):
        if col_idx in log_columns:
            filtered_data[:, col_idx] = torch.log(filtered_data[:, col_idx])

    return filtered_data, filtered_det_labels, filtered_weights, feature_names


def createMultiJetMultiNuanData(
    root_dir: str,
    jet_num: Literal[0, 1, 2],
    useTestData: bool,
    parquet_filename: str = "FAIR_Universe_HiggsML_data.parquet",
    metadata_filename: str = "FAIR_Universe_HiggsML_data_metadata.json",
    set_mu: int = 3,
    seed: int = 0,
    n_param: List[int] = None,
):
    """
    Create multi-jet multi-nuisance data by processing multiple sub-datasets.

    Parameters:
        jet_num (int): The jet number to filter.
        useTestData (bool): Whether to use test data.
        set_mu (int, optional): Mu parameter for bootstrapping. Defaults to 3.
        seed (int, optional): Random seed. Defaults to 0.
        n_param (list, optional): List of systematic parameters. Defaults to [1,1,1,1,1,0].

    Returns:
        tuple[torch.Tensor, torch.Tensor, np.ndarray, list]:
            Processed data tensor, label tensor, weights, and feature names.
    """
    if not n_param:
        n_param = [1, 1, 1, 1, 1, 0]

    data = Data(
        input_dir=root_dir,
        parquet_filename=parquet_filename,
        metadata_filename=metadata_filename,
    )
    data.load_train_set()
    data.load_test_set()

    random_state = np.random.RandomState(seed)
    test_set = data.get_test_set()

    # Create a pseudo-experimental dataset using bootstrapping
    pseudo_exp_data = get_bootstrapped_dataset(
        test_set,
        mu=set_mu,
        ttbar_scale=n_param[0],
        diboson_scale=n_param[1],
        bkg_scale=n_param[2],
        seed=seed,
    )

    weights = np.ones(pseudo_exp_data.shape[0])
    detailed_labels = pseudo_exp_data["Label"]
    pseudo_exp_data.drop(columns="Label", inplace=True)
    labels = detailed_labels[detailed_labels == "htautau"]

    print("det lab")
    print(detailed_labels)

    # Apply systematics to the pseudo-experimental data
    data_vis = systematics(
        data_set={
            "data": pseudo_exp_data,
            "weights": weights,
            "detailed_labels": detailed_labels,
            "labels": labels,
        },
        tes=n_param[3],
        jes=n_param[4],
        soft_met=n_param[5],
    )

    filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(jet_num, data_vis)
    temp_labels = filtered_det_labels.values == "htautau"
    temp_labels = torch.tensor([int(val) for val in temp_labels])

    if not useTestData:
        # Compute background ratios relative to non-signal events
        ratio_ztt = len(filtered_data[filtered_det_labels == "ztautau"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )
        ratio_ttbar = len(filtered_data[filtered_det_labels == "ttbar"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )
        ratio_diboson = len(filtered_data[filtered_det_labels == "diboson"]) / len(
            filtered_data[filtered_det_labels != "htautau"]
        )

        data_vis_train = data.get_train_set()
        sub_dataset = []
        sub_labels = []
        MAX_SUB_EVENTS = 10000  # Subset size per iteration

        data_vis = {
            "data": data_vis_train,
            "weights": data_vis_train["weights"],
            "detailed_labels": data_vis_train["detailed_labels"],
            "labels": data_vis_train["labels"],
        }

        num_subdatasets: int = 499

        for i in tqdm(range(num_subdatasets), total=num_subdatasets):
            # Create a copy for the sub-dataset
            data_vis_sub = data_vis.copy()

            for key in data_vis.keys():
                if key != "settings":
                    try:
                        temp_df = data_vis_sub[key]
                        temp_df = temp_df.iloc[MAX_SUB_EVENTS * i : MAX_SUB_EVENTS * (i + 1)].reset_index(drop=True)
                        data_vis_sub[key] = temp_df
                    except Exception:
                        data_vis_sub[key] = data_vis_sub[key][MAX_SUB_EVENTS * i : MAX_SUB_EVENTS * (i + 1)]

            if data_vis_sub["data"].empty:
                # Case where len(data_vis_sub["data"]) < MAX_SUB_EVENTS (e.g. dataset smaller then subset)
                break

            # Apply random systematic shifts for this subset
            tes_val = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            jes_val = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            soft_met_val = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)

            data_vis_sub_sys = systematics(
                data_set=data_vis_sub,
                tes=tes_val,
                jes=jes_val,
                soft_met=soft_met_val,
                dopostprocess=False,
            )

            (
                filtered_data,
                filtered_det_labels,
                filtered_weights,
                feature_names,
            ) = filterbyjet(jet_num, data_vis_sub_sys)

            count_ztt = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ztt)
            count_ttbar = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ttbar)
            count_diboson = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_diboson)

            temp_labels = []
            signal_data = filtered_data[filtered_det_labels == "htautau"]
            temp_labels.extend([1] * len(signal_data))
            ztt_data = filtered_data[filtered_det_labels == "ztautau"][:count_ztt]
            temp_labels.extend([0] * len(ztt_data))
            ttbar_data = filtered_data[filtered_det_labels == "ttbar"][:count_ttbar]
            temp_labels.extend([0] * len(ttbar_data))
            diboson_data = filtered_data[filtered_det_labels == "diboson"][:count_diboson]
            temp_labels.extend([0] * len(diboson_data))

            filtered_data = pd.concat((signal_data, ztt_data, ttbar_data, diboson_data), ignore_index=True)
            filtered_data = torch.tensor(filtered_data.values)
            filtered_det_labels = torch.tensor(temp_labels)

            mask = torch.any(filtered_data == -25, dim=1)
            filtered_data = filtered_data[~mask]
            filtered_det_labels = filtered_det_labels[~mask]

            # Determine columns for logarithm transform based on jet type
            if jet_num == 1:
                log_columns = [0, 3, 6, 9, 10, 13, 14, 16, 17]
            elif jet_num == 0:
                log_columns = [0, 3, 6, 9, 10, 12, 13]
            else:
                log_columns = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]

            for col_idx in range(filtered_data.shape[1]):
                if col_idx in log_columns:
                    filtered_data[:, col_idx] = torch.log(filtered_data[:, col_idx])

            sub_dataset.append(filtered_data)
            sub_labels.append(filtered_det_labels)

        # Concatenate all sub-datasets
        filtered_data = torch.cat(sub_dataset)
        filtered_det_labels = torch.cat(sub_labels)

    return filtered_data, filtered_det_labels, filtered_weights, feature_names


class Dataset1j2j(Dataset):
    """
    Custom Dataset to hold paired 1-jet and 2-jet data samples.

    Each sample is a dictionary containing:
        - 'x_2j': Data for 2-jet events.
        - 'x_1j': Data for 1-jet events.
        - 'l_2j': Labels for 2-jet events.
        - 'l_1j': Labels for 1-jet events.
    """

    def __init__(self, data_sys_list_2j, data_sys_list_1j, label_list_2j, label_list_1j):
        self.samples = []
        for i in range(len(data_sys_list_2j)):
            self.samples.append(
                {
                    "x_2j": data_sys_list_2j[i],
                    "x_1j": data_sys_list_1j[i],
                    "l_2j": label_list_2j[i],
                    "l_1j": label_list_1j[i],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def return1j2j(
    alljet_data: Dict[str, torch.Tensor] | pd.DataFrame,
    models: torch.nn.ModuleDict,
    nevents: int = -1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, ...]:
    """
    Process the input data for 1-jet and 2-jet events, apply feature transforms,
    and append normalizing flow (NF) features computed from the given models.

    Parameters:
        alljet_data (Dict[str, torch.Tensor]): Dictionary containing the combined jet data.
        models (torch.nn.ModuleDict): Dictionary of pre-trained models for NF feature extraction.
            Expected keys are:
             - nf_signal_1jet&c_0p5
             - nf_background_1jet&c_0p5
             - nf_signal_1jet&c_2p0
             - nf_background_1jet&c_2p0
             - nf_signal_2jet&c_0p5
             - nf_background_2jet&c_0p5
             - nf_signal_2jet&c_2p0
             - nf_background_2jet&c_2p0.
        cut (bool, optional): Whether to limit the number of events. Defaults to False.
        nevents (int, optional): Number of events to use when cut is True. Defaults to 10.
        device (str, optional): Device to move tensors to. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - 2-jet data with NF features appended.
            - 1-jet data with NF features appended.
            - 2-jet labels (binary: 1 for htautau, 0 otherwise).
            - 1-jet labels (binary: 1 for htautau, 0 otherwise).
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

    if nevents > 0:
        data_1j = data_1j[:nevents]
        data_2j = data_2j[:nevents]
        label_2j = label_2j[:nevents]
        label_1j = label_1j[:nevents]

    data_1j = data_1j.to(device).to(torch.float32)
    data_2j = data_2j.to(device).to(torch.float32)
    label_1j = label_1j.to(device)
    label_2j = label_2j.to(device)
    models.to(device).eval().to(torch.float32)

    with torch.no_grad():
        try:
            NF_s1j_0p5 = torch.sigmoid(models["nf_signal_1jet&c_0p5"](data_1j)).unsqueeze(1)
            NF_b1j_0p5 = torch.sigmoid(models["nf_background_1jet&c_0p5"](data_1j)).unsqueeze(1)
            NF_s1j_2p0 = torch.sigmoid(models["nf_signal_1jet&c_2p0"](data_1j)).unsqueeze(1)
            NF_b1j_2p0 = torch.sigmoid(models["nf_background_1jet&c_2p0"](data_1j)).unsqueeze(1)
            NF_s2j_0p5 = torch.sigmoid(models["nf_signal_2jet&c_0p5"](data_2j)).unsqueeze(1)
            NF_b2j_0p5 = torch.sigmoid(models["nf_background_2jet&c_0p5"](data_2j)).unsqueeze(1)
            NF_s2j_2p0 = torch.sigmoid(models["nf_signal_2jet&c_2p0"](data_2j)).unsqueeze(1)
            NF_b2j_2p0 = torch.sigmoid(models["nf_background_2jet&c_2p0"](data_2j)).unsqueeze(1)
        except KeyError as e:
            raise KeyError(f"No key `{e}` found in model Dict. Available keys are {models.keys()}")
        # Append the NF features to the original data
        data_1j = torch.cat([data_1j, NF_s1j_0p5, NF_s1j_2p0, NF_b1j_0p5, NF_b1j_2p0], dim=1)
        data_2j = torch.cat([data_2j, NF_s2j_0p5, NF_s2j_2p0, NF_b2j_0p5, NF_b2j_2p0], dim=1)

    return data_2j, data_1j, label_2j, label_1j
