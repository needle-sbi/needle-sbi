"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import json
from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch

from ..models.classifier import CombinedClassifier
from ..models.classifier_datamodule import ClassifierDatamodule
from ..tasks.histogram import HistogramTask
from ..utils.selection import Data
from .selection import createJetData, return1j2j
from .stats import (
    compute_signal_fraction,
    fit_2D_splines_bin_by_bin_from_dict,
    get_confidence_interval,
    load_bias_data,
    string_to_tuple_str,
)

logger = Logger("predict")


def _signal_fraction_from_labels(alljet_data, models, device: str) -> float:
    data_2j, data_1j, label_2j, label_1j = return1j2j(alljet_data, models, device=device)
    labels = np.concatenate([label_2j.detach().cpu().numpy(), label_1j.detach().cpu().numpy()])
    if len(labels) == 0:
        raise RuntimeError("Cannot compute nominal signal fraction from an empty 1j/2j sample")
    return float(np.mean(labels))


def _nominal_signal_fraction(data: Data | None, root_dir: str | None, models, device: str) -> float:
    kwargs: Dict[str, Any]
    if data:
        kwargs = {"loaded_data": data}
    elif root_dir:
        kwargs = {"root_dir": root_dir}
    else:
        raise ValueError("Either set the argument `data` or `root_dir` to load the FAIR Universe Data")

    alljet_data_nominal, _ = createJetData(
        "all",
        True,
        set_mu=1.0,
        seed=0,
        n_param=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        useRand=False,
        **kwargs,
    )
    return _signal_fraction_from_labels(alljet_data_nominal, models, device)


def predict(
    mu: float,
    hist_path: str,
    neyman_path: str,
    snapshot_path: str,
    root_dir: str | None = None,
    data: Data | None = None,
    device: str = None,
    predict_num_events: int = 0,
    nuissance_parameters: List[float | None] = None,
) -> Dict:
    """
    Run prediction pipeline for parameter estimation or event classification.

    Args:
        mu: Signal strength parameter.
        hist_path: Path to histogram data file in JSON format.
        neyman_path: Path to Neyman construction bias correction data.
        snapshot_path: Path to model checkpoint snapshot.
        root_dir: Directory containing FAIR Universe ROOT data files. Either this or `data` must be provided.
        data: Preloaded FAIR Universe data. Either this or `root_dir` must be provided.
        device: Compute device ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        predict_num_events: Number of events for classifier evaluation. If 0, runs in mu estimation mode.
        nuissance_parameters: Nuissance parameter values for signal generation.
            Defaults to [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]. Order is
            [ttbar_scale, diboson_scale, bkg_scale, TES, JES, soft MET]

    Returns:
        Dictionary containing:
            - "mu": Input signal strength parameter.
            - If predict_num_events == 0 (estimation mode):
                - "real_mu": Estimated mu value
                - "mu_hat": MLE point estimate
                - "p16", "p84": Confidence interval bounds
                - "delta_mu_hat": Half-width of confidence interval
            - If predict_num_events > 0 (classification mode):
                - "scores_2j": Classifier scores for 2-jet events
                - "labels_2j": Labels for 2-jet events
                - "scores_1j": Classifier scores for 1-jet events
                - "labels_1j": Labels for 1-jet events

    Raises:
        ValueError: If neither `data` nor `root_dir` is provided.
    """
    if nuissance_parameters is None:
        nuissance_parameters = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    else:
        nuissance_parameters = list(nuissance_parameters)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Running pipeline on test dataset with {mu=}")

    with open(hist_path, "r") as f:
        serializable_dict: Dict = json.load(f)

    hist_dict = {key: (np.array(v["sig"]), np.array(v["bg"])) for key, v in serializable_dict.items()}

    # Create dictionaries mapping parameter tuples to signal and background arrays.
    S_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][0] for i in hist_dict.keys()}
    B_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][1] for i in hist_dict.keys()}

    # Fit 2D splines bin-by-bin using the dictionaries.
    bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
    bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)

    # loading Neyman data
    std_corrected_interp, a, b = load_bias_data(neyman_path)

    nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(snapshot_path)
    models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device).eval().to(torch.float32)
    class_model_load = (
        CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"]).to(device).eval().to(torch.float32)
    )
    seed = 31415

    if data:
        alljet_data, _ = createJetData(
            "all",
            True,
            set_mu=mu,
            seed=seed,
            n_param=nuissance_parameters,
            useRand=False,
            loaded_data=data,
        )
    elif root_dir:
        alljet_data, _ = createJetData(
            "all",
            True,
            set_mu=mu,
            seed=seed,
            n_param=nuissance_parameters,
            useRand=False,
            root_dir=root_dir,
        )
    else:
        raise ValueError("Either set the argument `data` or `root_dir` to load the FAIR Universe Data")

    results: Dict[str, Any] = {"mu": mu}

    if not predict_num_events:
        logger.info("Running in prediction mode")
        data_2j, data_1j, label_2j, label_1j = return1j2j(alljet_data, models, device=device)

        f_s_hat = compute_signal_fraction(
            test_data_2j=data_2j,
            test_data_1j=data_1j,
            dnn_model=class_model_load,
            bin_splines_S=bin_splines_S_class,
            bin_splines_BG=bin_splines_BG_class,
            eval_device=device,
        )
        f_s_nominal = _nominal_signal_fraction(data, root_dir, models, device)
        mu_observed = f_s_hat / f_s_nominal
        mu_MLE, mu_lower, mu_upper = get_confidence_interval(mu_observed, std_corrected_interp, a, b)

        results.update(
            {
                "real_mu": float(mu_observed),
                "f_s_hat": float(f_s_hat),
                "f_s_nominal": float(f_s_nominal),
                "mu_hat": float(mu_MLE),
                "p16": float(mu_lower),
                "p84": float(mu_upper),
                "delta_mu_hat": float(abs(mu_upper - mu_lower) / 2),
            }
        )

    else:
        logger.info(f"Running classifier (not as 'mu' estimator) for {predict_num_events} events")
        data_2j, data_1j, label_2j, label_1j = return1j2j(
            alljet_data,
            models,
            nevents=predict_num_events,
            device=device,
        )

        with torch.no_grad():
            scores_2j = torch.sigmoid(class_model_load(data_2j, 2)).cpu().numpy()
            scores_1j = torch.sigmoid(class_model_load(data_1j, 1)).cpu().numpy()

        results.update(
            {
                "scores_2j": scores_2j.tolist(),
                "labels_2j": label_2j.tolist(),
                "scores_1j": scores_1j.tolist(),
                "labels_1j": label_1j.tolist(),
            }
        )

    return results
