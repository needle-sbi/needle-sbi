"""
Original author: I. Elsharkawy
Based on https://github.com/ibrahimEls/CNFParameterEstimation
Adapted by K. Schmidt
"""

import ast
import json
import logging
import os
import re
import warnings
from typing import Tuple

import numpy as np
import torch
from scipy.interpolate import SmoothBivariateSpline, interp1d
from scipy.optimize import curve_fit, minimize
from tabulate import tabulate
from tqdm import tqdm

logger = logging.Logger("stats")

ROOT_DIR = os.getcwd()


def string_to_tuple_str(s: str) -> Tuple | None:
    """
    Converts a stringified tuple into an actual tuple of strings.

    Supports tuples of length 2 and 3.

    Parameters:
    - s (str): The string representation of the tuple, e.g., "('A1', 'B2')" or "('A1', 'B2', 'C3')"

    Returns:
    - tuple: A tuple of strings, e.g., ('A1', 'B2') or ('A1', 'B2', 'C3')

    Raises:
    - ValueError: If the string does not represent a tuple of length 2 or 3.
    """
    try:
        # Safely evaluate the string to a Python tuple
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
        parsed = ast.literal_eval(cleaned)

        # Check if the result is a tuple with length 2 or 3
        if isinstance(parsed, tuple) and len(parsed) in [2, 3]:
            # Convert each element to string
            return tuple(str(element) for element in parsed)
        else:
            raise ValueError("The string does not represent a tuple of length 2 or 3.")

    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string to tuple: {e}\n -> '{s}' is not a valid python literal structure")
        return None


# TODO: same as above pretty much so one needs to be removed
def neg_log_prior_theta(theta: float, mu_theta: float = 1.0, sigma_theta: float = 0.05) -> float:
    """Negative log of a Gaussian prior.

    The 0.5 * log(2*pi) constant is dropped since it doesn't affect the minimum.
    """
    return 0.5 * ((theta - mu_theta) / sigma_theta) ** 2 + np.log(sigma_theta)


def _grid_bounds_from_splines(bin_splines) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read the x,y training extent off any one bin spline.

    SmoothBivariateSpline.get_knots() returns the knot grid where its
    min/max bound the region where the fit was actually constrained by data.
    Outside that box the spline extrapolates and the optim. can find bad minima.
    """
    tx, ty = bin_splines[0].get_knots()
    return (float(tx.min()), float(tx.max())), (float(ty.min()), float(ty.max()))


def compute_signal_fraction(
    test_data_2j,
    test_data_1j,
    dnn_model,
    bin_splines_S,
    bin_splines_BG,
    eval_device: str = "cpu",
    nuisance_priors: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.05), (1.0, 0.05)),
    initial_f_s: float = 0.001,
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """
    Perform a simultaneous MLE of the global signal fraction `f_s`
    and a single nuisance parameter `theta`, using three categories
    (0-jet, 1-jet, 2-jet).

    Arguments
    ---------
    model : torch.nn.Module
        Your trained model (used for classifier scores).
    test_data_{0,1,2} : torch datasets
        The data for each category (0-jet, 1-jet, 2-jet).
    hist_dict_{0,1,2} : dict
        Each must provide the S and B template histograms keyed by theta values:
            hist_dict_c[theta] = (S_hist, B_hist)

    Returns
    -------
    A tuple (f_s_hat, theta_hat)
    """
    test_data_2j = test_data_2j.to(eval_device).to(torch.float32)
    test_data_1j = test_data_1j.to(eval_device).to(torch.float32)
    dnn_model = dnn_model.to(eval_device).eval().to(torch.float32)

    nbins = 200
    bins = np.linspace(0, 1, num=nbins)
    bin_widths = np.diff(bins)

    with torch.no_grad():
        scores_2j = torch.sigmoid(dnn_model(test_data_2j, 2)).cpu().numpy()
        scores_1j = torch.sigmoid(dnn_model(test_data_1j, 1)).cpu().numpy()

    total_score = np.concatenate([scores_2j, scores_1j])
    hist_data, _ = np.histogram(total_score, bins=bins)

    N_total = len(total_score)

    # derive nuisance bounds from the spline training grid. This prevents the
    # optimiser from escaping into extrapolated (nu1, nu2), where the
    # morphed templates are unphysical and produce bad minima
    nu1_bounds, nu2_bounds = _grid_bounds_from_splines(bin_splines_S)
    (mu1, sig1), (mu2, sig2) = nuisance_priors

    param_bounds = [
        (1e-6, 1.0),  # strictly positive lower bound to avoid log(0)
        nu1_bounds,
        nu2_bounds,
    ]

    def neg_log_likelihood(params):
        f_s, nu1, nu2 = params

        if not (param_bounds[0][0] <= f_s <= param_bounds[0][1]):
            return np.inf
        if not (nu1_bounds[0] <= nu1 <= nu1_bounds[1]):
            return np.inf
        if not (nu2_bounds[0] <= nu2 <= nu2_bounds[1]):
            return np.inf

        S = morph_histogram_2D_spline([nu1, nu2], bin_splines_S)
        B = morph_histogram_2D_spline([nu1, nu2], bin_splines_BG)

        E = N_total * (f_s * S + (1 - f_s) * B) * bin_widths
        E = np.clip(E, a_min=1e-10, a_max=None)
        nll = np.sum(E - hist_data * np.log(E))

        nll_prior = neg_log_prior_theta(nu1, mu1, sig1) + neg_log_prior_theta(nu2, mu2, sig2)
        return nll + nll_prior

    initial_params = [initial_f_s, mu1, mu2]

    # We'll use L-BFGS-B
    opt_result = minimize(neg_log_likelihood, x0=initial_params, method="L-BFGS-B", bounds=param_bounds)
    f_s_hat, nu1_hat, nu2_hat = opt_result.x

    if verbose:
        tqdm.write(
            tabulate(
                [
                    [f"f_s", f_s_hat],
                    ["nu1", nu1_hat],
                    ["nu2", nu2_hat],
                ],
                headers=["Parameter", "Estimated"],
                floatfmt=".6g",
            )
        )
        tqdm.write(f"Converged={opt_result.success}, {opt_result.message}")

    edge_tol = 1e-3
    nu1_hit_bound = abs(nu1_hat - nu1_bounds[0]) < edge_tol or abs(nu1_hat - nu1_bounds[1]) < edge_tol
    nu2_hit_bound = abs(nu2_hat - nu2_bounds[0]) < edge_tol or abs(nu2_hat - nu2_bounds[1]) < edge_tol

    if nu1_hit_bound:
        logger.warning(f"nu1 hit bound {nu1_bounds} at {nu1_hat:.6f}")
    if nu2_hit_bound:
        logger.warning(f"nu2 hit bound {nu2_bounds} at {nu2_hat:.6f}")

    if return_diagnostics:
        return {
            "f_s_hat": float(f_s_hat),
            "nu1_hat": float(nu1_hat),
            "nu2_hat": float(nu2_hat),
            "nu1_bounds": tuple(float(value) for value in nu1_bounds),
            "nu2_bounds": tuple(float(value) for value in nu2_bounds),
            "nu1_hit_bound": bool(nu1_hit_bound),
            "nu2_hit_bound": bool(nu2_hit_bound),
            "success": bool(opt_result.success),
            "message": str(opt_result.message),
        }

    return float(f_s_hat)


def fit_2D_splines_bin_by_bin_from_dict(param_hist_dict, s=0, kx=3, ky=3):
    """
    Fits a SmoothBivariateSpline for each bin (univariate in x,y) using
    the dictionary:
        {
            ("0.9","0.9"): hist_array,   # shape (nbins,)
            ("1.0","1.1"): hist_array,
            ...
        }
    The first element of the tuple is nu1, the second is nu2.
    Each hist_array has the bin contents (length nbins) at that (nu1, nu2).

    We treat each bin b as a function z_b(nu1, nu2), and fit a 2D spline.

    Parameters
    ----------
    param_hist_dict : dict
        Keys: (str, str) => e.g. ("0.9","1.0")
        Values: np.ndarray (nbins,) => histogram bin contents
    s : float, optional
        Smoothing factor for the spline (0 => interpolate exactly).
        Increase if you have noise and want a smoother fit.
    kx, ky : int, optional
        Spline degrees in x and y (e.g. 1=linear, 3=cubic).

    Returns
    -------
    bin_splines : list of SmoothBivariateSpline
        bin_splines[b] is the spline fitted to bin b's content as a function
        of (nu1, nu2).
    """

    # 1) Parse dictionary into arrays: param_array of shape (M,2), hist_array of shape (M, nbins)
    param_list = []
    hist_list = []

    for key_tuple, hist_data in param_hist_dict.items():
        # key_tuple might be ("0.9","1.0"), so convert to float
        nu_values = [float(x) for x in key_tuple]  # [nu1, nu2]
        param_list.append(nu_values)
        hist_list.append(hist_data)  # shape (nbins,)

    param_array = np.array(param_list)  # shape (M, 2)
    hist_array = np.array(hist_list)  # shape (M, nbins)

    M, nbins = hist_array.shape
    # param_array[:, 0] => all nu1 values
    # param_array[:, 1] => all nu2 values

    xvals = param_array[:, 0]
    yvals = param_array[:, 1]

    # 2) Fit a 2D spline per bin
    bin_splines = []
    with warnings.catch_warnings(action="ignore"):
        for b in range(nbins):
            # z-values for bin b across all M points
            zvals = hist_array[:, b]

            # Fit a bivariate spline, z = f(x,y)
            # *Note*: SmoothBivariateSpline expects 1D arrays x, y, z (scattered points).
            spline = SmoothBivariateSpline(xvals, yvals, zvals, kx=kx, ky=ky, s=s)
            bin_splines.append(spline)

    return bin_splines


def morph_histogram_2D_spline(params, bin_splines):
    """
    Evaluate the 2D spline morphing at (nu1, nu2).
    bin_splines[b] is a SmoothBivariateSpline for bin b.

    Parameters
    ----------
    nu1, nu2 : float
        The new parameter values at which to evaluate the histogram.
    bin_splines : list of SmoothBivariateSpline
        The fitted splines from fit_2D_splines_bin_by_bin_from_dict().

    Returns
    -------
    morphed_hist : np.ndarray, shape (nbins,)
        The bin contents at (nu1, nu2).
    """
    nu1, nu2 = params
    nbins = len(bin_splines)
    morphed = np.zeros(nbins, dtype=float)

    # Evaluate each bin's spline
    for b in range(nbins):
        # SmoothBivariateSpline.__call__(x, y, grid=False) returns a small 2D array
        # if x,y are arrays. With scalars, we get a shape (1,1). We'll take [0,0].
        val_2d = bin_splines[b](nu1, nu2, grid=False)
        morphed[b] = val_2d  # extract the single scalar

    # Clip negative bin contents if appropriate
    morphed = np.clip(morphed, 0, None)
    return morphed


def load_bias_data(path: str):
    with open(path, "r", encoding="utf-8") as file:
        MLE_ratio_arr = json.load(file)

    # Example data (replace with your actual data)
    mu_real_values = np.sort(np.array(list(MLE_ratio_arr.keys())))  # Your mu_real keys
    mu_obs_distributions = {mu_real: np.array(MLE_ratio_arr[mu_real], dtype=float) for mu_real in mu_real_values}

    mu_obs_means = []
    mu_obs_stds = []

    for mu_real in mu_real_values:
        mu_obs = mu_obs_distributions[mu_real]
        mu_obs_means.append(np.mean(mu_obs))
        mu_obs_stds.append(np.std(mu_obs))

    mu_obs_means = np.array(mu_obs_means)
    mu_obs_stds = np.array(mu_obs_stds)
    mu_real_values = np.array(mu_real_values, dtype=float)

    def bias_func(mu_real, a, b):
        return a * mu_real + b

    params, _ = curve_fit(bias_func, mu_real_values, mu_obs_means)
    a, b = params

    mu_obs_stds_corrected = mu_obs_stds / abs(a)
    std_corrected_interp = interp1d(mu_real_values, mu_obs_stds_corrected, kind="linear", fill_value="extrapolate")

    return std_corrected_interp, a, b


def inverse_bias_func(mu_obs_mean, a, b):
    return (mu_obs_mean - b) / a


def compute_posterior(mu_obs, mu_real_range, std_corrected_interp, a, b):
    mu_obs_corrected = inverse_bias_func(mu_obs, a, b)
    likelihood = np.exp(-0.5 * ((mu_obs_corrected - mu_real_range) / std_corrected_interp(mu_real_range)) ** 2)
    posterior = likelihood

    if np.any([np.isclose(mu_real, 0) for mu_real in mu_real_range]):
        raise RuntimeError(f"Found divide by zero in 'mu_real_range': {mu_real_range}")
    try:
        with warnings.catch_warnings(action="error"):
            posterior /= np.trapezoid(posterior, mu_real_range)
    except RuntimeWarning as e:
        raise RuntimeWarning(f"{e}\nFor posterior:\n{posterior}\nand 'mu_real_range':\n{mu_real_range}")
    return mu_obs_corrected, posterior


def get_confidence_interval(mu_obs, std_corrected_interp, a, b):
    mu_real_range = np.linspace(0.00001, 3, 1000)
    mu_obs_corrected, posterior = compute_posterior(mu_obs, mu_real_range, std_corrected_interp, a, b)
    cdf = np.cumsum(posterior)
    cdf /= cdf[-1]
    lower_idx = np.searchsorted(cdf, 0.16)
    upper_idx = np.searchsorted(cdf, 0.84)
    return mu_obs_corrected, mu_real_range[lower_idx], mu_real_range[upper_idx]
