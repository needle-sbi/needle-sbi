""" chep debugginggggg


- build local debug histograms and Neyman-style calibration on a bounded subset and 
  avoid Luigi multiprocessing (Macbook is not a fan of luigi mp)
  
"""

import argparse
import ast
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

from fair_universe_demo.models.classifier import CombinedClassifier
from fair_universe_demo.models.classifier_datamodule import ClassifierDatamodule
from fair_universe_demo.tasks.histogram import HistogramTask
from fair_universe_demo.tasks.plotting_mixin import PlottingMixin
from fair_universe_demo.utils.dataset import Data
from fair_universe_demo.utils.selection import createJetData, return1j2j
from fair_universe_demo.utils.stats import (
    _grid_bounds_from_splines,
    compute_signal_fraction,
    fit_2D_splines_bin_by_bin_from_dict,
    morph_histogram_2D_spline,
    neg_log_prior_theta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        default=os.environ.get(
            
            "FAIR_UNIVERSE_DATA", 
            "/Users/levievans/Dev/needle/fair-universe-data/data"),
        
        help="FAIR Universe data root, train/data directory, or data.parquet file.",
    )
    
    parser.add_argument(
        "--snapshot-path",
        default="runs/fair_universe_demo_fixed_normalization/dag_snapshot.json",
        help="Snapshot JSON containing trained NF/classifier checkpoints.",
    )
    
    parser.add_argument(
        "--output-dir",
        default="runs/fair_universe_demo_debug_subset",
        help="Directory for hist.json, neyman.json, and summary.json.",
    )
    
    parser.add_argument("--subset-size", type=int, default=20_000)
    parser.add_argument("--max-source-rows", type=int, default=200_000)
    parser.add_argument("--subset-seed", type=int, default=12345)
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=100.0,
        help="Scale non-signal sidecar event weights for local pseudo-experiments on small subsets.",
    )
    
    parser.add_argument(
        "--signal-weight-scale",
        type=float,
        default=10_000.0,
        help="Scale htautau sidecar weights separately so small local subsets contain enough signal events.",
    )
    
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--neyman-samples", type=int, default=3)
    parser.add_argument("--mu-values", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read output-dir/summary.json and regenerate plots without rerunning inference.",
    )
    
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Use existing output-dir/hist.json to run nuisance scans without rebuilding templates.",
    )
    
    parser.add_argument("--scan-samples", type=int, default=3)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Use existing output-dir/hist.json to run a nominal profile-likelihood scan.",
    )
    
    parser.add_argument("--profile-seed", type=int, default=31415)
    parser.add_argument("--profile-mu-min", type=float, default=0.5)
    parser.add_argument("--profile-mu-max", type=float, default=1.5)
    parser.add_argument("--profile-points", type=int, default=41)
    return parser.parse_args()


def load_debug_data(args: argparse.Namespace) -> Data:
    data = Data(args.root_dir, test_size=args.subset_size)
    data.load_test_set(
        test_size=args.subset_size,
        random_seed=args.subset_seed,
        max_source_rows=args.max_source_rows,
    )
    for label, subset in data.get_test_set().items():
        if "weights" in subset:
            scale = args.signal_weight_scale if label == "htautau" else args.weight_scale
            subset.loc[:, "weights"] = subset["weights"] * scale
    return data


def score_sample(
    *,
    data: Data,
    nf_models: torch.nn.ModuleDict,
    classifier: torch.nn.Module,
    device: str,
    mu: float,
    n_param: list[float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    alljet_data, _ = createJetData(
        jet_num="all",
        useTestData=True,
        loaded_data=data,
        set_mu=mu,
        seed=seed,
        n_param=list(n_param),
        useRand=False,
    )
    data_2j, data_1j, label_2j, label_1j = return1j2j(
        alljet_data,
        models=nf_models,
        device=device,
    )

    with torch.no_grad():
        scores_2j = torch.sigmoid(classifier(data_2j, 2)).detach().cpu().numpy().reshape(-1)
        scores_1j = torch.sigmoid(classifier(data_1j, 1)).detach().cpu().numpy().reshape(-1)

    labels = np.concatenate([label_2j.detach().cpu().numpy(), label_1j.detach().cpu().numpy()])
    scores = np.concatenate([scores_2j, scores_1j])
    return data_2j, data_1j, labels, scores


def build_histograms(
    *,
    data: Data,
    nf_models: torch.nn.ModuleDict,
    classifier: torch.nn.Module,
    device: str,
    grid_size: int,
    bins: np.ndarray,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    s_templates: dict[tuple[str, str], np.ndarray] = {}
    b_templates: dict[tuple[str, str], np.ndarray] = {}

    grid = np.linspace(0.9, 1.1, grid_size)
    for tes in tqdm(grid, desc="TES grid"):
        for jes in tqdm(grid, desc="JES grid", leave=False):
            _, _, labels, scores = score_sample(
                data=data,
                nf_models=nf_models,
                classifier=classifier,
                device=device,
                mu=1.0,
                n_param=[1.0, 1.0, 1.0, float(tes), float(jes), 0.0],
                seed=0,
            )
            if not np.any(labels == 1) or not np.any(labels == 0):
                raise RuntimeError(f"Missing signal/background events for template point {(tes, jes)}")

            s_hist, _ = np.histogram(scores[labels == 1], bins=bins, density=True)
            b_hist, _ = np.histogram(scores[labels == 0], bins=bins, density=True)
            key = (str(float(tes)), str(float(jes)))
            s_templates[key] = s_hist
            b_templates[key] = b_hist

    return s_templates, b_templates


def write_hist_json(output_path: Path, s_templates: dict, b_templates: dict) -> None:
    serializable_dict = {
        str(key): {
            "sig": s_templates[key].tolist(),
            "bg": b_templates[key].tolist(),
        }
        for key in s_templates
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(serializable_dict, file)


def load_hist_json(hist_path: Path) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    with open(hist_path, "r", encoding="utf-8") as file:
        hist_dict = json.load(file)

    s_templates: dict[tuple[str, str], np.ndarray] = {}
    b_templates: dict[tuple[str, str], np.ndarray] = {}
    for key, value in hist_dict.items():
        tes, jes = ast.literal_eval(key)
        template_key = (str(float(tes)), str(float(jes)))
        s_templates[template_key] = np.array(value["sig"])
        b_templates[template_key] = np.array(value["bg"])

    return s_templates, b_templates


def _group_closure_rows(summary: dict) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for row in summary["closure_rows"]:
        grouped.setdefault(float(row["mu_true"]), []).append(row)
    return grouped


def save_needle_plot(fig, path: Path) -> str:
    fig = PlottingMixin.set_needle_plot_style(fig)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def write_debug_plots(summary: dict, output_dir: Path) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped = _group_closure_rows(summary)
    mu_values = np.array(sorted(grouped))

    true_mean = np.array([np.mean([row["signal_fraction_true"] for row in grouped[mu]]) for mu in mu_values])
    true_std = np.array([np.std([row["signal_fraction_true"] for row in grouped[mu]]) for mu in mu_values])
    fit_mean = np.array([np.mean([row["signal_fraction_fit"] for row in grouped[mu]]) for mu in mu_values])
    fit_std = np.array([np.std([row["signal_fraction_fit"] for row in grouped[mu]]) for mu in mu_values])
    neyman_mean = np.array([np.mean(summary["neyman"][str(float(mu))]) for mu in mu_values])
    neyman_std = np.array([np.std(summary["neyman"][str(float(mu))]) for mu in mu_values])
    n_events_mean = np.array([np.mean([row["n_events"] for row in grouped[mu]]) for mu in mu_values])
    n_events_std = np.array([np.std([row["n_events"] for row in grouped[mu]]) for mu in mu_values])

    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(mu_values, true_mean, yerr=true_std, marker="o", capsize=3, label="True label fraction")
    ax.errorbar(mu_values, fit_mean, yerr=fit_std, marker="s", capsize=3, label="Fitted fraction")
    ax.set_xlabel(r"$\mu_\mathrm{true}$")
    ax.set_ylabel("Signal fraction")
    ax.legend()
    path = plot_dir / "signal_fraction_closure.png"
    paths.append(save_needle_plot(fig, path))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(mu_values, neyman_mean, yerr=neyman_std, marker="o", capsize=3, label="Subset Neyman mean")
    ax.plot(mu_values, mu_values, "k--", label="Ideal")
    ax.set_xlabel(r"$\mu_\mathrm{true}$")
    ax.set_ylabel(r"$f_s^\mathrm{fit} / f_s^\mathrm{nominal}$")
    ax.legend()
    path = plot_dir / "neyman_calibration.png"
    paths.append(save_needle_plot(fig, path))

    fig, ax = plt.subplots(figsize=(5, 4))
    residual = neyman_mean - mu_values
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.errorbar(mu_values, residual, yerr=neyman_std, marker="o", capsize=3)
    ax.set_xlabel(r"$\mu_\mathrm{true}$")
    ax.set_ylabel(r"Observed $f_s/f_s^\mathrm{nominal} - \mu_\mathrm{true}$")
    path = plot_dir / "neyman_calibration_residuals.png"
    paths.append(save_needle_plot(fig, path))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(mu_values, n_events_mean, yerr=n_events_std, marker="o", capsize=3)
    ax.set_xlabel(r"$\mu_\mathrm{true}$")
    ax.set_ylabel("Pseudo-experiment events")
    path = plot_dir / "event_counts.png"
    paths.append(save_needle_plot(fig, path))

    return paths


def run_nuisance_scans(
    *,
    data: Data,
    nf_models: torch.nn.ModuleDict,
    classifier: torch.nn.Module,
    device: str,
    splines_s,
    splines_b,
    f_s_nominal_true: float,
    subset_seed: int,
    scan_samples: int,
) -> list[dict]:
    scan_points = {
        "nominal": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "tes_low": [1.0, 1.0, 1.0, 0.95, 1.0, 0.0],
        "tes_high": [1.0, 1.0, 1.0, 1.05, 1.0, 0.0],
        "jes_low": [1.0, 1.0, 1.0, 1.0, 0.95, 0.0],
        "jes_high": [1.0, 1.0, 1.0, 1.0, 1.05, 0.0],
        "soft_met": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "ttbar_up": [1.10, 1.0, 1.0, 1.0, 1.0, 0.0],
        "diboson_up": [1.0, 1.25, 1.0, 1.0, 1.0, 0.0],
        "bkg_up": [1.0, 1.0, 1.01, 1.0, 1.0, 0.0],
    }

    rows: list[dict] = []
    for scan_name, n_param in tqdm(scan_points.items(), desc="Nuisance scan"):
        for i in range(scan_samples):
            seed = subset_seed + 20_000 + i
            data_2j, data_1j, labels, _ = score_sample(
                data=data,
                nf_models=nf_models,
                classifier=classifier,
                device=device,
                mu=1.0,
                n_param=n_param,
                seed=seed,
            )
            fit = compute_signal_fraction(
                test_data_2j=data_2j,
                test_data_1j=data_1j,
                dnn_model=classifier,
                bin_splines_S=splines_s,
                bin_splines_BG=splines_b,
                eval_device=device,
                initial_f_s=0.1,
                verbose=False,
                return_diagnostics=True,
            )
            rows.append(
                {
                    "scan": scan_name,
                    "seed": seed,
                    "n_param": n_param,
                    "n_events": int(len(labels)),
                    "signal_fraction_true": float(np.mean(labels)),
                    "mu_observed": float(fit["f_s_hat"] / f_s_nominal_true),
                    **fit,
                }
            )

    return rows


def write_nuisance_scan_plot(summary: dict, output_dir: Path) -> str | None:
    rows = summary.get("nuisance_scan_rows")
    if not rows:
        return None

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scan_names = list(dict.fromkeys(row["scan"] for row in rows))
    x = np.arange(len(scan_names))
    nu1 = np.array([np.mean([row["nu1_hat"] for row in rows if row["scan"] == name]) for name in scan_names])
    nu1_std = np.array([np.std([row["nu1_hat"] for row in rows if row["scan"] == name]) for name in scan_names])
    nu2 = np.array([np.mean([row["nu2_hat"] for row in rows if row["scan"] == name]) for name in scan_names])
    nu2_std = np.array([np.std([row["nu2_hat"] for row in rows if row["scan"] == name]) for name in scan_names])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x - 0.05, nu1, yerr=nu1_std, marker="o", linestyle="", capsize=3, label="nu1")
    ax.errorbar(x + 0.05, nu2, yerr=nu2_std, marker="s", linestyle="", capsize=3, label="nu2")
    ax.axhline(0.9, color="k", linestyle=":", linewidth=1)
    ax.axhline(1.1, color="k", linestyle=":", linewidth=1)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(scan_names, rotation=30, ha="right")
    ax.set_ylabel("Fitted nuisance")
    ax.legend()
    path = plot_dir / "nuisance_scan.png"
    return save_needle_plot(fig, path)


def write_nuisance_impact_plot(summary: dict, output_dir: Path) -> str | None:
    rows = summary.get("nuisance_scan_rows")
    if not rows:
        return None

    grouped = {
        scan_name: [row for row in rows if row["scan"] == scan_name]
        for scan_name in dict.fromkeys(row["scan"] for row in rows)
    }
    nominal_mu = np.mean([row["mu_observed"] for row in grouped["nominal"]])
    impact_rows = []
    for scan_name, scan_rows in grouped.items():
        if scan_name == "nominal":
            continue
        mean_mu = np.mean([row["mu_observed"] for row in scan_rows])
        impact_rows.append((scan_name, mean_mu - nominal_mu))

    impact_rows.sort(key=lambda item: abs(item[1]))
    labels = [name.replace("_", " ") for name, _ in impact_rows]
    impacts = np.array([impact for _, impact in impact_rows])
    colors = np.where(impacts >= 0, "C0", "C3")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, impacts, color=colors, alpha=0.8)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel(r"Shift in observed $f_s/f_s^\mathrm{nominal}$")
    ax.set_ylabel("Nuisance variation")
    path = plot_dir / "nuisance_impact_ranking.png"
    return save_needle_plot(fig, path)


def run_profile_likelihood_scan(
    *,
    data: Data,
    nf_models: torch.nn.ModuleDict,
    classifier: torch.nn.Module,
    device: str,
    splines_s,
    splines_b,
    f_s_nominal_true: float,
    seed: int,
    mu_min: float,
    mu_max: float,
    points: int,
    bins: np.ndarray,
) -> dict:
    _, _, labels, scores = score_sample(
        data=data,
        nf_models=nf_models,
        classifier=classifier,
        device=device,
        mu=1.0,
        n_param=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        seed=seed,
    )
    hist_data, _ = np.histogram(scores, bins=bins)
    bin_widths = np.diff(bins)
    n_total = len(scores)
    nu1_bounds, nu2_bounds = _grid_bounds_from_splines(splines_s)

    def nll_for_params(f_s: float, nu1: float, nu2: float) -> float:
        s_template = morph_histogram_2D_spline([nu1, nu2], splines_s)
        b_template = morph_histogram_2D_spline([nu1, nu2], splines_b)
        expected = n_total * (f_s * s_template + (1 - f_s) * b_template) * bin_widths
        expected = np.clip(expected, a_min=1e-10, a_max=None)
        nll = np.sum(expected - hist_data * np.log(expected))
        nll += neg_log_prior_theta(nu1) + neg_log_prior_theta(nu2)
        return float(nll)

    profile_rows = []
    mu_grid = np.linspace(mu_min, mu_max, points)
    for mu_observed in tqdm(mu_grid, desc="Profile scan"):
        f_s = float(np.clip(mu_observed * f_s_nominal_true, 1e-6, 1.0))

        def objective(params):
            nu1, nu2 = params
            return nll_for_params(f_s, nu1, nu2)

        opt = minimize(
            objective,
            x0=[1.0, 1.0],
            method="L-BFGS-B",
            bounds=[nu1_bounds, nu2_bounds],
        )
        profile_rows.append(
            {
                "mu_observed": float(mu_observed),
                "f_s": f_s,
                "nll": float(opt.fun),
                "nu1_hat": float(opt.x[0]),
                "nu2_hat": float(opt.x[1]),
                "success": bool(opt.success),
                "message": str(opt.message),
            }
        )

    min_nll = min(row["nll"] for row in profile_rows)
    for row in profile_rows:
        row["delta_nll"] = float(row["nll"] - min_nll)
        row["minus2_delta_log_likelihood"] = float(2 * row["delta_nll"])

    best = min(profile_rows, key=lambda row: row["nll"])
    return {
        "seed": seed,
        "n_events": int(len(labels)),
        "signal_fraction_true": float(np.mean(labels)),
        "best_mu_observed": float(best["mu_observed"]),
        "best_f_s": float(best["f_s"]),
        "best_nu1": float(best["nu1_hat"]),
        "best_nu2": float(best["nu2_hat"]),
        "rows": profile_rows,
    }


def write_profile_likelihood_plot(summary: dict, output_dir: Path) -> str | None:
    profile = summary.get("profile_likelihood")
    if not profile:
        return None

    rows = profile["rows"]
    mu_values = np.array([row["mu_observed"] for row in rows])
    q_values = np.array([row["minus2_delta_log_likelihood"] for row in rows])
    best_mu = profile["best_mu_observed"]

    gaussian_mu = None
    gaussian_q = None
    fit_mask = q_values < 4
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = q_values < 9
    if np.count_nonzero(fit_mask) >= 3:
        coeff = np.polyfit(mu_values[fit_mask] - best_mu, q_values[fit_mask], deg=2)
        curvature = coeff[0]
        if curvature > 0:
            gaussian_mu = np.linspace(mu_values.min(), mu_values.max(), 300)
            gaussian_q = curvature * (gaussian_mu - best_mu) ** 2

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mu_values, q_values, marker="o", markersize=3, label="Profile scan")
    if gaussian_mu is not None and gaussian_q is not None:
        ax.plot(gaussian_mu, gaussian_q, color="C3", linestyle="--", label="Gaussian approximation")
    ax.axvline(best_mu, color="C1", linestyle="--", label="Best fit")
    ax.axhline(1.0, color="0.5", linestyle=":", label="68% CL")
    ax.axhline(4.0, color="0.7", linestyle=":", label="95% CL")
    ax.set_xlabel(r"$f_s^\mathrm{fit} / f_s^\mathrm{nominal}$")
    ax.set_ylabel(r"$-2\Delta\log L$")
    y_max = float(np.nanmax(q_values))
    ax.set_ylim(0, 8 if y_max > 12 else max(4.5, 1.05 * y_max))
    visible_mask = q_values <= ax.get_ylim()[1]
    if y_max > 12 and np.count_nonzero(visible_mask) >= 2:
        visible_mu = mu_values[visible_mask]
        span = max(float(visible_mu.max() - visible_mu.min()), 0.05)
        ax.set_xlim(visible_mu.min() - 0.35 * span, visible_mu.max() + 0.35 * span)
    ax.legend()
    path = plot_dir / "profile_likelihood_scan.png"
    return save_needle_plot(fig, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        summary["plot_paths"] = write_debug_plots(summary, output_dir)
        nuisance_plot = write_nuisance_scan_plot(summary, output_dir)
        if nuisance_plot and nuisance_plot not in summary["plot_paths"]:
            summary["plot_paths"].append(nuisance_plot)
        nuisance_impact_plot = write_nuisance_impact_plot(summary, output_dir)
        if nuisance_impact_plot and nuisance_impact_plot not in summary["plot_paths"]:
            summary["plot_paths"].append(nuisance_impact_plot)
        profile_plot = write_profile_likelihood_plot(summary, output_dir)
        if profile_plot and profile_plot not in summary["plot_paths"]:
            summary["plot_paths"].append(profile_plot)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(json.dumps({"plot_paths": summary["plot_paths"]}, indent=2))
        return

    device = args.device
    nf_ckpts, classifier_ckpt = HistogramTask.parse_snapshot(args.snapshot_path)
    nf_models = ClassifierDatamodule.load_nf_models(nf_ckpts).to(device).eval().to(torch.float32)
    classifier = (
        CombinedClassifier.load_from_checkpoint(classifier_ckpt["classifier"]).to(device).eval().to(torch.float32)
    )

    data = load_debug_data(args)
    bins = np.linspace(0, 1, num=args.bins)

    if args.scan_only:
        s_templates, b_templates = load_hist_json(output_dir / "hist.json")
        splines_s = fit_2D_splines_bin_by_bin_from_dict(s_templates)
        splines_b = fit_2D_splines_bin_by_bin_from_dict(b_templates)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        summary["nuisance_scan_rows"] = run_nuisance_scans(
            data=data,
            nf_models=nf_models,
            classifier=classifier,
            device=device,
            splines_s=splines_s,
            splines_b=splines_b,
            f_s_nominal_true=float(summary["f_s_nominal_true"]),
            subset_seed=args.subset_seed,
            scan_samples=args.scan_samples,
        )
        nuisance_plot = write_nuisance_scan_plot(summary, output_dir)
        if nuisance_plot:
            summary.setdefault("plot_paths", [])
            if nuisance_plot not in summary["plot_paths"]:
                summary["plot_paths"].append(nuisance_plot)
        nuisance_impact_plot = write_nuisance_impact_plot(summary, output_dir)
        if nuisance_impact_plot:
            summary.setdefault("plot_paths", [])
            if nuisance_impact_plot not in summary["plot_paths"]:
                summary["plot_paths"].append(nuisance_impact_plot)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(json.dumps({"nuisance_scan_rows": summary["nuisance_scan_rows"]}, indent=2))
        return

    if args.profile_only:
        s_templates, b_templates = load_hist_json(output_dir / "hist.json")
        splines_s = fit_2D_splines_bin_by_bin_from_dict(s_templates)
        splines_b = fit_2D_splines_bin_by_bin_from_dict(b_templates)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        summary["profile_likelihood"] = run_profile_likelihood_scan(
            data=data,
            nf_models=nf_models,
            classifier=classifier,
            device=device,
            splines_s=splines_s,
            splines_b=splines_b,
            f_s_nominal_true=float(summary["f_s_nominal_true"]),
            seed=args.profile_seed,
            mu_min=args.profile_mu_min,
            mu_max=args.profile_mu_max,
            points=args.profile_points,
            bins=bins,
        )
        profile_plot = write_profile_likelihood_plot(summary, output_dir)
        if profile_plot:
            summary.setdefault("plot_paths", [])
            if profile_plot not in summary["plot_paths"]:
                summary["plot_paths"].append(profile_plot)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(json.dumps({"profile_likelihood": summary["profile_likelihood"]}, indent=2))
        return

    s_templates, b_templates = build_histograms(
        data=data,
        nf_models=nf_models,
        classifier=classifier,
        device=device,
        grid_size=args.grid_size,
        bins=bins,
    )
    hist_path = output_dir / "hist.json"
    write_hist_json(hist_path, s_templates, b_templates)

    splines_s = fit_2D_splines_bin_by_bin_from_dict(s_templates)
    splines_b = fit_2D_splines_bin_by_bin_from_dict(b_templates)

    _, _, nominal_labels, _ = score_sample(
        data=data,
        nf_models=nf_models,
        classifier=classifier,
        device=device,
        mu=1.0,
        n_param=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        seed=0,
    )
    f_s_nominal_true = float(np.mean(nominal_labels))

    neyman: dict[str, list[float]] = {}
    closure_rows = []
    for mu in tqdm(args.mu_values, desc="Neyman mu"):
        neyman[str(float(mu))] = []
        for i in tqdm(range(args.neyman_samples), desc="Neyman seed", leave=False):
            seed = args.subset_seed + 10_000 + i
            data_2j, data_1j, labels, _ = score_sample(
                data=data,
                nf_models=nf_models,
                classifier=classifier,
                device=device,
                mu=float(mu),
                n_param=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                seed=seed,
            )
            f_s_hat = compute_signal_fraction(
                test_data_2j=data_2j,
                test_data_1j=data_1j,
                dnn_model=classifier,
                bin_splines_S=splines_s,
                bin_splines_BG=splines_b,
                eval_device=device,
                initial_f_s=0.1,
                verbose=False,
            )
            f_s_true = float(np.mean(labels))
            closure_rows.append(
                {
                    "mu_true": float(mu),
                    "seed": seed,
                    "n_events": int(len(labels)),
                    "signal_fraction_true": f_s_true,
                    "signal_fraction_fit": float(f_s_hat),
                }
            )

    for row in closure_rows:
        neyman[str(row["mu_true"])].append(float(row["signal_fraction_fit"] / f_s_nominal_true))

    neyman_path = output_dir / "neyman.json"
    with open(neyman_path, "w", encoding="utf-8") as file:
        json.dump(neyman, file, indent=2)

    summary = {
        "args": vars(args),
        "hist_path": str(hist_path),
        "neyman_path": str(neyman_path),
        "f_s_nominal_true": f_s_nominal_true,
        "closure_rows": closure_rows,
        "neyman": neyman,
    }
    summary["plot_paths"] = write_debug_plots(summary, output_dir)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
