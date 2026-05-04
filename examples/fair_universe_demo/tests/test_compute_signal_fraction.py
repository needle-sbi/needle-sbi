
# temporary unit tests for fitting and neyman to validate changes.
# Majorly clanker assisted

import numpy as np
import pytest
import torch

from fair_universe_demo.utils.stats import (
    _grid_bounds_from_splines,
    compute_signal_fraction,
    fit_2D_splines_bin_by_bin_from_dict,
    neg_log_prior_theta,
)

SEED = 12345
RNG = np.random.default_rng(SEED)
NBINS = 200
BINS = np.linspace(0, 1, num=NBINS)
BIN_WIDTHS = np.diff(BINS)

class _CannedScoreModel(torch.nn.Module):
    """classifier that returns precomputed logits, ignoring its input.
    """

    def __init__(self, logits_2j: torch.Tensor, logits_1j: torch.Tensor):
        super().__init__()
        self._logits = {1: logits_1j, 2: logits_2j}

    def forward(self, x: torch.Tensor, jet_category: int) -> torch.Tensor:
        return self._logits[jet_category]


def _signal_score_density(nu1: float, nu2: float) -> np.ndarray:
    """Signal score PDF: Beta peaked near 1, slightly shifted by nuisances.

    Normalised so that sum(pdf * bin_width) == 1.
    """
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    a = 8.0 * nu1
    b = 2.0 * nu2
    pdf = centers ** (a - 1) * (1 - centers) ** (b - 1)
    pdf /= np.sum(pdf * BIN_WIDTHS)
    return pdf


def _background_score_density(nu1: float, nu2: float) -> np.ndarray:
    """Background score PDF: Beta peaked near 0, lightly shifted by nuisances."""
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    a = 2.0 * nu2
    b = 8.0 * nu1
    pdf = centers ** (a - 1) * (1 - centers) ** (b - 1)
    pdf /= np.sum(pdf * BIN_WIDTHS)
    return pdf


def _build_template_dict(density_fn) -> dict[tuple[str, str], np.ndarray]:
    """Make {(nu1_str, nu2_str): pdf-array} on a 5x5 grid centred at (1,1).

    Cubic SmoothBivariateSpline (kx=ky=3) requires at least (3+1)^2 = 16 points.
    """
    grid = ["0.9", "0.95", "1.0", "1.05", "1.1"]
    return {
        (s1, s2): density_fn(float(s1), float(s2))
        for s1 in grid
        for s2 in grid
    }


def _sample_scores_from_pdf(pdf: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling from a binned PDF on [0, 1]."""
    cdf = np.cumsum(pdf * BIN_WIDTHS)
    cdf /= cdf[-1]
    u = rng.uniform(size=n)
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, NBINS - 2)
    return 0.5 * (BINS[idx] + BINS[idx + 1])


def _make_canned_model(
    f_s_true: float,
    nu1_true: float,
    nu2_true: float,
    n_total: int,
    rng: np.random.Generator,
) -> _CannedScoreModel:
    """Build a classifier whose pre-sigmoid logits, when sigmoided, yield
    samples drawn from the known mixture f_s * S(nu) + (1-f_s) * B(nu).

    We split events ~half/half between 1-jet and 2-jet for realism.
    """
    s_pdf = _signal_score_density(nu1_true, nu2_true)
    b_pdf = _background_score_density(nu1_true, nu2_true)

    n_signal = rng.binomial(n_total, f_s_true)
    n_bkg = n_total - n_signal

    sig_scores = _sample_scores_from_pdf(s_pdf, n_signal, rng)
    bkg_scores = _sample_scores_from_pdf(b_pdf, n_bkg, rng)
    scores = np.concatenate([sig_scores, bkg_scores])
    rng.shuffle(scores)

    # sigmoid(logit) = score  =>  logit = log(score / (1 - score))
    eps = 1e-6
    scores = np.clip(scores, eps, 1 - eps)
    logits = np.log(scores / (1 - scores))

    split = n_total // 2
    return _CannedScoreModel(
        logits_2j=torch.tensor(logits[:split], dtype=torch.float32),
        logits_1j=torch.tensor(logits[split:], dtype=torch.float32),
    )


@pytest.fixture(scope="module")
def bin_splines():
    """Fit S and B splines on a 3x3 grid of nuisance points, once per module."""
    s_dict = _build_template_dict(_signal_score_density)
    b_dict = _build_template_dict(_background_score_density)
    splines_s = fit_2D_splines_bin_by_bin_from_dict(s_dict)
    splines_b = fit_2D_splines_bin_by_bin_from_dict(b_dict)
    return splines_s, splines_b


def test_neg_log_prior_minimum_at_mean():
    """Prior should be minimised at theta == mu_theta and grow quadratically."""
    at_mean = neg_log_prior_theta(1.0, mu_theta=1.0, sigma_theta=0.05)
    off_mean = neg_log_prior_theta(1.05, mu_theta=1.0, sigma_theta=0.05)
    far_off = neg_log_prior_theta(1.10, mu_theta=1.0, sigma_theta=0.05)
    assert off_mean > at_mean
    assert far_off > off_mean
    # 1-sigma off should add exactly 0.5 to the NLL.
    assert np.isclose(off_mean - at_mean, 0.5, atol=1e-9)


def test_grid_bounds_match_training_extent(bin_splines):
    splines_s, _ = bin_splines
    nu1_bounds, nu2_bounds = _grid_bounds_from_splines(splines_s)
    # Grid was 0.9, 1.0, 1.1 in both dimensions.
    assert nu1_bounds[0] == pytest.approx(0.9, abs=1e-9)
    assert nu1_bounds[1] == pytest.approx(1.1, abs=1e-9)
    assert nu2_bounds[0] == pytest.approx(0.9, abs=1e-9)
    assert nu2_bounds[1] == pytest.approx(1.1, abs=1e-9)


@pytest.mark.parametrize("f_s_true", [0.05, 0.10, 0.20])
def test_recovers_injected_signal_fraction(bin_splines, f_s_true):
    """Fit recovers injected f_s within 30% (relative) at high stats.

    Tolerance is generous because: 
    (a) limited spline grid (3x3), 
    (b) finite sample, 
    (c) shape mismatch between sampling and the splined templates.
    
    The point is to verify the fit moves toward truth — not to benchmark it.
    """
    splines_s, splines_b = bin_splines
    rng = np.random.default_rng(SEED + int(f_s_true * 1000))

    n_total = 50_000
    nu1_true, nu2_true = 1.0, 1.0
    model = _make_canned_model(f_s_true, nu1_true, nu2_true, n_total, rng)

    dummy_2j = torch.zeros((n_total // 2, 1))
    dummy_1j = torch.zeros((n_total - n_total // 2, 1))

    f_s_hat = compute_signal_fraction(
        test_data_2j=dummy_2j,
        test_data_1j=dummy_1j,
        dnn_model=model,
        bin_splines_S=splines_s,
        bin_splines_BG=splines_b,
        eval_device="cpu",
        nuisance_priors=((1.0, 0.05), (1.0, 0.05)),
        initial_f_s=0.01,
        verbose=False,
    )

    rel_err = abs(f_s_hat - f_s_true) / f_s_true
    assert rel_err < 0.30, (
        f"f_s recovery off: injected={f_s_true:.4f}, fitted={f_s_hat:.4f}, "
        f"rel_err={rel_err:.2%}"
    )


def test_fit_does_not_escape_spline_grid(bin_splines):
    """Even with a deliberately bad starting point, the fit must stay inside
    the spline training box (this was the f_s≈0.8, nu≈1.17 failure mode)."""
    splines_s, splines_b = bin_splines
    rng = np.random.default_rng(SEED + 999)

    f_s_true = 0.10
    model = _make_canned_model(f_s_true, 1.0, 1.0, 30_000, rng)
    dummy = torch.zeros((15_000, 1))

    # loose prior + bad initial
    f_s_hat = compute_signal_fraction(
        test_data_2j=dummy,
        test_data_1j=dummy,
        dnn_model=model,
        bin_splines_S=splines_s,
        bin_splines_BG=splines_b,
        eval_device="cpu",
        nuisance_priors=((1.0, 0.5), (1.0, 0.5)),  # very loose
        initial_f_s=0.5,
        verbose=False,
    )
    # should still converge to something physical
    assert 0.0 < f_s_hat < 0.5, f"fit appears to have escaped: f_s_hat={f_s_hat}"
