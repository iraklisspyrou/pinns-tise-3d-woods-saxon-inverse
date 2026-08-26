"""Selection-free physical-space parameter inference."""

import torch

from .parameters import (
    DEFAULT_BOUNDS,
    ParameterBounds,
    physical_mean_from_mu_sigma,
    physical_stats_from_mu_sigma,
)
from .wavefunctions import make_global_param_context

@torch.no_grad()
def infer_global_parameters(
    wave_net,
    param_net,
    samples,
    n_samples=5000,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=None,
):
    wave_net.eval()
    param_net.eval()

    X_global = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
    )

    mu, sigma, _ = param_net(X_global)
    return physical_stats_from_mu_sigma(
        mu,
        sigma,
        n_samples=n_samples,
        bounds=parameter_bounds,
        seed=seed,
    )


@torch.no_grad()
def infer_physical_mean_parameters(
    wave_net,
    param_net,
    samples,
    n_samples=10_000,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=0,
):
    """Return the primary estimator E[g(z)] in physical parameter space."""
    wave_net.eval()
    param_net.eval()
    context = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
    )
    mu, sigma, _ = param_net(context)
    return physical_mean_from_mu_sigma(
        mu,
        sigma,
        n_samples=n_samples,
        bounds=parameter_bounds,
        seed=seed,
    )


@torch.no_grad()
def infer_global_raw_latent(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    radial_normalization="full_separable",
    sign_probe_index=5,
):
    wave_net.eval()
    param_net.eval()

    X_global = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
    )
    mu, sigma, logvar = param_net(X_global)
    return mu.detach().cpu().tolist(), sigma.detach().cpu().tolist(), logvar.detach().cpu().tolist()
