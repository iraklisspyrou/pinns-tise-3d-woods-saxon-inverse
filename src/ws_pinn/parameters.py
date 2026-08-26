"""Six-parameter representation and latent-to-physical mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch


PARAMETER_NAMES = ("V0", "kappa", "r0", "a", "lam_so", "r0_so")


@dataclass
class WSParams:
    V0: torch.Tensor
    kappa: torch.Tensor
    r0: torch.Tensor
    a: torch.Tensor
    lam_so: torch.Tensor
    r0_so: torch.Tensor


@dataclass(frozen=True)
class ParameterBounds:
    """Lower and upper bounds for the physical parameter mapping."""

    V0: tuple[float, float] = (40.0, 65.0)
    kappa: tuple[float, float] = (0.30, 1.00)
    r0: tuple[float, float] = (1.15, 1.35)
    a: tuple[float, float] = (0.55, 0.75)
    lam_so: tuple[float, float] = (15.0, 40.0)
    r0_so: tuple[float, float] = (0.90, 1.35)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "ParameterBounds":
        if values is None:
            return cls()
        converted: dict[str, tuple[float, float]] = {}
        for name in PARAMETER_NAMES:
            pair = values.get(name)
            if pair is None:
                continue
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Bounds for {name} must contain [lower, upper].")
            lower, upper = float(pair[0]), float(pair[1])
            if not lower < upper:
                raise ValueError(f"Bounds for {name} must satisfy lower < upper.")
            converted[name] = (lower, upper)
        return cls(**converted)

    def to_dict(self) -> dict[str, tuple[float, float]]:
        return asdict(self)


DEFAULT_BOUNDS = ParameterBounds()


def _bounded_component(raw: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    lower, upper = interval
    return lower + (upper - lower) * torch.sigmoid(raw)


def params_from_raw_six(
    raw_params: torch.Tensor,
    bounds: ParameterBounds = DEFAULT_BOUNDS,
) -> WSParams:
    """Map a six-dimensional latent vector into bounded physical space."""
    values = {
        name: _bounded_component(raw_params[index], getattr(bounds, name))
        for index, name in enumerate(PARAMETER_NAMES)
    }
    return WSParams(**values)


@torch.no_grad()
def physical_samples_from_mu_sigma(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_samples: int = 1000,
    bounds: ParameterBounds = DEFAULT_BOUNDS,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw latent samples and transform them to physical parameter space."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=mu.device)
        generator.manual_seed(seed)
    eps = torch.randn(
        n_samples,
        len(PARAMETER_NAMES),
        device=mu.device,
        generator=generator,
    )
    raw = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    columns = [
        _bounded_component(raw[:, index], getattr(bounds, name))
        for index, name in enumerate(PARAMETER_NAMES)
    ]
    return torch.stack(columns, dim=1)


@torch.no_grad()
def physical_stats_from_mu_sigma(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_samples: int = 1000,
    bounds: ParameterBounds = DEFAULT_BOUNDS,
    seed: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Return physical-space means and model-derived output spreads."""
    samples = physical_samples_from_mu_sigma(mu, sigma, n_samples, bounds, seed)
    return {
        name: (
            float(samples[:, index].mean()),
            float(samples[:, index].std(unbiased=False)),
        )
        for index, name in enumerate(PARAMETER_NAMES)
    }


@torch.no_grad()
def physical_mean_from_mu_sigma(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_samples: int = 10_000,
    bounds: ParameterBounds = DEFAULT_BOUNDS,
    seed: int = 0,
) -> WSParams:
    """Construct the selection-free estimator E[g(z)] in physical space."""
    samples = physical_samples_from_mu_sigma(mu, sigma, n_samples, bounds, seed)
    means = samples.mean(dim=0)
    return WSParams(**{
        name: means[index]
        for index, name in enumerate(PARAMETER_NAMES)
    })
