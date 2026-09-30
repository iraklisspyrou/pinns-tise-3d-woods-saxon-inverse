#!/usr/bin/env python3
"""One FD + least-squares + MCMC baseline for spectra and spatial wavefunctions.

The likelihood uses only the 42 experimental identification energies. The
radial, polar, and azimuthal equations are solved numerically; posterior draws
also yield complete separable 3D wavefunctions on a recorded spherical grid.
The polar/azimuthal modes are reused across parameter proposals because their
equations do not depend on the Woods--Saxon interaction parameters.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.optimize import least_squares

from lsq_fit import (
    PARAMETER_NAMES,
    SEMINOLE_REFERENCE,
    build_pinn_identification_set,
    load_npz_dataset,
    solver_nucleus,
)
from ws_pinn.fd_solver import FDParameters, radial_eigenpairs, radial_eigenvalues
from ws_pinn.fd_spatial import azimuthal_mode, polar_mode, spatial_density


@dataclass(frozen=True)
class Channel:
    A: int
    Z: int
    is_proton: bool
    l: int
    j: float
    entries: tuple[tuple[int, int], ...]  # (dataframe position, radial index)

    @property
    def n_eigenvalues(self) -> int:
        return max(8, max(nr for _, nr in self.entries) + 4)


class SpatialForwardModel:
    """Evaluate energies and, when requested, radial eigenfunctions on states."""

    def __init__(self, states: pd.DataFrame, r_max: float, n_grid: int, prescription: str):
        if prescription not in ("direct", "schwierz"):
            raise ValueError("prescription must be direct or schwierz")
        self.states = states.reset_index(drop=True)
        self.r_max = r_max
        self.n_grid = n_grid
        self.calls = 0
        self.channel_solves = 0
        grouped: dict[tuple[int, int, bool, int, float], list[tuple[int, int]]] = {}
        for idx, row in self.states.iterrows():
            A, Z = solver_nucleus(row, prescription)
            key = (A, Z, bool(row.is_proton), int(row.l), float(row.j))
            grouped.setdefault(key, []).append((idx, int(row.nr)))
        self.channels = [Channel(*key, tuple(entries)) for key, entries in grouped.items()]

    def predict(
        self, parameters: np.ndarray, wave_indices: tuple[int, ...] = ()
    ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
        p = FDParameters(*map(float, parameters))
        requested = set(wave_indices)
        if any(idx < 0 or idx >= len(self.states) for idx in requested):
            raise ValueError("wave_indices contains an invalid state index")
        energies = np.empty(len(self.states), dtype=float)
        waves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.calls += 1
        for channel in self.channels:
            args = (
                channel.A, channel.Z, channel.is_proton, channel.l, channel.j,
                p, "seminole",
            )
            kwargs = dict(
                n_eigenvalues=channel.n_eigenvalues,
                r_max=self.r_max,
                n_grid=self.n_grid,
            )
            need_vectors = any(idx in requested for idx, _ in channel.entries)
            if need_vectors:
                values, grid, functions = radial_eigenpairs(*args, **kwargs)
            else:
                values = radial_eigenvalues(*args, **kwargs)
            self.channel_solves += 1
            for idx, nr in channel.entries:
                energies[idx] = values[nr]
                if idx in requested:
                    waves[idx] = (grid, functions[:, nr].copy())
        return energies, waves


def energy_metrics(prediction: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    error = prediction - observations
    return {
        "mae_mev": float(np.mean(np.abs(error))),
        "rmse_mev": float(np.sqrt(np.mean(error**2))),
        "max_abs_mev": float(np.max(np.abs(error))),
    }


def bounded_walkers(
    rng: np.random.Generator, center: np.ndarray,
    low: np.ndarray, high: np.ndarray, n_walkers: int,
) -> np.ndarray:
    scale = 0.015 * (high - low)
    positions = np.empty((n_walkers, len(center)))
    for i in range(n_walkers):
        for _ in range(10000):
            point = center + rng.normal(size=len(center)) * scale
            if np.all(point > low) and np.all(point < high):
                positions[i] = point
                break
        else:
            raise RuntimeError("Could not initialize walkers inside parameter bounds")
    return positions


def config_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["potential"] != "seminole":
        raise ValueError("The experimental baseline currently uses Seminole physics")
    bounds = config["parameter_bounds"]
    names = ("V0", "kappa", "r0", "a", "lam_so", "r0_so")
    limits = np.asarray([bounds[name] for name in names], dtype=float)
    if limits.shape != (6, 2) or not np.all(limits[:, 0] < limits[:, 1]):
        raise ValueError("Expected six strictly ordered parameter bounds")
    return limits[:, 0], limits[:, 1]


def full_selection_mask(full: pd.DataFrame, selected: pd.DataFrame) -> np.ndarray:
    columns = ["A", "Z", "is_proton", "nr", "l", "j", "energy"]
    full_index = pd.MultiIndex.from_frame(full[columns])
    selected_index = pd.MultiIndex.from_frame(selected[columns])
    mask = np.asarray(full_index.isin(selected_index), dtype=bool)
    if mask.sum() != len(selected) or selected_index.has_duplicates:
        raise RuntimeError("Identification levels cannot be uniquely matched to full data")
    return mask


def parse_indices(value: str, n_states: int) -> tuple[int, ...]:
    indices = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not indices or any(idx < 0 or idx >= n_states for idx in indices):
        raise ValueError(f"Spatial indices must be between 0 and {n_states - 1}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/seminole.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/experimental_dataset.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fd_mcmc"))
    parser.add_argument("--prescription", choices=("direct", "schwierz"), default="direct")
    parser.add_argument("--sigma-mev", type=float, default=1.0,
                        help="Assumed independent Gaussian energy error; includes model discrepancy")
    parser.add_argument("--r-max", type=float, default=25.0)
    parser.add_argument("--n-grid", type=int, default=2400)
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--n-r-spatial", type=int, default=64)
    parser.add_argument("--spatial-indices", default="36",
                        help="Comma-separated indices into selected_states.csv")
    parser.add_argument("--posterior-draws", type=int, default=24)
    parser.add_argument("--walkers", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.sigma_mev <= 0 or args.r_max <= 0 or args.n_grid < 4:
        parser.error("sigma-mev and r-max must be positive and n-grid >= 4")
    if args.walkers < 12 or args.walkers % 2 or args.steps <= args.burn_in or args.burn_in < 0:
        parser.error("Use an even number of walkers >= 12 and steps > burn-in >= 0")
    if args.posterior_draws < 1 or args.n_starts < 1 or args.max_nfev < 1:
        parser.error("posterior-draws, n-starts and max-nfev must be positive")
    if args.n_r_spatial < 4:
        parser.error("n-r-spatial must be >= 4")

    try:
        import emcee
    except ImportError as exc:
        raise SystemExit("Install the Bayesian extra: python -m pip install -e '.[bayesian]'") from exc

    total_start = perf_counter()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # emcee 3 uses NumPy's legacy RNG for proposals
    low, high = config_bounds(args.config)
    full = load_npz_dataset(args.dataset).reset_index(drop=True)
    selected = build_pinn_identification_set(full)
    selected_mask = full_selection_mask(full, selected)
    spatial_indices = parse_indices(args.spatial_indices, len(selected))
    observations = selected.energy.to_numpy(dtype=float)
    full_observations = full.energy.to_numpy(dtype=float)
    model = SpatialForwardModel(selected, args.r_max, args.n_grid, args.prescription)
    full_model = SpatialForwardModel(full, args.r_max, args.n_grid, args.prescription)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_states.csv", index=False)
    full.assign(used_for_identification=selected_mask).to_csv(
        args.output_dir / "all_states_and_split.csv", index=False
    )

    # The numerical angular operators use exactly the same l and m_l=0 for
    # every parameter proposal. This cache belongs to the integrated model.
    angular_start = perf_counter()
    polar_solutions = {l: polar_mode(l, args.n_theta) for l in sorted(set(selected.l))}
    phi_grid, phi_mode, phi_eigenvalue = azimuthal_mode(args.n_phi)
    angular_seconds = perf_counter() - angular_start

    fit_start = perf_counter()
    fit_runs = []
    for start_id in range(args.n_starts):
        x0 = np.clip(SEMINOLE_REFERENCE, low + 1e-6, high - 1e-6) if start_id == 0 else rng.uniform(low, high)

        def residual(x: np.ndarray) -> np.ndarray:
            return (model.predict(x)[0] - observations) / args.sigma_mev

        fit = least_squares(
            residual, x0, bounds=(low, high), method="trf",
            max_nfev=args.max_nfev, x_scale="jac",
        )
        fit_runs.append(fit)
        print(f"LSQ start {start_id + 1}/{args.n_starts}: cost={fit.cost:.5f}, "
              f"nfev={fit.nfev}, success={fit.success}", flush=True)
    best_fit = min(fit_runs, key=lambda result: result.cost)
    fit_seconds = perf_counter() - fit_start
    best_x = best_fit.x

    posterior_calls = 0
    rejected_by_prior = 0

    def log_probability(x: np.ndarray) -> float:
        nonlocal posterior_calls, rejected_by_prior
        if np.any(x <= low) or np.any(x >= high) or not np.all(np.isfinite(x)):
            rejected_by_prior += 1
            return -np.inf
        posterior_calls += 1
        predicted, _ = model.predict(x)
        residual_vector = (predicted - observations) / args.sigma_mev
        return -0.5 * float(np.dot(residual_vector, residual_vector))

    mcmc_start = perf_counter()
    starting_positions = bounded_walkers(rng, best_x, low, high, args.walkers)
    sampler = emcee.EnsembleSampler(args.walkers, 6, log_probability)
    sampler.run_mcmc(starting_positions, args.steps, progress=False)
    mcmc_seconds = perf_counter() - mcmc_start
    chain = sampler.get_chain()
    posterior = sampler.get_chain(discard=args.burn_in, flat=True)
    draws = posterior[rng.choice(len(posterior), size=min(args.posterior_draws, len(posterior)), replace=False)]
    try:
        tau = sampler.get_autocorr_time(discard=args.burn_in, tol=0).tolist()
        steps_per_tau = ((args.steps - args.burn_in) / np.asarray(tau)).tolist()
    except Exception:
        tau, steps_per_tau = None, None

    # All 96 levels are predicted here, but the 54 held-out observations have
    # never influenced LSQ, the energy likelihood, or posterior draw selection.
    predict_start = perf_counter()
    lsq_full, _ = full_model.predict(best_x)
    posterior_full = np.stack([full_model.predict(x)[0] for x in draws])
    mean_full = posterior_full.mean(axis=0)
    predictive_seconds = perf_counter() - predict_start
    state_table = full.assign(
        used_for_identification=selected_mask,
        lsq_fd_mev=lsq_full,
        posterior_mean_fd_mev=mean_full,
        posterior_q05_fd_mev=np.quantile(posterior_full, 0.05, axis=0),
        posterior_q95_fd_mev=np.quantile(posterior_full, 0.95, axis=0),
    )
    state_table.to_csv(args.output_dir / "posterior_state_predictions.csv", index=False)

    spatial_start = perf_counter()
    radial_points = (np.arange(args.n_r_spatial) + 0.5) * args.r_max / args.n_r_spatial
    _, lsq_waves = model.predict(best_x, spatial_indices)
    all_density: dict[int, list[np.ndarray]] = {idx: [] for idx in spatial_indices}
    all_radial: dict[int, list[np.ndarray]] = {idx: [] for idx in spatial_indices}
    for draw_id, parameters in enumerate(draws):
        _, waves = model.predict(parameters, spatial_indices)
        for idx in spatial_indices:
            row = selected.iloc[idx]
            theta_grid, theta_mode, theta_eigenvalue = polar_solutions[int(row.l)]
            grid, reduced = waves[idx]
            radial = np.interp(radial_points, grid, reduced) / radial_points
            all_radial[idx].append(radial)
            all_density[idx].append(
                spatial_density(grid, reduced, radial_points, theta_mode, phi_mode)
            )
    for idx in spatial_indices:
        row = selected.iloc[idx]
        theta_grid, theta_mode, theta_eigenvalue = polar_solutions[int(row.l)]
        grid, reduced = lsq_waves[idx]
        radial_lsq = np.interp(radial_points, grid, reduced) / radial_points
        density_lsq = spatial_density(grid, reduced, radial_points, theta_mode, phi_mode)
        densities = np.stack(all_density[idx])
        radial_draws = np.stack(all_radial[idx])
        # The real m_l=0 wavefunction of any draw is the outer product of its
        # saved radial component with these saved angular components.
        np.savez_compressed(
            args.output_dir / f"spatial_state_{idx}.npz",
            r=radial_points, theta=theta_grid, phi=phi_grid,
            radial_lsq=radial_lsq, radial_draws=radial_draws,
            polar=theta_mode, azimuthal=phi_mode,
            psi_lsq=radial_lsq[:, None, None] * theta_mode[None, :, None] * phi_mode[None, None, :],
            psi_representative=radial_draws[0, :, None, None] * theta_mode[None, :, None] * phi_mode[None, None, :],
            density_lsq=density_lsq,
            density_mean=densities.mean(axis=0),
            density_std=densities.std(axis=0),
            density_q05=np.quantile(densities, 0.05, axis=0),
            density_q95=np.quantile(densities, 0.95, axis=0),
            polar_eigenvalue=theta_eigenvalue,
            azimuthal_eigenvalue=phi_eigenvalue,
        )
    spatial_seconds = perf_counter() - spatial_start

    np.savez_compressed(
        args.output_dir / "posterior_chain.npz",
        chain=chain,
        log_probability=sampler.get_log_prob(),
        posterior_draws=draws,
        energy_predictions_all_states=posterior_full,
        parameter_names=np.asarray(PARAMETER_NAMES),
    )
    mean = posterior.mean(axis=0)
    std = posterior.std(axis=0, ddof=1)
    summary = {
        "dataset": str(args.dataset), "config": str(args.config),
        "prescription": args.prescription,
        "observations": len(selected), "held_out": int((~selected_mask).sum()),
        "assumptions": {
            "potential": "seminole", "magnetic_projection": 0,
            "prior": "independent uniform within config parameter_bounds",
            "likelihood": "independent Gaussian energy residuals",
            "sigma_mev": args.sigma_mev,
            "note": "Sigma is an assumed effective error scale, not measured calibration.",
        },
        "grids": {
            "radial": args.n_grid, "r_max_fm": args.r_max,
            "polar": args.n_theta, "azimuthal": args.n_phi,
            "spatial_radial": args.n_r_spatial,
        },
        "lsq": {
            "parameters": dict(zip(PARAMETER_NAMES, map(float, best_x))),
            "nfev": int(best_fit.nfev), "success": bool(best_fit.success),
            "n_starts": args.n_starts,
            "identification_metrics": energy_metrics(lsq_full[selected_mask], full_observations[selected_mask]),
            "held_out_metrics": energy_metrics(lsq_full[~selected_mask], full_observations[~selected_mask]),
        },
        "mcmc": {
            "walkers": args.walkers, "steps": args.steps, "burn_in": args.burn_in,
            "seed": args.seed, "likelihood_evaluations": posterior_calls,
            "prior_rejections": rejected_by_prior,
            "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "autocorrelation_time_steps_estimate": tau,
            "post_burnin_steps_per_tau_estimate": steps_per_tau,
            "convergence_verified": False,
            "convergence_note": "Inspect longer chains, autocorrelation stability, and multiple seeds before interpreting intervals as reliable.",
            "posterior_mean": dict(zip(PARAMETER_NAMES, mean.tolist())),
            "posterior_std": dict(zip(PARAMETER_NAMES, std.tolist())),
            "posterior_q05": dict(zip(PARAMETER_NAMES, np.quantile(posterior, 0.05, axis=0).tolist())),
            "posterior_q95": dict(zip(PARAMETER_NAMES, np.quantile(posterior, 0.95, axis=0).tolist())),
            "posterior_correlation": np.corrcoef(posterior, rowvar=False).tolist(),
            "draws_used_for_predictive_outputs": len(draws),
            "posterior_mean_identification_metrics": energy_metrics(mean_full[selected_mask], full_observations[selected_mask]),
            "posterior_mean_held_out_metrics": energy_metrics(mean_full[~selected_mask], full_observations[~selected_mask]),
        },
        "spatial_state_indices": list(spatial_indices),
        "angular_eigenvalues": {
            str(l): value for l, (_, _, value) in polar_solutions.items()
        },
        "azimuthal_eigenvalue": phi_eigenvalue,
        "timing_seconds": {
            "angular_odes": angular_seconds, "lsq": fit_seconds,
            "mcmc": mcmc_seconds, "posterior_predictive_all_96": predictive_seconds,
            "spatial_wavefunctions": spatial_seconds,
            "total": perf_counter() - total_start,
        },
        "forward_calls": model.calls + full_model.calls,
        "channel_eigensolves": model.channel_solves + full_model.channel_solves,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "emcee": emcee.__version__,
        },
        "device": "CPU (SciPy)",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Completed in {summary['timing_seconds']['total']:.1f} s", flush=True)
    print(f"Mean acceptance fraction: {summary['mcmc']['mean_acceptance_fraction']:.3f}")
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
