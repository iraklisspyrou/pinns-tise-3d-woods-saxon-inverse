"""Numerical angular ODEs and the spatial wavefunction for the paper's m_l=0 model.

No tabulated spherical harmonics or PINN outputs enter this reference solver.
The polar Sturm--Liouville equation is discretized in conservative form on
cell centers, and the azimuthal equation uses a periodic finite difference.
Both components are cached because the Woods--Saxon parameters affect only
the radial operator.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse import diags


@lru_cache(maxsize=None)
def polar_mode(l: int, n_theta: int = 128) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve -(sin(theta) T')' = lambda sin(theta) T for m_l=0.

    Half-cell centers avoid the coordinate singularities. Zero endpoint flux
    enforces the regular m_l=0 solution, with l nodes for the l-th eigenmode.
    The mode is normalized using sum(sin(theta) |T|^2 dtheta) = 1.
    """
    if l < 0 or n_theta < max(8, l + 2):
        raise ValueError("Need l >= 0 and n_theta >= max(8, l + 2)")
    step = np.pi / n_theta
    theta = (np.arange(n_theta) + 0.5) * step
    weight = np.sin(theta)
    flux = np.sin(np.arange(n_theta + 1) * step)
    diagonal = (flux[:-1] + flux[1:]) / (step * step * weight)
    off = -flux[1:-1] / (step * step * np.sqrt(weight[:-1] * weight[1:]))
    eigenvalue, vectors = eigh_tridiagonal(
        diagonal, off, select="i", select_range=(l, l),
        eigvals_only=False, check_finite=False,
    )
    polar = vectors[:, 0] / np.sqrt(weight * step)
    if polar[0] < 0:
        polar *= -1.0
    return theta, polar, float(eigenvalue[0])


@lru_cache(maxsize=None)
def azimuthal_mode(n_phi: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve -Phi'' = m_l^2 Phi on a periodic grid for m_l=0.

    WaveNet uses this magnetic projection in the reported experiments. The
    numerical null eigenmode is real; its global phase is set positive.
    """
    if n_phi < 8:
        raise ValueError("n_phi must be at least 8")
    step = 2.0 * np.pi / n_phi
    phi = np.arange(n_phi) * step
    laplacian = diags(
        (-np.ones(n_phi - 1), 2.0 * np.ones(n_phi), -np.ones(n_phi - 1)),
        offsets=(-1, 0, 1), format="lil",
    )
    laplacian[0, -1] = -1.0
    laplacian[-1, 0] = -1.0
    eigenvalue, vectors = eigh(
        laplacian.toarray() / step**2, subset_by_index=(0, 0),
        check_finite=False,
    )
    azimuthal = vectors[:, 0] / np.sqrt(step)
    if azimuthal[0] < 0:
        azimuthal *= -1.0
    return phi, azimuthal, float(eigenvalue[0])


def spatial_density(
    radial_grid: np.ndarray,
    reduced_radial: np.ndarray,
    radial_points: np.ndarray,
    polar: np.ndarray,
    azimuthal: np.ndarray,
) -> np.ndarray:
    """Evaluate |u(r)/r * Theta(theta) * Phi(phi)|^2 on a 3D grid."""
    if np.any(radial_points <= 0):
        raise ValueError("spatial evaluation points must have r > 0")
    radial = np.interp(radial_points, radial_grid, reduced_radial) / radial_points
    return (
        radial[:, None, None] ** 2
        * polar[None, :, None] ** 2
        * azimuthal[None, None, :] ** 2
    )
