"""Physical and data-split checks for the independent joint FD baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from fd_mcmc import SpatialForwardModel, full_selection_mask  # noqa: E402
from lsq_fit import (  # noqa: E402
    ForwardModel, SEMINOLE_REFERENCE, build_pinn_identification_set,
    load_npz_dataset,
)
from ws_pinn.fd_solver import (  # noqa: E402
    FDParameters, radial_eigenpairs, radial_eigenvalues,
)
from ws_pinn.fd_spatial import (  # noqa: E402
    azimuthal_mode, polar_mode, spatial_density,
)


class FDPhysicsTests(unittest.TestCase):
    def test_angular_modes_satisfy_known_eigenvalues_and_normalization(self):
        for l in range(6):
            theta, mode, eigenvalue = polar_mode(l, 128)
            step = np.pi / len(theta)
            self.assertAlmostEqual(eigenvalue, l * (l + 1), delta=0.05)
            self.assertAlmostEqual(np.sum(mode**2 * np.sin(theta)) * step, 1.0, places=10)
            expected = np.sqrt((2 * l + 1) / 2) * eval_legendre(l, np.cos(theta))
            self.assertLess(np.max(np.abs(mode - expected)), 0.035)
        _, _, eigenvalue_l7 = polar_mode(7, 128)
        self.assertAlmostEqual(eigenvalue_l7, 56.0, delta=0.2)
        phi, mode, eigenvalue = azimuthal_mode(64)
        self.assertAlmostEqual(eigenvalue, 0.0, delta=1e-10)
        self.assertLess(np.ptp(mode), 1e-10)
        self.assertAlmostEqual(np.sum(mode**2) * (2 * np.pi / len(phi)), 1.0, places=10)

    def test_radial_eigenfunctions_reproduce_energies_and_spatial_norm(self):
        params = FDParameters(*SEMINOLE_REFERENCE)
        values, grid, reduced = radial_eigenpairs(
            48, 20, False, 1, 1.5, params, "seminole",
            n_eigenvalues=4, n_grid=360,
        )
        energies = radial_eigenvalues(
            48, 20, False, 1, 1.5, params, "seminole",
            n_eigenvalues=4, n_grid=360,
        )
        np.testing.assert_allclose(values, energies, atol=1e-8, rtol=0)
        self.assertTrue(np.all(reduced[[0, -1]] == 0))
        dr = grid[1] - grid[0]
        self.assertAlmostEqual(np.sum(reduced[:, 0] ** 2) * dr, 1.0, places=9)
        theta, polar, _ = polar_mode(1, 64)
        phi, azimuthal, _ = azimuthal_mode(32)
        radial = grid[1:-1]
        rho = spatial_density(grid, reduced[:, 0], radial, polar, azimuthal)
        norm = np.sum(
            rho * radial[:, None, None] ** 2
            * np.sin(theta)[None, :, None]
        ) * dr * (np.pi / len(theta)) * (2 * np.pi / len(phi))
        self.assertAlmostEqual(norm, 1.0, places=7)

    def test_optimized_42_level_model_matches_legacy_at_two_parameter_sets(self):
        full = load_npz_dataset(ROOT / "data" / "experimental_dataset.npz")
        selected = build_pinn_identification_set(full)
        mask = full_selection_mask(full, selected)
        self.assertEqual((int(mask.sum()), int((~mask).sum())), (42, 54))
        fast = SpatialForwardModel(selected, 25.0, 320, "direct")
        old = ForwardModel(25.0, 320, "direct")
        for params in (SEMINOLE_REFERENCE, np.array([53.3, 0.70, 1.29, 0.65, 26.0, 1.20])):
            np.testing.assert_allclose(fast.predict(params)[0], old.predict(selected, params), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
