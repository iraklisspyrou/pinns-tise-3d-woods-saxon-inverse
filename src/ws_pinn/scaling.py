"""Dimensionless input scaling used by WaveNet and ParamNet."""

import torch

from .runtime import device
from .constants import R_MAX, PI, TWOPI

def scale_energy(E):
    return E / 100.0


def scale_r(r):
    return r / R_MAX


def scale_theta(th):
    return th / PI


def scale_phi(ph):
    return ph / TWOPI


def scale_nucleus(A, Z, is_proton):
    return torch.tensor(
        [float(A) / 250.0, float(Z) / 100.0, 1.0 if is_proton else 0.0],
        dtype=torch.float32,
        device=device,
    )


def scale_quantum(nr, l, j):
    return torch.tensor(
        [float(nr) / 5.0, float(l) / 8.0, float(j) / 8.0],
        dtype=torch.float32,
        device=device,
    )
