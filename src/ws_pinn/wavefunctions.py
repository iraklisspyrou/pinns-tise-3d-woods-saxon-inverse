"""Automatic differentiation, quadrature, normalization, and ParamNet probes."""

from functools import lru_cache
import numpy as np
import torch

from .normalization import phi_reference_key

from .constants import PI, R_MAX, TWOPI
from .runtime import device
from .scaling import scale_energy, scale_nucleus, scale_quantum


POLAR_QUADRATURE_RULE = "log_gauss_legendre_v1"
POLAR_EPS = 1e-5


@lru_cache(maxsize=32)
def _polar_rule_numpy(n):
    """Gauss rule in t=log(tan(theta/2)), on the original polar interval."""
    if n < 2:
        raise ValueError("Polar quadrature requires at least two points.")
    x, w = np.polynomial.legendre.leggauss(n)
    limit = -np.log(np.tan(POLAR_EPS / 2.0))
    theta = 2.0 * np.arctan(np.exp(limit * x))
    return theta, limit * w


@lru_cache(maxsize=64)
def _polar_rule_torch(n, target_device, dtype):
    theta, weights_dt = _polar_rule_numpy(n)
    return (
        torch.tensor(theta, device=target_device, dtype=dtype),
        torch.tensor(weights_dt, device=target_device, dtype=dtype),
    )


def polar_integral(values, theta, rule=POLAR_QUADRATURE_RULE):
    """Integrate values dtheta on the polar nodes returned by make_grids.

    dtheta=sin(theta) dt. The weighted rule resolves both endpoint layers;
    in particular, the 1/sin(theta) energy factor cancels this Jacobian.
    Coordinates and quadrature weights are fixed with respect to model
    parameters. Gradients through values, including coordinate derivatives
    of the learned wavefunctions, remain available.
    """
    theta = theta.reshape(-1)
    if rule == "uniform_trapezoid_v1":
        return torch.trapz(values, theta)
    if rule != POLAR_QUADRATURE_RULE:
        raise ValueError(f"Unknown polar quadrature: {rule}")
    if values.ndim != 1 or values.shape != theta.shape:
        raise ValueError("Polar integrand and nodes must be matching vectors.")
    _, weights_dt = _polar_rule_torch(theta.numel(), theta.device, theta.dtype)
    return torch.sum(values * torch.sin(theta.detach()) * weights_dt)


def d(f, x):
    return torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


def make_grids(Nr, Nth, Nph, requires_grad=True, polar_rule=POLAR_QUADRATURE_RULE):
    """Construct differentiable spherical-coordinate quadrature grids.

    Production calls pass ``Nr``, ``Nth``, and ``Nph`` from the YAML
    ``quadrature`` section.  The polar endpoints are displaced from the
    coordinate singularities, and the duplicated periodic endpoint at
    ``2*pi`` is omitted from the azimuthal grid.
    """
    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    if polar_rule == POLAR_QUADRATURE_RULE:
        polar_nodes, _ = _polar_rule_torch(int(Nth), r.device, r.dtype)
        th = polar_nodes.clone().unsqueeze(1)
    elif polar_rule == "uniform_trapezoid_v1":
        th = torch.linspace(POLAR_EPS, PI-POLAR_EPS, Nth, device=device).unsqueeze(1)
    else:
        raise ValueError(f"Unknown polar quadrature: {polar_rule}")
    ph = torch.linspace(0.0, TWOPI, Nph + 1, device=device)[:-1].unsqueeze(1)
    r.requires_grad_(requires_grad)
    th.requires_grad_(requires_grad)
    ph.requires_grad_(requires_grad)
    return r, th, ph


def eval_R(wave_net, r, E_vec, sample, st):
    return wave_net.R(r, E_vec, sample, st)


def eval_Theta(wave_net, th, E_vec, sample, st):
    return wave_net.Theta(th, E_vec, sample, st)


def eval_Phi(wave_net, ph, E_vec, sample, st):
    return wave_net.Phi(ph, E_vec, sample, st)


def norm_parts(wave_net, E_vec, sample, st, Nr, Nth, Nph, eps=1e-12):
    """Return separated norm integrals and the full normalization coefficient.

    The volume element is ``r^2 sin(theta) dr dtheta dphi``.  Following the
    manuscript convention, the single coefficient
    ``alpha=(I_R I_theta I_phi)^(-1/2)`` is later applied to the radial
    component only; it is not three independent normalization operations.
    """
    rule = getattr(wave_net, "polar_quadrature", POLAR_QUADRATURE_RULE)
    r, th, ph = make_grids(Nr, Nth, Nph, requires_grad=True, polar_rule=rule)

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)

    r1, th1, ph1 = r.squeeze(), th.squeeze(), ph.squeeze()

    IR = torch.trapz((Rv.squeeze() ** 2) * (r1**2), r1)
    sinth = torch.sin(th1).clamp_min(1e-8)
    Ith = polar_integral((Thv.squeeze() ** 2) * sinth, th1, rule)
    Phi2 = a_re.squeeze() ** 2 + b_im.squeeze() ** 2
    Iphi = torch.trapz(Phi2, ph1)

    Iang = Ith * Iphi
    I = (IR * Iang).clamp_min(eps)
    s = torch.rsqrt(I)

    return IR, Ith, Iphi, Iang, I, s, r, th, ph


def psi_scale_only(wave_net, E_vec, sample, st, Nr, Nth, Nph):
    IR, Ith, Iphi, Iang, I, s, *_ = norm_parts(
        wave_net,
        E_vec,
        sample,
        st,
        Nr,
        Nth,
        Nph,
    )
    return IR, Ith, Iphi, Iang, I, s


def eval_R_norm(wave_net, r, E_vec, sample, st, Nr, Nth, Nph):
    _, _, _, _, _, s = psi_scale_only(wave_net, E_vec, sample, st, Nr, Nth, Nph)
    return s * eval_R(wave_net, r, E_vec, sample, st), s


def eval_psi_norm(wave_net, r, th, ph, E_vec, sample, st, Nr, Nth, Nph):
    _, _, _, _, _, s = psi_scale_only(wave_net, E_vec, sample, st, Nr, Nth, Nph)
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    return s * Rv * Thv * a_re, s * Rv * Thv * b_im, s


def phi_norm_ratio(wave_net, E_vec, sample, st, current_phi):
    """I_phi/I_phi_initial; one for explicitly restored legacy models."""
    refs = getattr(wave_net, "phi_reference_norms", None)
    if refs is None:
        return current_phi.new_ones(())
    key = phi_reference_key(sample, st)
    if key not in refs:
        raise ValueError("No saved Phi reference for this identification context/state.")
    return current_phi / current_phi.new_tensor(refs[key])


def make_param_input_from_radial_psi(
    wave_net,
    sample,
    E_vec,
    n_r_points,
    Nr_norm,
    Nth_norm,
    Nph_norm,
    radial_normalization="full_separable",
    sign_probe_index=5,
):
    """
    One system-level context vector:
      normalized radial probe values for all selected states
      + target energies
      + nucleus descriptors
      + quantum numbers.

    ``full_separable`` implements the manuscript convention

        alpha = (I_R I_theta I_phi)^(-1/2),  R_tilde = alpha R,

    with the single complete-wavefunction coefficient applied to the radial
    component only. For corrected models these probes additionally receive
    sqrt(I_phi/I_phi_initial), removing only their Phi-amplitude dependence.
    ``radial_l2`` reproduces the uploaded legacy script and is
    retained only for evaluating checkpoints trained with that input convention.
    """
    # ParamNet sees the wavefunction at a fixed, ordered radial probe grid.
    # Its resolution is configured independently from the norm quadrature.
    r = torch.linspace(0.0, R_MAX, n_r_points, device=device).unsqueeze(1)
    radial_parts = []

    if radial_normalization not in {"full_separable", "radial_l2"}:
        raise ValueError(
            "radial_normalization must be 'full_separable' or 'radial_l2'."
        )

    for st in sample["states"]:
        IR, _, _, _, _, alpha = psi_scale_only(
            wave_net,
            E_vec,
            sample,
            st,
            Nr_norm,
            Nth_norm,
            Nph_norm,
        )
        R_raw = eval_R(wave_net, r, E_vec, sample, st)
        if radial_normalization == "full_separable":
            R_normalized = alpha * R_raw
        else:
            R_normalized = R_raw / torch.sqrt(IR.clamp_min(1e-12))

        R_vector = R_normalized.squeeze()
        index = min(int(sign_probe_index), R_vector.numel() - 1)
        sign = torch.sign(R_vector[index].detach())
        if sign.item() == 0.0:
            sign = torch.tensor(1.0, device=device)
        radial_parts.append(sign * R_vector)

    psi_vec = torch.cat(radial_parts)
    E_scaled = scale_energy(E_vec)
    nuc = scale_nucleus(sample["A"], sample["Z"], sample["is_proton"])
    q_vec = torch.cat(
        [scale_quantum(st["nr"], st["l"], st["j"]) for st in sample["states"]]
    )

    context = torch.cat([psi_vec, E_scaled, nuc, q_vec])
    if radial_normalization == "full_separable" and getattr(wave_net, "phi_reference_norms", None) is not None:
        # Match the working Colab evaluation order and autograd graph exactly.
        gains = []
        for st in sample["states"]:
            Iphi = norm_parts(wave_net, E_vec, sample, st, Nr_norm, Nth_norm, Nph_norm)[2]
            gains.append(torch.sqrt(phi_norm_ratio(wave_net, E_vec, sample, st, Iphi)))
        k = len(sample["states"])
        radial = context[:k*n_r_points].reshape(k, n_r_points) * torch.stack(gains)[:, None]
        context = torch.cat([radial.flatten(), context[k*n_r_points:]])
    return context


def make_global_param_context(
    wave_net,
    samples,
    n_r_points,
    Nr_norm,
    Nth_norm,
    Nph_norm,
    radial_normalization="full_separable",
    sign_probe_index=5,
):
    """
    Matrix of sample-level contexts.

    Shape:
        (N_samples, input_dim)

    ParamNet will pool this matrix and output one global q(z).
    """
    x_list = []

    for sample in samples:
        E_vec = torch.tensor(
            [float(st["energy"]) for st in sample["states"]],
            dtype=torch.float32,
            device=device,
        )
        x_sample = make_param_input_from_radial_psi(
            wave_net=wave_net,
            sample=sample,
            E_vec=E_vec,
            n_r_points=n_r_points,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            radial_normalization=radial_normalization,
            sign_probe_index=sign_probe_index,
        )
        x_list.append(x_sample)

    return torch.stack(x_list, dim=0)
