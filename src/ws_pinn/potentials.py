"""Seminole and Wahlborn Woods--Saxon Hamiltonian terms.

The training configuration selects one expression before optimization.  The
two expressions differ only in the isospin-dependent central depth and in the
strength multiplying the spin--orbit form factor.
"""

import torch

from .constants import hc, u, e2
from .parameters import WSParams

_POTENTIAL_EXPRESSION = "seminole"


def set_potential_expression(name: str) -> None:
    """Select the central-depth/spin--orbit convention used by the PINN."""
    global _POTENTIAL_EXPRESSION
    value = str(name).strip().lower()
    if value not in {"seminole", "wahlborn"}:
        raise ValueError("potential must be 'seminole' or 'wahlborn'")
    _POTENTIAL_EXPRESSION = value


def get_potential_expression() -> str:
    return _POTENTIAL_EXPRESSION

def reduced_mass(A: int, is_proton: bool):
    m_core = float(A - 1) * u
    m_nucl = (1.007276466621 * u) if is_proton else (1.00866491588 * u)
    return (m_core * m_nucl) / (m_core + m_nucl)


def K_value(mu):
    return hc**2 / (2.0 * mu)


def l_dot_s(l: int, j: float):
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def R_central(A, params: WSParams):
    return params.r0 * (float(A) ** (1.0 / 3.0))


def R_spin_orbit(A, params: WSParams):
    return params.r0_so * (float(A) ** (1.0 / 3.0))


def f_ws(r, R, a):
    x = ((r - R) / a).clamp(-80.0, 80.0)
    return 1.0 / (1.0 + torch.exp(x))


def df_ws_dr(r, R, a):
    x = ((r - R) / a).clamp(-80.0, 80.0)
    ex = torch.exp(x)

    return -(ex / (1.0 + ex) ** 2) * (1.0 / a)


def tdotT_expectation(A: int, Z: int, is_proton: bool):
    """
    Seminole / Schwierz expectation value:

        <t . T_core> = -1/4 * rhs

    with the proton/neutron dependence defined through the
    isospin of the core.
    """
    N = int(A) - int(Z)

    if N == Z:
        rhs = 3.0

    elif N > Z:
        rhs = (
            (N - Z + 1)
            if is_proton
            else -(N - Z + 1)
        ) + 2.0

    else:
        rhs = (
            (N - Z - 1)
            if is_proton
            else -(N - Z - 1)
        ) + 2.0

    return -0.25 * float(rhs)


def V_depth(A, Z, is_proton, params: WSParams):
    """Return the species-dependent central depth.

    Seminole:
        D_tau = V0 [1 - 4 kappa <t.T_core>/A]

    Wahlborn:
        D_tau = V0 [1 + eta_tau kappa (N-Z)/A],
        with eta_p=+1 and eta_n=-1.
    """
    if _POTENTIAL_EXPRESSION == "wahlborn":
        N = int(A) - int(Z)
        eta_tau = 1.0 if is_proton else -1.0
        return params.V0 * (
            1.0 + eta_tau * params.kappa * float(N - int(Z)) / float(A)
        )

    tdotT = tdotT_expectation(
        A,
        Z,
        is_proton,
    )

    return params.V0 * (
        1.0
        - (
            4.0
            * params.kappa
            / float(A)
        )
        * tdotT
    )


def V_central(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    """
    Central Woods-Saxon potential:

        Vcentral(r) = -V_tau * f(r, R, a)
    """
    Vtau = V_depth(
        A,
        Z,
        is_proton,
        params,
    )

    R = R_central(A, params)

    return -Vtau * f_ws(
        r,
        R,
        params.a,
    )


def Vcentral_r2(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    return (r**2) * V_central(
        r,
        A,
        Z,
        is_proton,
        params,
    )


def Vc_r2(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    """
    r^2-scaled Coulomb potential for a uniformly charged sphere.
    """
    if not is_proton:
        return torch.zeros_like(r)

    Z_core = max(int(Z) - 1, 0)
    Zc = float(Z_core)

    R = R_central(A, params)
    rc = r.clamp_min(0.0)

    inside = rc <= R
    outside = ~inside

    out = torch.empty_like(rc)

    out[inside] = (
        Zc
        * e2
        / (2.0 * R)
        * (
            3.0 * rc[inside] ** 2
            - rc[inside] ** 4 / R**2
        )
    )

    out[outside] = (
        Zc
        * e2
        * rc[outside]
    )

    return out


def Vso_r2(
    r,
    A,
    Z,
    is_proton,
    l,
    j,
    params: WSParams,
):
    """Return the r^2-scaled spin--orbit potential.

    Seminole uses lambda_SO V0, while Wahlborn uses lambda_SO D_tau.
    """
    mu = reduced_mass(
        A,
        is_proton,
    )

    rc = r.clamp_min(1e-10)
    Rso = R_spin_orbit(A, params)

    depth = (
        V_depth(A, Z, is_proton, params)
        if _POTENTIAL_EXPRESSION == "wahlborn"
        else params.V0
    )
    Vtilde = params.lam_so * depth

    dVt_dr = (
        Vtilde
        * df_ws_dr(
            rc,
            Rso,
            params.a,
        )
    )

    pref_r2 = (
        rc
        * hc**2
        / (2.0 * mu**2)
    )

    return (
        pref_r2
        * dVt_dr
        * float(
            l_dot_s(l, j)
        )
    )


def Veff_r2(
    r,
    A,
    Z,
    is_proton,
    l,
    j,
    params: WSParams,
):
    """
    Full r^2-scaled effective potential:

        r^2 Veff
        =
        r^2 Vcentral
        + r^2 VC
        + r^2 VSO
    """
    return (
        Vcentral_r2(
            r,
            A,
            Z,
            is_proton,
            params,
        )
        + Vc_r2(
            r,
            A,
            Z,
            is_proton,
            params,
        )
        + Vso_r2(
            r,
            A,
            Z,
            is_proton,
            l,
            j,
            params,
        )
    )
