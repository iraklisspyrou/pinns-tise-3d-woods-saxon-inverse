"""Physics-informed objective terms and multi-system aggregation."""

import numpy as np
import torch

from .constants import PRIOR_STD, R_MAX, PI, TWOPI
from .runtime import device
from .parameters import DEFAULT_BOUNDS, ParameterBounds, params_from_raw_six
from .potentials import K_value, Veff_r2, reduced_mass
from .wavefunctions import (
    d,
    eval_Phi,
    eval_R,
    eval_R_norm,
    eval_Theta,
    eval_psi_norm,
    make_global_param_context,
    norm_parts,
    psi_scale_only,
)

def energy_rayleigh_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    params,
    Nr=512,
    Nth=256,
    Nph=256,
    eps=1e-12,
):
    A, Z, is_proton = sample["A"], sample["Z"], sample["is_proton"]
    ell, j = int(st["l"]), float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    IR, Ith, Iphi, Iang, _, s, r, th, ph = norm_parts(
        wave_net,
        E_vec,
        sample,
        st,
        Nr,
        Nth,
        Nph,
        eps,
    )

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)

    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)

    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = d(a_re, ph), d(b_im, ph)

    r1, th1, ph1 = r.squeeze(), th.squeeze(), ph.squeeze()

    IRprime = torch.trapz((Rr.squeeze() ** 2) * (r1**2), r1)
    IR0 = torch.trapz(Rv.squeeze() ** 2, r1)

    Veff_r2_vals = Veff_r2(r, A, Z, is_proton, ell, j, params).squeeze()
    IVR = torch.trapz(Veff_r2_vals * (Rv.squeeze() ** 2), r1)

    sinth = torch.sin(th1).clamp_min(1e-8)
    Ith = torch.trapz((Thv.squeeze() ** 2) * sinth, th1)
    Ithprime = torch.trapz((Th_th.squeeze() ** 2) * sinth, th1)
    Ith_over_sin = torch.trapz((Thv.squeeze() ** 2) / sinth, th1)

    Phi2 = a_re.squeeze() ** 2 + b_im.squeeze() ** 2
    Iphi = torch.trapz(Phi2, ph1)
    PhiP2 = ap.squeeze() ** 2 + bp.squeeze() ** 2
    Iphi_prime = torch.trapz(PhiP2, ph1)

    A0 = Ith * Iphi

    T_r = s * s * IRprime * A0
    T_th = s * s * IR0 * Ithprime * Iphi
    T_ph = s * s * IR0 * Ith_over_sin * Iphi_prime
    U = s * s * IVR * A0

    num = K * (T_r + T_th + T_ph) + U
    den = (s * s * IR * Iang).clamp_min(eps)

    return (num / den).squeeze()


def loss_pde_radial_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    params,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    N=256,
):
    A, Z, is_proton = sample["A"], sample["Z"], sample["is_proton"]
    ell, j = int(st["l"]), float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    r = torch.rand(N, 1, device=device) * R_MAX
    r.requires_grad_(True)

    _, _, _, _, _, s = psi_scale_only(
        wave_net,
        E_vec,
        sample,
        st,
        Nr_norm,
        Nth_norm,
        Nph_norm,
    )

    E_det = energy_rayleigh_full3d(
        wave_net,
        E_vec,
        sample,
        st,
        params,
        Nr_norm,
        Nth_norm,
        Nph_norm,
    ).detach()

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)
    Rrr = d(Rr, r)

    rc = r.clamp_min(1e-8)
    Rn, Rnr, Rnrr = s * Rv, s * Rr, s * Rrr

    Veff_r2_vals = Veff_r2(rc, A, Z, is_proton, ell, j, params)

    residual = (
        -K * ((rc**2) * Rnrr + 2.0 * rc * Rnr - ell * (ell + 1.0) * Rn)
        + (Veff_r2_vals - (rc**2) * E_det) * Rn
    )

    return torch.mean(residual**2)


def loss_pde_theta_full3d(wave_net, E_vec, sample, st, emm=0, N=256):
    ell = int(st["l"])

    th = (torch.rand(N, 1, device=device) * (1.0 - 2e-3) + 1e-3) * PI
    th.requires_grad_(True)

    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)
    Th_thth = d(Th_th, th)

    sinth, costh = torch.sin(th), torch.cos(th)
    sin2 = sinth**2

    op = (
        sin2 * Th_thth
        + sinth * costh * Th_th
        + (ell * (ell + 1.0) * sin2 - float(emm**2)) * Thv
    )

    return torch.mean(op**2)


def loss_pde_phi_full3d(wave_net, E_vec, sample, st, emm=0, N=256, Np_bc=64):
    ph = torch.rand(N, 1, device=device) * TWOPI
    ph.requires_grad_(True)

    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = d(a_re, ph), d(b_im, ph)
    app, bpp = d(ap, ph), d(bp, ph)

    L_ode = ((app + float(emm**2) * a_re) ** 2 + (bpp + float(emm**2) * b_im) ** 2).mean()

    z0 = torch.zeros(Np_bc, 1, device=device, requires_grad=True)
    z2 = torch.full_like(z0, TWOPI, requires_grad=True)

    a0, b0 = eval_Phi(wave_net, z0, E_vec, sample, st)
    a2, b2 = eval_Phi(wave_net, z2, E_vec, sample, st)

    ap0, bp0 = d(a0, z0), d(b0, z0)
    ap2, bp2 = d(a2, z2), d(b2, z2)

    L_per = ((a0 - a2) ** 2 + (b0 - b2) ** 2 + (ap0 - ap2) ** 2 + (bp0 - bp2) ** 2).mean()
    L_gauge = ((a0 - 1.0) ** 2 + b0**2 + ap0**2 + (bp0 - float(emm)) ** 2).mean()

    return L_ode + L_per + 2.0 * L_gauge


def loss_bc_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    Nb=64,
):
    ell = int(st["l"])

    rR = torch.full((Nb, 1), R_MAX, device=device, requires_grad=True)
    RnR, _ = eval_R_norm(wave_net, rR, E_vec, sample, st, Nr_norm, Nth_norm, Nph_norm)
    L_Rmax = torch.mean(RnR**2)

    r0 = torch.zeros((Nb, 1), device=device, requires_grad=True)
    Rn0, _ = eval_R_norm(wave_net, r0, E_vec, sample, st, Nr_norm, Nth_norm, Nph_norm)

    if ell == 0:
        dRn0 = d(Rn0, r0)
        L_0 = torch.mean(dRn0**2)
    elif ell == 1:
        L_0 = torch.mean(Rn0**2)
    else:
        dRn0 = d(Rn0, r0)
        L_0 = torch.mean(Rn0**2) + torch.mean(dRn0**2)

    return L_Rmax + L_0


def loss_orthogonality_full3d(
    wave_net,
    sample,
    E_vec,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    N=512,
):
    """
    Selective orthogonality only for same (l,j) and different nr.
    Spin-orbit partners are not penalized here.
    """
    states = sample["states"]
    K_states = len(states)

    if K_states < 2:
        return torch.tensor(0.0, device=device)

    r = torch.rand(N, 1, device=device) * R_MAX
    th = torch.rand(N, 1, device=device) * PI
    ph = torch.rand(N, 1, device=device) * TWOPI

    weight = (r.squeeze() ** 2) * torch.sin(th.squeeze()).clamp_min(1e-12)
    volume = R_MAX * PI * TWOPI

    psis = []
    for st in states:
        Re, Im, _ = eval_psi_norm(
            wave_net,
            r,
            th,
            ph,
            E_vec,
            sample,
            st,
            Nr=Nr_norm,
            Nth=Nth_norm,
            Nph=Nph_norm,
        )
        psis.append((Re.squeeze(), Im.squeeze()))

    total = torch.tensor(0.0, device=device)
    n_pairs = 0

    for i in range(K_states):
        st_i = states[i]
        nr_i = int(st_i["nr"])
        l_i = int(st_i["l"])
        j_i = float(st_i["j"])
        Re_i, Im_i = psis[i]

        for j_idx in range(i):
            st_j = states[j_idx]
            nr_j = int(st_j["nr"])
            l_j = int(st_j["l"])
            j_j = float(st_j["j"])

            same_lj_different_nr = (
                l_i == l_j
                and abs(j_i - j_j) < 1e-6
                and nr_i != nr_j
            )

            if not same_lj_different_nr:
                continue

            Re_j, Im_j = psis[j_idx]

            overlap_re_density = Re_j * Re_i + Im_j * Im_i
            overlap_im_density = Re_j * Im_i - Im_j * Re_i

            overlap_re = volume * torch.mean(overlap_re_density * weight)
            overlap_im = volume * torch.mean(overlap_im_density * weight)

            total = total + overlap_re**2 + overlap_im**2
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    return total / n_pairs


def loss_spin_orbit_splitting(pred_energy_map, target_energy_map, states):
    """
    Explicit splitting loss for pairs:
        (nr, l, j=l-1/2) and (nr, l, j=l+1/2).

    This helps constrain lambda_SO and r0_SO.
    """
    LSO = torch.tensor(0.0, device=device)
    n_pairs = 0
    seen = set()

    for st in states:
        nr = int(st["nr"])
        l = int(st["l"])

        if l == 0:
            continue

        jm = round(float(l) - 0.5, 6)
        jp = round(float(l) + 0.5, 6)

        key_m = (nr, l, jm)
        key_p = (nr, l, jp)
        pair_id = (nr, l)

        if pair_id in seen:
            continue

        if key_m in pred_energy_map and key_p in pred_energy_map:
            pred_split = pred_energy_map[key_p] - pred_energy_map[key_m]
            target_split = target_energy_map[key_p] - target_energy_map[key_m]
            LSO = LSO + (pred_split - target_split) ** 2
            n_pairs += 1
            seen.add(pair_id)

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    return LSO / n_pairs


def gaussian_kl_to_isotropic_prior(mu, logvar, prior_std=PRIOR_STD):
    prior_std_tensor = torch.as_tensor(
        prior_std,
        dtype=mu.dtype,
        device=mu.device,
    )

    if prior_std_tensor.item() <= 0.0:
        raise ValueError("prior_std must be strictly positive.")

    return 0.5 * torch.sum(
        (logvar.exp() + mu.pow(2)) / prior_std_tensor.pow(2)
        - 1.0
        + 2.0 * torch.log(prior_std_tensor)
        - logvar
    )


def compute_sample_physics_loss_full3d(
    wave_net,
    sample,
    params,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    emm=0,
    wE=1.0,
    wR=10.0,
    wTh=5.0,
    wPh=5.0,
    wBC=5.0,
    wORTH=0.0,
    wSO=0.0,
    orth_points=512,
):
    states = sample["states"]
    K_states = len(states)

    E_vec = torch.tensor(
        [float(st["energy"]) for st in states],
        dtype=torch.float32,
        device=device,
    )

    LE = torch.tensor(0.0, device=device)
    LR = torch.tensor(0.0, device=device)
    LTH = torch.tensor(0.0, device=device)
    LPH = torch.tensor(0.0, device=device)
    LBC = torch.tensor(0.0, device=device)

    rows = []
    pred_energy_map = {}
    target_energy_map = {}

    for i, st in enumerate(states):
        E_target = E_vec[i]

        E_pred = energy_rayleigh_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            params,
            Nr_norm,
            Nth_norm,
            Nph_norm,
        )

        LE = LE + (E_pred - E_target) ** 2

        LR = LR + loss_pde_radial_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            params,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            N=256,
        )

        LTH = LTH + loss_pde_theta_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            emm=emm,
            N=256,
        )

        LPH = LPH + loss_pde_phi_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            emm=emm,
            N=256,
            Np_bc=64,
        )

        LBC = LBC + loss_bc_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            Nb=64,
        )

        key = (int(st["nr"]), int(st["l"]), round(float(st["j"]), 6))
        pred_energy_map[key] = E_pred
        target_energy_map[key] = E_target

        rows.append(
            (
                st,
                float(E_pred.detach().cpu()),
                float(E_target.detach().cpu()),
            )
        )

    LE = LE / K_states
    LR = LR / K_states
    LTH = LTH / K_states
    LPH = LPH / K_states
    LBC = LBC / K_states

    if wORTH > 0.0:
        LORTH = loss_orthogonality_full3d(
            wave_net,
            sample,
            E_vec,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            N=orth_points,
        )
    else:
        LORTH = torch.tensor(0.0, device=device)

    if wSO > 0.0:
        LSO = loss_spin_orbit_splitting(pred_energy_map, target_energy_map, states)
    else:
        LSO = torch.tensor(0.0, device=device)

    physics_loss = (
        wE * LE
        + wR * LR
        + wTh * LTH
        + wPh * LPH
        + wBC * LBC
        + wORTH * LORTH
        + wSO * LSO
    )

    return physics_loss, LE, LR, LTH, LPH, LBC, LORTH, LSO, rows


def compute_multinucleus_loss_full3d(
    wave_net,
    param_net,
    samples,
    batch_size=None,
    prior_std=PRIOR_STD,
    wKL=0.0,
    is_training=True,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    **loss_kwargs,
):
    """
    Build global context from all samples, infer one global q(z),
    then use the same physical parameters for every sample in the loss batch.
    """
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

    raw_params = mu + sigma * torch.randn_like(sigma) if is_training else mu
    params = params_from_raw_six(raw_params, bounds=parameter_bounds)

    L_KL = gaussian_kl_to_isotropic_prior(mu, logvar, prior_std=prior_std)

    if batch_size is None or batch_size >= len(samples):
        batch = samples
    else:
        idx = np.random.choice(len(samples), size=batch_size, replace=False)
        batch = [samples[i] for i in idx]

    total_physics_loss = torch.tensor(0.0, device=device)
    metrics = {k: 0.0 for k in ["LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LSO"]}
    per_sample_rows = []

    for sample in batch:
        (
            sample_loss,
            LE,
            LR,
            LTH,
            LPH,
            LBC,
            LORTH,
            LSO,
            rows,
        ) = compute_sample_physics_loss_full3d(
            wave_net=wave_net,
            sample=sample,
            params=params,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            **loss_kwargs,
        )

        total_physics_loss = total_physics_loss + sample_loss

        values = {
            "LE": LE,
            "LR": LR,
            "LTH": LTH,
            "LPH": LPH,
            "LBC": LBC,
            "LORTH": LORTH,
            "LSO": LSO,
        }
        for key, value in values.items():
            metrics[key] += float(value.detach().cpu())

        per_sample_rows.append(
            {
                "sample": sample,
                "rows": rows,
                "loss": float(sample_loss.detach().cpu()),
            }
        )

    B = len(batch)
    total_physics_loss = total_physics_loss / B
    for key in metrics:
        metrics[key] /= B

    total_loss = total_physics_loss + wKL * L_KL
    metrics["LKL"] = float(L_KL.detach().cpu())

    return total_loss, metrics, per_sample_rows
