"""WaveNet and global-context ParamNet architectures."""

import torch
import torch.nn as nn

from .scaling import (
    scale_energy,
    scale_nucleus,
    scale_phi,
    scale_quantum,
    scale_r,
    scale_theta,
)


class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=128, depth=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class WaveNet3D(nn.Module):
    """Shared conditional networks for separated ``R``, ``Theta`` and ``Phi``.

    Every component is conditioned on the complete selected energy vector,
    nucleus descriptors and the state's ``(nr, l, j)`` quantum numbers.  The
    radial output includes the analytic near-origin factor ``r**l`` and an
    exponential tail, leaving the MLP to learn the remaining shape.
    """
    def __init__(self, n_states, hidden=128, depth=5, beta=0.6):
        super().__init__()
        self.n_states = n_states
        self.beta = beta
        input_dim = 1 + n_states + 3 + 3
        self.R_net = ConditionalMLP(input_dim, 1, hidden, depth)
        self.Th_net = ConditionalMLP(input_dim, 1, hidden, depth)
        self.Ph_net = ConditionalMLP(input_dim, 2, hidden, depth)

    def _context(self, x_coord, E_vec, sample, st, coord_scaler):
        N = x_coord.shape[0]
        x_scaled = coord_scaler(x_coord)
        E_scaled = scale_energy(E_vec).unsqueeze(0).repeat(N, 1)
        nuc = scale_nucleus(
            sample["A"],
            sample["Z"],
            sample["is_proton"],
        ).unsqueeze(0).repeat(N, 1)
        q = scale_quantum(st["nr"], st["l"], st["j"]).unsqueeze(0).repeat(N, 1)
        return torch.cat([x_scaled, E_scaled, nuc, q], dim=1)

    def R(self, r, E_vec, sample, st):
        x = self._context(r, E_vec, sample, st, scale_r)
        raw = self.R_net(x)
        rc = r.clamp_min(1e-8)
        l = int(st["l"])
        # Hard architectural envelope: regular at r=0 and decaying at large r.
        return (rc**l) * torch.exp(-self.beta * rc) * raw

    def Theta(self, th, E_vec, sample, st):
        x = self._context(th, E_vec, sample, st, scale_theta)
        return self.Th_net(x)

    def Phi(self, ph, E_vec, sample, st):
        x = self._context(ph, E_vec, sample, st, scale_phi)
        out = self.Ph_net(x)
        return out[:, :1], out[:, 1:]


class GlobalContextParamNet(nn.Module):
    """
    Context-informed global parameter network.

    Input:
        X : shape (N_samples, input_dim)
            one row per nucleus/species sample.

    Output:
        one global q(z), not one q(z|sample).
    """

    def __init__(self, input_dim, hidden=256, depth=3, n_parameters=6):
        super().__init__()
        if n_parameters != 6:
            raise ValueError("Expected six Woods-Saxon parameters.")

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]

        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, n_parameters)
        self.logvar_head = nn.Linear(hidden, n_parameters)

        self.apply(self._init_weights)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -3.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, X):
        h = self.encoder(X)
        # Mean pooling makes the inferred global distribution invariant to the
        # order in which nucleus/species systems are supplied.
        h_global = h.mean(dim=0)
        mu = self.mu_head(h_global)
        logvar = self.logvar_head(h_global)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma, logvar
