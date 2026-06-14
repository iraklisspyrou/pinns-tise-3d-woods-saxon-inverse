# Physics-Informed Inverse Modeling of the Woods–Saxon Potential

This repository contains a physics-informed neural network framework for solving the three-dimensional time-independent Schrödinger equation and inferring Woods–Saxon mean-field parameters from nuclear single-particle spectra.

The implementation combines:

- a separable wavefunction representation,
- physics-informed residual losses,
- Rayleigh-quotient energy estimation,
- boundary and orthogonality constraints,
- probabilistic parameter inference,
- Monte Carlo uncertainty estimation,
- diagnostic plots for wavefunctions, energies, overlaps and parameter convergence.

## Repository structure

```text
.
├── inverse_3d_tise_woods_saxon.py
├── plotting_diagnostics_results.py
├── ws_fd_dataset.npz
└── README.md
```

### `inverse_3d_tise_woods_saxon.py`

Main implementation of the inverse PINN. It defines:

- the Woods–Saxon mean-field potential,
- the central, Coulomb, spin–orbit and isospin-dependent contributions,
- the conditional radial, polar and azimuthal neural networks,
- the probabilistic parameter network,
- normalization and Rayleigh-energy calculations,
- radial and angular Schrödinger residuals,
- boundary and selective orthogonality losses,
- single-nucleus training and parameter inference.

The inferred parameter vector is

$$
\left(V_0,\ r_0,\ a,\ \lambda_{\mathrm{SO}}\right).
$$

The remaining potential parameters are fixed to the adopted reference values:

$$
\kappa=0.639,\qquad
r_{0,\mathrm{SO}}=1.16\ \mathrm{fm},\qquad
a_{\mathrm{SO}}=0.662\ \mathrm{fm}.
$$

### `plotting_diagnostics_results.py`

Experiment and diagnostics module. It provides:

- Weights & Biases logging,
- parameter and loss histories,
- Monte Carlo uncertainty estimation,
- random radial-probe sensitivity tests,
- wavefunction slices and marginal probability densities,
- angular probability heatmaps,
- predicted-versus-target energy plots,
- overlap matrices,
- single-nucleus and multi-nucleus experiments,
- CSV, JSON, PNG and PyTorch checkpoint outputs.

### `ws_fd_dataset.npz`

Finite-difference reference dataset containing nuclear single-particle states and energies. Each sample is identified by:

- mass number \(A\),
- proton number \(Z\),
- particle type,
- radial quantum number \(n_r\),
- orbital angular momentum \(l\),
- total angular momentum \(j\),
- reference energy.

## Model overview

The scalar wavefunction is represented in separable form:

$$
\Psi(r,\theta,\phi) = R(r)\Theta(\theta)\Phi(\phi).
$$

The radial component uses the physics-guided ansatz

$$
R(r) = r^l e^{-\beta r}N_R(r),
$$

where \(N_R\) is a neural network. The factor \(r^l\) enforces the regular near-origin behavior and the exponential factor encourages bound-state decay.

The total training objective combines

$$
\mathcal{L} =
w_E\mathcal{L}_E
+
w_R\mathcal{L}_R
+
w_\theta\mathcal{L}_\theta
+
w_\phi\mathcal{L}_\phi
+
w_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}}
+
w_{\mathrm{orth}}\mathcal{L}_{\mathrm{orth}}
+
w_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}.
$$

The terms correspond to:

- energy reconstruction,
- radial Schrödinger residual,
- polar and azimuthal equation residuals,
- radial and periodic boundary conditions,
- selective orthogonality between radial excitations,
- variational regularization of parameter uncertainty.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <your-repository-url>
cd <your-repository-name>

python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas matplotlib torch wandb
```

`wandb` is optional. The diagnostic script can run with `use_wandb=False`.

## Quick start

### 1. Run the default single-nucleus inverse problem

The default configuration in `inverse_3d_tise_woods_saxon.py` trains a neutron model for \(^{56}\mathrm{Ni}\):

```bash
python inverse_3d_tise_woods_saxon.py
```

The example configuration uses:

```python
A = 56
Z = 28
is_proton = False
max_states = 7
epochs = 15000
```

At the end of training, the script prints the inferred parameter means and standard deviations:

```text
V0     = ... ± ... MeV
r0     = ... ± ... fm
a      = ... ± ... fm
lam_so = ... ± ...
```

## Running a custom single-nucleus experiment

Import the training function and specify the nucleus and hyperparameters:

```python
from inverse_3d_tise_woods_saxon import (
    train_single_nucleus_full3d,
    infer_parameters_full3d,
)

wave_net, param_net, history, sample = train_single_nucleus_full3d(
    dataset_path="ws_fd_dataset.npz",
    A=56,
    Z=28,
    is_proton=False,
    max_states=9,
    epochs=15000,
    lr_wave=5e-4,
    lr_param=1e-3,
    n_r_points=96,
    Nr_norm=1024,
    Nth_norm=512,
    Nph_norm=512,
    hidden_wave=256,
    hidden_param=256,
    emm=0,
    wE=0.5,
    wR=10.0,
    wTh=5.0,
    wPh=5.0,
    wBC=5.0,
    wORTH=5.0,
    wKL=1e-3,
    print_every=100,
)

parameters = infer_parameters_full3d(
    wave_net,
    param_net,
    sample,
    n_samples=2000,
    n_r_points=96,
    Nr_norm=1024,
    Nth_norm=512,
    Nph_norm=512,
)

print(parameters)
```

For protons, set:

```python
is_proton=True
```

## Running the diagnostics and multi-nucleus benchmark

The diagnostics script imports the base PINN module through the `PINN_BASE_MODULE` environment variable.

The module name must be provided **without** the `.py` extension.

### Linux or macOS

```bash
export PINN_BASE_MODULE=inverse_3d_tise_woods_saxon
python plotting_diagnostics_results.py
```

### Windows PowerShell

```powershell
$env:PINN_BASE_MODULE="inverse_3d_tise_woods_saxon"
python plotting_diagnostics_results.py
```

The script runs:

```python
run_full_multi_nucleus_benchmark()
```

with the benchmark configuration defined near the end of `plotting_diagnostics_results.py`.

## Running diagnostics from Python or Jupyter

```python
import os

os.environ["PINN_BASE_MODULE"] = "inverse_3d_tise_woods_saxon"

from plotting_diagnostics_results import run_many_nuclei_experiment

cases = [
    (56, 28, False),
    (56, 28, True),
    (60, 28, False),
    (60, 28, True),
]

summary_df = run_many_nuclei_experiment(
    dataset_path="ws_fd_dataset.npz",
    cases=cases,
    max_states=7,
    epochs=15000,
    out_root="benchmark_outputs",
    wandb_project="inverse-ws-pinn",
    wandb_group="custom-benchmark",
    use_wandb=False,
    common_train_kwargs={
        "lr_wave": 5e-4,
        "lr_param": 1e-3,
        "n_r_points": 96,
        "Nr_norm": 1024,
        "Nth_norm": 512,
        "Nph_norm": 256,
        "hidden_wave": 128,
        "hidden_param": 128,
        "emm": 0,
        "wE": 0.5,
        "wR": 10.0,
        "wTh": 5.0,
        "wPh": 5.0,
        "wBC": 5.0,
        "wORTH": 1.0,
        "wKL": 1e-3,
        "print_every": 100,
        "log_every": 100,
        "plot_every": None,
    },
)

display(summary_df)
```

Each case is represented by:

```python
(A, Z, is_proton)
```

For example:

```python
(56, 28, False)  # neutron states in 56Ni
(56, 28, True)   # proton states in 56Ni
```

## Example: one instrumented experiment

To train one nucleus and automatically generate diagnostics:

```python
import os

os.environ["PINN_BASE_MODULE"] = "inverse_3d_tise_woods_saxon"

from plotting_diagnostics_results import train_single_nucleus_instrumented

wave_net, param_net, history, sample, summary = (
    train_single_nucleus_instrumented(
        dataset_path="ws_fd_dataset.npz",
        A=56,
        Z=28,
        is_proton=False,
        max_states=7,
        epochs=15000,
        lr_wave=5e-4,
        lr_param=1e-3,
        n_r_points=96,
        Nr_norm=1024,
        Nth_norm=512,
        Nph_norm=256,
        hidden_wave=128,
        hidden_param=128,
        emm=0,
        wE=0.5,
        wR=10.0,
        wTh=5.0,
        wPh=5.0,
        wBC=5.0,
        wORTH=1.0,
        wKL=1e-3,
        out_root="single_nucleus_outputs",
        use_wandb=False,
    )
)

print(summary)
```

## Output files

For each nucleus, the diagnostics pipeline creates a folder such as:

```text
paper_outputs_final/A56_Z28_n/
```

Typical outputs include:

```text
A56_Z28_n_training_history.csv
A56_Z28_n_final_parameters.json
A56_Z28_n_summary.json
A56_Z28_n_energy_table.csv
A56_Z28_n_wave_net.pt
A56_Z28_n_param_net.pt
A56_Z28_n_overlap_matrix.png
A56_Z28_n_energy_pred_vs_target.png
A56_Z28_n_energy_residuals.png
A56_Z28_n_loss_components.png
A56_Z28_n_param_V0_vs_epoch.png
A56_Z28_n_param_r0_vs_epoch.png
A56_Z28_n_param_a_vs_epoch.png
A56_Z28_n_param_lam_so_vs_epoch.png
```

The random radial-probe diagnostic additionally generates:

```text
A56_Z28_n_random_radial_probe_parameter_inference.csv
A56_Z28_n_random_radial_probe_summary.json
```

## Random radial-probe inference

The parameter network is normally conditioned on radial wavefunction values sampled on a fixed grid.

The diagnostic module can repeat inference using independently sampled radial points:

\[
r_i\sim U(0,R_{\max}).
\]

This test evaluates whether parameter estimates remain stable when the trained radial functions are queried at different locations.

It should be interpreted as a **sampling-sensitivity diagnostic**, not as a new physical dataset.

## Weights & Biases

To enable online experiment tracking:

```bash
wandb login
```

Then run an experiment with:

```python
use_wandb=True
```

Logged quantities include:

- total and component losses,
- inferred parameter means,
- inferred parameter uncertainties,
- learning rates,
- energy RMSE and MAE,
- maximum off-diagonal overlap,
- wavefunction and parameter plots.

## Reproducibility notes

- NumPy and PyTorch random seeds are fixed in the base implementation.
- Training may still vary slightly across GPU architectures and CUDA versions.
- The network architecture, number of selected states and radial sampling resolution must remain consistent when loading saved checkpoints.
- Full benchmark runs are computationally expensive. Reduce `epochs`, grid sizes and hidden dimensions for initial testing.

A smaller test configuration could use:

```python
epochs = 100
Nr_norm = 256
Nth_norm = 128
Nph_norm = 128
hidden_wave = 64
hidden_param = 64
```

This configuration is suitable only for checking that the pipeline runs successfully.


## Citation

When using this repository in academic work, cite the associated thesis or publication. A formal citation entry can be added here once the paper is available.

## License

Add the license selected for the repository, for example:

```text
MIT License
```

