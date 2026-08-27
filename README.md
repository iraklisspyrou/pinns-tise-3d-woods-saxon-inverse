# Coupled forward--inverse Woods--Saxon PINN

Reproducible code for the paper **Probabilistic Physics-Informed Neural
Solvers for Woods--Saxon Parameter Identification: A Coupled Forward--Inverse
Approach**.

The implementation learns separated radial, polar and azimuthal spatial
wavefunction components with WaveNet and a dataset-conditioned output
distribution over six global Woods--Saxon parameters with ParamNet. The code
supports both the Seminole and Wahlborn potential expressions.

## Files

```text
configs/
  seminole.yaml                 Seminole synthetic/experimental settings
  wahlborn.yaml                 Wahlborn synthetic/experimental settings
data/
  seminole_synthetic_dataset.npz
  wahlborn_synthetic_dataset.npz
  experimental_dataset.npz
  README.md                     provenance, schema and identification subsets
src/ws_pinn/
  models.py                     MLP, WaveNet and ParamNet declarations
  data_loader.py                NPZ loader and state selection
  training.py                   joint forward--inverse training
  fd_solver.py                  independent radial finite-difference solver
  potentials.py                Seminole and Wahlborn Hamiltonians
  losses.py                     physics-informed objective
scripts/
  train.py                      training command
  generate_data.py              synthetic-data generation
  inspect_data.py               dataset tables and overview plots
  fd_validation.py              independent closure validation
  lsq_fit.py                    Levenberg--Marquardt least-squares baseline
visualization/
  training_visualization.py     loss and parameter histories
  results_visualization.py      energy and residual figures
  spectrum_plots.py             publication level-scheme figures
checkpoints/
  README.md                     checkpoint contents and resume behavior
docs/data/
  dataset_state_counts.*        generated system-size overview
  dataset_energy_overview.*     generated energy-coverage overview
```

The supplied datasets are stored in `data/`:

- `data/seminole_synthetic_dataset.npz`
- `data/wahlborn_synthetic_dataset.npz`
- `data/experimental_dataset.npz`

All data needed by the public scripts are included in this repository. The
detailed [dataset guide](data/README.md) documents their origin, object-array
schema, level counts, identification subsets and regeneration commands.

| Dataset | Systems | Levels |
|---|---:|---:|
| Seminole synthetic | 16 | 219 |
| Wahlborn synthetic | 34 | 353 |
| Experimental | 15 | 96 |

Inspect all archives and export human-readable tables and overview plots with:

```bash
python scripts/inspect_data.py --output-dir outputs/data_inspection
```

![Number of bound levels in each supplied system](docs/data/dataset_state_counts.png)

![Energy coverage of the supplied datasets](docs/data/dataset_energy_overview.png)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Install optional Weights & Biases support with:

```bash
pip install -e ".[tracking]"
```

## Choose the experiment

There are exactly two configuration files, one per potential expression. In
either file, change only:

```yaml
data:
  mode: synthetic
```

to:

```yaml
data:
  mode: experimental
```

The corresponding dataset path, selected systems, number of states, epochs
and output directory are then selected automatically.

The YAML file is the single source of truth for model architecture, parameter
bounds, training and normalization grids, loss weights, schedulers, inference,
diagnostic sampling, logging and checkpoint behavior. `training.py` does not
maintain an independent set of experiment defaults.

## Train

```bash
python scripts/train.py --config configs/seminole.yaml
python scripts/train.py --config configs/wahlborn.yaml
```

The paper normalization is retained: one coefficient computed from the full
separable wavefunction is applied to the radial component only. Training saves
updated weights and resumable checkpoints in the configured output directory.
It also copies the input YAML and writes a complete `run_manifest.json` with
resolved settings, derived dimensions and software/hardware metadata.

## Generate synthetic data

```bash
python scripts/generate_data.py --config configs/seminole.yaml \
  --output data/generated_seminole_dataset.npz
```

Use `configs/wahlborn.yaml` for the Wahlborn expression.

The nuclear systems, maximum quantum numbers, radial box and FD grid used by
the generator are recorded under `generation:` in each YAML. Command-line
options such as `--n-grid` override those recorded values only when explicitly
provided.

## Checkpoints and resume

Every run writes the following files under its output directory:

```text
checkpoints/latest_checkpoint.pt
checkpoints/final_checkpoint.pt
checkpoints/best_training_loss.pt
wave_net_global_multinucleus.pt
param_net_global_multinucleus.pt
input_config.yaml
run_manifest.json
```

`latest_checkpoint.pt` contains both networks, optimizers, schedulers, random
states, history and the resolved run manifest. With `output.resume: true`, a
new invocation resumes automatically from that file. Use a fresh output
directory or set `resume: false` after changing bounds, architecture,
normalization, state cardinality or potential expression. See the complete
[checkpoint guide](checkpoints/README.md).

Verified manuscript `.pt` files are not currently stored in Git. They should
be distributed through the permanent Zenodo archive or a versioned GitHub
Release so figures can be regenerated without repeating full training.

## Independent finite-difference validation

Validate the reference values from the configuration:

```bash
python scripts/fd_validation.py --config configs/seminole.yaml \
  --dataset data/seminole_synthetic_dataset.npz
```

Validate a trained distribution-mean estimate:

```bash
python scripts/fd_validation.py --config configs/seminole.yaml \
  --parameters outputs/seminole_synthetic/final_global_parameter_prediction.json \
  --dataset data/seminole_synthetic_dataset.npz
```

## Least-squares baseline

```bash
python scripts/lsq_fit.py --dataset data/experimental_dataset.npz
```

This reproduces the separate finite-difference Levenberg--Marquardt comparison
on the same 42 identification levels; it is not the original Seminole fit.

## Figures

```bash
python visualization/training_visualization.py \
  outputs/seminole_synthetic/training_history.csv

python visualization/results_visualization.py \
  outputs/seminole_synthetic/plots/final/energies/all_nuclei_energy_table.csv

python scripts/generate_data.py --config configs/seminole.yaml \
  --output outputs/seminole_reference_spectrum.npz

python scripts/generate_data.py --config configs/seminole.yaml \
  --parameters outputs/seminole_synthetic/final_global_parameter_prediction.json \
  --output outputs/seminole_pinn_spectrum.npz

python visualization/spectrum_plots.py \
  --experimental data/experimental_dataset.npz \
  --seminole outputs/seminole_reference_spectrum.npz \
  --pinn outputs/seminole_pinn_spectrum.npz
```

The visualization commands save PDF by default where applicable, so plots can
be regenerated without retraining when the corresponding CSV/NPZ result files
are available.

## Reviewer-style verification

Starting from a fresh clone, the shortest end-to-end checks are:

```bash
python scripts/train.py --help
python scripts/generate_data.py --help
python scripts/fd_validation.py --help
python scripts/lsq_fit.py --help

python scripts/inspect_data.py --output-dir outputs/data_inspection

python scripts/fd_validation.py --config configs/seminole.yaml \
  --dataset data/seminole_synthetic_dataset.npz \
  --output outputs/seminole_reference_fd.csv

python scripts/lsq_fit.py --dataset data/experimental_dataset.npz \
  --output-dir outputs/lsq_smoke --n-grid 400 --max-nfev 8 --verbose 1
```

To test the actual training path, run either public configuration and interrupt
after the first printed epoch. A full reproduction should retain the YAML
unchanged, complete the configured epochs and then pass the resulting physical-
space mean JSON to `fd_validation.py`.

## Reproducibility notes

- NumPy and PyTorch seeds are set through the YAML files.
- The exact input YAML and complete resolved run manifest are saved per run.
- The finite-difference solver is independent of the neural-network code.
- Training/normalization quadrature and plotting grids are configured
  separately; changing plot resolution does not change the objective.
- GPU results may differ slightly across CUDA and hardware versions.
- Full paper runs are computationally expensive; use fewer epochs and smaller
  quadrature grids only for a smoke test.

## Key Results

### Synthetic closure

All six global parameters were recovered with sub-percent relative errors.
After the inferred distribution means were reintroduced into the independent
finite-difference solver, the spectral errors were:

| Parameterization | FD closure MAE [MeV] |
|---|---:|
| Seminole | 0.0109 |
| Wahlborn | 0.0131 |

### Experimental spectra

The primary result is the physical-space distribution mean, which does not
require selecting an individual parameter sample.

| Potential expression | Reference MAE [MeV] | PINN mean MAE [MeV] |
|---|---:|---:|
| Seminole | 0.7969 | 0.8068 |
| Wahlborn | 1.0783 | 0.8303 |

The experimental identification set contains 42 levels: 24 neutron and 18
proton levels from `40Ca`, `48Ca`, `132Sn` and `208Pb`. The complete
experimental dataset contains 96 entries, and `90Zr` is retained as an
out-of-calibration case.

For the common 94-state Seminole comparison, the finite-difference
least-squares fit gives MAE/RMSE values of 0.8101/1.1807 MeV, while the
benchmark-selected PINN sample gives 0.8056/1.1525 MeV. This sampled result is
included only as a supplementary benchmark; it is not treated as an unbiased
test estimator. The work therefore does not claim that the PINN is globally
more accurate than least squares.

The standard deviations produced by ParamNet are model-derived output spreads
and qualitative indicators of concentration or parameter stiffness.

### Representative spectra

The level schemes compare experiment, the Seminole reference interaction and
the PINN-inferred interaction. The displayed PINN spectra use the
benchmark-selected parameter sample for illustration; the distribution mean
remains the primary selection-free estimator.

#### 48Ca

![Experimental, Seminole and PINN single-particle spectra for calcium-48](docs/figures/spectrum_48Ca.jpg)

#### 208Pb

![Experimental, Seminole and PINN single-particle spectra for lead-208](docs/figures/spectrum_208Pb.jpg)

### Forward-solution examples

These panels illustrate learned separated spatial wavefunction
representations for proton states in $^{208}$Pb. They are scalar spatial
components and should not be interpreted as complete spinor wavefunctions.

#### Radial probability densities

![Learned radial probability densities for proton states in lead-208](docs/figures/radial_probability_208Pb_proton.jpg)

#### Angular wavefunction slice

![Angular wavefunction probability slice for proton states in lead-208](docs/figures/angular_slice_208Pb_proton.jpg)

#### Angular probability heatmap

![Angular probability heatmap for a proton state in lead-208](docs/figures/angular_heatmap_208Pb_proton.jpg)
