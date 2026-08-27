# Datasets

The repository contains every numerical dataset used by the training and
validation scripts.  No external download is required.

| File | Origin | Systems | Levels |
|---|---|---:|---:|
| `seminole_synthetic_dataset.npz` | Independent radial finite-difference calculation using the Seminole/Schwierz reference parameters | 16 | 219 |
| `wahlborn_synthetic_dataset.npz` | Independent radial finite-difference calculation using the Wahlborn reference parameters | 34 | 353 |
| `experimental_dataset.npz` | Experimental single-particle energies and assignments tabulated from the sources cited in the manuscript | 15 | 96 |

A *system* is one `(A, Z, species)` combination.  Neutron and proton sectors
of the same nucleus are counted separately.

## Archive schema

Each file is a compressed NumPy archive containing an object array named
`data`.  Each system has the form:

```python
{
    "A": 208,
    "Z": 82,
    "is_proton": False,
    "states": [
        {"nr": 0, "l": 0, "j": 0.5, "energy": -39.1},
        # ...
    ],
}
```

Energies are in MeV.  The synthetic states contain `nr`, `l`, `j`, and
`energy`.  Experimental states additionally retain the tabulated orbital,
particle/hole classification, excitation energy, spectroscopic information,
uncertainty, and source-note fields when those values are available.

The archives are loaded with `allow_pickle=True` because they store nested
Python dictionaries.  Only load trusted copies, such as the files distributed
with this repository or its permanent archive.

## Identification subsets

The complete archives contain more levels than the inverse model uses for
identification.  State selection is deterministic and pair-preserving; its
implementation is in `src/ws_pinn/data_loader.py`.

- Seminole synthetic: 4 neutron systems x 7 levels = 28 identification levels.
- Wahlborn synthetic: 8 neutron/proton systems x 7 levels = 56 identification levels.
- Experimental: 7 neutron/proton systems x 6 levels = 42 identification levels
  (24 neutron and 18 proton).  The remaining 54 levels are used only for the
  broader evaluation.

The selected systems are recorded explicitly in `configs/seminole.yaml` and
`configs/wahlborn.yaml`.

## Synthetic generation

The YAML files record the reference parameters, nuclear systems, radial box,
grid resolution, and maximum quantum numbers used by the public generator.
Generate a fresh archive with:

```bash
python scripts/generate_data.py --config configs/seminole.yaml \
  --output data/generated_seminole_synthetic_dataset.npz

python scripts/generate_data.py --config configs/wahlborn.yaml \
  --output data/generated_wahlborn_synthetic_dataset.npz
```

The generator solves the Dirichlet problem for `u(r)=rR(r)` independently of
WaveNet and ParamNet.  Small last-digit differences can occur when the radial
grid or SciPy version changes; these are finite-difference discretization
effects rather than changes to the reference interaction.

Omitting `--parameters` uses the reference values stored in the YAML. To
generate the spectrum associated with a trained physical-space mean, pass its
JSON explicitly:

```bash
python scripts/generate_data.py --config configs/seminole.yaml \
  --parameters outputs/seminole_synthetic/final_global_parameter_prediction.json \
  --output outputs/seminole_pinn_spectrum.npz
```

## Inspecting the data

Generate human-readable CSV tables and overview plots with:

```bash
python scripts/inspect_data.py --output-dir outputs/data_inspection
```

The command writes:

- `all_levels.csv`
- `dataset_summary.csv`
- `dataset_state_counts.png` and `.pdf`
- `dataset_energy_overview.png` and `.pdf`

The two overview plots committed under `docs/data/` were produced with this
same command.
