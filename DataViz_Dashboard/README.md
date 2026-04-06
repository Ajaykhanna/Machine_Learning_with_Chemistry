# Molecular Trajectory Dashboard

Offline Python dashboard for exploring molecular dynamics trajectory data stored in NumPy arrays. The package serves a local interactive web app for:

- energy trajectories and energy distributions
- force norms
- dipole magnitudes
- NACR norms
- dENACR norms

It is designed for large trajectory datasets, supports metric selection from the CLI or an input file, and can export publication-ready `.png` and `.pdf` figures.

## Features

- Fully offline workflow: no internet access, no cloud services, no npm build step.
- Local Python server with a browser-based interactive dashboard.
- Handles trajectories from a few thousand to hundreds of thousands or millions of snapshots.
- Automatic cache generation for derived 1D series.
- Progress bar in the terminal when cache generation is expensive for large datasets.
- Interactive zoom by mouse wheel, drag-to-zoom, double-click reset, and global reset button.
- Hover tooltips with values rounded to 2 decimals.
- PNG/PDF export for line plots and histograms.
- CLI metric selection and input-file driven configuration.
- CLI options take priority over input-file settings and print a warning when both are provided.

## What This Package Expects

The dashboard reads NumPy `.npy` files from a trajectory data directory. It prefers combined tensors when available, and falls back to per-state or per-pair files when they are not.

Expected combined files:

- `acn_S.npy`
- `acn_F.npy`
- `acn_D.npy`
- `acn_NACR.npy`
- `acn_dENACR.npy`

Expected fallback naming:

- energies: `acn_S0.npy`, `acn_S1.npy`, ...
- forces: `acn_F0.npy`, `acn_F1.npy`, ...
- dipoles: `acn_D0.npy`, `acn_D1.npy`, ...
- NACRs: `acn_NACR12.npy`, `acn_NACR13.npy`, ...
- dENACRs: `acn_dENACR12.npy`, `acn_dENACR13.npy`, ...

State labels are inferred automatically from the available energy files. Excited-state pair labels are inferred automatically from the discovered states.

## Expected Array Shapes

Typical combined tensor shapes are:

- energies: `(n_snapshots, n_states)`
- forces: `(n_snapshots, n_states, n_atoms, 3)`
- dipoles: `(n_snapshots, n_states, 3)`
- NACRs: `(n_snapshots, n_pairs, n_atoms, 3)`
- dENACRs: `(n_snapshots, n_pairs, n_components)`

For the example dataset included during development, these were:

- `S`: `(50000, 11)`
- `F`: `(50000, 11, 15, 3)`
- `D`: `(50000, 11, 3)`
- `NACR`: `(50000, 45, 15, 3)`
- `dENACR`: `(50000, 45, 45)`

## Units

Default units in the package are:

- energy: `eV`
- force norm: `eV/Angstrom`
- dipole magnitude: `atomic units`
- NACR norm: `1/Angstrom`
- dENACR norm: `eV/Angstrom`

Axis labels are rendered in bracket style, for example `Energy [eV]`.

## Required Python Libraries

External Python dependencies:

- `numpy`
- `matplotlib`

Standard-library modules are used for:

- CLI parsing
- local HTTP serving
- JSON/config handling
- browser launch
- threading and concurrent execution

Frontend assets are bundled locally:

- `dashboard_static/vendor/d3-lite.js`

No separate JavaScript installation is required.

## Recommended Environment

This package was built for a Conda-style workflow and can automatically re-execute itself inside a detected `ml_env` environment.

Recommended:

```powershell
conda activate ml_env
```

Minimal Python requirement:

- Python 3.10+ recommended

## Package Structure

Minimum dashboard package structure:

```text
.
+-- README.md
+-- .gitignore
+-- run_dashboard.py
+-- precompute_dashboard.py
+-- dashboard_common.py
+-- dashboard_data.py
+-- dashboard_server.py
+-- dashboard_config.json
+-- dashboard_input_example.inp
+-- test_dashboard_common.py
+-- test_dashboard_data.py
+-- dashboard_static/
|   +-- index_v2.html
|   +-- app_v2.js
|   +-- styles.css
|   \-- vendor/
|       \-- d3-lite.js
\-- trajectory data files (*.npy)
```

Runtime-generated directories:

- `dashboard_cache/`
- `exports/`

These are created automatically when needed and are excluded in `.gitignore`.

## Quick Start

Launch with default behavior, which plots energies only:

```powershell
python run_dashboard.py --data_dir "C:\path\to\dataset"
```

Launch all metrics:

```powershell
python run_dashboard.py --data_dir "C:\path\to\dataset" --all
```

Launch a selected subset:

```powershell
python run_dashboard.py --data_dir "C:\path\to\dataset" --energies --forces_norm --dipole_magnitude
```

Precompute the cache first:

```powershell
python precompute_dashboard.py --data_dir "C:\path\to\dataset" --all --rebuild
```

## Input File Workflow

You can configure the dashboard using an input file:

```powershell
python run_dashboard.py --input_file "dashboard_input_example.inp"
```

You can still override parts of that configuration from the CLI:

```powershell
python run_dashboard.py --input_file "dashboard_input_example.inp" --energies --port 8141
```

If both input-file settings and CLI options are provided for the same setting, the CLI value wins and a warning is printed.

## Input File Format

Lines beginning with `#` are treated as comments.

Example:

```ini
# Core paths
data_dir = C:\path\to\dataset
output_dir = exports
port = 8141

# Metric selection
metrics = energies, forces_norm, dipole_magnitude

# Export formats
save_formats = both

# Optional combined tensor filenames
energy_file = acn_S.npy
force_file = acn_F.npy
dipole_file = acn_D.npy
nacr_file = acn_NACR.npy
denacr_file = acn_dENACR.npy

# Optional fallback prefixes
energy_prefix = acn_S
force_prefix = acn_F
dipole_prefix = acn_D
nacr_prefix = acn_NACR
denacr_prefix = acn_dENACR

# Optional axis label overrides
energy_axis_label = Energy
force_axis_label = Force Norm
dipole_axis_label = Dipole Magnitude
nacr_axis_label = NACR Norm
denacr_axis_label = dENACR Norm
```

## Supported Input File Keys

Core runtime keys:

- `data_dir`
- `output_dir`
- `port`
- `host`
- `metrics`
- `save_formats`

Metric selection values:

- `energies`
- `forces_norm`
- `dipole_magnitude`
- `nacrs_norm`
- `scaled_nacrs`
- `all`

Combined-file override keys:

- `energy_file`
- `force_file`
- `dipole_file`
- `nacr_file`
- `denacr_file`

Fallback-prefix override keys:

- `energy_prefix`
- `force_prefix`
- `dipole_prefix`
- `nacr_prefix`
- `denacr_prefix`

Axis-label override keys:

- `energy_axis_label`
- `force_axis_label`
- `dipole_axis_label`
- `nacr_axis_label`
- `denacr_axis_label`

## CLI Options

Main launcher:

```powershell
python run_dashboard.py --help
```

Current options:

- `--input_file`
- `--data_dir`
- `--host`
- `--port`
- `--no-browser`
- `--all`
- `--energies`
- `--forces_norm`
- `--dipole_magnitude`
- `--nacrs_norm`
- `--scaled_nacrs`

Precompute helper:

```powershell
python precompute_dashboard.py --help
```

Additional precompute options:

- `--rebuild`
- `--workers`

## Outputs

Generated during runtime:

- `dashboard_cache/`
  - cached reduced 1D arrays
  - multiresolution overview packs
  - metadata
- `exports/`
  - exported plot images and PDFs

The dashboard itself serves:

- `/api/metadata`
- `/api/series`
- `/api/histogram`
- `/api/statistics`
- `/api/export`

## Interactivity

The dashboard supports:

- state selection
- NAC pair selection
- snapshot window selection
- histogram bin control
- line and histogram hover tooltips
- mouse-wheel zoom
- drag-to-zoom
- double-click per-plot zoom reset
- global reset zoom button
- PNG/PDF export buttons

## Performance Notes

- Combined tensor files are preferred because they simplify loading and metadata discovery.
- Derived metrics are cached into `dashboard_cache/`.
- If the dataset exceeds `50K` snapshots, the package displays a terminal progress bar during cache generation.
- The dashboard uses downsampling for line plots so large trajectories remain interactive.

## Testing

Run the packaged tests with:

```powershell
python -B -m unittest test_dashboard_common.py test_dashboard_data.py -v
```

## Troubleshooting

If the first load is slow:

- wait for cache generation to finish
- for large datasets, watch the terminal progress bar

If plots do not reflect recent code changes:

- restart the server
- hard refresh the browser

If custom filenames are wrong:

- the dashboard prints a warning
- it falls back to the default expected naming when possible

If ports conflict:

- change `port` in the input file
- or pass `--port` on the CLI

## Example Commands

Energies only:

```powershell
python run_dashboard.py --data_dir "C:\path\to\dataset" --energies
```

All metrics from an input file:

```powershell
python run_dashboard.py --input_file "dashboard_input_example.inp"
```

Input file plus CLI override:

```powershell
python run_dashboard.py --input_file "dashboard_input_example.inp" --energies --port 9000
```

Precompute cache before launch:

```powershell
python precompute_dashboard.py --input_file "dashboard_input_example.inp" --rebuild
```

## Repository Notes

Before pushing to GitHub:

- keep the dashboard code and frontend assets
- keep `dashboard_input_example.inp`
- keep tests if you want a reusable package
- do not commit `dashboard_cache/`, `exports/`, or `__pycache__/`
- decide separately whether large `.npy` trajectory files should live in the repository, be stored with Git LFS, or stay outside the repo

## License and Attribution

Add your preferred project license and citation instructions here before publishing.
