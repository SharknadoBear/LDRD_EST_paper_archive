# WaterPACT Paper Data and Analysis Archive

This repository contains the data, configuration files, generated figures, and analysis scripts used for the WaterPACT/LDRD 2025 microphysics model paper workflow.

The archive is intended to be self-contained after Python environment setup. Run scripts and notebooks from the repository root because the analysis uses relative paths such as `./data`, `./config`, and `./output`.

## Contents

- `data/`: experimental CSV files, including time vectors, flow-rate files, PPD and PPDQ mass data, concentration data, and supporting values.
- `config/`: WMCR model configuration JSON files for the 9 PPDQ experiment conditions.
- `output/`: archived WMCR optimization outputs and plots, including `optimization_specs_*.json`, `optimization_result_*.png`, and `parameter_boxplot_*.png`.
- `n_figure_2.ipynb` through `n_figure_6.ipynb`: notebooks used to generate manuscript Figures 2-6.
- `d_optimization_solve.py`: serial WMCR Bayesian-optimization script.
- `d_optimization_solve_parallel.py`: parallel WMCR Bayesian-optimization script.
- `Figure_1.png` through `Figure_6.png`: archived manuscript figure images.

## Environment Setup

Conda is recommended for Windows users because it provides prebuilt scientific Python packages.

```powershell
conda env create -f environment.yml
conda activate waterpact-paper-archive
python -m ipykernel install --user --name waterpact-paper-archive --display-name "WaterPACT paper archive"
```

If Conda is not available, use `requirements.txt` in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name waterpact-paper-archive --display-name "WaterPACT paper archive"
```

When opening the notebooks, select the `WaterPACT paper archive` kernel or another environment with the packages listed in `environment.yml` or `requirements.txt`.

## Reproducing Figures

Open the notebooks from the repository root and run all cells.

Suggested order:

1. `n_figure_2.ipynb`
2. `n_figure_3.ipynb`
3. `n_figure_4.ipynb`
4. `n_figure_5.ipynb`
5. `n_figure_6.ipynb`

`n_figure_4.ipynb`, `n_figure_5.ipynb`, and `n_figure_6.ipynb` read archived optimization specifications from `output/`. The figure notebooks save `Figure_*.png` files in the repository root, so rerunning them can overwrite the archived figure images.

## Running Optimization Scripts

Both Python scripts prompt for the number of valid optimization clusters to collect. Larger values give more complete uncertainty estimates but can take longer.

Serial optimizer:

```powershell
python -u d_optimization_solve.py
```

Parallel optimizer:

```powershell
$env:WMCR_N_WORKERS = "4"
python -u d_optimization_solve_parallel.py
```

On Linux or macOS, set the worker count with:

```bash
export WMCR_N_WORKERS=4
python -u d_optimization_solve_parallel.py
```

The scripts write figures and JSON files into `output/` and may overwrite archived optimization results. To preserve the archived 100-cluster outputs exactly, run optimization scripts in a copy of the repository.

## Selecting An Experiment

To run a different experiment, update the hard-coded experiment tags before starting the optimizer. In `d_optimization_solve_parallel.py`, change lines 287-288:

```python
exp_tag = "4C_S"
exp_full_tag = "PPDQ_4C_S"
```

The serial script uses the same pattern in `d_optimization_solve.py` on lines 170-171.

Use one of these 9 experiment cases:

| Case | `exp_tag` | `exp_full_tag` |
| --- | --- | --- |
| 4 C, small particles | `4C_S` | `PPDQ_4C_S` |
| 4 C, medium particles | `4C_M` | `PPDQ_4C_M` |
| 4 C, large particles | `4C_L` | `PPDQ_4C_L` |
| 20 C, small particles | `20C_S` | `PPDQ_20C_S` |
| 20 C, medium particles | `20C_M` | `PPDQ_20C_M` |
| 20 C, large particles | `20C_L` | `PPDQ_20C_L` |
| 40 C, small particles | `40C_S` | `PPDQ_40C_S` |
| 40 C, medium particles | `40C_M` | `PPDQ_40C_M` |
| 40 C, large particles | `40C_L` | `PPDQ_40C_L` |

Each case corresponds to matching files in `data/` and `config/`. For example, `exp_full_tag = "PPDQ_20C_M"` reads `data/PPDQ_20C_M.csv`, `data/C_PPDQ_20C_M.csv`, and `config/meta_PPDQ_20C_M.json`.

## Validation

The archive was checked with:

- Python syntax compilation for the optimizer scripts.
- Import checks for the required scientific Python packages.
- Notebook execution checks for `n_figure_2.ipynb` through `n_figure_6.ipynb`.
- Optimizer smoke tests with `N_cluster = 1`.

## Citation And License

Before public reuse or redistribution, add the final manuscript citation, preferred license, and any required data-use statement.
