# Sensitivity-Aware Process Modeling

This repository contains the public code and result artifacts used for the thesis *Sensitivity-aware process modeling from time series data under hidden disturbances*.

The repository is organized as follows:

- `code/data_generation`: scripts used to generate the benchmark process data used in the thesis.
- `code/data_processing`: scripts for preparing generated process data.
- `code/gradient_estimation`: standardized gradient-estimation implementations.
- `code/graph_learning`: graph-learning scripts used for the benchmark graphs.
- `code/standardized_workflow`: workflow entry points for rerunning the standardized gradient process when input data are supplied.
- `data/thesis_table_and_plot_data`: final processed CSV, TeX and PNG artifacts used for thesis figures, result tables and appendix tables.

Raw exploratory outputs, local run logs, full method-output directories, private reference PDFs, thesis source files and presentation files are intentionally not included. The goal is a compact public release for inspecting the benchmark definitions, rerunning the code and checking the final processed data behind the thesis figures and tables.

## Standardized Workflow

The main workflow entry point is:

```powershell
python code/standardized_workflow/run_standardized_process.py --help
```

The workflow writes new method outputs under `results/`. The raw consolidated process and graph inputs are not included in this compact release, so reruns require supplying compatible input data first.

The thesis-facing benchmark names are `lumped`, `low_dim` and `two_column`. Some files and internal method outputs keep older stable identifiers such as `heat`, `multi_stage` and `_pure` for script compatibility. The workflow entry points accept the thesis-facing aliases, and the processed result tables use the thesis display names.

The graph-learning folder contains the LPCMCI scripts used for the shared steam-header example, the thermal benchmark, the low-dimensional nonlinear benchmark and the two-column benchmark.

## Software

The Python dependencies used by the included workflow scripts are listed in `requirements.txt`. Some benchmark data-generation scripts are MATLAB scripts and require MATLAB when rerunning those generators.

## Notes

The data-generation and gradient-estimation scripts are research code. Paths in older scripts may still need to be adapted when running outside the original local directory layout. The included `data/thesis_table_and_plot_data` folder is the reference copy for the thesis figures and tables.
