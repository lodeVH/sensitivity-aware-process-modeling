# Standardized Gradient Summaries

This folder contains the current thesis-side copies of the standardized gradient
summary outputs used to build the appendix gradient-error tables.

The files are sanitized aggregate exports from the thesis-side standardized
workflow runs.

The GP method names in these summaries follow the current thesis notation:
`gp_rbf` is the RBF GP with analytic derivative and `gp_matern` is the Matern GP
with finite-difference derivative. Older `gp_average` and `gp_operating` names
should not be used for new appendix tables.

Each benchmark/regime folder contains:

- `standardized_gradient_summary.csv`: aggregate gradient estimates used for the
  appendix tables.
- `standardized_job_status.csv`: per-job status and runtime metadata.
- `standardized_manifest.json`: standardized-process run metadata.

Older pre-standardized gradient-result files and intermediate GP-name migration
copies are not included in this public release.
