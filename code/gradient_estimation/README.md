# Unified Gradient Approximation Runners

This directory contains the active gradient-estimation code used by the thesis.

Use the single entry point:

```powershell
python "run_gradient_method.py" --method average_lse --system two_column --data-csv "<process.csv>"
```

Graph-informed variants require `--graph-csv`.

## Active Methods

The runner accepts thesis-facing method aliases. The corresponding internal IDs are kept stable for reproducible output folders.

| Thesis name | Accepted alias | Internal ID |
| --- | --- | --- |
| Linear state-space estimator | `linear_state_space` | `state_space` |
| Neural ODE | `neural_ode` | `neural_ode` |
| LSTM | `lstm` | `lstm` |
| RBF GP | `rbf_gp` | `gp_rbf` |
| Matern GP | `matern_gp` | `gp_matern` |
| Linear residual DML | `linear_residual_dml` | `dml_linear_pure` |
| Linear residual DML + graph zero | `linear_residual_dml_graph_zero` | `dml_linear_graph_zero` |
| Linear residual DML + graph control | `linear_residual_dml_graph_control` | `dml_linear_graph_control` |
| Expanded DML | `expanded_dml` | `dml_basis_pure` |
| Expanded DML + graph zero | `expanded_dml_graph_zero` | `dml_basis_graph_zero` |
| Expanded DML + graph control | `expanded_dml_graph_control` | `dml_basis_graph_control` |
| Average LSE | `average_lse` | `lsr_linear_pure` |
| Average LSE + graph zero | `average_lse_graph_zero` | `lsr_linear_graph_zero` |
| Average LSE + graph control | `average_lse_graph_control` | `lsr_linear_graph_control` |
| Expanded LSE | `expanded_lse` | `lsr_basis_pure` |
| Expanded LSE + graph zero | `expanded_lse_graph_zero` | `lsr_basis_graph_zero` |
| Expanded LSE + graph control | `expanded_lse_graph_control` | `lsr_basis_graph_control` |

The `_pure` suffix appears only in internal IDs and output folders. It means the no-graph baseline used before applying the graph-zero or graph-control variants.

## Outputs

Each run writes one summary CSV, one JSON file, and one text log. The JSON stores
the run metadata, including the measured runtime in seconds, source files,
hyperparameters, and any steady-state extraction settings used by the method.
No figure, heatmap, MATLAB plot-job, or gradient-map files are produced by these
method runners.

By default outputs are written under:

```text
code/gradient_estimation/method outputs/<method>/<system>/<data-stem>/
```

Use `--output-root <dir>` to send outputs somewhere else.

## Smoke Testing

All active methods support `--smoke`. This reduces expensive training settings
only for validation runs. It should not be used for thesis experiments.
