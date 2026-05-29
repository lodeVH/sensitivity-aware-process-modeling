# Standardized Gradient Workflow

Use this workflow to run the standardized gradient-estimation process for one benchmark system and one input regime.

The workflow writes new outputs under `results/`. This compact public release does not include the raw consolidated process and graph input folders, so reruns require supplying compatible input data first.

## Single Run

From the repository root:

```powershell
python code/standardized_workflow/run_standardized_process.py lumped zero --jobs 4
```

Examples:

```powershell
python code/standardized_workflow/run_standardized_process.py lumped var --jobs 4
python code/standardized_workflow/run_standardized_process.py low_dim return_to_baseline --jobs 4
python code/standardized_workflow/run_standardized_process.py two_column var --jobs 4
```

## Batch Run

```powershell
python code/standardized_workflow/run_standardized_batch.py lumped:return_to_baseline lumped:var low_dim:return_to_baseline low_dim:var --jobs 8
```

## Quick Smoke Check

```powershell
python code/standardized_workflow/run_standardized_process.py lumped return_to_baseline --methods linear_state_space average_lse_graph_zero --max-files 1 --smoke
```

## Supported Names

Systems:

- `lumped`, `heat`, `thermal`
- `low_dim`
- `two_column`, `multi_stage`, `multi`

Input regimes:

- `return_to_baseline`, `zero`, `to_zero`, `back_to_zero`
- `var`, `var_baseline`, `variable_baseline`

The output folders keep the compact internal regime keys `zero` and `var`.

## Method Name Aliases

The command accepts thesis-facing aliases such as:

- `linear_state_space`
- `neural_ode`
- `lstm`
- `rbf_gp`
- `matern_gp`
- `average_lse`, `average_lse_graph_zero`, `average_lse_graph_control`
- `expanded_lse`, `expanded_lse_graph_zero`, `expanded_lse_graph_control`
- `linear_residual_dml`, `linear_residual_dml_graph_zero`, `linear_residual_dml_graph_control`
- `expanded_dml`, `expanded_dml_graph_zero`, `expanded_dml_graph_control`

Internal output folders still use stable method IDs such as `lsr_linear_pure` and `dml_basis_graph_control`; the `_pure` suffix means the corresponding no-graph baseline.

## Output Layout

New runs are written to:

```text
results/standardized_results_windows/<system>/<regime>/<run_id>/
results/standardized_results_linux/<system>/<regime>/<run_id>/
```

Each run contains a manifest, job-status CSV, aggregated gradient-summary CSV and per-method output folders.

Graph-based methods only run when the matching graph file exists in the supplied graph-data folder. If no graph file exists, no replacement graph is invented and the graph-based job is skipped.
