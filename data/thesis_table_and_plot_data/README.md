# Thesis Table and Plot Data

This folder contains the compact public copy of the processed data used for thesis figures, result tables and appendix tables.

The most important subfolders are:

- `combined analysis`: cross-system comparison summaries and figures.
- `graph metrics`: graph-score summaries and learned-graph figures.
- `lumped analysis`, `low dim analysis` and `two column analysis`: benchmark-specific result summaries and figures.
- `standardized gradient summaries`: sanitized aggregate gradient summaries used by the table-generation scripts.
- `two_column`: corrected steady-state rows, graph files and standardized summaries for the two-column extraction benchmark.
- `method_name_mapping.csv`: mapping from workflow method IDs and summary `method`/`variant` pairs to thesis display names.

Some CSV rows keep the internal key `multi_stage` because that was the stable system ID used by the runners when the raw method outputs were generated. The same applies to internal method IDs ending in `_pure`, which denote the corresponding no-graph baseline.
