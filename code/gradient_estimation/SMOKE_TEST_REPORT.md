# Smoke Test Report

Smoke tests were run on one representative file for each system:

- Heat: `Input 13 (heat N1)`, `Hbrn_N1`
- Low-dimensional nonlinear: `Input 28`, `Hbrn_N1`
- Two-column extraction: `Input 10`

All active method IDs completed and produced a CSV and JSON output in smoke mode.
The smoke outputs were removed from the active code tree after verification.

The graph-zero reuse path was also checked separately for expanded DML and
expanded LSE: the no-graph output was generated first, and the graph-zero run
reused it instead of refitting the same baseline model.

Smoke mode intentionally reduces expensive settings such as neural epochs and
random-forest tree counts. The runtime values stored in real experiment JSON files
should therefore be used for actual method-runtime comparisons.
