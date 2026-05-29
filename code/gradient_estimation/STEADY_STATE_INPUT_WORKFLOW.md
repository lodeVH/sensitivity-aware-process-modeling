# Steady-State Input Convention

The active unified runners keep the Chapter 3 convention:

- Dynamic methods use the raw generated trajectory for training or response
  extraction: `state_space`, `neural_ode`, `lstm`, and LSR.
- `neural_ode` trains on the raw trajectory, but its reported gain is a learned
  steady-state Jacobian. It therefore extracts steady-state rows for the final
  operating points before solving the learned steady-state balance.
- `lstm` trains on raw trajectory sequences and reports a finite-difference
  rollout gain. It does not extract steady-state rows for the gradient step.
  Instead, it starts from measured history windows, rolls the learned one-step
  model forward with the original input and with one perturbed input, and averages
  the resulting response slopes. By default, the rollout horizon is the same
  benchmark-specific response window used by LSR.
- Static response methods use steady-state rows: `gp_rbf`,
  `gp_matern`, `dml_linear_*`, and `dml_basis_*`.

If a static method receives a file whose name already contains `_ST_DATA`, that
file is used directly. Otherwise the runner extracts steady-state rows before
fitting:

- Heat: take the row 6 samples after an input step.
- Two-column extraction: take the row 80 samples after an input step.
- Low-dimensional nonlinear system: take the row immediately before the next
  input step.

Hidden disturbance columns are not part of the method specifications. The active
system definitions use only manipulated inputs and measured outputs.
