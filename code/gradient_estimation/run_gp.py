import argparse
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from common_methods import (
    RuntimeTimer,
    add_common_arguments,
    ensure_ss_dataframe,
    make_record,
    method_output_dir,
    resolve_common_args,
    write_summary_outputs,
)


ALPHA_FLOOR = 1e-10


def rbf_kernel(n_inputs):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )


def matern_kernel(n_inputs, name):
    if name == "rbf":
        signal = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(n_inputs),
            length_scale_bounds=(1e-3, 1e3),
        )
    elif name == "matern_rbf":
        signal = ConstantKernel(1.0, (1e-3, 1e3)) * (
            Matern(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-3, 1e3), nu=1.5)
            + RBF(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-3, 1e3))
            + DotProduct()
        )
    else:
        signal = ConstantKernel(1.0, (1e-3, 1e3)) * (
            Matern(length_scale=np.ones(n_inputs), length_scale_bounds=(1e-3, 1e3), nu=1.5)
            + DotProduct()
        )
    return signal + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))


def fit_gp(X_raw, y_raw, kernel, restarts):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA_FLOOR,
        normalize_y=False,
        n_restarts_optimizer=restarts,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gp.fit(X_scaled, y_scaled)
    return gp, x_scaler, y_scaler


def predict_raw(gp, x_scaler, y_scaler, X_raw):
    y_scaled = gp.predict(x_scaler.transform(X_raw))
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def rbf_local_gradients(gp, x_scaler, y_scaler, X_raw):
    X_scaled = x_scaler.transform(X_raw)
    signal_kernel = gp.kernel_.k1 if hasattr(gp.kernel_, "k1") else gp.kernel_
    amplitude = float(signal_kernel.k1.constant_value) if hasattr(signal_kernel, "k1") else 1.0
    fitted_rbf_kernel = signal_kernel.k2 if hasattr(signal_kernel, "k2") else signal_kernel
    length_scales = np.asarray(fitted_rbf_kernel.length_scale, dtype=float)
    if length_scales.ndim == 0:
        length_scales = np.full(X_scaled.shape[1], float(length_scales))
    X_train = gp.X_train_
    alpha = gp.alpha_.reshape(-1)
    inv_l2 = 1.0 / np.square(length_scales)
    gradients_scaled = np.zeros_like(X_scaled)
    for row_index, x_point in enumerate(X_scaled):
        deltas = X_train - x_point
        sq = np.sum(np.square(deltas / length_scales), axis=1)
        k_vec = amplitude * np.exp(-0.5 * sq)
        gradients_scaled[row_index, :] = np.sum((alpha * k_vec)[:, None] * deltas * inv_l2, axis=0)
    return gradients_scaled * (y_scaler.scale_[0] / x_scaler.scale_)


def fd_steps(X_raw, relative_step):
    spans = np.ptp(X_raw, axis=0)
    return np.maximum(spans * relative_step, 1e-6)


def finite_difference_gradients(gp, x_scaler, y_scaler, X_raw, steps):
    gradients = np.zeros_like(X_raw, dtype=float)
    for idx in range(X_raw.shape[1]):
        plus = X_raw.copy()
        minus = X_raw.copy()
        plus[:, idx] += steps[idx]
        minus[:, idx] -= steps[idx]
        gradients[:, idx] = (predict_raw(gp, x_scaler, y_scaler, plus) - predict_raw(gp, x_scaler, y_scaler, minus)) / (2.0 * steps[idx])
    return gradients


def run(args):
    timer = RuntimeTimer()
    spec, data_csv, graph_csv, output_root = resolve_common_args(args)
    restarts = 0 if args.smoke else args.restarts
    method_id = "gp_rbf" if args.variant == "rbf" else "gp_matern"
    out_dir = method_output_dir(method_id, spec.name, data_csv, output_root)
    ss_df, ss_path, ss_meta = ensure_ss_dataframe(data_csv, spec, out_dir / "steady_state_data")
    if args.smoke:
        ss_df = ss_df.iloc[: min(len(ss_df), 40)].copy()

    X_raw = ss_df[list(spec.input_vars)].to_numpy(dtype=float)
    operating_point = np.array([float(ss_df[name].mean()) for name in spec.input_vars], dtype=float)
    records = []
    local_records = []
    kernels = {}
    steps = fd_steps(X_raw, args.fd_relative_step)

    for output in spec.output_vars:
        y_raw = ss_df[output].to_numpy(dtype=float)
        kernel = rbf_kernel(X_raw.shape[1]) if args.variant == "rbf" else matern_kernel(X_raw.shape[1], args.kernel)
        gp, x_scaler, y_scaler = fit_gp(X_raw, y_raw, kernel, restarts)
        kernels[output] = str(gp.kernel_)
        if args.variant == "rbf":
            local = rbf_local_gradients(gp, x_scaler, y_scaler, X_raw)
            operating_local = rbf_local_gradients(gp, x_scaler, y_scaler, operating_point.reshape(1, -1))[0]
        else:
            local = finite_difference_gradients(gp, x_scaler, y_scaler, X_raw, steps)
            operating_local = finite_difference_gradients(gp, x_scaler, y_scaler, operating_point.reshape(1, -1), steps)[0]
        summary_values = operating_local
        summary_std = np.std(local, axis=0, ddof=0)
        for input_idx, input_name in enumerate(spec.input_vars):
            records.append(
                make_record(
                    method_id,
                    args.variant,
                    spec,
                    output,
                    input_name,
                    summary_values[input_idx],
                    summary_std[input_idx],
                    len(ss_df),
                    data_csv,
                    graph_csv,
                    timer.elapsed(),
                    kernel=str(gp.kernel_),
                    ss_csv="" if ss_path is None else str(ss_path),
                    operating_input=json.dumps(dict(zip(spec.input_vars, operating_point.tolist()))),
                    fd_relative_step="" if args.variant == "rbf" else args.fd_relative_step,
                )
            )
        if args.store_local:
            for sample_idx, row in enumerate(local):
                for input_idx, input_name in enumerate(spec.input_vars):
                    local_records.append({"output": output, "input": input_name, "sample_index": sample_idx, "local_gradient": float(row[input_idx])})

    runtime = timer.elapsed()
    for record in records:
        record["runtime_seconds"] = runtime
    metadata = {
        "method": method_id,
        "variant": args.variant,
        "source_csv": str(data_csv),
        "ss_csv": "" if ss_path is None else str(ss_path),
        "ss_extraction": ss_meta,
        "kernel_choice": "rbf" if args.variant == "rbf" else args.kernel,
        "alpha_floor": ALPHA_FLOOR,
        "restarts": restarts,
        "fd_relative_step": None if args.variant == "rbf" else args.fd_relative_step,
        "gradient_readout": "analytic_rbf_derivative" if args.variant == "rbf" else "symmetric_finite_difference",
        "runtime_seconds": runtime,
        "kernels": kernels,
    }
    paths = write_summary_outputs(
        method_id=method_id,
        system=spec.name,
        source_csv=data_csv,
        output_dir=out_dir,
        records=records,
        metadata=metadata,
        local_records=local_records if args.store_local else None,
    )
    print(json.dumps({"csv": str(paths[0]), "json": str(paths[1]), "runtime_seconds": runtime}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Gaussian-process response-model gradient estimator.")
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=["rbf", "matern"], default="rbf")
    parser.add_argument("--kernel", choices=["matern", "matern_rbf", "rbf"], default="matern")
    parser.add_argument("--fd-relative-step", type=float, default=0.02)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--store-local", action="store_true", help="Store per-sample local GP gradients in the JSON output.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
