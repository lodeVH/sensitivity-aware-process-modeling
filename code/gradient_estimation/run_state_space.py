import argparse
import json

import numpy as np

from common_methods import (
    RuntimeTimer,
    add_common_arguments,
    make_record,
    method_output_dir,
    read_dataframe,
    resolve_common_args,
    write_summary_outputs,
)


METHOD_ID = "state_space"
VARIANT = "ridge_delta_state"
RIDGE = 1e-3


def fit_state_space(df, spec, ridge=RIDGE):
    y = df[list(spec.output_vars)].to_numpy(dtype=float)
    u = df[list(spec.input_vars)].to_numpy(dtype=float)
    y0 = y[:-1]
    y1 = y[1:]
    u0 = u[:-1]
    y_bar = y0.mean(axis=0)
    u_bar = u0.mean(axis=0)
    dy = y1 - y0
    phi = np.column_stack([np.ones(len(y0)), y0 - y_bar, u0 - u_bar])
    eye = np.eye(phi.shape[1])
    theta = np.linalg.solve(phi.T @ phi + ridge * eye, phi.T @ dy)
    p = len(spec.output_vars)
    d = theta[0, :]
    f_matrix = theta[1 : 1 + p, :].T
    g_matrix = theta[1 + p :, :].T
    try:
        ss_gain = -np.linalg.solve(f_matrix, g_matrix)
        solve_method = "solve"
    except np.linalg.LinAlgError:
        ss_gain = -np.linalg.pinv(f_matrix) @ g_matrix
        solve_method = "pinv"
    return {
        "d": d,
        "F": f_matrix,
        "G": g_matrix,
        "gain": ss_gain,
        "y_bar": y_bar,
        "u_bar": u_bar,
        "solve_method": solve_method,
        "transition_count": len(y0),
    }


def run(args):
    timer = RuntimeTimer()
    spec, data_csv, graph_csv, output_root = resolve_common_args(args)
    df = read_dataframe(data_csv, spec)
    if args.smoke:
        df = df.iloc[: min(len(df), 200)].copy()
    fit = fit_state_space(df, spec)
    runtime = timer.elapsed()

    records = []
    gain = fit["gain"]
    for out_idx, output in enumerate(spec.output_vars):
        local_values = gain[out_idx, :]
        for in_idx, input_name in enumerate(spec.input_vars):
            records.append(
                make_record(
                    METHOD_ID,
                    VARIANT,
                    spec,
                    output,
                    input_name,
                    gain[out_idx, in_idx],
                    "",
                    fit["transition_count"],
                    data_csv,
                    graph_csv,
                    runtime,
                    ridge=RIDGE,
                    solve_method=fit["solve_method"],
                )
            )

    metadata = {
        "method": METHOD_ID,
        "variant": VARIANT,
        "ridge": RIDGE,
        "source_csv": str(data_csv),
        "runtime_seconds": runtime,
        "transition_count": fit["transition_count"],
        "output_vars": list(spec.output_vars),
        "input_vars": list(spec.input_vars),
        "operating_output": fit["y_bar"].tolist(),
        "operating_input": fit["u_bar"].tolist(),
        "F": fit["F"].tolist(),
        "G": fit["G"].tolist(),
        "d": fit["d"].tolist(),
    }
    out_dir = method_output_dir(METHOD_ID, spec.name, data_csv, output_root)
    paths = write_summary_outputs(
        method_id=METHOD_ID,
        system=spec.name,
        source_csv=data_csv,
        output_dir=out_dir,
        records=records,
        metadata=metadata,
    )
    print(json.dumps({"csv": str(paths[0]), "json": str(paths[1]), "runtime_seconds": runtime}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified linear state-space gradient estimator.")
    add_common_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
