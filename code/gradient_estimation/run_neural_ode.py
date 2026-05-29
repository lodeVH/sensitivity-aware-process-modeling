import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common_methods import (
    RuntimeTimer,
    add_common_arguments,
    ensure_ss_dataframe,
    infer_dt,
    make_record,
    method_output_dir,
    read_dataframe,
    resolve_common_args,
    write_summary_outputs,
)


METHOD_ID = "neural_ode"


class Standardizer:
    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean


class ODEModel(nn.Module):
    def __init__(self, n_states, n_inputs, hidden, depth):
        super().__init__()
        layers = []
        width_in = n_states + n_inputs
        for _ in range(depth):
            layers.append(nn.Linear(width_in, hidden))
            layers.append(nn.Tanh())
            width_in = hidden
        layers.append(nn.Linear(width_in, n_states))
        self.net = nn.Sequential(*layers)

    def forward(self, y, u):
        return self.net(torch.cat([y, u], dim=-1))


def solve_learned_steady_states(model, y_initial, u_scaled, max_steps, learning_rate, tolerance):
    y_ss = y_initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([y_ss], lr=learning_rate)
    steps_done = 0
    final_mean = float("nan")
    final_max = float("nan")
    for step in range(max_steps):
        optimizer.zero_grad()
        residual = model(y_ss, u_scaled)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()
        steps_done = step + 1
        with torch.no_grad():
            final_mean = float(torch.mean(torch.abs(residual)).item())
            final_max = float(torch.max(torch.abs(residual)).item())
        if final_max < tolerance:
            break

    with torch.no_grad():
        residual = model(y_ss, u_scaled)
        final_mean = float(torch.mean(torch.abs(residual)).item())
        final_max = float(torch.max(torch.abs(residual)).item())
    return y_ss.detach(), steps_done, final_mean, final_max


def batched_jacobians(model, y_ss, u_scaled):
    try:
        from torch.func import jacrev, vmap

        def vector_field_single(y_row, u_row):
            return model(y_row.unsqueeze(0), u_row.unsqueeze(0)).squeeze(0)

        jac_y = vmap(jacrev(vector_field_single, argnums=0))(y_ss, u_scaled)
        jac_u = vmap(jacrev(vector_field_single, argnums=1))(y_ss, u_scaled)
        return jac_y.detach().cpu().numpy(), jac_u.detach().cpu().numpy(), "torch.func.vmap"
    except Exception:
        jac_y_rows = []
        jac_u_rows = []
        for row_idx in range(y_ss.shape[0]):
            y_row = y_ss[row_idx].detach().clone().requires_grad_(True)
            u_row = u_scaled[row_idx].detach().clone().requires_grad_(True)

            def vector_field_y(y_in):
                return model(y_in.unsqueeze(0), u_row.unsqueeze(0)).squeeze(0)

            def vector_field_u(u_in):
                return model(y_row.unsqueeze(0), u_in.unsqueeze(0)).squeeze(0)

            jac_y_rows.append(torch.autograd.functional.jacobian(vector_field_y, y_row))
            jac_u_rows.append(torch.autograd.functional.jacobian(vector_field_u, u_row))
        return (
            torch.stack(jac_y_rows).detach().cpu().numpy(),
            torch.stack(jac_u_rows).detach().cpu().numpy(),
            "torch.autograd.functional.jacobian",
        )


def run(args):
    timer = RuntimeTimer()
    spec, data_csv, graph_csv, output_root = resolve_common_args(args)
    out_dir = method_output_dir(METHOD_ID, spec.name, data_csv, output_root)
    dt = infer_dt(pd.read_csv(data_csv))
    df = read_dataframe(data_csv, spec)
    if args.smoke:
        df = df.iloc[: min(len(df), 180)].copy()
    y_raw = df[list(spec.output_vars)].to_numpy(dtype=np.float32)
    u_raw = df[list(spec.input_vars)].to_numpy(dtype=np.float32)
    y_scaler = Standardizer().fit(y_raw)
    u_scaler = Standardizer().fit(u_raw)
    y_scaled = y_scaler.transform(y_raw)
    u_scaled = u_scaler.transform(u_raw)
    y0 = torch.tensor(y_scaled[:-1], dtype=torch.float32)
    u0 = torch.tensor(u_scaled[:-1], dtype=torch.float32)
    dy = torch.tensor(y_scaled[1:] - y_scaled[:-1], dtype=torch.float32)
    epochs = 2 if args.smoke else args.epochs
    hidden = 16 if args.smoke else args.hidden
    batch_size = min(args.batch_size, len(y0))
    model = ODEModel(len(spec.output_vars), len(spec.input_vars), hidden, args.depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(y0, u0, dy), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch_y, batch_u, batch_dy in loader:
            optimizer.zero_grad()
            pred = dt * model(batch_y, batch_u)
            loss = torch.mean((pred - batch_dy) ** 2)
            loss.backward()
            optimizer.step()

    ss_df, ss_path, ss_meta = ensure_ss_dataframe(data_csv, spec, out_dir / "steady_state_data")
    if args.smoke:
        ss_df = ss_df.iloc[: min(len(ss_df), 40)].copy()
    ss_y_raw = ss_df[list(spec.output_vars)].to_numpy(dtype=np.float32)
    ss_u_raw = ss_df[list(spec.input_vars)].to_numpy(dtype=np.float32)
    ss_y_scaled = torch.tensor(y_scaler.transform(ss_y_raw), dtype=torch.float32)
    ss_u_scaled = torch.tensor(u_scaler.transform(ss_u_raw), dtype=torch.float32)

    model.eval()
    ss_steps = 25 if args.smoke else args.ss_max_steps
    y_ss, ss_steps_done, ss_residual_mean, ss_residual_max = solve_learned_steady_states(
        model,
        ss_y_scaled,
        ss_u_scaled,
        ss_steps,
        args.ss_learning_rate,
        args.ss_tolerance,
    )
    jac_y, jac_u, jacobian_backend = batched_jacobians(model, y_ss, ss_u_scaled)
    scale_ratio = y_scaler.scale[:, None] / u_scaler.scale[None, :]
    local = np.zeros((len(ss_df), len(spec.output_vars), len(spec.input_vars)), dtype=float)
    condition_numbers = []
    for row_idx in range(len(ss_df)):
        a_matrix = jac_y[row_idx]
        b_matrix = jac_u[row_idx]
        try:
            condition_numbers.append(float(np.linalg.cond(a_matrix)))
        except np.linalg.LinAlgError:
            condition_numbers.append(float("inf"))
        local_scaled = -np.linalg.pinv(a_matrix, rcond=args.pinv_rcond) @ b_matrix
        local[row_idx, :, :] = scale_ratio * local_scaled

    runtime = timer.elapsed()
    records = []
    for out_idx, output in enumerate(spec.output_vars):
        for in_idx, input_name in enumerate(spec.input_vars):
            vals = local[:, out_idx, in_idx]
            records.append(
                make_record(
                    METHOD_ID,
                    "steady_state_jacobian",
                    spec,
                    output,
                    input_name,
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=0)),
                    len(vals),
                    data_csv,
                    graph_csv,
                    runtime,
                    epochs=epochs,
                    hidden=hidden,
                    depth=args.depth,
                    batch_size=batch_size,
                    learning_rate=args.learning_rate,
                    ss_max_steps=ss_steps,
                    ss_steps_done=ss_steps_done,
                    ss_residual_mean=ss_residual_mean,
                    ss_residual_max=ss_residual_max,
                    pinv_rcond=args.pinv_rcond,
                )
            )
    metadata = {
        "method": METHOD_ID,
        "variant": "steady_state_jacobian",
        "source_csv": str(data_csv),
        "steady_state_csv": None if ss_path is None else str(ss_path),
        "epochs": epochs,
        "hidden": hidden,
        "depth": args.depth,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "dt": dt,
        "ss_extraction": ss_meta,
        "ss_max_steps": ss_steps,
        "ss_learning_rate": args.ss_learning_rate,
        "ss_tolerance": args.ss_tolerance,
        "ss_steps_done": ss_steps_done,
        "ss_residual_mean": ss_residual_mean,
        "ss_residual_max": ss_residual_max,
        "pinv_rcond": args.pinv_rcond,
        "jacobian_backend": jacobian_backend,
        "jacobian_condition_median": float(np.median(condition_numbers)) if condition_numbers else None,
        "jacobian_condition_max": float(np.max(condition_numbers)) if condition_numbers else None,
        "runtime_seconds": runtime,
    }
    paths = write_summary_outputs(method_id=METHOD_ID, system=spec.name, source_csv=data_csv, output_dir=out_dir, records=records, metadata=metadata)
    print(json.dumps({"csv": str(paths[0]), "json": str(paths[1]), "runtime_seconds": runtime}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Neural ODE state-space gradient estimator.")
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ss-max-steps", type=int, default=250)
    parser.add_argument("--ss-learning-rate", type=float, default=5e-2)
    parser.add_argument("--ss-tolerance", type=float, default=1e-5)
    parser.add_argument("--pinv-rcond", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
