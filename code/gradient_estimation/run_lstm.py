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
    infer_dt,
    make_record,
    method_output_dir,
    read_dataframe,
    resolve_common_args,
    write_summary_outputs,
)


METHOD_ID = "lstm"
DEFAULT_WINDOW = 3


class Standardizer:
    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean


class LSTMRolloutModel(nn.Module):
    def __init__(self, n_features, n_outputs, hidden, layers, dropout):
        super().__init__()
        lstm_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, sequence):
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :])


def build_sequences(y_scaled, u_scaled, window):
    features = np.column_stack([y_scaled, u_scaled])
    seqs = []
    targets = []
    for start in range(0, len(features) - window):
        end = start + window
        seqs.append(features[start:end])
        targets.append(y_scaled[end] - y_scaled[end - 1])
    if not seqs:
        raise ValueError("Not enough samples to build LSTM rollout sequences.")
    return np.asarray(seqs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def rollout_final_y(model, initial_sequence, constant_u_scaled, horizon, n_outputs):
    sequence = initial_sequence.clone()
    for _ in range(horizon):
        delta_y = model(sequence)
        current_y = sequence[:, -1, :n_outputs]
        next_y = current_y + delta_y
        next_features = torch.cat([next_y, constant_u_scaled], dim=1).unsqueeze(1)
        if sequence.shape[1] == 1:
            sequence = next_features
        else:
            sequence = torch.cat([sequence[:, 1:, :], next_features], dim=1)
    return sequence[:, -1, :n_outputs]


def sample_indices(sample_count, max_samples):
    if sample_count <= max_samples:
        return np.arange(sample_count)
    return np.linspace(0, sample_count - 1, max_samples, dtype=int)


def rollout_gradients(
    model,
    sequences,
    y_scaler,
    u_scaler,
    spec,
    horizon,
    fd_relative_step,
    fd_absolute_floor,
    max_samples,
):
    n_outputs = len(spec.output_vars)
    n_inputs = len(spec.input_vars)
    indices = sample_indices(len(sequences), max_samples)
    u_raw_all = sequences[:, -1, n_outputs:] * u_scaler.scale + u_scaler.mean
    fd_steps = np.maximum(np.ptp(u_raw_all, axis=0) * fd_relative_step, fd_absolute_floor)
    local = np.zeros((len(indices), n_outputs, n_inputs), dtype=float)
    model.eval()
    with torch.no_grad():
        for local_idx, seq_idx in enumerate(indices):
            base_sequence = torch.tensor(sequences[seq_idx : seq_idx + 1], dtype=torch.float32)
            base_u_scaled = base_sequence[:, -1, n_outputs:].clone()
            base_final_scaled = rollout_final_y(model, base_sequence, base_u_scaled, horizon, n_outputs)

            for input_idx in range(n_inputs):
                delta_u = float(fd_steps[input_idx])
                perturbed_u_scaled = base_u_scaled.clone()
                perturbed_u_scaled[:, input_idx] += delta_u / float(u_scaler.scale[input_idx])
                perturbed_final_scaled = rollout_final_y(
                    model,
                    base_sequence,
                    perturbed_u_scaled,
                    horizon,
                    n_outputs,
                )
                diff_scaled = (perturbed_final_scaled - base_final_scaled).cpu().numpy()[0]
                diff_raw = diff_scaled * y_scaler.scale
                local[local_idx, :, input_idx] = diff_raw / delta_u
    return local, len(indices)


def run(args):
    timer = RuntimeTimer()
    spec, data_csv, graph_csv, output_root = resolve_common_args(args)
    out_dir = method_output_dir(METHOD_ID, spec.name, data_csv, output_root)
    dt = infer_dt(pd.read_csv(data_csv))
    df = read_dataframe(data_csv, spec)
    if args.smoke:
        df = df.iloc[: min(len(df), 220)].copy()

    y_raw = df[list(spec.output_vars)].to_numpy(dtype=np.float32)
    u_raw = df[list(spec.input_vars)].to_numpy(dtype=np.float32)
    window = args.window or DEFAULT_WINDOW
    horizon = args.rollout_horizon or spec.lsr_window
    if args.smoke:
        window = min(window, 3)
        horizon = min(horizon, 2)

    y_scaler = Standardizer().fit(y_raw)
    u_scaler = Standardizer().fit(u_raw)
    y_scaled = y_scaler.transform(y_raw)
    u_scaled = u_scaler.transform(u_raw)
    seqs, targets = build_sequences(y_scaled, u_scaled, window)

    epochs = 2 if args.smoke else args.epochs
    hidden = 16 if args.smoke else args.hidden
    max_samples = 20 if args.smoke else args.max_rollout_samples
    batch_size = min(args.batch_size, len(seqs))
    model = LSTMRolloutModel(seqs.shape[2], len(spec.output_vars), hidden, args.layers, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.tensor(seqs), torch.tensor(targets)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = torch.mean((pred - batch_y) ** 2)
            loss.backward()
            optimizer.step()

    local, gradient_samples = rollout_gradients(
        model,
        seqs,
        y_scaler,
        u_scaler,
        spec,
        horizon,
        args.fd_relative_step,
        args.fd_absolute_floor,
        max_samples,
    )

    runtime = timer.elapsed()
    records = []
    for out_idx, output in enumerate(spec.output_vars):
        for in_idx, input_name in enumerate(spec.input_vars):
            vals = local[:, out_idx, in_idx]
            records.append(
                make_record(
                    METHOD_ID,
                    "rollout_fd",
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
                    layers=args.layers,
                    batch_size=batch_size,
                    learning_rate=args.learning_rate,
                    dropout=args.dropout,
                    window=window,
                    rollout_horizon=horizon,
                    max_rollout_samples=max_samples,
                    fd_relative_step=args.fd_relative_step,
                    fd_absolute_floor=args.fd_absolute_floor,
                    rollout_horizon_rule="manual" if args.rollout_horizon else "settling_window",
                    head="linear_tanh_linear",
                )
            )

    metadata = {
        "method": METHOD_ID,
        "variant": "rollout_fd",
        "source_csv": str(data_csv),
        "epochs": epochs,
        "hidden": hidden,
        "layers": args.layers,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "dropout": args.dropout,
        "window": window,
        "rollout_horizon": horizon,
        "rollout_horizon_rule": "manual" if args.rollout_horizon else "settling_window",
        "rollout_duration": horizon * dt,
        "max_rollout_samples": max_samples,
        "gradient_samples": gradient_samples,
        "fd_relative_step": args.fd_relative_step,
        "fd_absolute_floor": args.fd_absolute_floor,
        "head": "linear_tanh_linear",
        "dt": dt,
        "runtime_seconds": runtime,
    }
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
    parser = argparse.ArgumentParser(description="LSTM rollout finite-difference gradient estimator.")
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--rollout-horizon", type=int)
    parser.add_argument("--max-rollout-samples", type=int, default=250)
    parser.add_argument("--fd-relative-step", type=float, default=0.05)
    parser.add_argument("--fd-absolute-floor", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
