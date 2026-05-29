import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_methods import (
    RuntimeTimer,
    add_common_arguments,
    apply_graph_zero,
    build_feature_basis,
    graph_ancestor_controls,
    graph_reachable_inputs,
    make_record,
    method_output_dir,
    read_dataframe,
    residualize_with_controls,
    resolve_common_args,
    write_summary_outputs,
)


RIDGE_ALPHA = 1.0


def build_gradient_samples(df, spec, window):
    samples = []
    for input_name in spec.input_vars:
        values = df[input_name].to_numpy(dtype=float)
        for index in range(len(df) - 1):
            du = values[index + 1] - values[index]
            if abs(du) <= 1e-9:
                continue
            after_index = index + window
            if after_index >= len(df):
                continue
            row = {
                "input": input_name,
                "t0": int(index),
                "du": float(du),
                "window": int(window),
            }
            for output in spec.output_vars:
                y_delta = float(df.iloc[after_index][output] - df.iloc[index][output])
                row[f"dy_{output}"] = y_delta
                row[f"g_{output}"] = y_delta / du
                row[f"{output}_before"] = float(df.iloc[index][output])
            for input_col in spec.input_vars:
                row[f"{input_col}_before"] = float(df.iloc[index][input_col])
            samples.append(row)
    return samples


def sample_dataframe(samples):
    import pandas as pd

    return pd.DataFrame(samples)


def pure_context_features(spec):
    return [f"{name}_before" for name in (*spec.output_vars, *spec.input_vars)]


def graph_context_features(spec, output, input_name, graph_controls):
    cols = set()
    for name in graph_controls.get(output, set()):
        if name == output:
            continue
        if name in spec.output_vars or name in spec.input_vars:
            cols.add(f"{name}_before")
    cols.add(f"{output}_before")
    cols.add(f"{input_name}_before")
    return sorted(cols)


def linear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime):
    records = []
    allowed = graph_reachable_inputs(graph_csv, spec) if args.variant in {"graph_zero", "graph_control"} else None
    graph_controls = graph_ancestor_controls(graph_csv, spec) if args.variant == "graph_control" else None

    pure_records = []
    for output in spec.output_vars:
        for input_name in spec.input_vars:
            subset = gradients[gradients["input"] == input_name]
            if subset.empty:
                value = 0.0
                std = 0.0
                count = 0
                controls = []
            elif args.variant == "graph_control":
                is_allowed = input_name in allowed.get(output, set())
                controls = graph_context_features(spec, output, input_name, graph_controls)
                if not is_allowed:
                    records.append(
                        make_record("lsr_linear", args.variant, spec, output, input_name, 0.0, 0.0, len(subset), data_csv, graph_csv, runtime, allowed_by_graph=False, controls=json.dumps(controls))
                    )
                    continue
                control_matrix = subset[controls].to_numpy(dtype=float) if controls else np.zeros((len(subset), 0))
                d_res = residualize_with_controls(subset["du"].to_numpy(dtype=float), control_matrix)
                y_res = residualize_with_controls(subset[f"dy_{output}"].to_numpy(dtype=float), control_matrix)
                denom = float(np.dot(d_res, d_res))
                value = float(np.dot(d_res, y_res) / denom) if denom > 1e-14 else 0.0
                std = 0.0
                count = len(subset)
            else:
                values = subset[f"g_{output}"].to_numpy(dtype=float)
                value = float(np.mean(values))
                std = float(np.std(values, ddof=0))
                count = len(values)
                controls = []
            record = make_record("lsr_linear", "pure" if args.variant == "graph_zero" else args.variant, spec, output, input_name, value, std, count, data_csv, graph_csv, runtime, allowed_by_graph="" if args.variant != "graph_control" else True, controls=json.dumps(controls))
            if args.variant == "graph_zero":
                pure_records.append(record)
            else:
                records.append(record)
    if args.variant == "graph_zero":
        records = apply_graph_zero(pure_records, graph_csv, spec)
        for record in records:
            record["variant"] = "graph_zero"
    return records


def nonlinear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime):
    records = []
    allowed = graph_reachable_inputs(graph_csv, spec) if args.variant in {"graph_zero", "graph_control"} else None
    graph_controls = graph_ancestor_controls(graph_csv, spec) if args.variant == "graph_control" else None
    pure_records = []
    for output in spec.output_vars:
        for input_name in spec.input_vars:
            subset = gradients[gradients["input"] == input_name].copy()
            if subset.empty:
                record = make_record("lsr_basis", args.variant, spec, output, input_name, 0.0, 0.0, 0, data_csv, graph_csv, runtime, controls="[]")
                records.append(record)
                continue
            if args.variant == "graph_control":
                is_allowed = input_name in allowed.get(output, set())
                features = graph_context_features(spec, output, input_name, graph_controls)
                if not is_allowed:
                    records.append(make_record("lsr_basis", args.variant, spec, output, input_name, 0.0, 0.0, len(subset), data_csv, graph_csv, runtime, allowed_by_graph=False, controls=json.dumps(features)))
                    continue
            else:
                features = pure_context_features(spec)
            X, feature_names = build_feature_basis(subset, features)
            y = subset[f"g_{output}"].to_numpy(dtype=float)
            if X.shape[1] == 0:
                predictions = np.repeat(np.mean(y), len(y))
                value = float(np.mean(y))
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                model.fit(X, y)
                predictions = model.predict(X)
                mean_row = subset[features].mean(axis=0).to_frame().T
                X_op, _ = build_feature_basis(mean_row, features)
                value = float(model.predict(X_op)[0])
            std = float(np.std(predictions, ddof=0))
            record = make_record(
                "lsr_basis",
                "pure" if args.variant == "graph_zero" else args.variant,
                spec,
                output,
                input_name,
                value,
                std,
                len(subset),
                data_csv,
                graph_csv,
                runtime,
                allowed_by_graph=True if args.variant == "graph_control" else "",
                controls=json.dumps(features),
                basis_terms="linear,square,cube,tanh,signed_log1p",
                ridge_alpha=RIDGE_ALPHA,
            )
            if args.variant == "graph_zero":
                pure_records.append(record)
            else:
                records.append(record)
    if args.variant == "graph_zero":
        records = apply_graph_zero(pure_records, graph_csv, spec)
        for record in records:
            record["variant"] = "graph_zero"
    return records


def load_pure_summary_if_available(out_dir):
    pure_dir = out_dir / "pure"
    if not pure_dir.is_dir():
        return None
    json_files = sorted(pure_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not json_files:
        return None
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    return payload.get("summary")


def run(args):
    timer = RuntimeTimer()
    spec, data_csv, graph_csv, output_root = resolve_common_args(args)
    if args.variant in {"graph_zero", "graph_control"} and graph_csv is None:
        raise ValueError(f"LSR variant '{args.variant}' requires --graph-csv")
    df = read_dataframe(data_csv, spec)
    if args.smoke:
        df = df.iloc[: min(len(df), 200)].copy()
    samples = build_gradient_samples(df, spec, args.window or spec.lsr_window)
    gradients = sample_dataframe(samples)
    runtime_so_far = timer.elapsed()
    method_id = "lsr_linear" if args.kind == "linear" else "lsr_basis"
    base_out_dir = method_output_dir(method_id, spec.name, data_csv, output_root)
    if args.variant == "graph_zero":
        pure_records = load_pure_summary_if_available(base_out_dir)
        if pure_records is None:
            original_variant = args.variant
            args.variant = "pure"
            if args.kind == "linear":
                pure_records = linear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime_so_far)
            else:
                pure_records = nonlinear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime_so_far)
            args.variant = original_variant
            pure_metadata = {
                "method": method_id,
                "variant": "pure",
                "source_csv": str(data_csv),
                "graph_csv": "" if graph_csv is None else str(graph_csv),
                "window": args.window or spec.lsr_window,
                "gradient_sample_count": len(samples),
                "runtime_seconds": timer.elapsed(),
                "basis_terms": ["linear", "square", "cube", "tanh", "signed_log1p"] if args.kind == "basis" else None,
                "created_as_dependency_for": "graph_zero",
            }
            write_summary_outputs(
                method_id=method_id,
                system=spec.name,
                source_csv=data_csv,
                output_dir=base_out_dir / "pure",
                records=pure_records,
                metadata=pure_metadata,
                local_records=samples if args.store_samples else None,
            )
        records = apply_graph_zero(pure_records, graph_csv, spec)
        for record in records:
            record["variant"] = "graph_zero"
    elif args.kind == "linear":
        records = linear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime_so_far)
    else:
        records = nonlinear_lsr_records(args, spec, gradients, data_csv, graph_csv, runtime_so_far)
    runtime = timer.elapsed()
    for record in records:
        record["runtime_seconds"] = runtime
    metadata = {
        "method": method_id,
        "variant": args.variant,
        "source_csv": str(data_csv),
        "graph_csv": "" if graph_csv is None else str(graph_csv),
        "window": args.window or spec.lsr_window,
        "gradient_sample_count": len(samples),
        "runtime_seconds": runtime,
        "basis_terms": ["linear", "square", "cube", "tanh", "signed_log1p"] if args.kind == "basis" else None,
    }
    out_dir = base_out_dir / args.variant
    paths = write_summary_outputs(
        method_id=method_id,
        system=spec.name,
        source_csv=data_csv,
        output_dir=out_dir,
        records=records,
        metadata=metadata,
        local_records=samples if args.store_samples else None,
    )
    print(json.dumps({"csv": str(paths[0]), "json": str(paths[1]), "runtime_seconds": runtime}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified local sensitivity regression estimator.")
    add_common_arguments(parser)
    parser.add_argument("--kind", choices=["linear", "basis"], required=True)
    parser.add_argument("--variant", choices=["pure", "graph_zero", "graph_control"], default="pure")
    parser.add_argument("--window", type=int)
    parser.add_argument("--store-samples", action="store_true", help="Store raw local-gradient samples in JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
