import argparse
import json
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_methods import (
    RuntimeTimer,
    add_common_arguments,
    apply_graph_zero,
    basis_derivative,
    basis_matrix,
    centered_slope,
    ensure_ss_dataframe,
    graph_ancestor_controls,
    graph_reachable_inputs,
    make_record,
    method_output_dir,
    residualize_with_controls,
    resolve_common_args,
    write_summary_outputs,
)


def fold_count(n_samples, requested):
    return max(2, min(int(requested), int(n_samples)))


def lasso_model(cv):
    return make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=42, max_iter=20000, alphas=100),
    )


def rf_model(args):
    return RandomForestRegressor(
        n_estimators=args.rf_estimators,
        min_samples_leaf=args.rf_min_leaf,
        random_state=42,
        n_jobs=-1,
    )


def crossfit_residuals(y, treatments, controls, args, nonlinear=False):
    y = np.asarray(y, dtype=float).reshape(-1)
    treatments = np.asarray(treatments, dtype=float)
    if treatments.ndim == 1:
        treatments = treatments.reshape(-1, 1)
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1 and controls.size:
        controls = controls.reshape(-1, 1)
    if controls.size == 0 or controls.shape[1] == 0:
        y_res = y - np.mean(y)
        d_res = treatments - np.mean(treatments, axis=0)
        return y_res, d_res

    n = len(y)
    folds = fold_count(n, args.folds)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    y_res = np.zeros(n)
    d_res = np.zeros_like(treatments)
    for train_idx, test_idx in kf.split(controls):
        nuisance_cv = fold_count(len(train_idx), args.nuisance_cv)
        if nonlinear:
            y_model = rf_model(args)
            d_models = [rf_model(args) for _ in range(treatments.shape[1])]
        else:
            y_model = lasso_model(nuisance_cv)
            d_models = [lasso_model(nuisance_cv) for _ in range(treatments.shape[1])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            y_model.fit(controls[train_idx], y[train_idx])
            y_res[test_idx] = y[test_idx] - y_model.predict(controls[test_idx])
            for col_idx, model in enumerate(d_models):
                model.fit(controls[train_idx], treatments[train_idx, col_idx])
                d_res[test_idx, col_idx] = treatments[test_idx, col_idx] - model.predict(controls[test_idx])
    return y_res, d_res


def estimate_linear_effect(y, treatment, controls, args):
    y_res, d_res = crossfit_residuals(y, treatment.reshape(-1, 1), controls, args, nonlinear=False)
    return centered_slope(y_res, d_res[:, 0])


def estimate_basis_effect(y, treatment, controls, args):
    phi, names = basis_matrix(treatment)
    y_res, phi_res = crossfit_residuals(y, phi, controls, args, nonlinear=True)
    coef = np.linalg.pinv(phi_res) @ y_res
    operating = float(np.mean(treatment))
    dphi_op, _ = basis_derivative(np.array([operating]))
    local_dphi, _ = basis_derivative(treatment)
    local_gradients = local_dphi @ coef
    return {
        "coef": coef,
        "basis_names": names,
        "operating_input": operating,
        "value": float(dphi_op[0] @ coef),
        "local_std": float(np.std(local_gradients, ddof=0)),
    }


def pure_control_columns(spec, treatment_name):
    return [name for name in spec.input_vars if name != treatment_name]


def graph_control_columns(spec, output_name, treatment_name, graph_controls):
    candidates = []
    for name in sorted(graph_controls.get(output_name, set())):
        if name == output_name or name == treatment_name:
            continue
        if name in spec.input_vars or name in spec.output_vars:
            candidates.append(name)
    return candidates


def build_records(args, spec, ss_df, data_csv, graph_csv, runtime, variant):
    method_id = f"dml_{args.kind}"
    records = []
    allowed = graph_reachable_inputs(graph_csv, spec) if variant in {"graph_zero", "graph_control"} else None
    graph_controls = graph_ancestor_controls(graph_csv, spec) if variant == "graph_control" else None

    pure_records_for_zero = []
    for output in spec.output_vars:
        y = ss_df[output].to_numpy(dtype=float)
        for input_name in spec.input_vars:
            treatment = ss_df[input_name].to_numpy(dtype=float)
            if variant == "graph_control":
                is_allowed = input_name in allowed.get(output, set())
                controls = graph_control_columns(spec, output, input_name, graph_controls)
                if not is_allowed:
                    records.append(
                        make_record(
                            method_id,
                            variant,
                            spec,
                            output,
                            input_name,
                            0.0,
                            0.0,
                            len(ss_df),
                            data_csv,
                            graph_csv,
                            runtime,
                            allowed_by_graph=False,
                            controls=json.dumps(controls),
                        )
                    )
                    continue
            else:
                controls = pure_control_columns(spec, input_name)

            control_matrix = ss_df[controls].to_numpy(dtype=float) if controls else np.zeros((len(ss_df), 0))
            if args.kind == "linear":
                value = estimate_linear_effect(y, treatment, control_matrix, args)
                std = ""
                extra = {}
            else:
                fit = estimate_basis_effect(y, treatment, control_matrix, args)
                value = fit["value"]
                std = fit["local_std"]
                extra = {
                    "basis_terms": ",".join(fit["basis_names"]),
                    "operating_input": fit["operating_input"],
                    "basis_coefficients": json.dumps({name: float(coef) for name, coef in zip(fit["basis_names"], fit["coef"])}),
                }
            record = make_record(
                method_id,
                "pure" if variant == "graph_zero" else variant,
                spec,
                output,
                input_name,
                value,
                std,
                len(ss_df),
                data_csv,
                graph_csv,
                runtime,
                allowed_by_graph=True if variant == "graph_control" else "",
                controls=json.dumps(controls),
                **extra,
            )
            if variant == "graph_zero":
                pure_records_for_zero.append(record)
            else:
                records.append(record)

    if variant == "graph_zero":
        records = apply_graph_zero(pure_records_for_zero, graph_csv, spec)
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
    if args.smoke:
        args.folds = 2
        args.nuisance_cv = 2
        args.rf_estimators = min(args.rf_estimators, 5)
        args.rf_min_leaf = 1
    method_id = f"dml_{args.kind}"
    out_dir = method_output_dir(method_id, spec.name, data_csv, output_root)
    ss_df, ss_path, ss_meta = ensure_ss_dataframe(data_csv, spec, out_dir / "steady_state_data")
    if args.smoke:
        ss_df = ss_df.iloc[: min(len(ss_df), 60)].copy()
    if args.variant in {"graph_zero", "graph_control"} and graph_csv is None:
        raise ValueError(f"DML variant '{args.variant}' requires --graph-csv")

    if args.variant == "graph_zero":
        pure_records = load_pure_summary_if_available(out_dir)
        if pure_records is None:
            pure_records = build_records(args, spec, ss_df, data_csv, graph_csv, 0.0, "pure")
            pure_metadata = {
                "method": method_id,
                "variant": "pure",
                "source_csv": str(data_csv),
                "graph_csv": "" if graph_csv is None else str(graph_csv),
                "ss_csv": "" if ss_path is None else str(ss_path),
                "ss_extraction": ss_meta,
                "folds": args.folds,
                "nuisance_cv": args.nuisance_cv,
                "rf_estimators": args.rf_estimators if args.kind == "basis" else None,
                "rf_min_leaf": args.rf_min_leaf if args.kind == "basis" else None,
                "basis_terms": list(("linear", "square", "cube", "tanh", "signed_log1p")) if args.kind == "basis" else None,
                "runtime_seconds": timer.elapsed(),
                "created_as_dependency_for": "graph_zero",
            }
            write_summary_outputs(
                method_id=method_id,
                system=spec.name,
                source_csv=data_csv,
                output_dir=out_dir / "pure",
                records=pure_records,
                metadata=pure_metadata,
            )
        records = apply_graph_zero(pure_records, graph_csv, spec)
        for record in records:
            record["variant"] = "graph_zero"
    else:
        records = build_records(args, spec, ss_df, data_csv, graph_csv, 0.0, args.variant)
    runtime = timer.elapsed()
    for record in records:
        record["runtime_seconds"] = runtime
    metadata = {
        "method": method_id,
        "variant": args.variant,
        "source_csv": str(data_csv),
        "graph_csv": "" if graph_csv is None else str(graph_csv),
        "ss_csv": "" if ss_path is None else str(ss_path),
        "ss_extraction": ss_meta,
        "folds": args.folds,
        "nuisance_cv": args.nuisance_cv,
        "rf_estimators": args.rf_estimators if args.kind == "basis" else None,
        "rf_min_leaf": args.rf_min_leaf if args.kind == "basis" else None,
        "basis_terms": list(("linear", "square", "cube", "tanh", "signed_log1p")) if args.kind == "basis" else None,
        "runtime_seconds": runtime,
    }
    paths = write_summary_outputs(
        method_id=method_id,
        system=spec.name,
        source_csv=data_csv,
        output_dir=out_dir / args.variant,
        records=records,
        metadata=metadata,
    )
    print(json.dumps({"csv": str(paths[0]), "json": str(paths[1]), "runtime_seconds": runtime}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified DML gradient estimator.")
    add_common_arguments(parser)
    parser.add_argument("--kind", choices=["linear", "basis"], required=True)
    parser.add_argument("--variant", choices=["pure", "graph_zero", "graph_control"], default="pure")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--nuisance-cv", type=int, default=5)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--rf-min-leaf", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
