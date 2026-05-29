import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


METHOD_COMMANDS = {
    "state_space": ("run_state_space.py", []),
    "neural_ode": ("run_neural_ode.py", []),
    "lstm": ("run_lstm.py", []),
    "gp_rbf": ("run_gp.py", ["--variant", "rbf"]),
    "gp_matern": ("run_gp.py", ["--variant", "matern"]),
    "dml_linear_pure": ("run_dml.py", ["--kind", "linear", "--variant", "pure"]),
    "dml_linear_graph_zero": ("run_dml.py", ["--kind", "linear", "--variant", "graph_zero"]),
    "dml_linear_graph_control": ("run_dml.py", ["--kind", "linear", "--variant", "graph_control"]),
    "dml_basis_pure": ("run_dml.py", ["--kind", "basis", "--variant", "pure"]),
    "dml_basis_graph_zero": ("run_dml.py", ["--kind", "basis", "--variant", "graph_zero"]),
    "dml_basis_graph_control": ("run_dml.py", ["--kind", "basis", "--variant", "graph_control"]),
    "lsr_linear_pure": ("run_lsr.py", ["--kind", "linear", "--variant", "pure"]),
    "lsr_linear_graph_zero": ("run_lsr.py", ["--kind", "linear", "--variant", "graph_zero"]),
    "lsr_linear_graph_control": ("run_lsr.py", ["--kind", "linear", "--variant", "graph_control"]),
    "lsr_basis_pure": ("run_lsr.py", ["--kind", "basis", "--variant", "pure"]),
    "lsr_basis_graph_zero": ("run_lsr.py", ["--kind", "basis", "--variant", "graph_zero"]),
    "lsr_basis_graph_control": ("run_lsr.py", ["--kind", "basis", "--variant", "graph_control"]),
}


METHOD_ALIASES = {
    "state_space": "state_space",
    "linear_state_space": "state_space",
    "linear_state_space_estimator": "state_space",
    "neural_ode": "neural_ode",
    "lstm": "lstm",
    "lstm_sequence_model": "lstm",
    "gp_rbf": "gp_rbf",
    "rbf_gp": "gp_rbf",
    "gp_matern": "gp_matern",
    "matern_gp": "gp_matern",
    "dml_linear_pure": "dml_linear_pure",
    "dml_linear": "dml_linear_pure",
    "linear_residual_dml": "dml_linear_pure",
    "linear_residual_dml_no_graph": "dml_linear_pure",
    "dml_linear_graph_zero": "dml_linear_graph_zero",
    "linear_residual_dml_graph_zero": "dml_linear_graph_zero",
    "dml_linear_graph_control": "dml_linear_graph_control",
    "linear_residual_dml_graph_control": "dml_linear_graph_control",
    "dml_basis_pure": "dml_basis_pure",
    "dml_basis": "dml_basis_pure",
    "expanded_dml": "dml_basis_pure",
    "expanded_dml_no_graph": "dml_basis_pure",
    "dml_basis_graph_zero": "dml_basis_graph_zero",
    "expanded_dml_graph_zero": "dml_basis_graph_zero",
    "dml_basis_graph_control": "dml_basis_graph_control",
    "expanded_dml_graph_control": "dml_basis_graph_control",
    "lsr_linear_pure": "lsr_linear_pure",
    "lsr_linear": "lsr_linear_pure",
    "average_lse": "lsr_linear_pure",
    "average_lse_no_graph": "lsr_linear_pure",
    "lsr_linear_graph_zero": "lsr_linear_graph_zero",
    "average_lse_graph_zero": "lsr_linear_graph_zero",
    "lsr_linear_graph_control": "lsr_linear_graph_control",
    "average_lse_graph_control": "lsr_linear_graph_control",
    "lsr_basis_pure": "lsr_basis_pure",
    "lsr_basis": "lsr_basis_pure",
    "expanded_lse": "lsr_basis_pure",
    "expanded_lse_no_graph": "lsr_basis_pure",
    "lsr_basis_graph_zero": "lsr_basis_graph_zero",
    "expanded_lse_graph_zero": "lsr_basis_graph_zero",
    "lsr_basis_graph_control": "lsr_basis_graph_control",
    "expanded_lse_graph_control": "lsr_basis_graph_control",
}


SYSTEM_ALIASES = {
    "heat": "heat",
    "heating": "heat",
    "lumped": "heat",
    "lumped_heat": "heat",
    "lumped_thermal": "heat",
    "thermal": "heat",
    "low": "low_dim",
    "low_dim": "low_dim",
    "lowdim": "low_dim",
    "low-dimensional": "low_dim",
    "low_dimensional": "low_dim",
    "nonlinear": "low_dim",
    "multi": "multi_stage",
    "multistage": "multi_stage",
    "multi_stage": "multi_stage",
    "two_column": "multi_stage",
    "two-column": "multi_stage",
    "2column": "multi_stage",
    "2_column": "multi_stage",
    "2colum": "multi_stage",
    "2_colum": "multi_stage",
}


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_method(value: str) -> str:
    key = normalize_token(value)
    if key not in METHOD_ALIASES:
        valid = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown method '{value}'. Valid aliases include: {valid}")
    return METHOD_ALIASES[key]


def normalize_system(value: str) -> str:
    key = normalize_token(value)
    if key not in SYSTEM_ALIASES:
        valid = ", ".join(sorted(SYSTEM_ALIASES))
        raise ValueError(f"Unknown system '{value}'. Valid aliases include: {valid}")
    return SYSTEM_ALIASES[key]


def parse_args():
    parser = argparse.ArgumentParser(description="Single entry point for thesis gradient-estimation methods.")
    parser.add_argument("--method", required=True, help="Method alias, for example average_lse, expanded_dml, rbf_gp, or neural_ode.")
    parser.add_argument("--system", required=True, help="Benchmark alias, for example lumped, low_dim, two_column, heat, or multi_stage.")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--graph-csv")
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to the selected method runner after '--'.")
    return parser.parse_args()


def main():
    args = parse_args()
    method = normalize_method(args.method)
    system = normalize_system(args.system)
    script_name, fixed = METHOD_COMMANDS[method]
    command = [
        sys.executable,
        str(BASE_DIR / script_name),
        "--system",
        system,
        "--data-csv",
        args.data_csv,
        *fixed,
    ]
    if args.graph_csv:
        command.extend(["--graph-csv", args.graph_csv])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    if args.smoke:
        command.append("--smoke")
    extras = list(args.extra_args)
    if extras and extras[0] == "--":
        extras = extras[1:]
    command.extend(extras)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
