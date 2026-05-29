from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI


DEFAULT_EXCLUDE_TIME = True
DEFAULT_USE_H = False
INPUT_VAR_PATTERN = re.compile(
    r"^(L_A|G_A|X0_A|Y6_A|L_B|G_B|X0_B|Y5_B)$",
    re.IGNORECASE,
)
HIDDEN_VAR_PATTERN = re.compile(r"^H\d+$", re.IGNORECASE)
AUX_VAR_PATTERN = re.compile(r"^[sc]\d+$", re.IGNORECASE)
TIME_VAR_PATTERN = re.compile(r"^(time|time_hours|t)$", re.IGNORECASE)
X_A_PATTERN = re.compile(r"^X[1-5]A$", re.IGNORECASE)
Y_A_PATTERN = re.compile(r"^Y[1-5]A$", re.IGNORECASE)
X_B_PATTERN = re.compile(r"^X[1-4]B$", re.IGNORECASE)
Y_B_PATTERN = re.compile(r"^Y[1-4]B$", re.IGNORECASE)

IDEAL_LAYOUT = {
    "X0_A": (0.08, 0.82),
    "X1A": (0.24, 0.82),
    "X2A": (0.39, 0.82),
    "X3A": (0.54, 0.82),
    "X4A": (0.69, 0.82),
    "X5A": (0.84, 0.82),
    "Y1A": (0.24, 0.60),
    "Y2A": (0.39, 0.60),
    "Y3A": (0.54, 0.60),
    "Y4A": (0.69, 0.60),
    "Y5A": (0.84, 0.60),
    "Y6_A": (0.98, 0.60),
    "L_A": (0.39, 0.98),
    "G_A": (0.54, 0.24),
    "X0_B": (0.08, -0.02),
    "X1B": (0.24, -0.02),
    "X2B": (0.39, -0.02),
    "X3B": (0.54, -0.02),
    "X4B": (0.69, -0.02),
    "Y1B": (0.24, -0.24),
    "Y2B": (0.39, -0.24),
    "Y3B": (0.54, -0.24),
    "Y4B": (0.69, -0.24),
    "Y5_B": (0.84, -0.24),
    "L_B": (0.39, 0.18),
    "G_B": (0.54, -0.46),
}


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def find_single_csv(search_dirs: list[Path]) -> Path:
    for directory in search_dirs:
        csv_files = sorted(
            path
            for path in directory.glob("*.csv")
            if not path.name.endswith("_links.csv") and not path.name.endswith("_all_links.csv")
        )
        if not csv_files:
            continue
        if len(csv_files) > 1:
            names = ", ".join(path.name for path in csv_files)
            raise RuntimeError(
                f"Expected exactly one raw CSV file in {directory}, found {len(csv_files)}: {names}"
            )
        return csv_files[0]

    searched = ", ".join(str(path) for path in search_dirs)
    raise FileNotFoundError(f"No raw CSV file found in: {searched}")


def resolve_input_path(path_arg: str, script_dir: Path) -> Path:
    input_path = Path(path_arg).expanduser()
    if input_path.is_absolute():
        return input_path

    candidates = [
        script_dir / input_path,
        script_dir.parent / input_path,
        Path.cwd() / input_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / input_path).resolve()


def resolve_output_path(path_arg: str, script_dir: Path, default_name: str) -> Path:
    if not path_arg:
        return script_dir / default_name

    output_path = Path(path_arg).expanduser()
    if output_path.is_absolute():
        return output_path
    return (script_dir / output_path).resolve()


def parse_vars(vars_arg: str, columns: list[str]) -> list[str]:
    if not vars_arg.strip():
        return columns

    requested = [item.strip() for item in vars_arg.split(",") if item.strip()]
    if not requested:
        raise ValueError("--vars was provided but no valid variable names were parsed.")

    missing = [name for name in requested if name not in columns]
    if missing:
        raise ValueError(f"Unknown variable(s) in --vars: {missing}. Available columns: {columns}")

    return requested


def filter_selected_columns(columns: list[str], *, exclude_time: bool, use_h: bool) -> list[str]:
    filtered = []
    for column in columns:
        if exclude_time and TIME_VAR_PATTERN.match(column):
            continue
        if AUX_VAR_PATTERN.match(column):
            continue
        if not use_h and HIDDEN_VAR_PATTERN.match(column):
            continue
        filtered.append(column)
    return filtered


def build_stage_layout(var_names: list[str]) -> dict[str, np.ndarray]:
    x_coords = np.zeros(len(var_names), dtype=float)
    y_coords = np.zeros(len(var_names), dtype=float)
    fallback_names = [name for name in var_names if name not in IDEAL_LAYOUT]
    fallback_x_positions = np.linspace(0.10, 0.90, len(fallback_names)) if fallback_names else []

    for index, name in enumerate(var_names):
        if name in IDEAL_LAYOUT:
            x_coords[index], y_coords[index] = IDEAL_LAYOUT[name]
        else:
            fallback_index = fallback_names.index(name)
            x_coords[index] = fallback_x_positions[fallback_index]
            y_coords[index] = -0.40

    return {"x": x_coords, "y": y_coords}


def save_links_csv(results: dict, var_names: list[str], output_path: Path) -> None:
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "target",
                "lag",
                "p_value",
                "effect_size",
                "abs_effect_size",
                "connection_type",
                "connection_string",
            ]
        )

        n_vars, _, n_lags = graph.shape
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                for lag in range(n_lags):
                    if source_idx == target_idx:
                        continue

                    connection_type = graph[source_idx, target_idx, lag]
                    if connection_type == "":
                        continue

                    effect_size = float(val_matrix[source_idx, target_idx, lag])
                    writer.writerow(
                        [
                            var_names[source_idx],
                            var_names[target_idx],
                            lag,
                            float(p_matrix[source_idx, target_idx, lag]),
                            effect_size,
                            abs(effect_size),
                            connection_type,
                            f"({source_idx},{-lag}) {connection_type} ({target_idx},0)",
                        ]
                    )


def save_all_links_csv(results: dict, var_names: list[str], output_path: Path, pc_alpha: float) -> None:
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "target",
                "lag",
                "selected",
                "connection_type",
                "p_value",
                "effect_size",
                "abs_effect_size",
                "p_gt_pc_alpha",
                "selection_note",
                "connection_string",
            ]
        )

        n_vars, _, n_lags = graph.shape
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                for lag in range(n_lags):
                    if source_idx == target_idx:
                        continue

                    connection_type = graph[source_idx, target_idx, lag]
                    p_value = float(p_matrix[source_idx, target_idx, lag])
                    effect_size = float(val_matrix[source_idx, target_idx, lag])
                    selected = connection_type != ""
                    p_gt_alpha = p_value > pc_alpha

                    if selected:
                        selection_note = "selected"
                    elif p_gt_alpha:
                        selection_note = "not_selected_p_value_gt_pc_alpha"
                    else:
                        selection_note = "not_selected_other_or_constrained"

                    writer.writerow(
                        [
                            var_names[source_idx],
                            var_names[target_idx],
                            lag,
                            selected,
                            connection_type,
                            p_value,
                            effect_size,
                            abs(effect_size),
                            p_gt_alpha,
                            selection_note,
                            f"({source_idx},{-lag}) {connection_type if connection_type else '[none]'} ({target_idx},0)",
                        ]
                    )


def detect_input_indices(var_names: list[str]) -> set[int]:
    return {idx for idx, name in enumerate(var_names) if INPUT_VAR_PATTERN.match(name)}


def is_allowed_strict_lg_target(source_name: str, target_name: str) -> bool:
    if source_name.upper() == "L_A":
        return bool(X_A_PATTERN.match(target_name))
    if source_name.upper() == "G_A":
        return bool(Y_A_PATTERN.match(target_name))
    if source_name.upper() == "L_B":
        return bool(X_B_PATTERN.match(target_name))
    if source_name.upper() == "G_B":
        return bool(Y_B_PATTERN.match(target_name))
    return True


def build_link_assumptions(
    var_names: list[str],
    tau_max: int,
    *,
    strict_lg_targets: bool = False,
) -> dict[int, dict[tuple[int, int], str]]:
    input_indices = detect_input_indices(var_names)
    assumptions: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(len(var_names))}

    for target_idx in range(len(var_names)):
        for source_idx in range(len(var_names)):
            for lag in range(0, tau_max + 1):
                if lag == 0 and source_idx == target_idx:
                    continue

                key = (source_idx, -lag)
                source_name = var_names[source_idx]
                target_name = var_names[target_idx]
                source_is_input = source_idx in input_indices
                target_is_input = target_idx in input_indices

                if strict_lg_targets:
                    source_is_lg = source_name.upper() in {"L_A", "G_A", "L_B", "G_B"}
                    target_is_lg = target_name.upper() in {"L_A", "G_A", "L_B", "G_B"}
                    source_disallowed = source_is_lg and not target_is_input and not is_allowed_strict_lg_target(source_name, target_name)
                    target_disallowed = target_is_lg and not source_is_input and not is_allowed_strict_lg_target(target_name, source_name)
                    if source_disallowed or target_disallowed:
                        assumptions[target_idx][key] = ""
                        continue

                if lag == 0:
                    if source_is_input and target_is_input:
                        assumptions[target_idx][key] = ""
                    elif source_is_input:
                        assumptions[target_idx][key] = "-?>"
                    elif target_is_input:
                        assumptions[target_idx][key] = "<?-"
                    else:
                        assumptions[target_idx][key] = "o?o"
                    continue

                if source_is_input and target_is_input:
                    assumptions[target_idx][key] = ""
                elif source_is_input:
                    assumptions[target_idx][key] = "-?>"
                elif target_is_input:
                    assumptions[target_idx][key] = ""
                else:
                    assumptions[target_idx][key] = "o?>"

    return assumptions


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV path. If omitted, the script searches its own folder and then the parent two-column folder.",
    )
    parser.add_argument(
        "--exclude-time",
        type=str2bool,
        default=DEFAULT_EXCLUDE_TIME,
        help="Exclude time/time_hours/t columns if present (true/false).",
    )
    parser.add_argument(
        "--vars",
        type=str,
        default="",
        help="Comma-separated list of variables to include. By default all non-time, non-auxiliary variables are used.",
    )
    parser.add_argument(
        "--use-h",
        type=str2bool,
        default=DEFAULT_USE_H,
        help="Include hidden H variables in the analysis if present (true/false).",
    )
    parser.add_argument("--tau-max", type=int, default=1, help="Maximum lag for LPCMCI.")
    parser.add_argument("--pc-alpha", type=float, default=0.05, help="LPCMCI significance level.")
    parser.add_argument("--verbosity", type=int, default=0, help="LPCMCI verbosity level.")
    parser.add_argument(
        "--strict-lg-targets",
        type=str2bool,
        default=False,
        help="Restrict L_A/L_B edges to same-column X states and G_A/G_B edges to same-column Y states.",
    )
    parser.add_argument(
        "--plot",
        type=str2bool,
        default=True,
        help="Plot the learned DPAG using tigramite.plotting.plot_graph (true/false).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Optional output path for the plot image. Relative paths are resolved inside the current CI-test directory.",
    )
    parser.add_argument(
        "--save-links",
        type=str,
        default="",
        help="Optional output path for discovered links CSV. Relative paths are resolved inside the current CI-test directory.",
    )
    parser.add_argument(
        "--save-all-links",
        type=str,
        default="",
        help="Optional output path for all tested links CSV. Relative paths are resolved inside the current CI-test directory.",
    )
    return parser


def run_multi_stage_lpcmci(
    *,
    args: argparse.Namespace,
    cond_ind_test: object,
    ci_test_label: str,
    ci_test_summary: str,
    script_path: Path,
) -> None:
    script_dir = script_path.resolve().parent
    csv_path = (
        resolve_input_path(args.csv, script_dir)
        if args.csv
        else find_single_csv([script_dir, script_dir.parent])
    )

    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    columns = [column.strip() for column in header.split(",") if column.strip()]
    if not columns:
        raise ValueError(f"No columns found in CSV header for {csv_path.name}.")

    selected_columns = parse_vars(args.vars, columns)
    selected_columns = filter_selected_columns(
        selected_columns,
        exclude_time=args.exclude_time,
        use_h=args.use_h,
    )
    if not selected_columns:
        raise ValueError(
            "No variables left after excluding time/auxiliary/hidden variables. "
            "Use --use-h true or choose different --vars."
        )

    usecols = [columns.index(column) for column in selected_columns]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    dataframe = pp.DataFrame(data=data, var_names=selected_columns)
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=args.verbosity)
    link_assumptions = build_link_assumptions(
        selected_columns,
        tau_max=args.tau_max,
        strict_lg_targets=args.strict_lg_targets,
    )
    results = lpcmci.run_lpcmci(
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        link_assumptions=link_assumptions,
    )

    print(f"CI test: {ci_test_summary}")
    print(f"CSV used: {csv_path}")
    print(f"Variables used: {selected_columns}")
    print(
        "Input variables with structural assumptions:",
        [selected_columns[idx] for idx in sorted(detect_input_indices(selected_columns))],
    )
    print("\nLPCMCI graph shape:", results["graph"].shape)
    print("LPCMCI graph:")
    print(results["graph"])

    links_output_path = resolve_output_path(
        args.save_links,
        script_dir,
        f"{csv_path.stem}_{ci_test_label}_assumed_links.csv",
    )
    save_links_csv(results=results, var_names=selected_columns, output_path=links_output_path)
    print(f"Saved links CSV to: {links_output_path}")

    all_links_output_path = resolve_output_path(
        args.save_all_links,
        script_dir,
        f"{csv_path.stem}_{ci_test_label}_assumed_all_links.csv",
    )
    save_all_links_csv(
        results=results,
        var_names=selected_columns,
        output_path=all_links_output_path,
        pc_alpha=args.pc_alpha,
    )
    print(f"Saved all-links CSV to: {all_links_output_path}")

    if args.plot:
        node_pos = build_stage_layout(selected_columns)
        tp.plot_graph(
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            var_names=selected_columns,
            node_pos=node_pos,
            figsize=(18, 9),
            node_size=0.13,
            node_label_size=11,
        )
        if args.save_plot:
            plot_output_path = resolve_output_path(args.save_plot, script_dir, "")
            plot_output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_output_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {plot_output_path}")
            plt.close()
        else:
            plt.show()
