"""Run LPCMCI on heating-system CSV data using structural link assumptions.

Usage examples:
    python LPCMCI_heat_sys_link_assumptions.py
    python LPCMCI_heat_sys_link_assumptions.py --vars u1,u2,u3,u4,C --tau-max 2 --pc-alpha 0.05

This variant encodes prior knowledge that variables matching ``u<integer>`` are
input variables:
    * u variables are not influenced by non-u variables.
    * u variables do not influence each other.
    * Any allowed edge touching a u variable is oriented with a tail on the u side.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

DEFAULT_EXCLUDE_TIME = True
DEFAULT_USE_H = False
INPUT_VAR_PATTERN = re.compile(r"^u\d+$", re.IGNORECASE)  # Matches variable names like u1, u2, u3, etc. (case-insensitive)
DEFAULT_INPUTS_DIRNAME = "Multiple inputs"
DEFAULT_OUTPUTS_DIRNAME = "multiple outputs"


def str2bool(value: str) -> bool:
    """Parse a string into a boolean value for argparse."""
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")



def find_single_csv(script_dir: Path) -> Path:
    """Return the only raw data CSV in script_dir or raise a helpful error."""
    csv_files = sorted(
        path
        for path in script_dir.glob("*.csv")
        if not path.name.endswith("_links.csv") and not path.name.endswith("_all_links.csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {script_dir}. Place exactly one .csv file next to this script."
        )
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise RuntimeError(
            f"Expected exactly one CSV file in {script_dir}, found {len(csv_files)}: {names}"
        )
    return csv_files[0]


def find_csvs_in_directory(input_dir: Path) -> list[Path]:
    """Return all raw data CSV files in a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    csv_files = sorted(path.resolve() for path in input_dir.glob("*.csv") if is_raw_data_csv(path))
    if not csv_files:
        raise FileNotFoundError(f"No valid input CSV files found in {input_dir}")
    return csv_files


def is_raw_data_csv(path: Path) -> bool:
    """Return True for input CSVs and False for derived link-summary CSVs."""
    return path.suffix.lower() == ".csv" and not (
        path.name.endswith("_links.csv")
        or path.name.endswith("_all_links.csv")
        or path.name.endswith("_runtime.csv")
        or path.name == "runtime_summary.csv"
        or path.name == "batch_runtime_summary.csv"
    )


def select_csv_files_gui(initial_dir: Path) -> list[Path]:
    """Open a file picker so the user can select multiple CSV files."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is not available, so --select-files cannot open a file picker.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_files = filedialog.askopenfilenames(
        title="Select one or more CSV files for LPCMCI",
        initialdir=str(initial_dir),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    csv_paths = [Path(file_path).resolve() for file_path in selected_files]
    csv_paths = [path for path in csv_paths if is_raw_data_csv(path)]
    if not csv_paths:
        raise RuntimeError("No valid input CSV files were selected.")
    return csv_paths


def resolve_csv_paths(csv_args: list[str], script_dir: Path) -> list[Path]:
    """Resolve CLI CSV paths relative to the script directory when needed."""
    csv_paths: list[Path] = []
    for raw_path in csv_args:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (script_dir / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {path}")
        if not is_raw_data_csv(path):
            raise ValueError(f"Not a valid raw-data CSV input: {path}")
        csv_paths.append(path)
    return csv_paths


def make_unique_run_names(csv_paths: list[Path]) -> list[str]:
    """Create stable per-run folder names, even when stems repeat."""
    counts: dict[str, int] = {}
    run_names: list[str] = []
    for csv_path in csv_paths:
        base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", csv_path.stem).strip("._-") or "run"
        count = counts.get(base_name, 0)
        counts[base_name] = count + 1
        run_names.append(base_name if count == 0 else f"{base_name}_{count + 1}")
    return run_names


def write_runtime_csv(output_path: Path, summary: dict[str, object]) -> None:
    """Write runtime metadata for one CSV run."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "csv_file",
                "output_dir",
                "started_at",
                "finished_at",
                "runtime_seconds",
                "row_count",
                "column_count",
                "variables",
                "tau_max",
                "pc_alpha",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def resolve_output_file_path(
    cli_value: str,
    default_path: Path,
    batch_output_dir: Path | None,
) -> Path:
    """Resolve explicit output paths for both single-run and batch-run modes."""
    if not cli_value:
        return default_path
    if batch_output_dir is None:
        return Path(cli_value)
    cli_path = Path(cli_value)
    return cli_path if cli_path.is_absolute() else batch_output_dir / cli_path.name



def parse_vars(vars_arg: str, columns: list[str]) -> list[str]:
    """Parse optional --vars argument into an ordered column list."""
    if not vars_arg.strip():
        return columns

    requested = [item.strip() for item in vars_arg.split(",") if item.strip()]
    if not requested:
        raise ValueError("--vars was provided but no valid variable names were parsed.")

    missing = [name for name in requested if name not in columns]
    if missing:
        raise ValueError(f"Unknown variable(s) in --vars: {missing}. Available columns: {columns}")

    return requested



def save_links_csv(results: dict, var_names: list[str], output_path: Path) -> None:
    """Save discovered links to CSV with link type and metrics."""
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

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
    """Save all tested links (selected and non-selected) to CSV."""
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

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
    """Return indices of variables treated as exogenous u inputs."""
    return {idx for idx, name in enumerate(var_names) if INPUT_VAR_PATTERN.match(name)}



def build_link_assumptions(var_names: list[str], tau_max: int) -> dict[int, dict[tuple[int, int], str]]:
    """Build complete LPCMCI link assumptions using u variables as exogenous inputs.

    Missing entries in LPCMCI imply 'no link', so this helper explicitly lists every
    candidate pair/lag and only rules out the links that conflict with the prior knowledge.
    """
    input_indices = detect_input_indices(var_names)
    assumptions: dict[int, dict[tuple[int, int], str]] = {j: {} for j in range(len(var_names))}

    for target_idx in range(len(var_names)):
        for source_idx in range(len(var_names)):
            for lag in range(0, tau_max + 1):
                if lag == 0 and source_idx == target_idx:
                    continue

                key = (source_idx, -lag)
                source_is_input = source_idx in input_indices
                target_is_input = target_idx in input_indices

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


def run_single_csv(csv_path: Path, args: argparse.Namespace, batch_output_dir: Path | None = None) -> dict[str, object]:
    """Run LPCMCI for one CSV and write its outputs."""
    start_dt = datetime.now().astimezone()
    start_time = time.perf_counter()

    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    columns = [column.strip() for column in header.split(",") if column.strip()]
    if not columns:
        raise ValueError(f"No columns found in CSV header for {csv_path.name}.")

    working_columns = columns.copy()
    if args.exclude_time and working_columns[0].lower() == "t":
        working_columns = working_columns[1:]

    selected_columns = parse_vars(args.vars, working_columns)
    if not args.use_h and "H" in selected_columns:
        selected_columns = [column for column in selected_columns if column != "H"]
        if not selected_columns:
            raise ValueError(
                "No variables left after excluding H. Use --use-h true or choose different --vars."
            )

    usecols = [columns.index(column) for column in selected_columns]

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    dataframe = pp.DataFrame(data=data, var_names=selected_columns)
    parcorr = ParCorr(significance="analytic")
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=args.verbosity)
    link_assumptions = build_link_assumptions(selected_columns, tau_max=args.tau_max)

    results = lpcmci.run_lpcmci(
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        link_assumptions=link_assumptions,
        max_cond_px=args.max_cond_px,
        max_p_global=args.max_p_global,
        max_p_non_ancestral=args.max_p_non_ancestral,
        max_q_global=args.max_q_global,
        max_pds_set=args.max_pds_set,
        prelim_only=args.prelim_only,
    )

    default_links_path = (
        batch_output_dir / f"{csv_path.stem}_assumed_links.csv"
        if batch_output_dir is not None
        else csv_path.with_name(f"{csv_path.stem}_assumed_links.csv")
    )
    links_output_path = resolve_output_file_path(args.save_links, default_links_path, batch_output_dir)
    save_links_csv(results=results, var_names=selected_columns, output_path=links_output_path)

    default_all_links_path = (
        batch_output_dir / f"{csv_path.stem}_assumed_all_links.csv"
        if batch_output_dir is not None
        else csv_path.with_name(f"{csv_path.stem}_assumed_all_links.csv")
    )
    all_links_output_path = resolve_output_file_path(args.save_all_links, default_all_links_path, batch_output_dir)
    save_all_links_csv(
        results=results,
        var_names=selected_columns,
        output_path=all_links_output_path,
        pc_alpha=args.pc_alpha,
    )

    tp.plot_graph(
        graph=results["graph"],
        val_matrix=results["val_matrix"],
        var_names=selected_columns,
    )
    default_plot_path = (
        batch_output_dir / f"{csv_path.stem}_graph.png"
        if batch_output_dir is not None
        else csv_path.with_name(f"{csv_path.stem}_graph.png")
    )
    plot_output_path = resolve_output_file_path(args.save_plot, default_plot_path, batch_output_dir)
    plt.savefig(plot_output_path, dpi=200, bbox_inches="tight")
    plt.close()

    runtime_seconds = time.perf_counter() - start_time
    finished_dt = datetime.now().astimezone()

    default_runtime_path = (
        batch_output_dir / "runtime_summary.csv"
        if batch_output_dir is not None
        else csv_path.with_name(f"{csv_path.stem}_runtime.csv")
    )
    runtime_output_path = resolve_output_file_path(getattr(args, "save_runtime", ""), default_runtime_path, batch_output_dir)

    summary = {
        "status": "success",
        "csv_file": str(csv_path),
        "output_dir": str(batch_output_dir) if batch_output_dir is not None else str(csv_path.parent),
        "started_at": start_dt.isoformat(),
        "finished_at": finished_dt.isoformat(),
        "runtime_seconds": f"{runtime_seconds:.6f}",
        "row_count": int(data.shape[0]),
        "column_count": int(data.shape[1]),
        "variables": ",".join(selected_columns),
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "error": "",
    }
    write_runtime_csv(runtime_output_path, summary)

    print(f"CSV used: {csv_path}")
    print(f"Variables used: {selected_columns}")
    print(
        "Input variables with structural assumptions:",
        [selected_columns[idx] for idx in sorted(detect_input_indices(selected_columns))],
    )
    print("\nLPCMCI graph shape:", results["graph"].shape)
    print(
        "LPCMCI conditioning limits:",
        {
            "max_cond_px": args.max_cond_px,
            "max_p_global": args.max_p_global,
            "max_p_non_ancestral": args.max_p_non_ancestral,
            "max_q_global": args.max_q_global,
            "max_pds_set": args.max_pds_set,
            "prelim_only": args.prelim_only,
        },
    )
    print("LPCMCI graph:")
    print(results["graph"])
    print(f"Saved links CSV to: {links_output_path}")
    print(f"Saved all-links CSV to: {all_links_output_path}")
    print(f"Saved plot to: {plot_output_path}")
    print(f"Saved runtime CSV to: {runtime_output_path}")

    return summary


def run_single_csv_worker(csv_path_str: str, args_dict: dict[str, object], batch_output_dir_str: str) -> dict[str, object]:
    """Process-pool wrapper for one CSV run."""
    csv_path = Path(csv_path_str)
    batch_output_dir = Path(batch_output_dir_str)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**args_dict)

    try:
        return run_single_csv(csv_path=csv_path, args=args, batch_output_dir=batch_output_dir)
    except Exception as exc:
        finished_dt = datetime.now().astimezone()
        error_path = batch_output_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        summary = {
            "status": "failed",
            "csv_file": str(csv_path),
            "output_dir": str(batch_output_dir),
            "started_at": "",
            "finished_at": finished_dt.isoformat(),
            "runtime_seconds": "",
            "row_count": "",
            "column_count": "",
            "variables": "",
            "tau_max": args.tau_max,
            "pc_alpha": args.pc_alpha,
            "error": str(exc),
        }
        write_runtime_csv(batch_output_dir / "runtime_summary.csv", summary)
        return summary


def run_batch(csv_paths: list[Path], args: argparse.Namespace, script_dir: Path) -> None:
    """Run LPCMCI on multiple CSV files in parallel."""
    batch_root = Path(args.output_root) if args.output_root else script_dir / DEFAULT_OUTPUTS_DIRNAME
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = batch_root / f"lpcmci_batch_{batch_timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=False)

    run_names = make_unique_run_names(csv_paths)
    worker_count = args.workers if args.workers is not None else min(len(csv_paths), os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, len(csv_paths)))

    print(f"Starting batch run with {len(csv_paths)} CSV file(s) and {worker_count} worker(s).")
    print(f"Batch output directory: {batch_dir}")

    summary_rows: list[dict[str, object]] = []
    args_dict = vars(args).copy()

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {}
        for csv_path, run_name in zip(csv_paths, run_names):
            output_dir = batch_dir / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            future = executor.submit(
                run_single_csv_worker,
                str(csv_path),
                args_dict,
                str(output_dir),
            )
            future_map[future] = (csv_path, output_dir)

        for future in concurrent.futures.as_completed(future_map):
            csv_path, output_dir = future_map[future]
            summary = future.result()
            summary_rows.append(summary)
            print(
                f"[{summary['status']}] {csv_path.name} -> {output_dir} "
                f"(runtime_seconds={summary['runtime_seconds'] or 'n/a'})"
            )

    batch_summary_path = batch_dir / "batch_runtime_summary.csv"
    with batch_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "csv_file",
                "output_dir",
                "started_at",
                "finished_at",
                "runtime_seconds",
                "row_count",
                "column_count",
                "variables",
                "tau_max",
                "pc_alpha",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    failed_runs = [row for row in summary_rows if row["status"] != "success"]
    print(f"Saved batch runtime summary to: {batch_summary_path}")
    if failed_runs:
        print(f"Batch completed with {len(failed_runs)} failed run(s). See per-run error.txt files for details.")
    else:
        print("Batch completed successfully.")



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LPCMCI on the single CSV file that is in the same directory as this script, "
            "using link assumptions for u1/u2/u3/u4-style input variables."
        )
    )
    parser.add_argument(
        "--exclude-time",
        type=str2bool,
        default=DEFAULT_EXCLUDE_TIME,
        help="Exclude a leading 't' time column if present (true/false).",
    )
    parser.add_argument(
        "--vars",
        type=str,
        default="",
        help=(
            "Comma-separated list of variables to include in the analysis, e.g. u1,u2,C,H. "
            "By default all columns are used (except 't' when --exclude-time is true)."
        ),
    )
    parser.add_argument(
        "--use-h",
        type=str2bool,
        default=DEFAULT_USE_H,
        help="Include column H in the analysis if present (true/false).",
    )
    parser.add_argument("--tau-max", type=int, default=2, help="Maximum lag for LPCMCI.")
    parser.add_argument("--pc-alpha", type=float, default=0.05, help="LPCMCI significance level.")
    parser.add_argument("--max-cond-px", type=int, default=0, help="Maximum lagged-parent conditioning dimension.")
    parser.add_argument("--max-p-global", type=float, default=float("inf"), help="Maximum global conditioning dimension.")
    parser.add_argument(
        "--max-p-non-ancestral",
        type=float,
        default=float("inf"),
        help="Maximum conditioning dimension in the non-ancestral phase.",
    )
    parser.add_argument("--max-q-global", type=float, default=float("inf"), help="Maximum q conditioning dimension.")
    parser.add_argument("--max-pds-set", type=float, default=float("inf"), help="Maximum possible-D-sep set size.")
    parser.add_argument("--prelim-only", type=str2bool, default=False, help="Run only LPCMCI preliminary phase.")
    parser.add_argument("--verbosity", type=int, default=0, help="LPCMCI verbosity level.")
    parser.add_argument(
        "--csv-files",
        nargs="+",
        default=[],
        help=(
            "One or more CSV files to process. When multiple files are provided, they are run in parallel. "
            "Relative paths are resolved from this script's directory."
        ),
    )
    parser.add_argument(
        "--select-files",
        type=str2bool,
        default=False,
        help="Open a file picker to select one or more CSV files (true/false).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for batch runs. Defaults to min(number of files, CPU count).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Root directory for batch outputs. Defaults to the 'multiple outputs' folder next to this script.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Directory containing input CSV files for batch mode. Defaults to the 'Multiple inputs' folder next to this script.",
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
        help="Optional output path for the plot image (e.g. lpcmci_graph.png).",
    )
    parser.add_argument(
        "--save-links",
        type=str,
        default="",
        help="Optional output path for discovered links CSV. Defaults to suffix _links.csv.",
    )
    parser.add_argument(
        "--save-all-links",
        type=str,
        default="",
        help="Optional output path for all links CSV (selected + not selected). Defaults to suffix _all_links.csv.",
    )
    parser.add_argument(
        "--save-runtime",
        type=str,
        default="",
        help="Optional runtime CSV output path. In batch mode, a filename here is written inside each run folder.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / DEFAULT_INPUTS_DIRNAME
    selected_csvs = resolve_csv_paths(args.csv_files, script_dir) if args.csv_files else []
    input_dir = Path(args.input_dir) if args.input_dir else default_input_dir
    if not input_dir.is_absolute():
        input_dir = (script_dir / input_dir).resolve()
    else:
        input_dir = input_dir.resolve()

    if not selected_csvs and not args.select_files:
        selected_csvs.extend(find_csvs_in_directory(input_dir))
    if args.select_files:
        selected_csvs.extend(select_csv_files_gui(script_dir))

    unique_csvs: list[Path] = []
    seen_paths: set[Path] = set()
    for csv_path in selected_csvs:
        if csv_path not in seen_paths:
            unique_csvs.append(csv_path)
            seen_paths.add(csv_path)

    if len(unique_csvs) > 1:
        run_batch(unique_csvs, args, script_dir)
        return

    csv_path = unique_csvs[0] if unique_csvs else find_single_csv(script_dir)
    run_single_csv(csv_path=csv_path, args=args)


if __name__ == "__main__":
    main()
