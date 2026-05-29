from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DONE_DATA = REPOSITORY_ROOT / "data" / "done_data"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / (
    "standardized_results_windows" if os.name == "nt" else "standardized_results_linux"
)
GRADIENT_RUNNER = REPOSITORY_ROOT / "code" / "gradient_estimation" / "run_gradient_method.py"

DEFAULT_METHODS = [
    "state_space",
    "neural_ode",
    "lstm",
    "gp_rbf",
    "gp_matern",
    "dml_linear_pure",
    "dml_linear_graph_zero",
    "dml_linear_graph_control",
    "dml_basis_pure",
    "dml_basis_graph_zero",
    "dml_basis_graph_control",
    "lsr_linear_pure",
    "lsr_linear_graph_zero",
    "lsr_linear_graph_control",
    "lsr_basis_pure",
    "lsr_basis_graph_zero",
    "lsr_basis_graph_control",
]

GRAPH_METHODS = {method for method in DEFAULT_METHODS if "_graph_" in method}


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


@dataclass(frozen=True)
class SystemConfig:
    runner_system: str
    done_data_label: str
    output_key: str
    graph_suffix_priority: tuple[str, ...]


SYSTEMS = {
    "lumped": SystemConfig(
        runner_system="heat",
        done_data_label="lumped (heat)",
        output_key="lumped",
        graph_suffix_priority=("_assumed_links.csv",),
    ),
    "low_dim": SystemConfig(
        runner_system="low_dim",
        done_data_label="low dim",
        output_key="low_dim",
        graph_suffix_priority=("_gpdc_links.csv",),
    ),
    "multi_stage": SystemConfig(
        runner_system="multi_stage",
        done_data_label="2 colum (multi)",
        output_key="multi_stage",
        graph_suffix_priority=("_parcorr_linear_assumed_links.csv", "_assumed_links.csv"),
    ),
}

SYSTEM_ALIASES = {
    "heat": "lumped",
    "heating": "lumped",
    "lumped_heat": "lumped",
    "lumped_thermal": "lumped",
    "lumped": "lumped",
    "thermal": "lumped",
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

REGIME_ALIASES = {
    "zero": "zero",
    "to_zero": "zero",
    "return_to_zero": "zero",
    "back_to_zero": "zero",
    "return_to_baseline": "zero",
    "return_to_base": "zero",
    "baseline_return": "zero",
    "base": "zero",
    "var": "var",
    "variable": "var",
    "varying": "var",
    "var_baseline": "var",
    "variable_baseline": "var",
}


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_system(value: str) -> str:
    key = normalize_token(value)
    if key not in SYSTEM_ALIASES:
        valid = ", ".join(sorted(SYSTEM_ALIASES))
        raise ValueError(f"Unknown system '{value}'. Valid aliases include: {valid}")
    return SYSTEM_ALIASES[key]


def normalize_regime(value: str) -> str:
    key = normalize_token(value)
    if key not in REGIME_ALIASES:
        valid = ", ".join(sorted(REGIME_ALIASES))
        raise ValueError(f"Unknown input regime '{value}'. Valid aliases include: {valid}")
    return REGIME_ALIASES[key]


def normalize_method(value: str) -> str:
    key = normalize_token(value)
    if key not in METHOD_ALIASES:
        valid = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown method '{value}'. Valid aliases include: {valid}")
    return METHOD_ALIASES[key]


def parse_methods(method_values: list[str] | None) -> list[str]:
    if not method_values:
        return list(DEFAULT_METHODS)
    tokens: list[str] = []
    for value in method_values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(DEFAULT_METHODS)

    methods = [normalize_method(method) for method in tokens]

    # Preserve thesis/default dependency order, even if the user provides a shuffled list.
    requested = set(methods)
    return [method for method in DEFAULT_METHODS if method in requested]


def safe_name(value: str, max_len: int = 120) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return name[:max_len] if len(name) > max_len else name


def discover_process_files(done_data: Path, config: SystemConfig, regime: str, max_files: int | None) -> list[Path]:
    process_root = done_data / "Process data" / config.done_data_label / regime
    if not process_root.is_dir():
        raise FileNotFoundError(f"Missing process-data directory: {process_root}")
    files = sorted(path for path in process_root.rglob("*.csv") if path.is_file())
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No process-data CSV files found in {process_root}")
    return files


def graph_candidates_for(process_csv: Path, process_root: Path, graph_root: Path, config: SystemConfig) -> list[Path]:
    rel_parent = process_csv.relative_to(process_root).parent
    local_graph_dir = graph_root / rel_parent
    if not local_graph_dir.is_dir():
        return []

    candidates: list[Path] = []
    for suffix in config.graph_suffix_priority:
        exact = local_graph_dir / f"{process_csv.stem}{suffix}"
        if exact.is_file():
            candidates.append(exact)

    if not candidates:
        for path in sorted(local_graph_dir.glob(f"{process_csv.stem}*links.csv")):
            if "_all_links" in path.name:
                continue
            candidates.append(path)

    # Keep order but remove accidental duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def resolve_graph_file(done_data: Path, config: SystemConfig, regime: str, process_csv: Path) -> tuple[Path | None, str]:
    process_root = done_data / "Process data" / config.done_data_label / regime
    graph_root = done_data / "Graph data" / config.done_data_label / regime
    if not graph_root.is_dir():
        return None, "missing_graph_directory"

    candidates = graph_candidates_for(process_csv, process_root, graph_root, config)
    if len(candidates) == 1:
        return candidates[0], "exact_or_unique_match"
    if len(candidates) > 1:
        candidate_names = "; ".join(path.name for path in candidates)
        return None, f"ambiguous_graph_match:{candidate_names}"
    return None, "missing_graph_file"


def build_command(
    *,
    python_executable: str,
    method: str,
    config: SystemConfig,
    process_csv: Path,
    graph_csv: Path | None,
    output_root: Path,
    smoke: bool,
) -> list[str]:
    command = [
        python_executable,
        str(GRADIENT_RUNNER),
        "--method",
        method,
        "--system",
        config.runner_system,
        "--data-csv",
        str(process_csv),
        "--output-root",
        str(output_root),
    ]
    if graph_csv is not None:
        command.extend(["--graph-csv", str(graph_csv)])
    if smoke:
        command.append("--smoke")
    return command


def build_execution_command(command: list[str], job: dict) -> list[str]:
    properties: list[str] = []
    if job.get("job_memory_max"):
        properties.extend(["-p", f"MemoryMax={job['job_memory_max']}"])
    if job.get("job_memory_swap_max"):
        properties.extend(["-p", f"MemorySwapMax={job['job_memory_swap_max']}"])
    if job.get("job_cpu_quota"):
        properties.extend(["-p", f"CPUQuota={job['job_cpu_quota']}"])
    if not properties:
        return command
    if shutil.which("systemd-run") is None:
        return command
    return ["systemd-run", "--user", "--scope", *properties, *command]


def run_job(job: dict) -> dict:
    start = time.perf_counter()
    command = job["command"]
    execution_command = build_execution_command(command, job)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if job["dry_run"]:
        content = "DRY RUN\nCOMMAND:\n" + " ".join(command) + "\n"
        if execution_command != command:
            content += "\nEXECUTION_COMMAND:\n" + " ".join(execution_command) + "\n"
        log_path.write_text(content, encoding="utf-8")
        return {**job, "execution_command": execution_command, "status": "planned", "returncode": 0, "runtime_seconds": 0.0}

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(command))
        if execution_command != command:
            log_handle.write("\n\nEXECUTION_COMMAND:\n")
            log_handle.write(" ".join(execution_command))
        log_handle.write("\n\nSTDOUT_AND_STDERR:\n")
        log_handle.flush()
        completed = subprocess.run(
            execution_command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    runtime = time.perf_counter() - start
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n")
        log_handle.write(f"RETURNCODE: {completed.returncode}\n")
        log_handle.write(f"RUNTIME_SECONDS: {runtime:.6f}\n")
    status = "completed" if completed.returncode == 0 else "failed"
    return {**job, "execution_command": execution_command, "status": status, "returncode": completed.returncode, "runtime_seconds": runtime}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_gradient_summary(method_output_root: Path, output_csv: Path) -> int:
    rows: list[dict] = []
    for csv_path in sorted(method_output_root.rglob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["result_csv"] = str(csv_path.relative_to(method_output_root))
                rows.append(row)
    if rows:
        write_csv(output_csv, rows)
    else:
        output_csv.write_text("", encoding="utf-8")
    return len(rows)


def result_status_row(row: dict) -> dict:
    cleaned = dict(row)
    if isinstance(cleaned.get("command"), list):
        cleaned["command"] = " ".join(cleaned["command"])
    if isinstance(cleaned.get("execution_command"), list):
        cleaned["execution_command"] = " ".join(cleaned["execution_command"])
    cleaned.pop("dry_run", None)
    return cleaned


def skipped_status_row(row: dict, config: SystemConfig, regime: str) -> dict:
    return {
        "method": row["method"],
        "system": config.output_key,
        "regime": regime,
        "data_csv": row["data_csv"],
        "graph_csv": row["graph_csv"],
        "graph_status": row["reason"],
        "status": "skipped",
        "returncode": "",
        "runtime_seconds": 0.0,
        "command": "",
        "log_path": "",
    }


def prepare_jobs(args, run_dir: Path, methods: list[str], config: SystemConfig, regime: str, process_files: list[Path]) -> tuple[list[dict], list[dict]]:
    method_output_root = run_dir / "method outputs"
    logs_root = run_dir / "logs"
    skipped: list[dict] = []
    jobs: list[dict] = []
    graph_cache: dict[Path, tuple[Path | None, str]] = {}

    for process_csv in process_files:
        graph_cache[process_csv] = resolve_graph_file(args.done_data, config, regime, process_csv)

    for method in methods:
        for index, process_csv in enumerate(process_files, start=1):
            graph_csv, graph_status = graph_cache[process_csv]
            needs_graph = method in GRAPH_METHODS
            if needs_graph and graph_csv is None:
                skipped.append(
                    {
                        "method": method,
                        "data_csv": str(process_csv),
                        "reason": graph_status,
                        "graph_csv": "",
                    }
                )
                continue

            log_name = f"{index:04d}_{method}_{safe_name(process_csv.stem, 80)}.log"
            command = build_command(
                python_executable=args.python,
                method=method,
                config=config,
                process_csv=process_csv,
                graph_csv=graph_csv if needs_graph else None,
                output_root=method_output_root,
                smoke=args.smoke,
            )
            jobs.append(
                {
                    "method": method,
                    "system": config.output_key,
                    "regime": regime,
                    "data_csv": str(process_csv),
                    "graph_csv": "" if graph_csv is None else str(graph_csv),
                    "graph_status": graph_status,
                    "command": command,
                    "log_path": str(logs_root / method / log_name),
                    "dry_run": args.dry_run,
                    "job_memory_max": args.job_memory_max,
                    "job_memory_swap_max": args.job_memory_swap_max,
                    "job_cpu_quota": args.job_cpu_quota,
                }
            )
    return jobs, skipped


def execute_jobs(
    jobs: list[dict],
    methods: list[str],
    jobs_parallel: int,
    stop_on_failure: bool,
    *,
    incremental_status_csv: Path | None = None,
    skipped_rows: list[dict] | None = None,
) -> list[dict]:
    results: list[dict] = []

    def write_incremental_status() -> None:
        if incremental_status_csv is None:
            return
        status_rows = [result_status_row(row) for row in results]
        if skipped_rows:
            status_rows.extend(skipped_rows)
        write_csv(incremental_status_csv, status_rows)

    for method in methods:
        method_jobs = [job for job in jobs if job["method"] == method]
        if not method_jobs:
            continue
        if jobs_parallel <= 1:
            for job in method_jobs:
                result = run_job(job)
                results.append(result)
                write_incremental_status()
                if stop_on_failure and result["status"] == "failed":
                    return results
        else:
            with ThreadPoolExecutor(max_workers=jobs_parallel) as pool:
                future_map = {pool.submit(run_job, job): job for job in method_jobs}
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    write_incremental_status()
                    if stop_on_failure and result["status"] == "failed":
                        return results
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized thesis gradient-estimation process for one "
            "system and one input regime using DONE DATA."
        )
    )
    parser.add_argument("system", help="System alias, for example lumped, low_dim, two_column, heat, or multi_stage.")
    parser.add_argument("regime", help="Input regime alias, for example return_to_baseline, zero, var, or var_baseline.")
    parser.add_argument("--methods", nargs="*", help="Method aliases to run. Omit or use 'all' to run all thesis methods.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs per method. Default is 1.")
    parser.add_argument("--max-files", type=int, help="Limit the number of process CSV files. Useful for smoke checks.")
    parser.add_argument("--missing-graph-only", action="store_true", help="Run only process CSVs without a resolvable graph CSV.")
    parser.add_argument("--run-id", help="Output run id. Defaults to the current timestamp.")
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to each method runner.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands without running methods.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed job.")
    parser.add_argument("--job-memory-max", help="Run each method job in a systemd user scope with this MemoryMax, for example 11G.")
    parser.add_argument("--job-memory-swap-max", help="Run each method job in a systemd user scope with this MemorySwapMax, for example 1G.")
    parser.add_argument("--job-cpu-quota", help="Run each method job in a systemd user scope with this CPUQuota, for example 90%%.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run method scripts.")
    parser.add_argument("--done-data", type=Path, default=DEFAULT_DONE_DATA, help="DONE DATA directory.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Root directory for standardized results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.done_data = args.done_data.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    system_key = normalize_system(args.system)
    regime = normalize_regime(args.regime)
    config = SYSTEMS[system_key]
    methods = parse_methods(args.methods)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / config.output_key / regime / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    process_files = discover_process_files(args.done_data, config, regime, args.max_files)
    process_file_count_before_filter = len(process_files)
    if args.missing_graph_only:
        process_files = [
            process_csv
            for process_csv in process_files
            if resolve_graph_file(args.done_data, config, regime, process_csv)[0] is None
        ]
        if not process_files:
            raise FileNotFoundError("No process-data CSV files without graph data matched the requested system/regime.")
    jobs, skipped = prepare_jobs(args, run_dir, methods, config, regime, process_files)

    job_status_csv = run_dir / "standardized_job_status.csv"
    skipped_status_rows = [skipped_status_row(row, config, regime) for row in skipped]
    if skipped_status_rows:
        write_csv(job_status_csv, skipped_status_rows)

    start = time.perf_counter()
    results = execute_jobs(
        jobs,
        methods,
        max(1, args.jobs),
        args.stop_on_failure,
        incremental_status_csv=job_status_csv,
        skipped_rows=skipped_status_rows,
    )
    total_runtime = time.perf_counter() - start

    job_rows: list[dict] = [result_status_row(row) for row in results]
    job_rows.extend(skipped_status_rows)

    write_csv(job_status_csv, job_rows)
    if skipped:
        write_csv(run_dir / "skipped_graph_jobs.csv", skipped)

    aggregate_csv = run_dir / "standardized_gradient_summary.csv"
    aggregate_rows = collect_gradient_summary(run_dir / "method outputs", aggregate_csv)

    completed = sum(1 for row in results if row["status"] == "completed")
    failed = sum(1 for row in results if row["status"] == "failed")
    manifest = {
        "run_id": run_id,
        "system": config.output_key,
        "runner_system": config.runner_system,
        "done_data_label": config.done_data_label,
        "regime": regime,
        "done_data": str(args.done_data),
        "results_dir": str(run_dir),
        "method_output_root": str(run_dir / "method outputs"),
        "process_file_count": len(process_files),
        "process_file_count_before_filter": process_file_count_before_filter,
        "missing_graph_only": args.missing_graph_only,
        "methods": methods,
        "jobs_requested": len(jobs),
        "jobs_completed": completed,
        "jobs_failed": failed,
        "jobs_skipped": len(skipped),
        "aggregate_rows": aggregate_rows,
        "total_runtime_seconds": total_runtime,
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "job_status_csv": str(job_status_csv),
        "aggregate_csv": str(aggregate_csv),
    }
    (run_dir / "standardized_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
