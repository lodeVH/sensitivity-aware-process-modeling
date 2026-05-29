import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMNS = {"time", "t", "time_hours"}
INPUT_CHANGE_THRESHOLD = 1e-9
BASIS_SUFFIXES = ("linear", "square", "cube", "tanh", "signed_log1p")


@dataclass(frozen=True)
class SystemSpec:
    name: str
    input_vars: tuple[str, ...]
    output_vars: tuple[str, ...]
    st_mode: str
    st_rows_after_change: int = 0
    lsr_window: int = 5
    neural_window: int = 8


SYSTEM_SPECS = {
    "heat": SystemSpec(
        name="heat",
        input_vars=("u1", "u2", "u3", "u4"),
        output_vars=("T1", "T2", "T3", "T4", "C", "P"),
        st_mode="rows_after_change",
        st_rows_after_change=6,
        lsr_window=5,
        neural_window=8,
    ),
    "low_dim": SystemSpec(
        name="low_dim",
        input_vars=("u",),
        output_vars=("x1", "x2"),
        st_mode="pre_step",
        st_rows_after_change=0,
        lsr_window=4,
        neural_window=4,
    ),
    "multi_stage": SystemSpec(
        name="multi_stage",
        input_vars=("L_A", "G_A", "X0_A", "Y6_A", "L_B", "G_B", "X0_B", "Y5_B"),
        output_vars=(
            "X1A", "Y1A", "X2A", "Y2A", "X3A", "Y3A", "X4A", "Y4A", "X5A", "Y5A",
            "X1B", "Y1B", "X2B", "Y2B", "X3B", "Y3B", "X4B", "Y4B",
        ),
        st_mode="rows_after_change",
        st_rows_after_change=80,
        lsr_window=4,
        neural_window=5,
    ),
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


METHOD_OUTPUT_ROOT = Path(__file__).resolve().parent / "method outputs"


def normalize_system(system: str) -> str:
    key = system.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in SYSTEM_ALIASES:
        valid = ", ".join(sorted(SYSTEM_ALIASES))
        raise KeyError(f"Unknown system '{system}'. Valid aliases include: {valid}")
    return SYSTEM_ALIASES[key]


def get_spec(system: str) -> SystemSpec:
    key = normalize_system(system)
    if key not in SYSTEM_SPECS:
        raise KeyError(f"Unknown system '{system}'. Valid systems: {sorted(SYSTEM_SPECS)}")
    return SYSTEM_SPECS[key]


def compact_source_stem(stem):
    compact = str(stem)
    replacements = [
        ("heat_system", "heat"),
        ("multi_stage", "ms"),
        ("low_dim_nonlinear", "lowdim"),
        ("varying_baseline", "var"),
        ("var_baseline", "var"),
        ("zero_return", "zero"),
        ("random_pulse_schedule", "rps"),
        ("random_pulse", "rp"),
        ("_latentH", "_H"),
        ("_duration", "_dur"),
        ("_dur", "_d"),
        ("_Tend", "_T"),
        ("_seed", "_s"),
        ("_ST_DATA", "_SS"),
    ]
    for old, new in replacements:
        compact = compact.replace(old, new)
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact.strip("_")


def resolve_path(path_text) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def read_dataframe(csv_path: Path, spec: SystemSpec) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    first_col = str(df.columns[0]).strip().lower()
    if first_col in TIME_COLUMNS:
        df = df.drop(columns=[df.columns[0]])
    missing = [name for name in (*spec.input_vars, *spec.output_vars) if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {spec.name}: {missing}")
    return df


def detect_input_changes(df: pd.DataFrame, input_vars: tuple[str, ...]) -> list[int]:
    indices = []
    for row_index in range(len(df) - 1):
        for input_name in input_vars:
            if abs(float(df.iloc[row_index + 1][input_name]) - float(df.iloc[row_index][input_name])) > INPUT_CHANGE_THRESHOLD:
                indices.append(row_index)
                break
    return indices


def ensure_ss_dataframe(csv_path: Path, spec: SystemSpec, output_dir: Path | None = None) -> tuple[pd.DataFrame, Path | None, dict]:
    df = read_dataframe(csv_path, spec)
    if "_ST_DATA" in csv_path.stem or "_SS_DATA" in csv_path.stem:
        return df, csv_path, {"ss_mode": "already_ss", "selected_indices": None}

    change_indices = detect_input_changes(df, spec.input_vars)
    selected_indices = []
    if spec.st_mode == "pre_step":
        selected_indices = change_indices
    elif spec.st_mode == "rows_after_change":
        for index in change_indices:
            target = index + spec.st_rows_after_change
            if target < len(df):
                selected_indices.append(target)
    else:
        raise ValueError(f"Unsupported SS extraction mode: {spec.st_mode}")

    if not selected_indices:
        raise ValueError(f"No steady-state-style rows could be extracted from {csv_path}")

    ss_df = df.iloc[selected_indices].copy()
    ss_df.insert(0, "source_row_index", selected_indices)
    ss_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ss_path = output_dir / f"{csv_path.stem}_ST_DATA.csv"
        ss_df.to_csv(ss_path, index=False)

    return ss_df, ss_path, {
        "ss_mode": spec.st_mode,
        "rows_after_change": spec.st_rows_after_change,
        "selected_indices": selected_indices,
    }


def method_output_dir(method_id: str, system: str, source_csv: Path, output_root: Path | None = None) -> Path:
    root = output_root or METHOD_OUTPUT_ROOT
    out_dir = root / method_id / get_spec(system).name / compact_source_stem(source_csv.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_summary_outputs(
    *,
    method_id: str,
    system: str,
    source_csv: Path,
    output_dir: Path,
    records: list[dict],
    metadata: dict,
    local_records: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{method_id}_{get_spec(system).name}_{compact_source_stem(source_csv.stem)}"
    csv_path = output_dir / f"{base}.csv"
    json_path = output_dir / f"{base}.json"
    log_path = output_dir / f"{base}.log"

    preferred = [
        "method",
        "variant",
        "system",
        "output",
        "input",
        "value",
        "local_gradient_std",
        "sample_count",
        "allowed_by_graph",
        "controls",
        "source_csv",
        "graph_csv",
        "runtime_seconds",
    ]
    fieldnames = list(preferred)
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "metadata": metadata,
        "summary": records,
    }
    if local_records is not None:
        payload["local_gradients"] = local_records
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log_lines = [
        f"method={method_id}",
        f"system={system}",
        f"source_csv={source_csv}",
        f"csv={csv_path}",
        f"json={json_path}",
        f"runtime_seconds={metadata.get('runtime_seconds')}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return csv_path, json_path, log_path


def load_graph_edges(graph_csv: Path | None) -> list[tuple[str, str]]:
    if graph_csv is None:
        return []
    rows = pd.read_csv(graph_csv)
    edges = []
    for _, row in rows.iterrows():
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        ctype = str(row.get("connection_type", "-->")).strip()
        if not source or not target:
            continue
        if ctype in {"-->", "o->", ""}:
            edges.append((source, target))
        elif ctype in {"<--", "<-o"}:
            edges.append((target, source))
        elif ctype == "<->":
            edges.append((source, target))
            edges.append((target, source))
    return edges


def graph_reachable_inputs(graph_csv: Path | None, spec: SystemSpec) -> dict[str, set[str]]:
    edges = load_graph_edges(graph_csv)
    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)

    allowed = {output: set() for output in spec.output_vars}
    for input_name in spec.input_vars:
        stack = [input_name]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in spec.output_vars:
                allowed[node].add(input_name)
            for nxt in adjacency.get(node, set()):
                if nxt not in seen:
                    stack.append(nxt)
    return allowed


def graph_ancestor_controls(graph_csv: Path | None, spec: SystemSpec) -> dict[str, set[str]]:
    edges = load_graph_edges(graph_csv)
    reverse: dict[str, set[str]] = {}
    for source, target in edges:
        reverse.setdefault(target, set()).add(source)

    controls = {output: set() for output in spec.output_vars}
    for output in spec.output_vars:
        stack = [output]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for parent in reverse.get(node, set()):
                if parent not in seen:
                    controls[output].add(parent)
                    stack.append(parent)
    return controls


def apply_graph_zero(records: list[dict], graph_csv: Path | None, spec: SystemSpec) -> list[dict]:
    allowed = graph_reachable_inputs(graph_csv, spec)
    masked = []
    for record in records:
        output = record["output"]
        input_name = record["input"]
        keep = input_name in allowed.get(output, set())
        new_record = dict(record)
        new_record["raw_value"] = record["value"]
        new_record["allowed_by_graph"] = keep
        new_record["value"] = record["value"] if keep else 0.0
        masked.append(new_record)
    return masked


def signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def basis_matrix(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=float).reshape(-1)
    columns = [
        values,
        values ** 2,
        values ** 3,
        np.tanh(values),
        signed_log1p(values),
    ]
    return np.column_stack(columns), list(BASIS_SUFFIXES)


def basis_derivative(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=float).reshape(-1)
    columns = [
        np.ones_like(values),
        2.0 * values,
        3.0 * values ** 2,
        1.0 - np.tanh(values) ** 2,
        1.0 / (1.0 + np.abs(values)),
    ]
    return np.column_stack(columns), list(BASIS_SUFFIXES)


def build_feature_basis(df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    matrices = []
    names = []
    for name in feature_names:
        matrix, suffixes = basis_matrix(df[name].to_numpy(dtype=float))
        matrices.append(matrix)
        names.extend([f"{name}_{suffix}" for suffix in suffixes])
    if not matrices:
        return np.zeros((len(df), 0)), []
    return np.column_stack(matrices), names


def centered_slope(y: np.ndarray, d: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    d = np.asarray(d, dtype=float).reshape(-1)
    y = y - np.mean(y)
    d = d - np.mean(d)
    denom = float(np.dot(d, d))
    if denom <= 1e-14:
        return 0.0
    return float(np.dot(d, y) / denom)


def residualize_with_controls(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if controls.size == 0:
        return values - np.mean(values)
    X = np.column_stack([np.ones(len(controls)), controls])
    coef = np.linalg.pinv(X) @ values
    return values - X @ coef


def infer_dt(df: pd.DataFrame) -> float:
    for col in df.columns:
        if str(col).strip().lower() in TIME_COLUMNS:
            t = df[col].to_numpy(dtype=float)
            if len(t) > 1:
                diffs = np.diff(t)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if len(diffs):
                    return float(np.median(diffs))
    return 1.0


class RuntimeTimer:
    def __init__(self):
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.start)


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--system",
        required=True,
        help="Benchmark alias, for example lumped, low_dim, two_column, heat, or multi_stage.",
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--graph-csv")
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true", help="Use reduced settings for quick validation.")


def resolve_common_args(args):
    spec = get_spec(args.system)
    data_csv = resolve_path(args.data_csv)
    graph_csv = resolve_path(args.graph_csv) if getattr(args, "graph_csv", None) else None
    output_root = Path(args.output_root).expanduser().resolve() if getattr(args, "output_root", None) else METHOD_OUTPUT_ROOT
    return spec, data_csv, graph_csv, output_root


def make_record(method_id, variant, spec, output, input_name, value, std, sample_count, source_csv, graph_csv, runtime, **extra):
    if std in (None, ""):
        std_value = ""
    else:
        std_value = float(std)
    record = {
        "method": method_id,
        "variant": variant,
        "system": spec.name,
        "output": output,
        "input": input_name,
        "value": float(value),
        "local_gradient_std": std_value,
        "sample_count": int(sample_count),
        "source_csv": source_csv.name,
        "graph_csv": "" if graph_csv is None else graph_csv.name,
        "runtime_seconds": float(runtime),
    }
    record.update(extra)
    return record


def smoke_slice(df: pd.DataFrame, max_rows: int = 80) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.iloc[:max_rows].copy()
