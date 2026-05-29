from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.gpdc import GPDC
from tigramite.lpcmci import LPCMCI


def save_links_csv(results: dict, var_names: list[str], output_path: Path) -> None:
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "lag", "p_value", "effect_size", "connection_type"])
        n_vars, _, n_lags = graph.shape
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                for lag in range(n_lags):
                    if source_idx == target_idx:
                        continue
                    connection_type = graph[source_idx, target_idx, lag]
                    if connection_type == "":
                        continue
                    writer.writerow(
                        [
                            var_names[source_idx],
                            var_names[target_idx],
                            lag,
                            float(p_matrix[source_idx, target_idx, lag]),
                            float(val_matrix[source_idx, target_idx, lag]),
                            connection_type,
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
                        ]
                    )


def build_link_assumptions(var_names: list[str], tau_max: int) -> dict[int, dict[tuple[int, int], str]]:
    input_indices = {idx for idx, name in enumerate(var_names) if name == "u"}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LPCMCI with GPDC on the Low-Dimensional Nonlinear Benchmark.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--tau-max", type=int, default=2)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--sig-samples", type=int, default=20)
    parser.add_argument("--gp-alpha", type=float, default=1e-6)
    parser.add_argument("--gp-restarts", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--save-links", type=str, default="")
    parser.add_argument("--save-all-links", type=str, default="")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=float)
    var_names = ["u", "x1", "x2"]
    dataframe = pp.DataFrame(data=data, var_names=var_names)
    cond_ind_test = GPDC(
        significance="shuffle_test",
        sig_samples=args.sig_samples,
        gp_params={"alpha": args.gp_alpha, "n_restarts_optimizer": args.gp_restarts},
    )
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=args.verbosity)
    results = lpcmci.run_lpcmci(
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        link_assumptions=build_link_assumptions(var_names, tau_max=args.tau_max),
    )
    print(f"CSV used: {csv_path}")
    print("Variables used:", var_names)
    print("LPCMCI graph shape:", results["graph"].shape)
    print(results["graph"])
    links_path = Path(args.save_links).resolve() if args.save_links else csv_path.with_name(f"{csv_path.stem}_gpdc_links.csv")
    save_links_csv(results, var_names, links_path)
    print(f"Saved links CSV to: {links_path}")
    all_links_path = Path(args.save_all_links).resolve() if args.save_all_links else csv_path.with_name(f"{csv_path.stem}_gpdc_all_links.csv")
    save_all_links_csv(results, var_names, all_links_path, pc_alpha=args.pc_alpha)
    print(f"Saved all-links CSV to: {all_links_path}")


if __name__ == "__main__":
    main()
