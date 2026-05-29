from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from two_column_simulator import (
    TwoColumnExtractionModel,
    estimate_nominal_physical_rk4_dt_limit,
    simulate,
    write_csv,
)


DEFAULT_PULSE_SECONDS = 1e4
DEFAULT_SAMPLE_TIME_SECONDS = 1e2
DEFAULT_INITIAL_BASELINE_SECONDS = 1e4
DEFAULT_FINAL_BASELINE_SECONDS = 1e4
DEFAULT_TOTAL_SIM_SECONDS = 1e7
DEFAULT_REPEAT_SEQUENCE = True
DEFAULT_STATE_NOISE_STD = 0.001
DEFAULT_INPUT_TARGET_RANGES = {
    "L_A": (6.4, 9.6),
    "G_A": (40.0, 60.0),
    "X0_A": (0.50, 0.70),
    "Y6_A": (0.00, 0.15),
    "L_B": (6.0, 9.0),
    "G_B": (40.0, 60.0),
    "X0_B": (0.50, 0.70),
    "Y5_B": (0.00, 0.15),
}
DEFAULT_INPUT_TARGET_SEED = 7


@dataclass(frozen=True)
class GeneratorConfig:
    pulse_seconds: float = DEFAULT_PULSE_SECONDS
    sample_time_seconds: float = DEFAULT_SAMPLE_TIME_SECONDS
    initial_baseline_seconds: float = DEFAULT_INITIAL_BASELINE_SECONDS
    final_baseline_seconds: float = DEFAULT_FINAL_BASELINE_SECONDS
    total_sim_seconds: float = DEFAULT_TOTAL_SIM_SECONDS
    repeat_sequence: bool = DEFAULT_REPEAT_SEQUENCE
    state_noise_std: float = DEFAULT_STATE_NOISE_STD
    input_target_ranges: dict[str, tuple[float, float]] | None = None
    input_target_seed: int = DEFAULT_INPUT_TARGET_SEED

    def resolved_input_target_ranges(self) -> dict[str, tuple[float, float]]:
        if self.input_target_ranges is None:
            return dict(DEFAULT_INPUT_TARGET_RANGES)
        return dict(self.input_target_ranges)


def default_config() -> GeneratorConfig:
    return GeneratorConfig(input_target_ranges=dict(DEFAULT_INPUT_TARGET_RANGES))


def format_token(value: float) -> str:
    if value != 0:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        if np.isclose(mantissa, round(mantissa), atol=1e-12):
            rounded_mantissa = int(round(mantissa))
            if rounded_mantissa in (1, 2, 5):
                return f"{rounded_mantissa}e{exponent}"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def experiment_tag(config: GeneratorConfig) -> str:
    return (
        f"p{format_token(config.pulse_seconds)}"
        f"_rand"
        f"_dt{format_token(config.sample_time_seconds)}"
        f"_ib{format_token(config.initial_baseline_seconds)}"
        f"_fb{format_token(config.final_baseline_seconds)}"
        f"_tot{format_token(config.total_sim_seconds)}"
        f"_seed{config.input_target_seed}"
        f"_ns{format_token(config.state_noise_std) if np.isscalar(config.state_noise_std) else 'vec'}"
    )


def scaled_range(baseline_value: float, min_value: float, max_value: float, scale: float) -> tuple[float, float]:
    return (
        float(baseline_value + scale * (min_value - baseline_value)),
        float(baseline_value + scale * (max_value - baseline_value)),
    )


def apply_range_scales(
    ranges: dict[str, tuple[float, float]],
    model: TwoColumnExtractionModel,
    *,
    l_scale: float = 1.0,
    g_scale: float = 1.0,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    per_input_scales: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    baseline = dict(zip(model.input_names(), model.default_input(), strict=True))
    scaled: dict[str, tuple[float, float]] = {}
    resolved_per_input_scales = {} if per_input_scales is None else {
        key: float(value) for key, value in per_input_scales.items()
    }
    for name, (min_value, max_value) in ranges.items():
        if name in resolved_per_input_scales:
            factor = resolved_per_input_scales[name]
        else:
            if name.startswith("L_"):
                factor = l_scale
            elif name.startswith("G_"):
                factor = g_scale
            elif name.startswith("X0_"):
                factor = x_scale
            elif name.startswith("Y"):
                factor = y_scale
            else:
                factor = 1.0
        scaled[name] = scaled_range(baseline[name], min_value, max_value, factor)
    return scaled


def validate_input_ranges(
    model: TwoColumnExtractionModel,
    input_ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    ranges = dict(input_ranges)
    missing = [name for name in model.input_names() if name not in ranges]
    if missing:
        raise KeyError(f"Missing input ranges for: {', '.join(missing)}")

    validated: dict[str, tuple[float, float]] = {}
    for name in model.input_names():
        min_value, max_value = ranges[name]
        min_value = float(min_value)
        max_value = float(max_value)
        if min_value > max_value:
            raise ValueError(f"Invalid range for {name}: min {min_value} is greater than max {max_value}.")
        validated[name] = (min_value, max_value)
    return validated


def build_pulse_schedule(
    model: TwoColumnExtractionModel,
    config: GeneratorConfig,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    baseline_u = model.default_input()
    input_names = model.input_names()
    input_ranges = validate_input_ranges(model, config.resolved_input_target_ranges())
    rng = np.random.default_rng(config.input_target_seed)

    pulse_steps = max(1, int(round(config.pulse_seconds / config.sample_time_seconds)))
    wait_steps = max(1, int(round(config.pulse_seconds / config.sample_time_seconds)))
    initial_steps = max(1, int(round(config.initial_baseline_seconds / config.sample_time_seconds)))
    final_steps = max(1, int(round(config.final_baseline_seconds / config.sample_time_seconds)))
    total_steps_limit = max(1, int(round(config.total_sim_seconds / config.sample_time_seconds)))

    segments: list[np.ndarray] = []
    events: list[dict[str, float | str]] = []
    current_step = 0
    cycle_index = 0

    def append_block(
        block_u: np.ndarray,
        block_steps: int,
        phase: str,
        input_name: str,
        cycle: int,
        *,
        target_value: float | None = None,
        baseline_value: float | None = None,
    ) -> None:
        nonlocal current_step
        if block_steps <= 0:
            return
        segments.append(np.tile(block_u, (block_steps, 1)))
        delta_value = (
            float(target_value - baseline_value)
            if target_value is not None and baseline_value is not None
            else 0.0
        )
        events.append(
            {
                "cycle": cycle,
                "phase": phase,
                "input": input_name,
                "start_time_seconds": current_step * config.sample_time_seconds,
                "end_time_seconds": (current_step + block_steps) * config.sample_time_seconds,
                "baseline_value": float(baseline_value) if baseline_value is not None else "",
                "target_value": float(target_value) if target_value is not None else "",
                "delta_value": delta_value if target_value is not None else "",
            }
        )
        current_step += block_steps

    initial_block_steps = min(initial_steps, total_steps_limit - current_step)
    append_block(baseline_u, initial_block_steps, "initial_baseline", "none", cycle_index)
    cycle_index += 1

    while current_step < total_steps_limit:
        for idx, input_name in enumerate(input_names):
            if current_step >= total_steps_limit:
                break

            pulse_u = baseline_u.copy()
            min_value, max_value = input_ranges[input_name]
            pulse_value = rng.uniform(min_value, max_value)
            pulse_u[idx] = pulse_value

            pulse_block_steps = min(pulse_steps, total_steps_limit - current_step)
            append_block(
                pulse_u,
                pulse_block_steps,
                "pulse",
                input_name,
                cycle_index,
                target_value=pulse_value,
                baseline_value=float(baseline_u[idx]),
            )
            if current_step >= total_steps_limit:
                break

            is_last_input = idx == len(input_names) - 1
            baseline_steps = final_steps if is_last_input else wait_steps
            baseline_block_steps = min(baseline_steps, total_steps_limit - current_step)
            phase = "final_baseline" if is_last_input else "baseline_wait"
            append_block(baseline_u, baseline_block_steps, phase, "none", cycle_index)

        if not config.repeat_sequence:
            break
        cycle_index += 1

    return np.vstack(segments), events


def write_schedule_events(events: list[dict[str, float | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)


def convert_time_columns(
    rows: list[dict[str, float]],
    *,
    seconds_per_hour: float = 3600.0,
) -> list[dict[str, float]]:
    converted: list[dict[str, float]] = []
    for row in rows:
        updated = dict(row)
        updated["time_hours"] = updated["time"]
        updated["time"] = updated["time"] * seconds_per_hour
        converted.append(updated)
    return converted


def generate_dataset(
    config: GeneratorConfig,
    *,
    output_dir: Path | None = None,
    model: TwoColumnExtractionModel | None = None,
    simulate_seed: int = 7,
) -> tuple[Path, Path, list[dict[str, float]]]:
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = output_dir if output_dir is not None else root_dir / "data"
    tag = experiment_tag(config)
    output_path = data_dir / f"non_overlapping_pulse_return_to_base_{tag}.csv"
    schedule_path = data_dir / f"non_overlapping_pulse_return_to_base_schedule_{tag}.csv"

    active_model = model if model is not None else TwoColumnExtractionModel(state_noise_std=config.state_noise_std)
    x0 = active_model.default_initial_state()
    input_schedule, events = build_pulse_schedule(active_model, config)
    dt_hours = config.sample_time_seconds / 3600.0

    rows = simulate(
        active_model,
        x0,
        input_schedule,
        dt=dt_hours,
        steps=input_schedule.shape[0],
        seed=simulate_seed,
    )
    rows = convert_time_columns(rows)
    write_csv(rows, output_path)
    write_schedule_events(events, schedule_path)
    return output_path, schedule_path, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate non-overlapping random pulse data for the two-column model."
    )
    parser.add_argument("--pulse-seconds", type=float, default=DEFAULT_PULSE_SECONDS)
    parser.add_argument("--sample-time-seconds", type=float, default=DEFAULT_SAMPLE_TIME_SECONDS)
    parser.add_argument("--initial-baseline-seconds", type=float, default=DEFAULT_INITIAL_BASELINE_SECONDS)
    parser.add_argument("--final-baseline-seconds", type=float, default=DEFAULT_FINAL_BASELINE_SECONDS)
    parser.add_argument("--total-sim-seconds", type=float, default=DEFAULT_TOTAL_SIM_SECONDS)
    parser.add_argument("--repeat-sequence", type=lambda x: x.strip().lower() in {"1", "true", "yes", "on"}, default=DEFAULT_REPEAT_SEQUENCE)
    parser.add_argument("--state-noise-std", type=float, default=DEFAULT_STATE_NOISE_STD)
    parser.add_argument("--input-target-seed", type=int, default=DEFAULT_INPUT_TARGET_SEED)
    parser.add_argument("--simulate-seed", type=int, default=7)
    parser.add_argument("--l-scale", type=float, default=1.0, help="Scale L_A/L_B deviations from baseline.")
    parser.add_argument("--g-scale", type=float, default=1.0, help="Scale G_A/G_B deviations from baseline.")
    parser.add_argument("--x-scale", type=float, default=1.0, help="Scale X0_A/X0_B deviations from baseline.")
    parser.add_argument("--y-scale", type=float, default=1.0, help="Scale Y6_A/Y5_B deviations from baseline.")
    parser.add_argument("--la-scale", type=float, default=None, help="Optional override for L_A deviation scale.")
    parser.add_argument("--ga-scale", type=float, default=None, help="Optional override for G_A deviation scale.")
    parser.add_argument("--x0a-scale", type=float, default=None, help="Optional override for X0_A deviation scale.")
    parser.add_argument("--y6a-scale", type=float, default=None, help="Optional override for Y6_A deviation scale.")
    parser.add_argument("--lb-scale", type=float, default=None, help="Optional override for L_B deviation scale.")
    parser.add_argument("--gb-scale", type=float, default=None, help="Optional override for G_B deviation scale.")
    parser.add_argument("--x0b-scale", type=float, default=None, help="Optional override for X0_B deviation scale.")
    parser.add_argument("--y5b-scale", type=float, default=None, help="Optional override for Y5_B deviation scale.")
    parser.add_argument("--output-dir", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = TwoColumnExtractionModel(state_noise_std=args.state_noise_std)
    per_input_scales = {
        name: value
        for name, value in {
            "L_A": args.la_scale,
            "G_A": args.ga_scale,
            "X0_A": args.x0a_scale,
            "Y6_A": args.y6a_scale,
            "L_B": args.lb_scale,
            "G_B": args.gb_scale,
            "X0_B": args.x0b_scale,
            "Y5_B": args.y5b_scale,
        }.items()
        if value is not None
    }
    scaled_ranges = apply_range_scales(
        DEFAULT_INPUT_TARGET_RANGES,
        model,
        l_scale=args.l_scale,
        g_scale=args.g_scale,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        per_input_scales=per_input_scales,
    )
    config = GeneratorConfig(
        pulse_seconds=args.pulse_seconds,
        sample_time_seconds=args.sample_time_seconds,
        initial_baseline_seconds=args.initial_baseline_seconds,
        final_baseline_seconds=args.final_baseline_seconds,
        total_sim_seconds=args.total_sim_seconds,
        repeat_sequence=args.repeat_sequence,
        state_noise_std=args.state_noise_std,
        input_target_ranges=scaled_ranges,
        input_target_seed=args.input_target_seed,
    )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    output_path, schedule_path, _rows = generate_dataset(
        config,
        output_dir=output_dir,
        model=model,
        simulate_seed=args.simulate_seed,
    )

    dt_info = estimate_nominal_physical_rk4_dt_limit(model)
    print(f"Wrote data CSV to {output_path}")
    print(f"Wrote pulse schedule to {schedule_path}")
    print("Configured pulse ranges:")
    for name, (min_value, max_value) in validate_input_ranges(model, config.resolved_input_target_ranges()).items():
        print(f"  {name}: [{min_value}, {max_value}]")
    print(f"Configured total simulation time = {config.total_sim_seconds} s")
    print(f"Configured sample time = {config.sample_time_seconds} s")
    print(f"Configured state noise std = {config.state_noise_std}")
    print(f"Nominal RK4 linear stability limit ~= {dt_info['dt_limit_hours'] * 3600.0:.2f} s")
    print(f"Conservative recommended dt ~= {dt_info['recommended_dt_hours'] * 3600.0:.2f} s")
    print(f"Fastest nominal modal tau ~= {dt_info['fastest_tau_hours'] * 3600.0:.2f} s")
    if config.sample_time_seconds / 3600.0 > dt_info["dt_limit_hours"]:
        print("WARNING: configured sample time is above the nominal RK4 stability limit.")
    elif config.sample_time_seconds / 3600.0 > dt_info["recommended_dt_hours"]:
        print("WARNING: configured sample time is below the formal RK4 stability limit but above the conservative recommendation.")


if __name__ == "__main__":
    main()
