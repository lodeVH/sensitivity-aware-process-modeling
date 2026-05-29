from __future__ import annotations

from pathlib import Path

import numpy as np

from data_gen_non_overlapping_pulse_return_to_base import (
    DEFAULT_INPUT_TARGET_RANGES,
    GeneratorConfig,
)
from multi_stage_noise_h_utils import (
    H_MODES,
    NOISE_MODES,
    DEFAULT_INPUT_TARGET_SEED,
    DEFAULT_PULSE_SECONDS,
    DEFAULT_SAMPLE_TIME_SECONDS,
    DEFAULT_INITIAL_BASELINE_SECONDS,
    DEFAULT_FINAL_BASELINE_SECONDS,
    DEFAULT_TOTAL_SIM_SECONDS,
    DEFAULT_STATE_NOISE_STD,
    apply_measurement_noise,
    build_var_baseline_schedule,
    experiment_tag,
    get_h_info,
    get_noise_info,
    hidden_rows,
    make_hidden_profiles,
    metadata_rows,
    simulate_with_hidden_profiles,
    write_case,
)
from two_column_simulator import TwoColumnExtractionModel


def main() -> None:
    config = GeneratorConfig(
        pulse_seconds=DEFAULT_PULSE_SECONDS,
        sample_time_seconds=DEFAULT_SAMPLE_TIME_SECONDS,
        initial_baseline_seconds=DEFAULT_INITIAL_BASELINE_SECONDS,
        final_baseline_seconds=DEFAULT_FINAL_BASELINE_SECONDS,
        total_sim_seconds=DEFAULT_TOTAL_SIM_SECONDS,
        repeat_sequence=True,
        state_noise_std=DEFAULT_STATE_NOISE_STD,
        input_target_ranges=dict(DEFAULT_INPUT_TARGET_RANGES),
        input_target_seed=DEFAULT_INPUT_TARGET_SEED,
    )

    model = TwoColumnExtractionModel(state_noise_std=DEFAULT_STATE_NOISE_STD)
    input_schedule, events = build_var_baseline_schedule(model, config)
    dt_hours = config.sample_time_seconds / 3600.0
    t_seconds = np.arange(input_schedule.shape[0] + 1, dtype=float) * config.sample_time_seconds
    tag = experiment_tag(config, prefix="varbase_")
    out_dir = Path(__file__).resolve().parents[3] / "Process data" / "two column" / "big G &L all H's and noise" / "var baseline"

    for h_mode in H_MODES:
        h_mode_tag, h_desc = get_h_info(h_mode)
        h1_vals, h2_vals, h3_vals, s1_vals, c1_vals, sc_rest, _, _ = make_hidden_profiles(
            model,
            t_seconds,
            h_mode=h_mode,
            seed=DEFAULT_INPUT_TARGET_SEED + h_mode,
        )
        s2_vals = sc_rest[:, 0]
        c2_vals = sc_rest[:, 1]
        s3_vals = sc_rest[:, 2]
        c3_vals = sc_rest[:, 3]

        base_rows = simulate_with_hidden_profiles(
            model,
            input_schedule,
            (h1_vals, h2_vals, h3_vals),
            (s1_vals, c1_vals, s2_vals, c2_vals, s3_vals, c3_vals),
            dt_hours=dt_hours,
        )
        hidden = hidden_rows(t_seconds, h1_vals, h2_vals, h3_vals)

        for noise_mode in NOISE_MODES:
            noise_mode_tag, noise_desc = get_noise_info(noise_mode)
            rows = apply_measurement_noise(
                base_rows,
                measurement_columns=model.physical_state_names(),
                dt_seconds=config.sample_time_seconds,
                noise_mode=noise_mode,
                seed=DEFAULT_INPUT_TARGET_SEED,
            )
            base_name = f"non_overlapping_pulse_var_baseline_{tag}_{h_mode_tag}_{noise_mode_tag}"
            write_case(
                rows,
                events,
                hidden,
                metadata_rows(
                    config,
                    h_mode=h_mode,
                    h_tag=h_mode_tag,
                    h_desc=h_desc,
                    noise_mode=noise_mode,
                    noise_tag=noise_mode_tag,
                    noise_desc=noise_desc,
                    schedule_mode="var_baseline",
                ),
                output_path=out_dir / f"{base_name}.csv",
                schedule_path=out_dir / f"{base_name}_schedule.csv",
                hidden_path=out_dir / f"{base_name}_hidden.csv",
                metadata_path=out_dir / f"{base_name}_meta.csv",
            )
            print(f"Wrote {base_name}.csv")


if __name__ == "__main__":
    main()
