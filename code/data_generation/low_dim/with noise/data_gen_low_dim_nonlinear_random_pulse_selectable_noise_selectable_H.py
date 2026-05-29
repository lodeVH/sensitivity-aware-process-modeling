from __future__ import annotations

from pathlib import Path

from low_dim_nonlinear_simulator import LowDimNonlinearBenchmarkModel, write_csv
from low_dim_noise_h_utils import (
    H_MODES,
    INPUT_TARGET_SEED,
    NOISE_MODES,
    SAMPLE_TIME_HOURS,
    STATE_NOISE_STD,
    TOTAL_SIM_HOURS,
    INPUT_TARGET_RANGE,
    PULSE_HOURS,
    build_metadata_rows,
    build_zero_return_schedule,
    format_token,
    hidden_profile_rows,
    h_tag,
    make_h_profile,
    noise_tag,
    simulate_with_hidden_profile,
    apply_measurement_noise,
    write_rows,
)


def experiment_tag() -> str:
    return (
        "latentH5"
        f"p{format_token(PULSE_HOURS)}"
        f"_dt{format_token(SAMPLE_TIME_HOURS)}"
        f"_tot{format_token(TOTAL_SIM_HOURS)}"
        f"_u{format_token(INPUT_TARGET_RANGE[0])}to{format_token(INPUT_TARGET_RANGE[1])}"
        f"_seed{INPUT_TARGET_SEED}"
        f"_ns{format_token(STATE_NOISE_STD)}"
    )


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[3]
    data_dir = repo_dir / "Process data" / "low dim" / "with noise and 5'hs" / "back to zero"
    tag = experiment_tag()

    model = LowDimNonlinearBenchmarkModel(state_noise_std=STATE_NOISE_STD)
    input_schedule, events = build_zero_return_schedule(seed=INPUT_TARGET_SEED)
    t_uniform = [k * SAMPLE_TIME_HOURS for k in range(input_schedule.shape[0] + 1)]

    for h_mode in H_MODES:
        h_vals, h_mode_tag, h_desc = make_h_profile(t_uniform, h_mode=h_mode, seed=INPUT_TARGET_SEED + h_mode)
        base_rows = simulate_with_hidden_profile(model, input_schedule, h_vals, dt=SAMPLE_TIME_HOURS)
        hidden_rows = hidden_profile_rows(t_uniform, h_vals)

        for noise_mode in NOISE_MODES:
            noise_mode_tag, noise_desc = noise_tag(noise_mode)
            output_path = data_dir / f"low_dim_nonlinear_random_pulse_{tag}_{h_mode_tag}_{noise_mode_tag}.csv"
            schedule_path = data_dir / f"low_dim_nonlinear_random_pulse_schedule_{tag}_{h_mode_tag}_{noise_mode_tag}.csv"
            metadata_path = data_dir / f"low_dim_nonlinear_random_pulse_{tag}_{h_mode_tag}_{noise_mode_tag}_meta.csv"
            hidden_path = data_dir / f"low_dim_nonlinear_random_pulse_{tag}_{h_mode_tag}_{noise_mode_tag}_hidden.csv"

            noisy_rows = apply_measurement_noise(base_rows, noise_mode=noise_mode, dt=SAMPLE_TIME_HOURS, seed=INPUT_TARGET_SEED)
            write_csv(noisy_rows, output_path)
            write_rows(events, schedule_path)
            write_rows(build_metadata_rows(h_mode, h_mode_tag, h_desc, noise_mode, noise_mode_tag, noise_desc), metadata_path)
            write_rows(hidden_rows, hidden_path)

            print(f"Wrote {len(noisy_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
