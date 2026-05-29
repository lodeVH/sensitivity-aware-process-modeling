from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from two_column_simulator import TwoColumnExtractionModel, write_csv
from data_gen_non_overlapping_pulse_return_to_base import (
    DEFAULT_FINAL_BASELINE_SECONDS,
    DEFAULT_INITIAL_BASELINE_SECONDS,
    DEFAULT_INPUT_TARGET_RANGES,
    DEFAULT_INPUT_TARGET_SEED,
    DEFAULT_PULSE_SECONDS,
    DEFAULT_SAMPLE_TIME_SECONDS,
    DEFAULT_TOTAL_SIM_SECONDS,
    GeneratorConfig,
    format_token,
    validate_input_ranges,
)


DEFAULT_STATE_NOISE_STD = 0.0
NOISE_MODES = (1, 2, 3)
H_MODES = (1, 2, 3, 4, 5)

DEFAULT_MEASUREMENT_GAUSSIAN_STD = 5e-3
DEFAULT_MEASUREMENT_QUANT_STEP = 0.002
DEFAULT_MEASUREMENT_RANDOM_WALK_SIGMA = 1.6e-7

DEFAULT_PRBS_HOLD_MIN_SECONDS = 1e4
DEFAULT_PRBS_HOLD_MAX_SECONDS = 5e4


def get_noise_info(noise_mode: int) -> tuple[str, str]:
    if noise_mode == 1:
        return "N1", "Gaussian measurement noise"
    if noise_mode == 2:
        return "N2", "quantized Gaussian measurement noise"
    if noise_mode == 3:
        return "N3", "random-walk measurement noise"
    raise ValueError(f"Unsupported noise_mode={noise_mode}. Choose 1, 2, or 3.")


def get_h_info(h_mode: int) -> tuple[str, str]:
    if h_mode == 1:
        return "Hsin", "sinusoidal H"
    if h_mode == 2:
        return "Hdec", "decreasing H"
    if h_mode == 3:
        return "Hbrn", "Brownian H with mean reversion"
    if h_mode == 4:
        return "Hstep", "midpoint step H"
    if h_mode == 5:
        return "Hprbs", "PRBS-like H"
    raise ValueError(f"Unsupported h_mode={h_mode}. Choose 1, 2, 3, 4, or 5.")


def rand_between(rng: np.random.Generator, a: float, b: float) -> float:
    return float(a + (b - a) * rng.random())


def clip_scalar(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def experiment_tag(config: GeneratorConfig, *, prefix: str) -> str:
    return (
        f"{prefix}"
        f"p{format_token(config.pulse_seconds)}"
        f"_rand"
        f"_dt{format_token(config.sample_time_seconds)}"
        f"_ib{format_token(config.initial_baseline_seconds)}"
        f"_fb{format_token(config.final_baseline_seconds)}"
        f"_tot{format_token(config.total_sim_seconds)}"
        f"_seed{config.input_target_seed}"
        f"_ns{format_token(config.state_noise_std) if np.isscalar(config.state_noise_std) else 'vec'}"
    )


def write_rows(rows: list[dict[str, float | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_return_to_zero_schedule(
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

    append_block(
        baseline_u,
        min(initial_steps, total_steps_limit - current_step),
        "initial_baseline",
        "none",
        cycle_index,
    )
    cycle_index += 1

    while current_step < total_steps_limit:
        for idx, input_name in enumerate(input_names):
            if current_step >= total_steps_limit:
                break

            pulse_u = baseline_u.copy()
            min_value, max_value = input_ranges[input_name]
            pulse_value = float(rng.uniform(min_value, max_value))
            pulse_u[idx] = pulse_value

            append_block(
                pulse_u,
                min(pulse_steps, total_steps_limit - current_step),
                "pulse",
                input_name,
                cycle_index,
                target_value=pulse_value,
                baseline_value=float(baseline_u[idx]),
            )
            if current_step >= total_steps_limit:
                break

            is_last_input = idx == len(input_names) - 1
            phase = "final_baseline" if is_last_input else "baseline_wait"
            baseline_steps = final_steps if is_last_input else wait_steps
            append_block(
                baseline_u,
                min(baseline_steps, total_steps_limit - current_step),
                phase,
                "none",
                cycle_index,
            )

        cycle_index += 1

    return np.vstack(segments), events


def build_var_baseline_schedule(
    model: TwoColumnExtractionModel,
    config: GeneratorConfig,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    baselines = model.default_input().astype(float).copy()
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
        baseline_before: float | None = None,
        target_value: float | None = None,
    ) -> None:
        nonlocal current_step
        if block_steps <= 0:
            return
        delta_value = (
            float(target_value - baseline_before)
            if target_value is not None and baseline_before is not None
            else 0.0
        )
        segments.append(np.tile(block_u, (block_steps, 1)))
        events.append(
            {
                "cycle": cycle,
                "phase": phase,
                "input": input_name,
                "start_time_seconds": current_step * config.sample_time_seconds,
                "end_time_seconds": (current_step + block_steps) * config.sample_time_seconds,
                "baseline_value": float(baseline_before) if baseline_before is not None else "",
                "target_value": float(target_value) if target_value is not None else "",
                "delta_value": delta_value if target_value is not None else "",
            }
        )
        current_step += block_steps

    append_block(
        baselines.copy(),
        min(initial_steps, total_steps_limit - current_step),
        "initial_baseline",
        "none",
        cycle_index,
    )
    cycle_index += 1

    while current_step < total_steps_limit:
        for idx, input_name in enumerate(input_names):
            if current_step >= total_steps_limit:
                break

            pulse_u = baselines.copy()
            min_value, max_value = input_ranges[input_name]
            baseline_before = float(baselines[idx])
            pulse_value = float(rng.uniform(min_value, max_value))
            pulse_u[idx] = pulse_value

            append_block(
                pulse_u,
                min(pulse_steps, total_steps_limit - current_step),
                "pulse",
                input_name,
                cycle_index,
                baseline_before=baseline_before,
                target_value=pulse_value,
            )
            baselines[idx] = pulse_value
            if current_step >= total_steps_limit:
                break

            is_last_input = idx == len(input_names) - 1
            phase = "final_baseline" if is_last_input else "baseline_hold"
            baseline_steps = final_steps if is_last_input else wait_steps
            append_block(
                baselines.copy(),
                min(baseline_steps, total_steps_limit - current_step),
                phase,
                "none",
                cycle_index,
            )

        cycle_index += 1

    return np.vstack(segments), events


def make_hidden_profiles(
    model: TwoColumnExtractionModel,
    t_seconds: np.ndarray,
    *,
    h_mode: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    h_tag, h_desc = get_h_info(h_mode)
    rng = np.random.default_rng(seed)
    t = np.asarray(t_seconds, dtype=float)
    total_seconds = float(t[-1]) if t.size else DEFAULT_TOTAL_SIM_SECONDS

    amps = np.array([model.amp_1, model.amp_2, model.amp_3], dtype=float)
    omegas = np.array([model.omega_1, model.omega_2, model.omega_3], dtype=float)
    phases = np.array([model.phi_1, model.phi_2, model.phi_3], dtype=float)

    h_vals = np.zeros((t.size, 3), dtype=float)
    s_vals = np.zeros((t.size, 3), dtype=float)
    c_vals = np.zeros((t.size, 3), dtype=float)

    if h_mode == 1:
        t_hours = t / 3600.0
        for j in range(3):
            s_vals[:, j] = np.sin(omegas[j] * t_hours + phases[j])
            c_vals[:, j] = np.cos(omegas[j] * t_hours + phases[j])
            h_vals[:, j] = amps[j] * s_vals[:, j]
        return h_vals[:, 0], h_vals[:, 1], h_vals[:, 2], s_vals[:, 0], c_vals[:, 0], np.column_stack((s_vals[:, 1], c_vals[:, 1], s_vals[:, 2], c_vals[:, 2])), h_tag, h_desc

    if h_mode == 2:
        t_half = 0.5 * total_seconds
        ramp = 1.0 - 2.0 * np.minimum(t / t_half, 1.0)
        for j in range(3):
            h_vals[:, j] = amps[j] * ramp
            s_vals[:, j] = np.clip(ramp, -1.0, 1.0)
        return h_vals[:, 0], h_vals[:, 1], h_vals[:, 2], s_vals[:, 0], c_vals[:, 0], np.column_stack((s_vals[:, 1], c_vals[:, 1], s_vals[:, 2], c_vals[:, 2])), h_tag, h_desc

    if h_mode == 3:
        mean_pull = 0.05
        sigma_scale = 0.18
        for j in range(3):
            for i in range(1, t.size):
                dt = max(float(t[i] - t[i - 1]), 0.0)
                proposal = h_vals[i - 1, j] + amps[j] * sigma_scale * np.sqrt(max(dt / 3600.0, 0.0)) * rng.standard_normal()
                proposal += mean_pull * (0.0 - h_vals[i - 1, j])
                h_vals[i, j] = clip_scalar(float(proposal), -amps[j], amps[j])
            s_vals[:, j] = np.divide(h_vals[:, j], amps[j], out=np.zeros_like(h_vals[:, j]), where=amps[j] != 0)
        return h_vals[:, 0], h_vals[:, 1], h_vals[:, 2], s_vals[:, 0], c_vals[:, 0], np.column_stack((s_vals[:, 1], c_vals[:, 1], s_vals[:, 2], c_vals[:, 2])), h_tag, h_desc

    if h_mode == 4:
        step_mask = t >= 0.5 * total_seconds
        for j in range(3):
            h_vals[:, j] = 0.0
            h_vals[step_mask, j] = amps[j]
            s_vals[:, j] = np.divide(h_vals[:, j], amps[j], out=np.zeros_like(h_vals[:, j]), where=amps[j] != 0)
        return h_vals[:, 0], h_vals[:, 1], h_vals[:, 2], s_vals[:, 0], c_vals[:, 0], np.column_stack((s_vals[:, 1], c_vals[:, 1], s_vals[:, 2], c_vals[:, 2])), h_tag, h_desc

    if h_mode == 5:
        for j in range(3):
            tcur = 0.0
            current_level = 0.0
            while tcur <= total_seconds:
                hold_time = rand_between(rng, DEFAULT_PRBS_HOLD_MIN_SECONDS, DEFAULT_PRBS_HOLD_MAX_SECONDS)
                sign_term = 1.0 if rng.random() > 0.5 else -1.0
                amp_term = rand_between(rng, 0.25 * amps[j], 1.0 * amps[j])
                next_level = clip_scalar(sign_term * amp_term, -amps[j], amps[j])
                mask = (t >= tcur) & (t < min(tcur + hold_time, total_seconds + np.finfo(float).eps))
                h_vals[mask, j] = next_level
                current_level = next_level
                tcur += hold_time
            if not np.any(h_vals[:, j]):
                h_vals[:, j] = current_level
            s_vals[:, j] = np.divide(h_vals[:, j], amps[j], out=np.zeros_like(h_vals[:, j]), where=amps[j] != 0)
        return h_vals[:, 0], h_vals[:, 1], h_vals[:, 2], s_vals[:, 0], c_vals[:, 0], np.column_stack((s_vals[:, 1], c_vals[:, 1], s_vals[:, 2], c_vals[:, 2])), h_tag, h_desc

    raise ValueError(f"Unsupported h_mode={h_mode}. Choose 1, 2, 3, 4, or 5.")


def physical_rhs_with_hidden(
    model: TwoColumnExtractionModel,
    x_phys: np.ndarray,
    u: np.ndarray,
    h1: float,
    h2: float,
    h3: float,
) -> np.ndarray:
    (
        x1a, y1a, x2a, y2a, x3a, y3a, x4a, y4a, x5a, y5a,
        x1b, y1b, x2b, y2b, x3b, y3b, x4b, y4b,
    ) = x_phys
    l_a, g_a, x0_a_in, y6_a_in, l_b, g_b, x0_b_in, y5_b_in = u

    x0_a_eff = x0_a_in + model.a_a1 * h1
    x0_b_eff = x0_b_in + model.a_b1 * h1
    kla_a_1 = model.kla_a * (1.0 + model.c_a1 * h2)
    kla_a_4 = model.kla_a * (1.0 + model.c_a4 * h2)
    kla_b_2 = model.kla_b * (1.0 + model.c_b2 * h3)
    kla_b_3 = model.kla_b * (1.0 + model.c_b3 * h3)

    xeq1a, xeq2a, xeq3a, xeq4a, xeq5a = y1a / model.m_a, y2a / model.m_a, y3a / model.m_a, y4a / model.m_a, y5a / model.m_a
    xeq1b, xeq2b, xeq3b, xeq4b = y1b / model.m_b, y2b / model.m_b, y3b / model.m_b, y4b / model.m_b

    q1a = kla_a_1 * (x1a - xeq1a) * model.vl_a
    q2a = model.kla_a * (x2a - xeq2a) * model.vl_a
    q3a = model.kla_a * (x3a - xeq3a) * model.vl_a
    q4a = kla_a_4 * (x4a - xeq4a) * model.vl_a
    q5a = model.kla_a * (x5a - xeq5a) * model.vl_a

    q1b = model.kla_b * (x1b - xeq1b) * model.vl_b
    q2b = kla_b_2 * (x2b - xeq2b) * model.vl_b
    q3b = kla_b_3 * (x3b - xeq3b) * model.vl_b
    q4b = model.kla_b * (x4b - xeq4b) * model.vl_b

    return np.array(
        [
            (l_a * (x0_a_eff - x1a) - q1a) / model.vl_a,
            (g_a * (y2a - y1a) + q1a) / model.vg_a,
            (l_a * (x1a - x2a) - q2a) / model.vl_a,
            (g_a * (y3a - y2a) + q2a) / model.vg_a,
            (l_a * (x2a - x3a) - q3a) / model.vl_a,
            (g_a * (y4a - y3a) + q3a) / model.vg_a,
            (l_a * (x3a - x4a) - q4a) / model.vl_a,
            (g_a * (y5a - y4a) + q4a) / model.vg_a,
            (l_a * (x4a - x5a) - q5a) / model.vl_a,
            (g_a * (y6_a_in - y5a) + q5a) / model.vg_a,
            (l_b * (x0_b_eff - x1b) - q1b) / model.vl_b,
            (g_b * (y2b - y1b) + q1b) / model.vg_b,
            (l_b * (x1b - x2b) - q2b) / model.vl_b,
            (g_b * (y3b - y2b) + q2b) / model.vg_b,
            (l_b * (x2b - x3b) - q3b) / model.vl_b,
            (g_b * (y4b - y3b) + q3b) / model.vg_b,
            (l_b * (x3b - x4b) - q4b) / model.vl_b,
            (g_b * (y5_b_in - y4b) + q4b) / model.vg_b,
        ],
        dtype=float,
    )


def simulate_with_hidden_profiles(
    model: TwoColumnExtractionModel,
    input_schedule: np.ndarray,
    h_profiles: tuple[np.ndarray, np.ndarray, np.ndarray],
    s_c_profiles: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    dt_hours: float,
) -> list[dict[str, float]]:
    u_array = np.asarray(input_schedule, dtype=float)
    h1_vals, h2_vals, h3_vals = [np.asarray(v, dtype=float) for v in h_profiles]
    s1_vals, c1_vals, s2_vals, c2_vals, s3_vals, c3_vals = [np.asarray(v, dtype=float) for v in s_c_profiles]

    n_steps = u_array.shape[0]
    if any(arr.shape[0] != n_steps + 1 for arr in (h1_vals, h2_vals, h3_vals, s1_vals, c1_vals, s2_vals, c2_vals, s3_vals, c3_vals)):
        raise ValueError("Hidden profiles must have one more sample than the input schedule.")

    x_phys = model.default_initial_state()[: model.physical_state_count].astype(float).copy()
    rows: list[dict[str, float]] = []

    for k in range(n_steps):
        t_hours = k * dt_hours
        u = u_array[k]
        row = {"time": t_hours}
        row.update(dict(zip(model.physical_state_names(), x_phys, strict=True)))
        row.update(
            {
                "s1": float(s1_vals[k]),
                "c1": float(c1_vals[k]),
                "s2": float(s2_vals[k]),
                "c2": float(c2_vals[k]),
                "s3": float(s3_vals[k]),
                "c3": float(c3_vals[k]),
            }
        )
        row.update(dict(zip(model.input_names(), u, strict=True)))
        row.update({"H1": float(h1_vals[k]), "H2": float(h2_vals[k]), "H3": float(h3_vals[k]), "time_hours": t_hours})
        rows.append(row)

        h0 = np.array([h1_vals[k], h2_vals[k], h3_vals[k]], dtype=float)
        h1 = np.array([h1_vals[k + 1], h2_vals[k + 1], h3_vals[k + 1]], dtype=float)
        h_mid = 0.5 * (h0 + h1)

        k1 = physical_rhs_with_hidden(model, x_phys, u, float(h0[0]), float(h0[1]), float(h0[2]))
        k2 = physical_rhs_with_hidden(model, x_phys + 0.5 * dt_hours * k1, u, float(h_mid[0]), float(h_mid[1]), float(h_mid[2]))
        k3 = physical_rhs_with_hidden(model, x_phys + 0.5 * dt_hours * k2, u, float(h_mid[0]), float(h_mid[1]), float(h_mid[2]))
        k4 = physical_rhs_with_hidden(model, x_phys + dt_hours * k3, u, float(h1[0]), float(h1[1]), float(h1[2]))
        x_phys = x_phys + (dt_hours / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(x_phys)):
            raise FloatingPointError(f"Simulation became non-finite at step {k + 1}.")

    final_time_hours = n_steps * dt_hours
    final_u = u_array[-1]
    final_row = {"time": final_time_hours}
    final_row.update(dict(zip(model.physical_state_names(), x_phys, strict=True)))
    final_row.update(
        {
            "s1": float(s1_vals[-1]),
            "c1": float(c1_vals[-1]),
            "s2": float(s2_vals[-1]),
            "c2": float(c2_vals[-1]),
            "s3": float(s3_vals[-1]),
            "c3": float(c3_vals[-1]),
        }
    )
    final_row.update(dict(zip(model.input_names(), final_u, strict=True)))
    final_row.update({"H1": float(h1_vals[-1]), "H2": float(h2_vals[-1]), "H3": float(h3_vals[-1]), "time_hours": final_time_hours})
    rows.append(final_row)
    return rows


def apply_measurement_noise(
    rows: list[dict[str, float]],
    *,
    measurement_columns: list[str],
    dt_seconds: float,
    noise_mode: int,
    seed: int,
) -> list[dict[str, float]]:
    noisy_rows = [dict(row) for row in rows]
    rng = np.random.default_rng(seed)

    if noise_mode == 1:
        for row in noisy_rows:
            for col in measurement_columns:
                row[col] += DEFAULT_MEASUREMENT_GAUSSIAN_STD * rng.standard_normal()
        return noisy_rows

    if noise_mode == 2:
        for row in noisy_rows:
            for col in measurement_columns:
                raw_value = row[col] + DEFAULT_MEASUREMENT_GAUSSIAN_STD * rng.standard_normal()
                row[col] = DEFAULT_MEASUREMENT_QUANT_STEP * np.round(raw_value / DEFAULT_MEASUREMENT_QUANT_STEP)
        return noisy_rows

    if noise_mode == 3:
        walks = {col: 0.0 for col in measurement_columns}
        for row in noisy_rows:
            for col in measurement_columns:
                row[col] += walks[col]
            for col in measurement_columns:
                walks[col] += DEFAULT_MEASUREMENT_RANDOM_WALK_SIGMA * dt_seconds * rng.standard_normal()
        return noisy_rows

    raise ValueError(f"Unsupported noise_mode={noise_mode}. Choose 1, 2, or 3.")


def hidden_rows(t_seconds: np.ndarray, h1_vals: np.ndarray, h2_vals: np.ndarray, h3_vals: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "time_seconds": float(t),
            "time_hours": float(t / 3600.0),
            "H1": float(h1),
            "H2": float(h2),
            "H3": float(h3),
        }
        for t, h1, h2, h3 in zip(t_seconds, h1_vals, h2_vals, h3_vals, strict=True)
    ]


def metadata_rows(config: GeneratorConfig, *, h_mode: int, h_tag: str, h_desc: str, noise_mode: int, noise_tag: str, noise_desc: str, schedule_mode: str) -> list[dict[str, float | str]]:
    return [
        {
            "schedule_mode": schedule_mode,
            "H_mode": h_mode,
            "H_tag": h_tag,
            "H_description": h_desc,
            "noise_mode": noise_mode,
            "noise_tag": noise_tag,
            "noise_description": noise_desc,
            "gaussian_std": DEFAULT_MEASUREMENT_GAUSSIAN_STD,
            "quantization_step": DEFAULT_MEASUREMENT_QUANT_STEP,
            "random_walk_sigma": DEFAULT_MEASUREMENT_RANDOM_WALK_SIGMA,
            "sample_time_seconds": config.sample_time_seconds,
            "pulse_seconds": config.pulse_seconds,
            "initial_baseline_seconds": config.initial_baseline_seconds,
            "final_baseline_seconds": config.final_baseline_seconds,
            "total_sim_seconds": config.total_sim_seconds,
        }
    ]


def write_case(
    rows: list[dict[str, float]],
    events: list[dict[str, float | str]],
    hidden: list[dict[str, float]],
    metadata: list[dict[str, float | str]],
    *,
    output_path: Path,
    schedule_path: Path,
    hidden_path: Path,
    metadata_path: Path,
) -> None:
    write_csv(rows, output_path)
    write_rows(events, schedule_path)
    write_rows(hidden, hidden_path)
    write_rows(metadata, metadata_path)
