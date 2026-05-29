from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from low_dim_nonlinear_simulator import LowDimNonlinearBenchmarkModel


PULSE_HOURS = 1.5
SAMPLE_TIME_HOURS = 0.05
INITIAL_BASELINE_HOURS = 1.5
FINAL_BASELINE_HOURS = 1.5
TOTAL_SIM_HOURS = 40.0
INPUT_TARGET_RANGE = (-2.0, 2.0)
INPUT_TARGET_SEED = 7
STATE_NOISE_STD = 0.0

NOISE_MODES = (1, 2, 3)
GAUSSIAN_STD = 0.04
QUANTIZED_GAUSSIAN_STD = 0.04
QUANTIZATION_STEP = 0.02
RANDOM_WALK_SIGMA = 0.014

H_MODES = (1, 2, 3, 4, 5)
H_SIN_MEAN = 2.2
H_SIN_AMP = 0.6
H_SIN_FREQ = 1 / 20.0
H_SIN_PHASE = 0.0
H_DECAY_START = 2.8
H_DECAY_END = 1.6
H_BROWN_MEAN = 2.2
H_BROWN_SIGMA = 0.12
H_BROWN_MIN = 1.6
H_BROWN_MAX = 2.8
H_BROWN_MEAN_PULL = 0.08
H_STEP_TIME = 0.5 * TOTAL_SIM_HOURS
H_STEP_INITIAL = 2.8
H_STEP_LEVEL = 1.6
H_PRBS_MEAN = 2.2
H_PRBS_AMP_MIN = 0.2
H_PRBS_AMP_MAX = 0.6
H_PRBS_HOLD_MIN = 3.0
H_PRBS_HOLD_MAX = 8.0
H_PRBS_MIN = 1.6
H_PRBS_MAX = 2.8


def format_token(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def noise_tag(noise_mode: int) -> tuple[str, str]:
    if noise_mode == 1:
        return "N1", "Gaussian measurement noise"
    if noise_mode == 2:
        return "N2", "quantized Gaussian measurement noise"
    if noise_mode == 3:
        return "N3", "random-walk measurement noise"
    raise ValueError(f"Unsupported NOISE_MODE={noise_mode}. Choose 1, 2, or 3.")


def h_tag(h_mode: int) -> tuple[str, str]:
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
    raise ValueError(f"Unsupported H_MODE={h_mode}. Choose 1, 2, 3, 4, or 5.")


def rand_between(rng: np.random.Generator, a: float, b: float) -> float:
    return float(a + (b - a) * rng.random())


def clip_scalar(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_zero_return_schedule(*, seed: int | None = INPUT_TARGET_SEED) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    baseline_u = np.array([[0.0]], dtype=float)
    rng = np.random.default_rng(seed)
    pulse_steps = max(1, int(round(PULSE_HOURS / SAMPLE_TIME_HOURS)))
    wait_steps = max(1, int(round(PULSE_HOURS / SAMPLE_TIME_HOURS)))
    initial_steps = max(1, int(round(INITIAL_BASELINE_HOURS / SAMPLE_TIME_HOURS)))
    final_steps = max(1, int(round(FINAL_BASELINE_HOURS / SAMPLE_TIME_HOURS)))
    total_steps_limit = max(1, int(round(TOTAL_SIM_HOURS / SAMPLE_TIME_HOURS)))

    segments: list[np.ndarray] = []
    events: list[dict[str, float | str]] = []
    current_step = 0
    pulse_index = 0

    def append_block(block_u: np.ndarray, block_steps: int, phase: str, *, target_value: float | None = None) -> None:
        nonlocal current_step
        if block_steps <= 0:
            return
        segments.append(np.tile(block_u, (block_steps, 1)))
        events.append(
            {
                "pulse_index": pulse_index if phase == "pulse" else "",
                "phase": phase,
                "start_time_hours": current_step * SAMPLE_TIME_HOURS,
                "end_time_hours": (current_step + block_steps) * SAMPLE_TIME_HOURS,
                "target_value": float(target_value) if target_value is not None else "",
            }
        )
        current_step += block_steps

    append_block(baseline_u, min(initial_steps, total_steps_limit), "initial_baseline")

    while current_step < total_steps_limit:
        target_value = float(rng.uniform(INPUT_TARGET_RANGE[0], INPUT_TARGET_RANGE[1]))
        pulse_u = np.array([[target_value]], dtype=float)
        append_block(pulse_u, min(pulse_steps, total_steps_limit - current_step), "pulse", target_value=target_value)
        if current_step >= total_steps_limit:
            break
        pulse_index += 1
        phase = "final_baseline" if current_step + wait_steps >= total_steps_limit else "baseline_wait"
        baseline_steps = final_steps if phase == "final_baseline" else wait_steps
        append_block(baseline_u, min(baseline_steps, total_steps_limit - current_step), phase)

    return np.vstack(segments), events


def build_varying_baseline_schedule(*, seed: int | None = INPUT_TARGET_SEED) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    rng = np.random.default_rng(seed)
    pulse_steps = max(1, int(round(PULSE_HOURS / SAMPLE_TIME_HOURS)))
    hold_steps = max(1, int(round(PULSE_HOURS / SAMPLE_TIME_HOURS)))
    initial_steps = max(1, int(round(INITIAL_BASELINE_HOURS / SAMPLE_TIME_HOURS)))
    final_steps = max(1, int(round(FINAL_BASELINE_HOURS / SAMPLE_TIME_HOURS)))
    total_steps_limit = max(1, int(round(TOTAL_SIM_HOURS / SAMPLE_TIME_HOURS)))

    current_baseline = 0.0
    segments: list[np.ndarray] = []
    events: list[dict[str, float | str]] = []
    current_step = 0
    pulse_index = 0

    def append_block(level: float, block_steps: int, phase: str, *, target_value: float | None = None, baseline_before: float | None = None) -> None:
        nonlocal current_step
        if block_steps <= 0:
            return
        segments.append(np.full((block_steps, 1), level, dtype=float))
        events.append(
            {
                "pulse_index": pulse_index if phase == "pulse" else "",
                "phase": phase,
                "start_time_hours": current_step * SAMPLE_TIME_HOURS,
                "end_time_hours": (current_step + block_steps) * SAMPLE_TIME_HOURS,
                "baseline_before": float(baseline_before) if baseline_before is not None else "",
                "level": float(level),
                "target_value": float(target_value) if target_value is not None else "",
            }
        )
        current_step += block_steps

    append_block(current_baseline, min(initial_steps, total_steps_limit), "initial_baseline", baseline_before=current_baseline)

    while current_step < total_steps_limit:
        baseline_before = current_baseline
        target_value = float(rng.uniform(INPUT_TARGET_RANGE[0], INPUT_TARGET_RANGE[1]))
        append_block(target_value, min(pulse_steps, total_steps_limit - current_step), "pulse", target_value=target_value, baseline_before=baseline_before)
        current_baseline = target_value
        if current_step >= total_steps_limit:
            break
        pulse_index += 1
        phase = "final_baseline" if current_step + hold_steps >= total_steps_limit else "baseline_hold"
        baseline_steps = final_steps if phase == "final_baseline" else hold_steps
        append_block(current_baseline, min(baseline_steps, total_steps_limit - current_step), phase, baseline_before=current_baseline)

    return np.vstack(segments), events


def make_h_profile(t_uniform: np.ndarray, *, h_mode: int, seed: int | None = INPUT_TARGET_SEED) -> tuple[np.ndarray, str, str]:
    t_force = np.asarray(t_uniform, dtype=float)
    tag, desc = h_tag(h_mode)
    rng = np.random.default_rng(seed)

    if h_mode == 1:
        h_vals = H_SIN_MEAN + H_SIN_AMP * np.sin(2 * np.pi * H_SIN_FREQ * t_force + H_SIN_PHASE)
        return h_vals, tag, desc

    if h_mode == 2:
        t_half = 0.5 * TOTAL_SIM_HOURS
        h_vals = H_DECAY_START + (H_DECAY_END - H_DECAY_START) * np.minimum(t_force / t_half, 1.0)
        return h_vals, tag, desc

    if h_mode == 3:
        h_vals = np.zeros_like(t_force)
        h_vals[0] = H_BROWN_MEAN
        for i in range(1, len(t_force)):
            dt = max(float(t_force[i] - t_force[i - 1]), 0.0)
            proposal = h_vals[i - 1] + H_BROWN_SIGMA * np.sqrt(dt) * rng.standard_normal()
            proposal += H_BROWN_MEAN_PULL * (H_BROWN_MEAN - h_vals[i - 1])
            h_vals[i] = clip_scalar(float(proposal), H_BROWN_MIN, H_BROWN_MAX)
        return h_vals, tag, desc

    if h_mode == 4:
        h_vals = np.full_like(t_force, H_STEP_INITIAL)
        h_vals[t_force >= H_STEP_TIME] = H_STEP_LEVEL
        return h_vals, tag, desc

    if h_mode == 5:
        h_vals = np.zeros_like(t_force)
        tcur = 0.0
        current_level = H_PRBS_MEAN
        while tcur <= TOTAL_SIM_HOURS:
            hold_time = rand_between(rng, H_PRBS_HOLD_MIN, H_PRBS_HOLD_MAX)
            sign_term = 1.0 if rng.random() > 0.5 else -1.0
            amp_term = rand_between(rng, H_PRBS_AMP_MIN, H_PRBS_AMP_MAX)
            next_level = clip_scalar(H_PRBS_MEAN + sign_term * amp_term, H_PRBS_MIN, H_PRBS_MAX)
            mask = (t_force >= tcur) & (t_force < min(tcur + hold_time, TOTAL_SIM_HOURS + np.finfo(float).eps))
            h_vals[mask] = next_level
            current_level = next_level
            tcur += hold_time
        if not np.any(h_vals):
            h_vals[:] = current_level
        return h_vals, tag, desc

    raise ValueError(f"Unsupported H_MODE={h_mode}. Choose 1, 2, 3, 4, or 5.")


def simulate_with_hidden_profile(
    model: LowDimNonlinearBenchmarkModel,
    input_schedule: np.ndarray,
    h_values: np.ndarray,
    *,
    dt: float,
    x0: np.ndarray | None = None,
) -> list[dict[str, float]]:
    u_array = np.asarray(input_schedule, dtype=float)
    h_array = np.asarray(h_values, dtype=float)
    if u_array.ndim != 2 or u_array.shape[1] != 1:
        raise ValueError(f"Expected input array shape (steps, 1), got {u_array.shape}.")
    if h_array.shape[0] != u_array.shape[0] + 1:
        raise ValueError("Hidden profile must have one more sample than the input schedule.")

    if x0 is None:
        x_state = model.default_initial_state()[:2].astype(float)
    else:
        x_state = np.asarray(x0, dtype=float).copy()

    def rhs(x: np.ndarray, u_value: float, h_value: float) -> np.ndarray:
        x1, x2 = x
        dx1 = -(x1 / model.tau1_hours) + model.a1 * np.tanh(model.b1 * u_value) + model.c1h * np.tanh(model.g1h * h_value)
        dx2 = -(x2 / model.tau2_hours) + model.a2h * np.tanh(model.g2h * h_value)
        return np.array([dx1, dx2], dtype=float)

    rows: list[dict[str, float]] = []
    for k in range(u_array.shape[0]):
        t = k * dt
        u_value = float(u_array[k, 0])
        h0 = float(h_array[k])
        h1 = float(h_array[k + 1])
        h_mid = 0.5 * (h0 + h1)

        rows.append({"time": t, "u": u_value, "x1": float(x_state[0]), "x2": float(x_state[1])})

        k1 = rhs(x_state, u_value, h0)
        k2 = rhs(x_state + 0.5 * dt * k1, u_value, h_mid)
        k3 = rhs(x_state + 0.5 * dt * k2, u_value, h_mid)
        k4 = rhs(x_state + dt * k3, u_value, h1)
        x_state = x_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(x_state)):
            raise FloatingPointError("Simulation became non-finite.")

    rows.append({"time": u_array.shape[0] * dt, "u": float(u_array[-1, 0]), "x1": float(x_state[0]), "x2": float(x_state[1])})
    return rows


def apply_measurement_noise(rows: list[dict[str, float]], *, noise_mode: int, dt: float, seed: int | None) -> list[dict[str, float]]:
    noisy_rows = [dict(row) for row in rows]
    rng = np.random.default_rng(seed)

    if noise_mode == 1:
        for row in noisy_rows:
            row["x1"] += GAUSSIAN_STD * rng.standard_normal()
            row["x2"] += GAUSSIAN_STD * rng.standard_normal()
        return noisy_rows

    if noise_mode == 2:
        for row in noisy_rows:
            x1_noisy = row["x1"] + QUANTIZED_GAUSSIAN_STD * rng.standard_normal()
            x2_noisy = row["x2"] + QUANTIZED_GAUSSIAN_STD * rng.standard_normal()
            row["x1"] = QUANTIZATION_STEP * np.round(x1_noisy / QUANTIZATION_STEP)
            row["x2"] = QUANTIZATION_STEP * np.round(x2_noisy / QUANTIZATION_STEP)
        return noisy_rows

    if noise_mode == 3:
        walk_x1 = 0.0
        walk_x2 = 0.0
        for row in noisy_rows:
            row["x1"] += walk_x1
            row["x2"] += walk_x2
            walk_x1 += RANDOM_WALK_SIGMA * dt * rng.standard_normal()
            walk_x2 += RANDOM_WALK_SIGMA * dt * rng.standard_normal()
        return noisy_rows

    raise ValueError(f"Unsupported NOISE_MODE={noise_mode}. Choose 1, 2, or 3.")


def build_metadata_rows(h_mode: int, h_mode_tag: str, h_desc: str, noise_mode: int, noise_mode_tag: str, noise_desc: str) -> list[dict[str, float | str]]:
    return [
        {
            "H_mode": h_mode,
            "H_tag": h_mode_tag,
            "H_description": h_desc,
            "noise_mode": noise_mode,
            "noise_tag": noise_mode_tag,
            "noise_description": noise_desc,
            "gaussian_std": GAUSSIAN_STD,
            "quantized_gaussian_std": QUANTIZED_GAUSSIAN_STD,
            "quantization_step": QUANTIZATION_STEP,
            "random_walk_sigma": RANDOM_WALK_SIGMA,
            "sample_time_hours": SAMPLE_TIME_HOURS,
        }
    ]


def hidden_profile_rows(t_uniform: np.ndarray, h_values: np.ndarray) -> list[dict[str, float]]:
    return [{"time": float(t), "H": float(h)} for t, h in zip(t_uniform, h_values, strict=True)]


def write_rows(rows: list[dict[str, float | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

