from __future__ import annotations

from dataclasses import dataclass, field, replace
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np


ArrayLike = np.ndarray


@dataclass(kw_only=True)
class TwoColumnExtractionModel:
    """Standalone pc-gym-style model for the two-column extraction system."""

    vl_a: float = 5.0
    vg_a: float = 5.0
    vl_b: float = 5.0
    vg_b: float = 5.0
    m_a: float = 1.0
    m_b: float = 1.0
    kla_a: float = 5.0
    kla_b: float = 5.0
    x0_a: float = 0.60
    y6_a: float = 0.05
    x0_b: float = 0.60
    y5_b: float = 0.05
    a_a1: float = 0.08
    a_b1: float = 0.08
    c_a1: float = 0.12
    c_a4: float = 0.10
    c_b2: float = 0.10
    c_b3: float = 0.08
    amp_1: float = 0.30
    amp_2: float = 0.20
    amp_3: float = 0.18
    omega_1: float = 0.18
    omega_2: float = 0.12
    omega_3: float = 0.15
    phi_1: float = 0.0
    phi_2: float = 0.6
    phi_3: float = 1.2
    state_noise_std: float | list[float] | np.ndarray = 0.0
    physical_state_count: int = field(init=False, default=18)

    def state_names(self) -> list[str]:
        return [
            "X1A", "Y1A", "X2A", "Y2A", "X3A", "Y3A", "X4A", "Y4A", "X5A", "Y5A",
            "X1B", "Y1B", "X2B", "Y2B", "X3B", "Y3B", "X4B", "Y4B",
            "s1", "c1", "s2", "c2", "s3", "c3",
        ]

    def physical_state_names(self) -> list[str]:
        return self.state_names()[: self.physical_state_count]

    def input_names(self) -> list[str]:
        return ["L_A", "G_A", "X0_A", "Y6_A", "L_B", "G_B", "X0_B", "Y5_B"]

    def hidden_names(self) -> list[str]:
        return ["H1", "H2", "H3"]

    def info(self) -> dict:
        return {
            "parameters": self.__dict__.copy(),
            "states": self.state_names(),
            "inputs": self.input_names(),
            "hidden_outputs": self.hidden_names(),
        }

    def default_initial_state(self) -> np.ndarray:
        return np.array(
            [
                0.55, 0.30, 0.45, 0.25, 0.40, 0.20, 0.35, 0.15, 0.25, 0.10,
                0.52, 0.28, 0.41, 0.22, 0.32, 0.16, 0.24, 0.09,
                math.sin(self.phi_1), math.cos(self.phi_1),
                math.sin(self.phi_2), math.cos(self.phi_2),
                math.sin(self.phi_3), math.cos(self.phi_3),
            ],
            dtype=float,
        )

    def default_input(self) -> np.ndarray:
        return np.array([8.0, 50.0, self.x0_a, self.y6_a, 7.5, 50.0, self.x0_b, self.y5_b], dtype=float)

    def hidden_signals(self, x: ArrayLike) -> tuple[float, float, float]:
        h1 = self.amp_1 * float(x[18])
        h2 = self.amp_2 * float(x[20])
        h3 = self.amp_3 * float(x[22])
        return h1, h2, h3

    def __call__(self, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        (
            x1a, y1a, x2a, y2a, x3a, y3a, x4a, y4a, x5a, y5a,
            x1b, y1b, x2b, y2b, x3b, y3b, x4b, y4b,
            s1, c1, s2, c2, s3, c3,
        ) = x
        l_a, g_a, x0_a_in, y6_a_in, l_b, g_b, x0_b_in, y5_b_in = u

        h1, h2, h3 = self.hidden_signals(x)
        x0_a_eff = x0_a_in + self.a_a1 * h1
        x0_b_eff = x0_b_in + self.a_b1 * h1
        kla_a_1 = self.kla_a * (1.0 + self.c_a1 * h2)
        kla_a_4 = self.kla_a * (1.0 + self.c_a4 * h2)
        kla_b_2 = self.kla_b * (1.0 + self.c_b2 * h3)
        kla_b_3 = self.kla_b * (1.0 + self.c_b3 * h3)

        xeq1a, xeq2a, xeq3a, xeq4a, xeq5a = y1a / self.m_a, y2a / self.m_a, y3a / self.m_a, y4a / self.m_a, y5a / self.m_a
        xeq1b, xeq2b, xeq3b, xeq4b = y1b / self.m_b, y2b / self.m_b, y3b / self.m_b, y4b / self.m_b

        q1a = kla_a_1 * (x1a - xeq1a) * self.vl_a
        q2a = self.kla_a * (x2a - xeq2a) * self.vl_a
        q3a = self.kla_a * (x3a - xeq3a) * self.vl_a
        q4a = kla_a_4 * (x4a - xeq4a) * self.vl_a
        q5a = self.kla_a * (x5a - xeq5a) * self.vl_a

        q1b = self.kla_b * (x1b - xeq1b) * self.vl_b
        q2b = kla_b_2 * (x2b - xeq2b) * self.vl_b
        q3b = kla_b_3 * (x3b - xeq3b) * self.vl_b
        q4b = self.kla_b * (x4b - xeq4b) * self.vl_b

        return np.array(
            [
                (l_a * (x0_a_eff - x1a) - q1a) / self.vl_a,
                (g_a * (y2a - y1a) + q1a) / self.vg_a,
                (l_a * (x1a - x2a) - q2a) / self.vl_a,
                (g_a * (y3a - y2a) + q2a) / self.vg_a,
                (l_a * (x2a - x3a) - q3a) / self.vl_a,
                (g_a * (y4a - y3a) + q3a) / self.vg_a,
                (l_a * (x3a - x4a) - q4a) / self.vl_a,
                (g_a * (y5a - y4a) + q4a) / self.vg_a,
                (l_a * (x4a - x5a) - q5a) / self.vl_a,
                (g_a * (y6_a_in - y5a) + q5a) / self.vg_a,
                (l_b * (x0_b_eff - x1b) - q1b) / self.vl_b,
                (g_b * (y2b - y1b) + q1b) / self.vg_b,
                (l_b * (x1b - x2b) - q2b) / self.vl_b,
                (g_b * (y3b - y2b) + q2b) / self.vg_b,
                (l_b * (x2b - x3b) - q3b) / self.vl_b,
                (g_b * (y4b - y3b) + q3b) / self.vg_b,
                (l_b * (x3b - x4b) - q4b) / self.vl_b,
                (g_b * (y5_b_in - y4b) + q4b) / self.vg_b,
                self.omega_1 * c1,
                -self.omega_1 * s1,
                self.omega_2 * c2,
                -self.omega_2 * s2,
                self.omega_3 * c3,
                -self.omega_3 * s3,
            ],
            dtype=float,
        )


def rk4_step(model: TwoColumnExtractionModel, x: ArrayLike, u: ArrayLike, dt: float) -> np.ndarray:
    k1 = model(x, u)
    k2 = model(x + 0.5 * dt * k1, u)
    k3 = model(x + 0.5 * dt * k2, u)
    k4 = model(x + dt * k3, u)
    return np.asarray(x, dtype=float) + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def physical_rhs(model: TwoColumnExtractionModel, x_phys: ArrayLike, u: ArrayLike) -> np.ndarray:
    x_phys = np.asarray(x_phys, dtype=float)
    x = np.concatenate([x_phys, np.zeros(6, dtype=float)])
    return model(x, u)[: model.physical_state_count]


def simulate_physical(
    model: TwoColumnExtractionModel,
    x0_phys: ArrayLike,
    u: ArrayLike,
    *,
    dt: float,
    steps: int,
) -> np.ndarray:
    x = np.concatenate([np.asarray(x0_phys, dtype=float), np.zeros(6, dtype=float)])
    for _ in range(steps):
        x = rk4_step(model, x, u, dt)
    return x[: model.physical_state_count]


def find_physical_steady_state(
    model: TwoColumnExtractionModel,
    u: ArrayLike,
    x0_phys: ArrayLike,
    *,
    dt: float = 0.05,
    chunk_steps: int = 400,
    max_chunks: int = 250,
    tol: float = 1e-9,
) -> np.ndarray:
    x_phys = np.asarray(x0_phys, dtype=float).copy()
    u = np.asarray(u, dtype=float)
    for _ in range(max_chunks):
        x_phys = simulate_physical(model, x_phys, u, dt=dt, steps=chunk_steps)
        residual = np.linalg.norm(physical_rhs(model, x_phys, u), ord=np.inf)
        if residual < tol:
            return x_phys
    raise RuntimeError("Failed to converge to a steady state. Try milder inputs or a longer horizon.")


def numerical_physical_jacobian(
    model: TwoColumnExtractionModel,
    x_ss: ArrayLike,
    u_ss: ArrayLike,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    x_ss = np.asarray(x_ss, dtype=float)
    u_ss = np.asarray(u_ss, dtype=float)
    n = x_ss.size
    jac = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        f_plus = physical_rhs(model, x_ss + dx, u_ss)
        f_minus = physical_rhs(model, x_ss - dx, u_ss)
        jac[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return jac


def rk4_stability_polynomial(z: ArrayLike) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    return 1.0 + z + (z**2) / 2.0 + (z**3) / 6.0 + (z**4) / 24.0


def estimate_rk4_dt_limit_from_matrix(
    a_matrix: ArrayLike,
    *,
    tol: float = 1e-10,
    max_dt: float = 1e6,
) -> float:
    eigenvalues = np.linalg.eigvals(np.asarray(a_matrix, dtype=float))
    if eigenvalues.size == 0:
        return np.inf

    def is_stable(dt: float) -> bool:
        return float(np.max(np.abs(rk4_stability_polynomial(dt * eigenvalues)))) <= 1.0 + tol

    if not is_stable(0.0):
        raise RuntimeError("The RK4 stability check failed at dt = 0.")

    upper = 1.0
    while upper < max_dt and is_stable(upper):
        upper *= 2.0

    if upper >= max_dt and is_stable(upper):
        return np.inf

    lower = 0.0
    for _ in range(80):
        mid = 0.5 * (lower + upper)
        if is_stable(mid):
            lower = mid
        else:
            upper = mid
    return lower


def estimate_nominal_physical_rk4_dt_limit(
    model: TwoColumnExtractionModel,
    *,
    steady_state_dt: float = 0.05,
    safety_factor: float = 0.8,
) -> dict[str, float]:
    nominal_model = replace(model, amp_1=0.0, amp_2=0.0, amp_3=0.0, state_noise_std=0.0)
    u_ss = nominal_model.default_input()
    x0_phys = nominal_model.default_initial_state()[: nominal_model.physical_state_count]
    x_ss = find_physical_steady_state(nominal_model, u_ss, x0_phys, dt=steady_state_dt)
    a_matrix = numerical_physical_jacobian(nominal_model, x_ss, u_ss)
    eigenvalues = np.linalg.eigvals(a_matrix)
    stable_real_parts = np.abs(np.real(eigenvalues[np.real(eigenvalues) < 0.0]))
    fastest_tau_hours = 1.0 / np.max(stable_real_parts) if stable_real_parts.size else np.inf
    dt_limit_hours = estimate_rk4_dt_limit_from_matrix(a_matrix)
    return {
        "dt_limit_hours": dt_limit_hours,
        "recommended_dt_hours": safety_factor * dt_limit_hours,
        "fastest_tau_hours": fastest_tau_hours,
    }


def constant_input_schedule(u: ArrayLike) -> Callable[[int, float], np.ndarray]:
    u = np.asarray(u, dtype=float)
    return lambda _k, _t: u.copy()


def _noise_vector(noise_std: float | list[float] | np.ndarray, size: int) -> np.ndarray:
    if np.isscalar(noise_std):
        return np.full(size, float(noise_std), dtype=float)
    arr = np.asarray(noise_std, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"Expected noise vector of shape {(size,)}, got {arr.shape}.")
    return arr


def simulate(
    model: TwoColumnExtractionModel,
    x0: ArrayLike,
    input_schedule: Callable[[int, float], np.ndarray] | ArrayLike,
    *,
    dt: float,
    steps: int,
    seed: int | None = None,
) -> list[dict[str, float]]:
    x = np.asarray(x0, dtype=float).copy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    noise_std = _noise_vector(model.state_noise_std, model.physical_state_count)

    if callable(input_schedule):
        schedule_fn = input_schedule
    else:
        u_array = np.asarray(input_schedule, dtype=float)
        if u_array.shape != (steps, len(model.input_names())):
            raise ValueError(f"Expected input array shape {(steps, len(model.input_names()))}, got {u_array.shape}.")
        schedule_fn = lambda k, _t: u_array[k]

    for k in range(steps):
        t = k * dt
        u = np.asarray(schedule_fn(k, t), dtype=float)
        if u.shape != (len(model.input_names()),):
            raise ValueError(f"Expected input vector shape {(len(model.input_names()),)}, got {u.shape}.")

        h1, h2, h3 = model.hidden_signals(x)
        row = {"time": t}
        row.update(dict(zip(model.state_names(), x)))
        row.update(dict(zip(model.input_names(), u)))
        row.update({"H1": h1, "H2": h2, "H3": h3})
        rows.append(row)

        x_next = rk4_step(model, x, u, dt)
        if np.any(noise_std > 0.0):
            x_next[: model.physical_state_count] += noise_std * math.sqrt(dt) * rng.standard_normal(model.physical_state_count)
        if not np.all(np.isfinite(x_next)):
            raise FloatingPointError(
                f"Simulation became non-finite at step {k + 1}. "
                "Try a smaller dt or milder flow inputs."
            )
        x = x_next

    final_time = steps * dt
    u = np.asarray(schedule_fn(steps - 1, (steps - 1) * dt), dtype=float)
    h1, h2, h3 = model.hidden_signals(x)
    row = {"time": final_time}
    row.update(dict(zip(model.state_names(), x)))
    row.update(dict(zip(model.input_names(), u)))
    row.update({"H1": h1, "H2": h2, "H3": h3})
    rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
