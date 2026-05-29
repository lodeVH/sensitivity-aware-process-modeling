from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


ArrayLike = np.ndarray


@dataclass(kw_only=True)
class LowDimNonlinearBenchmarkModel:
    tau1_hours: float = 1.4
    tau2_hours: float = 1.1
    a1: float = 0.90
    b1: float = 1.7
    c1h: float = 1.20
    a2h: float = 2.00
    g1h: float = 1.20
    g2h: float = 1.60
    h0: float = 2.8
    h_decay_rate: float = 0.03
    state_noise_std: float = 0.0

    def state_names(self) -> list[str]:
        return ["x1", "x2", "H"]

    def input_names(self) -> list[str]:
        return ["u"]

    def default_initial_state(self) -> np.ndarray:
        return np.array([0.05, -0.03, self.h0], dtype=float)

    def default_input(self) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def __call__(self, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        x1, x2, h = x
        u0 = float(u[0])
        dx1 = -(x1 / self.tau1_hours) + self.a1 * np.tanh(self.b1 * u0) + self.c1h * np.tanh(self.g1h * h)
        dx2 = -(x2 / self.tau2_hours) + self.a2h * np.tanh(self.g2h * h)
        dh = -self.h_decay_rate
        return np.array([dx1, dx2, dh], dtype=float)


def rk4_step(model: LowDimNonlinearBenchmarkModel, x: ArrayLike, u: ArrayLike, dt: float) -> np.ndarray:
    k1 = model(x, u)
    k2 = model(x + 0.5 * dt * k1, u)
    k3 = model(x + 0.5 * dt * k2, u)
    k4 = model(x + dt * k3, u)
    return np.asarray(x, dtype=float) + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    model: LowDimNonlinearBenchmarkModel,
    x0: ArrayLike,
    input_schedule: ArrayLike,
    *,
    dt: float,
    seed: int | None = None,
) -> list[dict[str, float]]:
    x = np.asarray(x0, dtype=float).copy()
    rng = np.random.default_rng(seed)
    u_array = np.asarray(input_schedule, dtype=float)
    if u_array.ndim != 2 or u_array.shape[1] != 1:
        raise ValueError(f"Expected input array shape (steps, 1), got {u_array.shape}.")

    rows: list[dict[str, float]] = []
    for k in range(u_array.shape[0]):
        t = k * dt
        u = u_array[k]
        rows.append({"time": t, "u": float(u[0]), "x1": float(x[0]), "x2": float(x[1])})
        x_next = rk4_step(model, x, u, dt)
        if model.state_noise_std > 0.0:
            x_next[:2] += model.state_noise_std * np.sqrt(dt) * rng.standard_normal(2)
        if not np.all(np.isfinite(x_next)):
            raise FloatingPointError("Simulation became non-finite.")
        x = x_next

    rows.append({"time": u_array.shape[0] * dt, "u": float(u_array[-1, 0]), "x1": float(x[0]), "x2": float(x[1])})
    return rows


def write_csv(rows: list[dict[str, float]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
