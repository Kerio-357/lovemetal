"""Cursor smoothing filters. All operate on 2D points and share one interface:

    filt.update((x, y), t) -> (x_smooth, y_smooth)
    filt.reset()
"""
from __future__ import annotations

import numpy as np


class EMAFilter:
    """Exponential moving average: s_t = a * x_t + (1 - a) * s_{t-1}.

    Cheap and predictable, but adds constant lag for any alpha < 1.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._s: np.ndarray | None = None

    def reset(self) -> None:
        self._s = None

    def update(self, p, t: float | None = None):
        p = np.asarray(p, dtype=np.float64)
        self._s = p.copy() if self._s is None else self.alpha * p + (1 - self.alpha) * self._s
        return float(self._s[0]), float(self._s[1])


class KalmanFilter2D:
    """Constant-velocity Kalman filter, state = [x, y, vx, vy].

    Compared to EMA it uses a motion model, so it lags less during fast
    sweeps while still suppressing jitter when the hand is nearly still.
    Tuning: raise `process_noise` for snappier, raise `meas_noise` for smoother.
    """

    def __init__(self, process_noise: float = 400.0, meas_noise: float = 25.0):
        self.q = process_noise
        self.r = meas_noise
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None      # state
        self.P = np.eye(4) * 1e3              # covariance
        self._t_prev: float | None = None
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.eye(2) * self.r

    def update(self, p, t: float):
        z = np.asarray(p, dtype=np.float64)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self._t_prev = t
            return float(z[0]), float(z[1])

        dt = max(1e-3, min(t - self._t_prev, 0.25))
        self._t_prev = t

        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Discrete white-noise acceleration model.
        Q = self.q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])


def make_smoother(cfg):
    if cfg.smoother == "ema":
        return EMAFilter(cfg.ema_alpha)
    if cfg.smoother == "kalman":
        return KalmanFilter2D(cfg.kalman_process_noise, cfg.kalman_meas_noise)
    raise ValueError(f"Unknown smoother '{cfg.smoother}' (use 'ema' or 'kalman')")
