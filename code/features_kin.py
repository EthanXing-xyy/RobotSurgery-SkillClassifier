"""Trial-level kinematic features.

Two blocks, both computed from the two patient-side (slave) manipulators, which are the
instruments that act on the tissue; the master manipulators are the surgeon's input devices
and their signals are close to duplicates of the slave signals under teleoperation.

K-simple  the statistics named in the task specification: duration, path length, velocity and
          gripper motion, plus two summary ratios (idle fraction, economy of motion) and two
          bimanual terms.
K-texture motion-texture features in the spirit of the background reference (Zia and Essa):
          approximate entropy of the speed and gripper signals, log dimensionless jerk and the
          spectral entropy of the speed signal. These describe how regular a movement is rather
          than how fast or how long it is.
"""
from __future__ import annotations

import numpy as np

from data import FPS, SLAVES, manipulator

VERSION = "kin-v1"

IDLE_SPEED = 0.005       # m/s (5 mm/s); below this the instrument is treated as not moving
APEN_HZ = 10.0           # approximate entropy is computed on a 10 Hz resampling
APEN_M = 2               # embedding dimension
APEN_R = 0.2             # tolerance, in units of the signal's standard deviation


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _mag(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _upward_crossings(x: np.ndarray) -> int:
    """Number of upward crossings of the signal's own mean: one per open-close cycle."""
    c = x - np.mean(x)
    s = np.sign(c)
    s[s == 0] = 1
    return int(np.sum((s[:-1] < 0) & (s[1:] > 0)))


def _approx_entropy(x: np.ndarray, m: int = APEN_M, r_frac: float = APEN_R) -> float:
    """Approximate entropy (Pincus). Lower values mean a more regular, repeatable signal."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sd = float(np.std(x))
    if n < m + 2 or sd == 0:
        return float("nan")
    r = r_frac * sd

    def phi(mm: int) -> float:
        k = n - mm + 1
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:k]
        # Chebyshev distance between every pair of embedded vectors, in blocks to cap memory
        counts = np.empty(k, dtype=np.float64)
        step = max(1, int(2e7 // max(k * mm, 1)))
        for i in range(0, k, step):
            d = np.abs(emb[i:i + step, None, :] - emb[None, :, :]).max(axis=2)
            counts[i:i + step] = np.sum(d <= r, axis=1)
        return float(np.mean(np.log(counts / k)))

    return float(phi(m) - phi(m + 1))


def _resample(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    step = max(1, int(round(src_hz / dst_hz)))
    return x[::step]


def _spectral_entropy(x: np.ndarray, fs: float = FPS) -> float:
    """Normalised Shannon entropy of the power spectrum: high for broadband, hesitant motion."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 64 or np.allclose(x, 0):
        return float("nan")
    from scipy.signal import welch
    _, p = welch(x, fs=fs, nperseg=min(512, len(x)))
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def _log_dimensionless_jerk(vel: np.ndarray, fs: float = FPS) -> float:
    """-log of the dimensionless squared jerk of the velocity profile (higher = smoother)."""
    speed = _mag(vel)
    if len(speed) < 5 or speed.max() <= 0:
        return float("nan")
    dt = 1.0 / fs
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    integral = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt)
    duration = len(speed) * dt
    peak = float(speed.max())
    if integral <= 0:
        return float("nan")
    return float(-np.log((duration ** 3 / peak ** 2) * integral))


# ----------------------------------------------------------------------
# feature blocks
# ----------------------------------------------------------------------

def simple_features(X: np.ndarray) -> dict[str, float]:
    """K-simple: duration, path length, velocity and gripper statistics (both slave arms)."""
    n = X.shape[0]
    f: dict[str, float] = {"duration_s": n / FPS}
    speeds = {}
    for name in SLAVES:
        m = manipulator(X, name)
        pos, vel, rotvel, grip = m["pos"], m["vel"], m["rotvel"], m["grip"]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path = float(step.sum())
        speed = _mag(vel)
        speeds[name] = speed
        span = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
        gs = _smooth(grip, int(0.5 * FPS))
        f.update({
            f"{name}_path_len": path,
            f"{name}_speed_mean": float(speed.mean()),
            f"{name}_speed_std": float(speed.std()),
            f"{name}_speed_p90": float(np.percentile(speed, 90)),
            f"{name}_rotspeed_mean": float(_mag(rotvel).mean()),
            f"{name}_rotspeed_std": float(_mag(rotvel).std()),
            f"{name}_idle_frac": float(np.mean(speed < IDLE_SPEED)),
            f"{name}_economy": float(path / span) if span > 0 else float("nan"),
            f"{name}_grip_mean": float(grip.mean()),
            f"{name}_grip_std": float(grip.std()),
            f"{name}_grip_range": float(grip.max() - grip.min()),
            f"{name}_grip_total_change": float(np.abs(np.diff(grip)).sum()),
            f"{name}_grip_cycles": float(_upward_crossings(gs)),
        })
    sl, sr = speeds["SL"], speeds["SR"]
    f["bimanual_speed_corr"] = float(np.corrcoef(sl, sr)[0, 1])
    f["bimanual_idle_both"] = float(np.mean((sl < IDLE_SPEED) & (sr < IDLE_SPEED)))
    f["log_path_ratio"] = float(np.log((f["SL_path_len"] + 1e-9) / (f["SR_path_len"] + 1e-9)))
    return f


def texture_features(X: np.ndarray) -> dict[str, float]:
    """K-texture: regularity and smoothness of the movement (motion texture)."""
    f: dict[str, float] = {}
    for name in SLAVES:
        m = manipulator(X, name)
        speed = _mag(m["vel"])
        rotspeed = _mag(m["rotvel"])
        grip = m["grip"]
        f[f"{name}_apen_speed"] = _approx_entropy(_resample(speed, FPS, APEN_HZ))
        f[f"{name}_apen_rotspeed"] = _approx_entropy(_resample(rotspeed, FPS, APEN_HZ))
        f[f"{name}_apen_grip"] = _approx_entropy(_resample(grip, FPS, APEN_HZ))
        f[f"{name}_ldj"] = _log_dimensionless_jerk(m["vel"])
        f[f"{name}_spec_ent_speed"] = _spectral_entropy(speed)
    return f
