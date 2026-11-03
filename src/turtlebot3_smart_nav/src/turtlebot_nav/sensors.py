"""Shared observation and command contracts, independent of ROS."""
from __future__ import annotations

import math
import numpy as np


class SensorError(RuntimeError):
    """Input cannot safely be interpreted as a valid observation."""


def lidar_bins(ranges, bins=24, max_range=3.5):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if not math.isfinite(max_range) or max_range <= 0:
        raise ValueError("max_range must be finite and positive")
    raw = np.asarray(ranges, dtype=np.float64)
    if raw.ndim != 1 or raw.size < bins:
        raise SensorError("scan must be one-dimensional with at least one reading per bin")
    if np.isnan(raw).any() or np.isneginf(raw).any() or (raw <= 0).any():
        raise SensorError("scan contains NaN, negative infinity or nonpositive ranges")
    raw = np.minimum(raw, max_range)  # +inf is the valid no-return convention.
    return np.asarray([part.min() for part in np.array_split(raw, bins)], dtype=np.float32)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def observation(scan, pose, goal, bins=24, max_range=3.5):
    values = np.asarray([*pose, *goal], dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise SensorError("pose and goal must contain finite x/y/yaw and x/y values")
    scan = lidar_bins(scan, bins, max_range)
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    distance = math.hypot(dx, dy)
    bearing = wrap_angle(math.atan2(dy, dx) - pose[2])
    state = np.concatenate((scan / max_range, [min(distance / 8.0, 1.0), bearing / math.pi]))
    return state.astype(np.float32), distance


def velocity(action, max_linear=0.22, max_angular=2.0):
    a = np.asarray(action, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or (np.abs(a) > 1.000001).any():
        raise ValueError("action must be two finite normalized values in [-1,1]")
    a = np.clip(a, -1, 1)
    return float((a[0] + 1) * 0.5 * max_linear), float(a[1] * max_angular)


def transition_reward(previous_distance, distance, obstacle, reached, timeout):
    reward = 5.0 * (previous_distance - distance) - 0.02
    if obstacle:
        return -10.0, True, False, "obstacle_stop"
    if reached:
        return reward + 10.0, True, False, "success"
    if timeout:
        return reward - 2.0, False, True, "timeout"
    return reward, False, False, "running"
