"""Small deterministic 2-D unicycle world for CPU integration checks.

Not Gazebo, a TurtleBot dynamics model or evidence of sim-to-real transfer.
"""
from __future__ import annotations

import math
import numpy as np
from .sensors import observation, transition_reward, velocity, wrap_angle

BOUNDS = 2.5
OBSTACLES = ((0.0, 0.0, 0.35), (1.1, -0.8, 0.28), (-1.0, 0.8, 0.28))


def clear_position(x, y, margin=0.30):
    return (abs(x) < BOUNDS - margin and abs(y) < BOUNDS - margin
            and all(math.hypot(x - ox, y - oy) > radius + margin
                    for ox, oy, radius in OBSTACLES))


def episode_layout(seed):
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(10000):
        p = rng.uniform(-2.0, 2.0, 2)
        if clear_position(*p) and (not points or np.linalg.norm(p - points[0]) > 0.8):
            points.append(p)
            if len(points) == 2:
                return np.array([*points[0], rng.uniform(-math.pi, math.pi)]), points[1]
    raise RuntimeError("could not sample an episode")


class KinematicEnv:
    backend = "cpu-kinematic-v1"

    def __init__(self, max_steps=250, bins=24, dt=0.2):
        if max_steps < 1 or bins < 4 or not 0 < dt <= 0.25:
            raise ValueError("invalid environment dimensions")
        self.max_steps, self.bins, self.dt = max_steps, bins, dt
        self.steps = 0
        self.finished = True

    def scan(self):
        angles = self.pose[2] + np.linspace(-math.pi, math.pi, self.bins, endpoint=False)
        dx, dy = np.cos(angles), np.sin(angles)
        x, y = self.pose[:2]
        result = np.full(self.bins, 3.5)
        for component, position in [(dx, x), (dy, y)]:
            for boundary in [-BOUNDS, BOUNDS]:
                t = np.full(self.bins, np.inf)
                np.divide(boundary - position, component, out=t, where=np.abs(component) > 1e-12)
                result = np.minimum(result, np.where(t > 0, t, np.inf))
        for ox, oy, radius in OBSTACLES:
            rx, ry = x - ox, y - oy
            b = dx * rx + dy * ry
            discriminant = b * b - (rx * rx + ry * ry - radius * radius)
            t = -b - np.sqrt(np.maximum(discriminant, 0))
            result = np.minimum(result, np.where((discriminant >= 0) & (t > 0), t, np.inf))
        return np.maximum(result, 1e-6)

    def reset(self, seed):
        self.pose, self.goal = episode_layout(seed)
        self.steps, self.finished = 0, False
        state, self.distance = observation(self.scan(), self.pose, self.goal, self.bins)
        return state

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset required before step")
        linear, angular = velocity(action)
        # Midpoint integration; finite, modest control timestep only.
        mid = self.pose[2] + angular * self.dt / 2
        self.pose[0] += linear * math.cos(mid) * self.dt
        self.pose[1] += linear * math.sin(mid) * self.dt
        self.pose[2] = wrap_angle(self.pose[2] + angular * self.dt)
        self.steps += 1
        scan = self.scan()
        state, distance = observation(scan, self.pose, self.goal, self.bins)
        geometry_contact = (not clear_position(*self.pose[:2], margin=0.105))
        obstacle = geometry_contact or float(scan.min()) < 0.15
        reward, terminated, truncated, outcome = transition_reward(
            self.distance, distance, obstacle, distance <= 0.20, self.steps >= self.max_steps)
        self.distance = distance
        self.finished = terminated or truncated
        info = {"outcome": outcome, "distance_m": distance, "steps": self.steps,
                "pose": self.pose.tolist(), "goal": self.goal.tolist(),
                "minimum_lidar_m": float(scan.min())}
        return state, reward, terminated, truncated, info

    def close(self):
        self.finished = True
