"""Small deterministic 2-D unicycle worlds for CPU training and integration checks.

Not Gazebo, a TurtleBot dynamics model or evidence of sim-to-real transfer.
Each world here is the single source of geometry: the Gazebo warehouse world
file is generated from the same definition (scripts/generate_world.py).
"""
from __future__ import annotations

import math
from collections import namedtuple
import numpy as np
from .sensors import observation, transition_reward, velocity, wrap_angle

# half_x/half_y: interior half-extents of the walled area (m).
# circles: (x, y, radius); boxes: (centre_x, centre_y, half_width_x, half_width_y).
# min_separation: minimum straight-line start-goal distance for a sampled episode.
World = namedtuple('World', 'name backend half_x half_y circles boxes min_separation')

NAVIGATION = World(
    name='navigation', backend='cpu-kinematic-v1', half_x=2.5, half_y=2.5,
    circles=((0.0, 0.0, 0.35), (1.1, -0.8, 0.28), (-1.0, 0.8, 0.28)),
    boxes=(), min_separation=0.8)

# 8 m x 6 m warehouse: four shelving racks forming two long aisles and a cross
# aisle, pallets and barrels in the loading area, and a column. Aisles are at
# least 1.1 m wide for a 0.18 m-wide TurtleBot3 Burger.
WAREHOUSE = World(
    name='warehouse', backend='cpu-kinematic-warehouse-v1', half_x=4.0, half_y=3.0,
    circles=((0.0, 0.70, 0.22), (-3.45, -0.95, 0.20), (2.45, -2.45, 0.20),
             (0.45, 2.40, 0.20), (-1.30, -2.40, 0.20)),
    boxes=((-2.0, 1.60, 1.30, 0.25), (2.0, 1.60, 1.30, 0.25),
           (-2.0, -0.20, 1.30, 0.25), (2.0, -0.20, 1.30, 0.25),
           (-2.55, -1.90, 0.30, 0.30), (-0.55, -1.55, 0.30, 0.30),
           (1.35, -2.10, 0.30, 0.30), (3.15, -1.35, 0.30, 0.30),
           (3.55, 2.55, 0.20, 0.20)),
    min_separation=2.0)

WORLDS = {w.name: w for w in (NAVIGATION, WAREHOUSE)}

# Backwards-compatible names for the original world.
BOUNDS = NAVIGATION.half_x
OBSTACLES = NAVIGATION.circles


def get_world(world):
    if isinstance(world, World):
        return world
    if world not in WORLDS:
        raise ValueError('world must be one of: '+', '.join(sorted(WORLDS)))
    return WORLDS[world]


def box_distance(x, y, box):
    cx, cy, hx, hy = box
    return math.hypot(max(abs(x - cx) - hx, 0.0), max(abs(y - cy) - hy, 0.0))


def clear_position(x, y, margin=0.30, world=NAVIGATION):
    world = get_world(world)
    return (abs(x) < world.half_x - margin and abs(y) < world.half_y - margin
            and all(math.hypot(x - ox, y - oy) > radius + margin
                    for ox, oy, radius in world.circles)
            and all(box_distance(x, y, box) > margin for box in world.boxes))


def episode_layout(seed, world=NAVIGATION):
    """Random start pose and goal point in free space, reproducible from the seed."""
    world = get_world(world)
    rng = np.random.default_rng(seed)
    points = []
    span = (world.half_x - 0.5, world.half_y - 0.5)
    for _ in range(10000):
        p = rng.uniform((-span[0], -span[1]), span) if world is not NAVIGATION else rng.uniform(-2.0, 2.0, 2)
        if clear_position(*p, world=world) and (not points or np.linalg.norm(p - points[0]) > world.min_separation):
            points.append(p)
            if len(points) == 2:
                return np.array([*points[0], rng.uniform(-math.pi, math.pi)]), points[1]
    raise RuntimeError("could not sample an episode")


def ray_distances(x, y, angles, world, max_range=3.5):
    """Exact ray casting against walls, cylinders and axis-aligned boxes."""
    dx, dy = np.cos(angles), np.sin(angles)
    result = np.full(len(angles), max_range)
    with np.errstate(divide='ignore', invalid='ignore'):
        for component, position, bound in [(dx, x, world.half_x), (dy, y, world.half_y)]:
            for boundary in [-bound, bound]:
                t = np.full(len(angles), np.inf)
                np.divide(boundary - position, component, out=t, where=np.abs(component) > 1e-12)
                result = np.minimum(result, np.where(t > 0, t, np.inf))
        for ox, oy, radius in world.circles:
            rx, ry = x - ox, y - oy
            b = dx * rx + dy * ry
            discriminant = b * b - (rx * rx + ry * ry - radius * radius)
            t = -b - np.sqrt(np.maximum(discriminant, 0))
            result = np.minimum(result, np.where((discriminant >= 0) & (t > 0), t, np.inf))
        for cx, cy, hx, hy in world.boxes:
            tx1, tx2 = (cx - hx - x) / dx, (cx + hx - x) / dx
            ty1, ty2 = (cy - hy - y) / dy, (cy + hy - y) / dy
            # A ray parallel to a slab either always or never lies inside it.
            inside_x = (x > cx - hx) & (x < cx + hx)
            inside_y = (y > cy - hy) & (y < cy + hy)
            lo_x = np.where(np.abs(dx) > 1e-12, np.minimum(tx1, tx2), np.where(inside_x, -np.inf, np.inf))
            hi_x = np.where(np.abs(dx) > 1e-12, np.maximum(tx1, tx2), np.where(inside_x, np.inf, -np.inf))
            lo_y = np.where(np.abs(dy) > 1e-12, np.minimum(ty1, ty2), np.where(inside_y, -np.inf, np.inf))
            hi_y = np.where(np.abs(dy) > 1e-12, np.maximum(ty1, ty2), np.where(inside_y, np.inf, -np.inf))
            near, far = np.maximum(lo_x, lo_y), np.minimum(hi_x, hi_y)
            hit = (far >= near) & (far > 0)
            result = np.minimum(result, np.where(hit, np.where(near > 0, near, 1e-6), np.inf))
    return np.maximum(result, 1e-6)


class KinematicEnv:
    def __init__(self, max_steps=250, bins=24, dt=0.2, world='navigation'):
        if max_steps < 1 or bins < 4 or not 0 < dt <= 0.25:
            raise ValueError("invalid environment dimensions")
        self.world = get_world(world)
        self.backend = self.world.backend
        self.max_steps, self.bins, self.dt = max_steps, bins, dt
        self.steps = 0
        self.finished = True

    def scan(self):
        angles = self.pose[2] + np.linspace(-math.pi, math.pi, self.bins, endpoint=False)
        return ray_distances(self.pose[0], self.pose[1], angles, self.world)

    def reset(self, seed):
        self.pose, self.goal = episode_layout(seed, self.world)
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
        geometry_contact = (not clear_position(*self.pose[:2], margin=0.105, world=self.world))
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
