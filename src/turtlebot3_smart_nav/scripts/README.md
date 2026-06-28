# Scripts — Code Architecture & Implementation Guide

> This file belongs at `src/turtlebot3_smart_nav/scripts/README.md` in the repo so GitHub renders it automatically when the scripts folder is opened.

This document covers the internal structure of the `turtlebot3_smart_nav` ROS2 package — what each script does, how the five DRL algorithms are implemented, and how to extend the project.

For setup and running instructions, see the [root README](../../../../README.md).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Docker Container                            │
│                                                                      │
│   Gazebo 11 Simulator                                                │
│   (TurtleBot3 Burger in obstacle environment)                        │
│              │                                                       │
│              │  /scan  <-- Velodyne LiDAR point cloud               │
│              │              filtered to N evenly-spaced readings     │
│              ▼                                                       │
│       env_bridge.py  (rclpy Node)                                    │
│              │                                                       │
│              │  state = [LiDAR readings | goal angle | distance]    │
│              ▼                                                       │
│       DRL Algorithm                                                  │
│       (DQN / DDPG / TD3 / SAC / CrossQ)                             │
│              │                                                       │
│              │  action = [linear_vel, angular_vel]                   │
│              ▼                                                       │
│       /cmd_vel  -->  robot moves in Gazebo                          │
│              │                                                       │
│              ▼                                                       │
│       Reward Signal                                                  │
│         +reward   : moved closer to goal                             │
│         -penalty  : collision detected (LiDAR ray < threshold)      │
│         +large    : goal reached, episode ends                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Sensor Visualisation in RViz2

![Velodyne RViz](../../../media/velodyne.png)

The image above shows the Velodyne LiDAR point cloud rendered in RViz2. Coloured segments represent obstacle distances detected in real time. This raw sensor data is what `env_bridge.py` converts into the agent's state vector at every timestep.

---

## File-by-File Breakdown

### env_bridge.py — Environment Interface

**Role:** The bridge between Gazebo (ROS2) and PyTorch. Implemented as a `rclpy` Node.

**Responsibilities:**
- Subscribes to `/scan` to receive raw Velodyne LiDAR data
- Downsamples 360 degree rays to N evenly-spaced readings to reduce input dimensionality
- Detects collisions when any LiDAR ray falls below a minimum safe distance
- Publishes velocity commands to `/cmd_vel` to move the robot
- Computes the per-step reward based on goal distance and collision status
- Resets the robot's pose in Gazebo at the start of each new episode

**Outputs passed to the agent at each step:**

| Output | Type | Description |
|--------|------|-------------|
| `state` | Tensor | LiDAR readings + relative goal angle + distance to goal |
| `reward` | Float | Scalar reward signal |
| `done` | Bool | True if goal reached or collision occurred |

---

### main_train.py — Training Loop

**Role:** Main entry point for training. Orchestrates the full environment-agent interaction loop.

**What it does:**
- Reads `config/training_params.yaml` to select the active algorithm
- Instantiates the correct algorithm class from `algorithms/`
- Runs the episode loop: reset → act → store transition → update
- Logs episode reward, step count, and success or failure to the terminal
- Saves model weights to `model/` at a configurable checkpoint interval

**Core loop:**

```python
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
```

---

### evaluate_agent.py — Evaluation Loop

**Role:** Loads pre-trained weights and runs the agent in pure inference mode with no weight updates.

**What it does:**
- Loads `.pt` model files from `model/` for the selected algorithm
- Runs the robot in deterministic mode with no exploration noise
- Reports success rate, average reward, and average steps over N evaluation episodes

---

## The Five DRL Algorithms

Each algorithm lives in its own file under `algorithms/` and exposes a common interface used by `main_train.py`:

- `select_action(state)` — returns an action given the current state
- `store_transition(s, a, r, s_, done)` — adds experience to the replay buffer
- `update()` — performs one gradient update step

Switch between algorithms by changing one line in `config/training_params.yaml` — no other code changes needed.

---

### DQN — Deep Q-Network

| Property | Detail |
|----------|--------|
| Action space | Discrete — fixed set of pre-defined velocity command pairs |
| Architecture | Single Q-network mapping state to Q-value per discrete action |
| Exploration | Epsilon-greedy, decays over episodes |
| Key mechanisms | Experience replay buffer + periodic hard target network updates |
| Role in project | Simplest baseline; establishes a performance lower bound |

---

### DDPG — Deep Deterministic Policy Gradient

| Property | Detail |
|----------|--------|
| Action space | Continuous — direct linear and angular velocity values |
| Architecture | Actor network + Critic network (separate) |
| Exploration | Ornstein-Uhlenbeck noise added to actions during training |
| Key mechanisms | Deterministic policy gradient; soft target network updates (Polyak averaging) |
| Role in project | Continuous-action baseline; known to be sensitive to hyperparameters |

---

### TD3 — Twin Delayed DDPG

| Property | Detail |
|----------|--------|
| Action space | Continuous |
| Architecture | One Actor + two Critic networks; target networks for all three |
| Exploration | Clipped Gaussian noise added to target actions |
| Key mechanisms | Takes minimum of two Q-values to reduce overestimation; delays actor updates |
| Role in project | More stable than DDPG; strong benchmark |

---

### SAC — Soft Actor-Critic

| Property | Detail |
|----------|--------|
| Action space | Continuous |
| Architecture | Stochastic Actor (outputs mean and std) + two Critic networks |
| Exploration | Built-in via entropy maximisation — no manual noise tuning required |
| Key mechanisms | Maximises both reward and policy entropy; automatic entropy coefficient |
| Role in project | Primary algorithm — best balance of stability and sample efficiency |

---

### CrossQ — Batch-Normalised Q-Learning

| Property | Detail |
|----------|--------|
| Action space | Continuous |
| Architecture | Actor + Critic networks with Batch Normalisation layers inside Q-networks |
| Exploration | Built-in, similar to SAC |
| Key mechanisms | Shares a data batch across current and target networks; far fewer target updates needed |
| Role in project | State-of-the-art comparison — strong performance with fewer environment interactions |

---

## Dependencies

All packages are installed automatically inside the Docker container.

| Package | Version | Purpose |
|---------|---------|---------|
| `rclpy` | ROS2 Foxy | `env_bridge.py` — ROS2 Node, topic pub/sub, Gazebo control |
| `torch` | 1.10.0 or higher | All algorithms — neural network training |
| `numpy` | 1.21.0 or higher | `env_bridge.py` — state array construction |
| `pyyaml` | 6.0 or higher | `main_train.py` — reading config files |

Note: `rclpy` is installed as part of the ROS2 Foxy Docker image, not via pip.

---

## Extending the Project

To add a new DRL algorithm:

1. Create `algorithms/your_algorithm.py` implementing `select_action()`, `store_transition()`, and `update()`
2. Add your algorithm name as a new option in `config/training_params.yaml`
3. Add an import and an `elif` branch in `main_train.py` where algorithms are instantiated
4. Run `docker-compose up` — no other changes needed
