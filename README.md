# Reinforcement-Learning-Based-Smart-Navigation-for-TurtleBot

## 📌 Project Overview

This project implements and compares **five Deep Reinforcement Learning (DRL) algorithms** for autonomous, mapless navigation of a simulated TurtleBot3 robot. The robot learns to navigate through a complex, obstacle-dense Gazebo environment using only raw LiDAR sensor data — no pre-built maps, no SLAM, no hand-crafted path planners. Trained models are also validated on a **real-world physical TurtleBot3**.

| Algorithm | Type | Key Characteristic |
|-----------|------|--------------------|
| **DQN** | Value-based, Discrete | Deep Q-Network baseline |
| **DDPG** | Policy-based, Continuous | Actor-Critic with deterministic policy |
| **TD3** | Policy-based, Continuous | Twin-Delayed DDPG (reduces overestimation bias) |
| **SAC** | Policy-based, Continuous | Entropy-regularised, sample-efficient |
| **CrossQ** | Policy-based, Continuous | Batch-Normalised Q-Learning (state-of-the-art) |

> 📂 For a detailed breakdown of code architecture and each script, see [`src/turtlebot3_smart_nav/scripts/README.md`](src/turtlebot3_smart_nav/scripts/README.md)

---

## 🌍 Simulation Environment

The robot operates inside a custom Gazebo world containing mixed static obstacles — furniture, walls, and boxes. The blue fan visible above is the robot's live LiDAR scan, which is the **only sensor input** used by the DRL agent to make navigation decisions.

---

## 📦 Pre-trained Weights, Results & Videos

All resources are available via Google Drive, linked from the [`model/`](model/) folder:

| Resource | Description |
|----------|-------------|
| 🧠 **Pre-trained weights** | Trained `.pt` files for all 5 algorithms |
| 📊 **Training & evaluation curves** | Reward-per-episode plots for each algorithm |
| 🎥 **Real-world deployment videos** | Physical TurtleBot3 navigating with trained weights |

---

## 📄 Research Paper

A full research paper in **IEEE format** is available in the [`paper/`](paper/) folder of this repository.

---

## ⚙️ System Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Ubuntu 20.04 / 22.04 (Linux required for X11 GUI) |
| **Docker** | Version 20.10 or higher |
| **Docker Compose** | Version 1.29 or higher |
| **RAM** | Minimum 8 GB recommended |
| **GPU** | NVIDIA GPU with CUDA support recommended |
| **Disk Space** | ~5 GB (ROS2 Foxy image + PyTorch) |
| **Display** | X11 display server (standard on Ubuntu Desktop) |

> **Note:** Windows and macOS are not natively supported. WSL2 with an X server (e.g. VcXsrv) can be used on Windows with additional setup.

---

## 🚀 Quick Start — 3 Steps

The entire stack (ROS2 Foxy, Gazebo 11, PyTorch) is **fully containerised**. You do **not** need to install ROS2 or PyTorch on your host machine.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/mathi0405/deep_rl_turtlebot_nav.git
cd deep_rl_turtlebot_nav
```

### Step 2 — Allow GUI Forwarding

```bash
xhost +local:root
```

This lets Docker display the Gazebo simulation window on your screen.

### Step 3 — Build and Launch

```bash
docker-compose up --build
```

> ⏱️ **First run only:** Docker downloads the ROS2 Foxy base image and installs PyTorch (~5–10 minutes depending on internet speed). Subsequent runs start in seconds.

---

## 🎛️ Configuring the Experiment

All settings are controlled from a single config file — no Python code changes required.

### Change the DRL Algorithm

Open `src/turtlebot3_smart_nav/config/training_params.yaml`:

```yaml
active_algorithm: "SAC"   # Options: "DQN", "DDPG", "TD3", "CrossQ"
```

Save and re-run `docker-compose up`. The system loads the new neural network automatically.

### Switch Between Training and Evaluation Mode

Open `docker-compose.yml` and find the `command:` line at the bottom:

```yaml
# Training from scratch (default):
command: bash -c "source install/setup.bash && ros2 launch turtlebot3_smart_nav start_rl_agent.launch.py mode:=train"

# Evaluation with pre-trained weights:
command: bash -c "source install/setup.bash && ros2 launch turtlebot3_smart_nav start_rl_agent.launch.py mode:=test"
```

To use pre-trained weights, download them from the [`model/`](model/) folder and place them in `model/` before switching to `mode:=test`.

---

## 🔍 Expected Output

Once launched, you should see:

1. **Terminal** — Episode number, total reward, step count, and success/failure logged per episode
2. **Gazebo window** — TurtleBot3 (burger model) appears in the obstacle environment and begins moving
3. **Training mode** — Robot explores randomly at first; purposeful goal-directed navigation emerges after several hundred episodes
4. **Test mode** — Robot navigates using pre-trained weights and consistently reaches the goal

---

## 🤖 Real-World Deployment

Trained models transfer directly to a physical TurtleBot3 via ROS2 topic remapping — no retraining required. Real-world navigation videos are available in the [`model/`](model/) folder on Google Drive.

---

## 🛑 Troubleshooting

| Problem | Solution |
|---------|----------|
| Gazebo window is black | Press `Ctrl+C`, run `xhost +local:root` again, then `docker-compose up` |
| Robot spinning in circles (train mode) | Normal early exploration behaviour — wait a few hundred episodes |
| `docker-compose` not found | `sudo apt install docker-compose` |
| Permission denied on Docker socket | `sudo usermod -aG docker $USER`, log out and back in |
| Gazebo crashes on startup | Ensure ≥ 4 GB free RAM; run `docker-compose down` then `docker-compose up` |
| `xhost` command not found | `sudo apt install x11-xserver-utils` |
| ROS2 packages not found | Run `source install/setup.bash` in every new terminal after `colcon build` |

---

## 🙏 Acknowledgements

This project builds upon the ROS2 DRL navigation framework by [tomasvr/turtlebot3_drlnav](https://github.com/tomasvr/turtlebot3_drlnav), which itself extends the foundational work of [Reinis Cimurs](https://github.com/reiniscimurs/DRL-robot-navigation). We thank both authors for making their work open source.

---

## 👥 Authors

**Mani chandan Mathi** — DRL implementation, ROS2 integration, simulation & real-world deployment
M.Sc. Intelligent Robotics, Deggendorf Institute of Technology
GitHub: [@mathi0405](https://github.com/mathi0405)

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
