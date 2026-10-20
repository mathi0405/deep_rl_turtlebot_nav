# TurtleBot3 reinforcement-learning navigation prototype

Python / PyTorch network definitions with a **ROS1 Noetic + Gazebo** environment bridge. This public checkout is an unfinished prototype; it is not currently a reproducible five-algorithm training benchmark.

## Implementation status

- Network definitions are provided for DQN, DDPG, TD3, SAC and CrossQ-style agents.
- `env_bridge.py` uses `rospy`, publishes velocity commands, reads LiDAR, and assigns collision/survival rewards.
- The training loop collects transitions but still has a TODO for optimization; it does not currently train the networks.
- Goal inputs are currently constants rather than odometry-derived distance and bearing. Goal-reaching reward and termination need implementation.
- The Dockerfile and Catkin package target ROS1 Noetic. Previous ROS2 Foxy instructions did not match the implementation.
- Evaluation needs a step limit, explicit missing-checkpoint failure, and machine-readable outcome metrics.
- External Drive links are provided in `model/` and `simulation_videos.txt`; hardware deployment and training results have not been independently verified from this checkout.
- No `paper/` directory is included in the current public tree.

## Container startup

On a Linux system with Docker and X11 configured:

```bash
git clone https://github.com/mathi0405/deep_rl_turtlebot_nav.git
cd deep_rl_turtlebot_nav
docker compose up --build
```

The container builds a Catkin workspace and launches `run_agent.launch` under ROS1. A successful simulation launch does not establish learning. The container startup correction has been reviewed statically; a complete Gazebo run still needs verification on a compatible host.

## Next implementation milestones

1. Implement and test one complete algorithm, including replay sampling, optimizer updates, exploration, target updates and checkpoint saving.
2. Add odometry-derived goal distance/bearing and goal-reaching termination.
3. Use identical held-out starts/goals for evaluation; report success, collisions, timeouts, path length and inference latency.
4. Publish exact dependencies, configurations and checkpoints with clear attribution.
5. Validate simulation before reporting any hardware-transfer result.

## Attribution

The project references [tomasvr/turtlebot3_drlnav](https://github.com/tomasvr/turtlebot3_drlnav) and [Reinis Cimurs's DRL robot navigation work](https://github.com/reiniscimurs/DRL-robot-navigation). Document specific reused components and your contributions when completing the implementation.

Mani Chandan Mathi — Master of Engineering student in Intelligent Robotics, Deggendorf Institute of Technology.
