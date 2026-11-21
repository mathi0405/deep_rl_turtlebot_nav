# Gazebo (ROS1 Noetic) execution results

Executed 22 November 2026 in the repository container (`docker compose`, headless
`gzserver`, TurtleBot3 Burger, `navigation.world`) on a Windows 11 host with Docker
Desktop. Runtime inside the container: Python 3.8.10, NumPy 1.24.4, PyTorch 2.2.2+cpu.
Machine-readable evidence, including policy, source and trajectory hashes and every
episode outcome, is in [`gazebo_evidence.json`](gazebo_evidence.json).

## Container fixes required to run the documented path

The container path had not been executed before this run. Three defects prevented it:

1. `docker/entrypoint.sh` used `set -u` before sourcing ROS. The ROS setup scripts read
   unset variables, so the container exited immediately. `nounset` is now enabled only
   after sourcing.
2. Ubuntu 20.04's pip 20 and setuptools 45 could not read the PEP 621 metadata in
   `pyproject.toml`. The package installed as `UNKNOWN` and the `turtlebot-nav` command
   was missing. The image now upgrades pip and builds with isolation, and the build
   fails if the command is missing.
3. `docker compose exec` bypasses the entrypoint, so the README commands ran without
   ROS sourced. The README now routes them through `/workspace/docker/entrypoint.sh`.

## Results

All Gazebo evaluations use the same 20 start/goal layouts (episode seeds
100000–100019), deterministic actor-mean actions and at most 250 steps per episode.

| Policy | Trained in | Evaluated in | Success | Obstacle stop | Timeout |
|---|---|---|---:|---:|---:|
| SAC, seed 7, 30,000 steps | CPU 2-D simulator | CPU 2-D simulator (50 episodes) | 49/50 | 1 | 0 |
| SAC, seed 7, 30,000 steps (`--allow-backend-transfer`) | CPU 2-D simulator | **Gazebo** | **19/20** | 1 | 0 |
| SAC, seed 7, 10,000 steps | Gazebo | Gazebo | 0/20 | 8 | 12 |
| Declared geometric baseline | – | Gazebo | 7/20 | 3 | 10 |

Observations:
- The policy trained in the CPU simulator transferred to Gazebo with 19/20 successes on
  these layouts. Its only failure, an obstacle stop, occurred on the same layout
  (seed 100015) in both the CPU and Gazebo evaluations. That makes it a consistent
  policy weakness, not a Gazebo-specific error.
- Retraining seed 7 inside this container gave 49/50 on the frozen CPU suite. The
  published November CPU result for seed 7 was 50/50 under a different Python/PyTorch
  runtime. This matches the documented limit that reproducibility holds only within a
  recorded runtime.
- 10,000 steps of training directly in Gazebo did not produce a working policy (0/20).
  This is a single short run and does not say how longer Gazebo training would do.
- The geometric baseline reached 7/20 in Gazebo, compared with 20/50 in the CPU
  simulator.

## Limits

These are 20 episodes in one fixed Gazebo world with one robot model, run once per
configuration. They show that the ROS1 adapter, the Gazebo services and checkpoint
transfer work end to end. They do not show robustness across worlds, sensor noise,
dynamic obstacles or physical hardware. Hardware execution remains unverified.
Run artifacts (`runs/`) are not checked in; their hashes are recorded in the evidence file.
