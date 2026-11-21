# Historical network definitions

`network_definitions/` retains the original DQN/DDPG/TD3/SAC/CrossQ-style
PyTorch architecture sketches for attribution and comparison. They are not
supported training algorithms or the current policy checkpoint format.
The former scripts guide incorrectly described complete training updates,
ROS2 interfaces and successful evaluation. Those claims are retired.

The supported implementation is `turtlebot_nav.sac`, using ROS1 Noetic for
the optional Gazebo adapter and an independent CPU backend for integration
checks. Do not load historical Drive weights into the new architecture.

`training_params.yaml` and `root-package.xml` preserve the former configuration
and duplicate package metadata. They are not used by the new CLI or Catkin
package. Configure the current learner with explicit CLI arguments; resolved
configuration is written to each new experiment directory.
