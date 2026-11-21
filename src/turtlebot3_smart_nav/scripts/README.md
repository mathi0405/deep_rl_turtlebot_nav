# ROS entry scripts

These scripts call the installed `turtlebot_nav` package. They contain no separate
algorithm implementation. `main_train.py` invokes the SAC training command;
`evaluate_agent.py` invokes bounded checkpoint evaluation. `rospy.myargv()`
removes ROS remapping arguments before CLI parsing.

The implementation is ROS1 Noetic, not ROS2. Full architecture, limitations and
CPU/Gazebo instructions are in the root README and `docs/ARCHITECTURE.md`.
Historical network sketches are in `legacy/network_definitions`, not a supported
five-algorithm training interface.
