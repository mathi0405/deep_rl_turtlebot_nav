cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_smart_nav)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs gazebo_msgs)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/env_bridge.py
  scripts/main_train.py
  scripts/evaluate_agent.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
