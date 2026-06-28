cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_smart_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  gazebo_msgs
)

catkin_package()

# This ensures the professor can run your scripts via rosrun/roslaunch
catkin_install_python(PROGRAMS
  scripts/dqn_model.py
  scripts/navigation_env.py
  scripts/train_agent.py
  scripts/goal_respawner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
