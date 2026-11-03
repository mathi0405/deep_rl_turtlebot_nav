cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_smart_nav)

find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs geometry_msgs nav_msgs gazebo_msgs std_srvs)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/main_train.py
  scripts/evaluate_agent.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch worlds DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
