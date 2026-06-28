#!/bin/bash
set -e

# Source standard ROS environment
source /opt/ros/noetic/setup.bash

# Build the workspace if it hasn't been built yet
if [ ! -d "/root/catkin_ws/devel" ]; then
    echo "Building Catkin Workspace..."
    cd /root/catkin_ws
    catkin_make
fi

# Source the custom workspace
source /root/catkin_ws/devel/setup.bash

# Execute whatever command was passed to Docker
exec "$@"
