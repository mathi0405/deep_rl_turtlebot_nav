#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
if [ ! -d "/root/catkin_ws/devel" ]; then
    echo "Building Catkin Workspace..."
    cd /root/catkin_ws && catkin_make
fi
source /root/catkin_ws/devel/setup.bash
exec "$@"
