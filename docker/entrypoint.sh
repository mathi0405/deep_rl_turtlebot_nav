#!/bin/bash
set -eo pipefail
# ROS setup scripts reference unset variables, so nounset is enabled only after sourcing.
source /opt/ros/noetic/setup.bash
source /workspace/install/setup.bash
set -u
exec "$@"
