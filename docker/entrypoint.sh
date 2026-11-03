#!/bin/bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
source /workspace/install/setup.bash
exec "$@"
