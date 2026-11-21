FROM osrf/ros:noetic-desktop-full
ENV DEBIAN_FRONTEND=noninteractive TURTLEBOT3_MODEL=burger
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-catkin-pkg ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations ros-noetic-xacro && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy==1.24.4" "setuptools<76" wheel && pip3 install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /workspace
COPY . /workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make install"
RUN pip3 install --no-deps --no-build-isolation .
RUN chmod +x /workspace/docker/entrypoint.sh
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["roslaunch", "turtlebot3_smart_nav", "simulation.launch", "gui:=false"]
