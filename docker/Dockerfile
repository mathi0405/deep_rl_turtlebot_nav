# Base Image: Official ROS Noetic Desktop Full
FROM osrf/ros:noetic-desktop-full

# Set non-interactive timezone to prevent apt-get hanging
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Update system and install essential build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-turtlebot3 \
    ros-noetic-turtlebot3-msgs \
    ros-noetic-turtlebot3-simulations \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Deep Learning Dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio numpy pyyaml tensorboard

# Set up the Catkin Workspace
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS

# Copy the custom entrypoint script
COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

# Set the TurtleBot model environment variable
ENV TURTLEBOT3_MODEL=burger

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]
