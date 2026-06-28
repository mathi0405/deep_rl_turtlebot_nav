# deep_rl_turtlebot_nav
# 🐢🧠 Deep Reinforcement Learning for TurtleBot3 Navigation

Welcome to the Master's project repository for Autonomous Mapless Navigation. 

This project trains a simulated TurtleBot3 to navigate through a complex, obstacle-dense maze using raw LiDAR data. Instead of using standard SLAM or pathfinding, the robot learns to drive using **five distinct Deep Reinforcement Learning (DRL) algorithms**:
1. **DQN** (Deep Q-Network)
2. **DDPG** (Deep Deterministic Policy Gradient)
3. **TD3** (Twin Delayed DDPG)
4. **SAC** (Soft Actor-Critic)
5. **CrossQ** (Batch-Normalized Q-Learning)

---

## 🚀 How to Run the Project (Zero-Setup Method)

To make evaluation as easy as possible, this entire project is containerized. **You do not need to install ROS, Gazebo, or PyTorch on your local machine.** As long as you have [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed on an Ubuntu/Linux machine, follow these 3 steps:

### Step 1: Allow the Simulation GUI to open on your screen
Open a terminal and run this command so Docker is allowed to pop open the Gazebo window:
```bash
xhost +local:root
### Step 2: Open the project folder
Navigate into the root folder of this repository where the docker-compose.yml file is located:

Bash
cd path/to/deep_rl_turtlebot_nav
### Step 3: Launch the magic command
Run the following command. It will automatically build the ROS environment, install PyTorch, compile the workspace, and launch the training simulation:

Bash
docker-compose up --build
(Note: The first time you run this, it will take a few minutes to download the ROS Noetic Docker image. Subsequent runs will be instant).

🎛️ How to Control the Experiment
We have externalized all the complex settings so you don't have to dig through Python code to change how the robot behaves.

1. How to change the AI Algorithm
By default, the robot uses SAC. If you want to watch a different algorithm (like DQN or TD3) drive the robot:

Open this file: src/turtlebot3_smart_nav/config/training_params.yaml

Look at line 3: active_algorithm: "SAC"

Change "SAC" to "DQN", "DDPG", "TD3", or "CrossQ".

Save the file and run docker-compose up. The system will automatically load the new neural network!

2. How to switch from "Training" to "Testing"
Right now, the robot is placed in the maze to learn from scratch (Training Mode). If you want to load our pre-trained neural network weights and watch the robot navigate flawlessly (Testing Mode):

Open the docker-compose.yml file in the root folder.

Scroll to the very bottom to the command: line.

Change mode:=train to mode:=test.

Save the file and run docker-compose up.

📂 Repository Blueprint
If you wish to explore the code, here is how the architecture is cleanly separated:

Plaintext
├── docker/                      # 🐳 Contains the custom Dockerfile that builds ROS + PyTorch
├── src/turtlebot3_smart_nav/    # 🤖 The actual ROS Package
│   ├── config/                  # ⚙️ Yaml files (Change your algorithm here!)
│   ├── launch/                  # 🚀 ROS Launch files linking Gazebo and Python
│   └── scripts/                 
│       ├── env_bridge.py        # 🌉 Translates Gazebo LiDAR data into PyTorch Tensors
│       ├── main_train.py        # 🏋️ The main execution loop for training
│       ├── evaluate_agent.py    # 🎯 The execution loop for testing pre-trained models
│       └── algorithms/          # 🧠 The 5 PyTorch Neural Network architectures isolated here
🛑 Troubleshooting
Gazebo opens but the screen is black: This is a common Docker graphics issue. Press Ctrl+C in your terminal to stop the container, run xhost +local:root again, and restart with docker-compose up.

Robot is spinning in circles: If running in train mode, this is normal! The robot is exploring its continuous action space. It will begin to figure out forward momentum after a few hundred episodes.
