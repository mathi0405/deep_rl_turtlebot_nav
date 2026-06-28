#!/usr/bin/env python3
import numpy as np
import rospy
import random
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class TurtleBot3NavigationEnv:
    def __init__(self, num_scan_samples=24):
        self.num_scan_samples = num_scan_samples
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        self.discrete_action_space = [[0.18, 0.0], [0.12, 0.75], [0.12, -0.75], [0.05, 1.5], [0.05, -1.5]]

    def get_lidar_state(self, scan_data):
        raw = np.array(scan_data.ranges)
        raw[raw == np.inf] = 3.5
        downsampled = []
        step = len(raw) // self.num_scan_samples
        for i in range(0, len(raw), step):
            downsampled.append(np.min(raw[i : i + step]))
        return np.array(downsampled[:self.num_scan_samples])

    def respawn_goal(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        state = ModelState()
        state.model_name = 'target_goal'
        state.pose.position.x = random.uniform(-2.0, 2.0)
        state.pose.position.y = random.uniform(-2.0, 2.0)
        try: self.state_service(state)
        except: pass

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try: self.unpause_sim()
        except: pass
            
        vel_cmd = Twist()
        # Handle Continuous Arrays vs Discrete Ints
        if isinstance(action, (list, np.ndarray, tuple)):
            vel_cmd.linear.x = max(0.0, action[0] * 0.22) # Prevent reverse driving
            vel_cmd.angular.z = action[1] * 2.0
        else:
            vel_cmd.linear.x = self.discrete_action_space[action][0]
            vel_cmd.angular.z = self.discrete_action_space[action][1]
            
        self.velocity_publisher.publish(vel_cmd)
        
        data = None
        while data is None and not rospy.is_shutdown():
            try: data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except: pass
                
        rospy.wait_for_service('/gazebo/pause_physics')
        try: self.pause_sim()
        except: pass
            
        state = self.get_lidar_state(data)
        done = bool(np.min(state) < 0.18)
        reward = -25.0 if done else 0.1
        
        return state, reward, done

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try: self.reset_proxy()
        except: pass
        self.respawn_goal()
        data = rospy.wait_for_message('/scan', LaserScan)
        return self.get_lidar_state(data)
