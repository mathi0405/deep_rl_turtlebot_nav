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
        
        self.action_space = [
            [0.18,  0.0],   
            [0.12,  0.75],  
            [0.12, -0.75],  
            [0.05,  1.5],   
            [0.05, -1.5]    
        ]

    def get_lidar_state(self, scan_data):
        raw_ranges = np.array(scan_data.ranges)
        raw_ranges[raw_ranges == np.inf] = 3.5
        
        downsampled = []
        step = len(raw_ranges) // self.num_scan_samples
        for i in range(0, len(raw_ranges), step):
            sector = raw_ranges[i : i + step]
            downsampled.append(np.min(sector))
            
        return np.array(downsampled[:self.num_scan_samples])

    def respawn_goal(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        new_state = ModelState()
        new_state.model_name = 'target_goal'
        new_state.pose.position.x = random.uniform(-2.0, 2.0)
        new_state.pose.position.y = random.uniform(-2.0, 2.0)
        new_state.pose.position.z = 0.0
        try:
            self.state_service(new_state)
        except rospy.ServiceException:
            pass

    def step(self, action_index):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_sim()
        except rospy.ServiceException:
            pass
            
        vel_cmd = Twist()
        vel_cmd.linear.x = self.action_space[action_index][0]
        vel_cmd.angular.z = self.action_space[action_index][1]
        self.velocity_publisher.publish(vel_cmd)
        
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=0.5)
            except:
                pass
                
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_sim()
        except rospy.ServiceException:
            pass
            
        state = self.get_lidar_state(data)
        min_distance = np.min(state)
        done = bool(min_distance < 0.18)
        reward = -25.0 if done else 0.1
        
        return state, reward, done

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            pass
            
        self.respawn_goal()
        data = rospy.wait_for_message('/scan', LaserScan)
        return self.get_lidar_state(data)
