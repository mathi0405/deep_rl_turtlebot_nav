"""
Gazebo Simulation Bridge and State Observation Wrapper.
"""
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

class TurtleBot3NavigationEnv:
    def __init__(self, num_scan_samples=24):
        rospy.init_node('turtlebot3_rl_env', anonymous=True)
        self.num_scan_samples = num_scan_samples
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        # Discrete Action Space Map: [linear_vel, angular_vel]
        self.action_space = [
            [0.18,  0.0],   # Forward
            [0.12,  0.75],  # Gentle Left
            [0.12, -0.75],  # Gentle Right
            [0.05,  1.5],   # Sharp Left
            [0.05, -1.5]    # Sharp Right
        ]

    def get_lidar_state(self, scan_data):
        raw_ranges = np.array(scan_data.ranges)
        raw_ranges[raw_ranges == np.inf] = 3.5
        
        # Downsample 360 beams to 24 discrete sectors
        downsampled = []
        step = len(raw_ranges) // self.num_scan_samples
        for i in range(0, len(raw_ranges), step):
            sector = raw_ranges[i : i + step]
            downsampled.append(np.min(sector))
            
        return np.array(downsampled[:self.num_scan_samples])

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
        data = rospy.wait_for_message('/scan', LaserScan)
        return self.get_lidar_state(data)
