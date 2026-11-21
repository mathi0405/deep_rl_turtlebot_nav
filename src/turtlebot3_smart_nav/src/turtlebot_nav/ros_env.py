"""ROS1/Gazebo adapter; physical robot use is deliberately unsupported."""
from __future__ import annotations

import math
import queue
import threading
import time
import numpy as np
from .sensors import SensorError, observation, transition_reward, velocity
from .simulation import episode_layout


class ROSNavigationEnv:
    backend = 'ros1-noetic-gazebo-world-v1'

    def __init__(self, max_steps=250, timeout=5.0):
        if max_steps < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid ROS environment limits')
        import rospy
        from gazebo_msgs.msg import ModelState
        from gazebo_msgs.srv import SetModelState
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from std_srvs.srv import Empty
        self.rospy, self.Twist, self.ModelState = rospy, Twist, ModelState
        if not rospy.core.is_initialized():
            rospy.init_node('turtlebot_nav', anonymous=True)
        if rospy.get_param('/use_sim_time', False) is not True:
            raise RuntimeError('ROS backend requires /use_sim_time=true; no physical robot mode')
        self.max_steps, self.timeout = max_steps, timeout
        self.condition = threading.Condition()
        self.samples, self.sequence = {}, {'scan': 0, 'odom': 0}
        self.finished = True
        self.publisher = rospy.Publisher('/turtlebot_lab/cmd_vel', Twist, queue_size=1)
        self.subscribers = [rospy.Subscriber('/turtlebot_lab/scan', LaserScan, self._scan_callback, queue_size=1),
                            rospy.Subscriber('/turtlebot_lab/odom', Odometry, self._odom_callback, queue_size=1)]
        self.services = {name: rospy.ServiceProxy('/gazebo/'+name, kind)
                         for name, kind in [('pause_physics', Empty), ('unpause_physics', Empty),
                                            ('reset_simulation', Empty), ('set_model_state', SetModelState)]}

    def _store(self, key, message):
        with self.condition:
            self.sequence[key] += 1
            self.samples[key] = message
            self.condition.notify_all()

    def _scan_callback(self, message):
        self._store('scan', message)

    def _odom_callback(self, message):
        self._store('odom', message)

    def _call(self, name, *args):
        self.rospy.wait_for_service('/gazebo/'+name, timeout=self.timeout)
        result = queue.Queue(maxsize=1)
        def invoke():
            try:
                result.put((True, self.services[name](*args)))
            except Exception as error:
                result.put((False, error))
        threading.Thread(target=invoke, daemon=True).start()
        try:
            ok, value = result.get(timeout=self.timeout)
        except queue.Empty:
            raise SensorError('Gazebo service timed out: '+name)
        if not ok:
            raise SensorError('Gazebo service failed: '+name+' '+str(value))
        if hasattr(value, 'success') and not value.success:
            raise SensorError('Gazebo rejected '+name+': '+value.status_message)
        return value

    def _command(self, action=None):
        message = self.Twist()
        if action is not None:
            message.linear.x, message.angular.z = velocity(action)
        self.publisher.publish(message)

    def _fresh(self, before):
        deadline = time.monotonic()+self.timeout
        with self.condition:
            while True:
                if self.rospy.is_shutdown():
                    raise SensorError('ROS shut down while waiting for sensors')
                if all(self.sequence[k] > before[k] for k in before):
                    scan, odom = self.samples['scan'], self.samples['odom']
                    skew = abs(scan.header.stamp.to_sec()-odom.header.stamp.to_sec())
                    if skew <= 0.15:
                        break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise SensorError('fresh aligned scan and odometry not received')
                self.condition.wait(min(remaining, 0.1))
        if (not math.isfinite(scan.angle_min) or not math.isfinite(scan.angle_increment) or scan.angle_increment <= 0
                or len(scan.ranges)*scan.angle_increment < 2*math.pi-0.1
                or not math.isfinite(scan.range_max) or scan.range_max < 3.5):
            raise SensorError('requires full-circle forward-aligned LiDAR with at least3.5m range')
        angles = scan.angle_min+np.arange(len(scan.ranges))*scan.angle_increment
        wrapped = (angles+math.pi) % (2*math.pi)-math.pi
        ranges = np.asarray(scan.ranges)[np.argsort(wrapped)]
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        quat = np.asarray([q.x, q.y, q.z, q.w])
        if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) > 0.02:
            raise SensorError('invalid odometry quaternion')
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        pose = np.array([p.x, p.y, yaw])
        state, distance = observation(ranges, pose, self.goal)
        return state, distance, pose

    def reset(self, seed):
        self.finished = True
        self._command()
        self._call('pause_physics')
        self._call('reset_simulation')
        start, self.goal = episode_layout(seed)
        model = self.ModelState()
        model.model_name = 'turtlebot3_burger'
        model.reference_frame = 'world'
        model.pose.position.x, model.pose.position.y, model.pose.position.z = float(start[0]), float(start[1]), 0.02
        model.pose.orientation.z, model.pose.orientation.w = math.sin(start[2]/2), math.cos(start[2]/2)
        self._call('set_model_state', model)
        with self.condition:
            before = self.sequence.copy()
        try:
            self._call('unpause_physics')
            state, self.distance, pose = self._fresh(before)
            if np.linalg.norm(pose[:2]-start[:2]) > 0.15:
                raise SensorError('odometry does not match world reset; use provided simulation launch')
            self.steps, self.finished = 0, False
            return state
        finally:
            self._command()
            self._call('pause_physics')

    def step(self, action):
        if self.finished:
            raise RuntimeError('reset required before step')
        velocity(action)
        try:
            self._call('unpause_physics')
            with self.condition:
                self._command(action)
                before = self.sequence.copy()
            state, distance, pose = self._fresh(before)
            self.steps += 1
            reward, terminated, truncated, outcome = transition_reward(
                self.distance, distance, float(state[:24].min())*3.5 < 0.15,
                distance <= 0.20, self.steps >= self.max_steps)
            self.distance, self.finished = distance, terminated or truncated
            return state, reward, terminated, truncated, dict(
                outcome=outcome, distance_m=distance, steps=self.steps,
                pose=pose.tolist(), goal=self.goal.tolist(), minimum_lidar_m=float(state[:24].min())*3.5)
        except Exception:
            self.finished = True
            raise
        finally:
            self._command()
            self._call('pause_physics')

    def close(self):
        self.finished = True
        self._command()
        if not self.rospy.is_shutdown():
            try:
                self._call('pause_physics')
            except Exception as error:
                self.rospy.logwarn('Cleanup could not confirm paused physics: %s', error)
        for subscriber in self.subscribers:
            subscriber.unregister()
