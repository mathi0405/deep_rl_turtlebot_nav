"""Mocked ROS fault contracts; these are not Gazebo integration evidence."""
import math
from types import SimpleNamespace as Obj
import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np
from turtlebot_nav.ros_env import ROSNavigationEnv
from turtlebot_nav.sensors import SensorError


def stamp(value):
    return Obj(to_sec=lambda: value)


def adapter():
    env = ROSNavigationEnv.__new__(ROSNavigationEnv)
    env.timeout = 0.02
    env.condition = threading.Condition()
    env.sequence = {'scan': 1, 'odom': 1}
    env.goal = [1, 0]
    env.rospy = Obj(is_shutdown=lambda: False)
    scan = Obj(header=Obj(stamp=stamp(1)), angle_min=0.0,
               angle_increment=2*math.pi/360, range_max=3.5, ranges=[2.0]*360)
    odom = Obj(header=Obj(stamp=stamp(1)), pose=Obj(pose=Obj(
        position=Obj(x=0.0, y=0.0), orientation=Obj(x=0., y=0., z=0., w=1.))))
    env.samples = {'scan': scan, 'odom': odom}
    return env


class ROSContractTests(unittest.TestCase):
    def test_non_simulation_master_is_refused_before_publishing(self):
        rospy = Obj(core=Obj(is_initialized=lambda: True), get_param=lambda *args: False,
                    Publisher=Mock())
        modules = {'rospy': rospy, 'gazebo_msgs.msg': Obj(ModelState=object),
                   'gazebo_msgs.srv': Obj(SetModelState=object), 'geometry_msgs.msg': Obj(Twist=object),
                   'nav_msgs.msg': Obj(Odometry=object), 'sensor_msgs.msg': Obj(LaserScan=object),
                   'std_srvs.srv': Obj(Empty=object)}
        with patch.dict('sys.modules', modules), self.assertRaises(RuntimeError):
            ROSNavigationEnv()
        rospy.Publisher.assert_not_called()

    def test_angle_ordering_puts_forward_in_center_bins(self):
        env = adapter()
        env.samples['scan'].ranges[0] = .1
        state, distance, pose = env._fresh({'scan': 0, 'odom': 0})
        self.assertAlmostEqual(float(state[12])*3.5, .1, places=6)
        self.assertEqual(distance, 1)
        np.testing.assert_array_equal(pose, [0, 0, 0])

    def test_stale_or_unsynchronized_messages_time_out(self):
        env = adapter()
        with self.assertRaises(SensorError):
            env._fresh(env.sequence.copy())
        env.samples['scan'].header.stamp = stamp(5)
        with self.assertRaises(SensorError):
            env._fresh({'scan': 0, 'odom': 0})

    def test_malformed_sensor_metadata_fails_closed(self):
        for field, bad in [('angle_min', float('nan')), ('angle_increment', 0.0), ('range_max', 1.0)]:
            env = adapter()
            setattr(env.samples['scan'], field, bad)
            with self.subTest(field=field), self.assertRaises(SensorError):
                env._fresh({'scan': 0, 'odom': 0})

    def test_sensor_error_stops_and_pauses(self):
        env = adapter()
        env.finished = False
        env._call = Mock()
        env._command = Mock()
        env._fresh = Mock(side_effect=SensorError('injected'))
        with self.assertRaises(SensorError):
            env.step([0, 0])
        self.assertTrue(env.finished)
        self.assertEqual(env._call.call_args_list[-1].args, ('pause_physics',))
        self.assertEqual(env._command.call_args_list[-1].args, ())

    def test_shutdown_is_an_error_not_a_none_scan(self):
        env = adapter()
        env.rospy.is_shutdown = lambda: True
        with self.assertRaises(SensorError):
            env._fresh({'scan': 0, 'odom': 0})

    def test_service_call_has_wall_clock_timeout(self):
        env = adapter()
        env.rospy.wait_for_service = Mock()
        env.services = {'pause_physics': lambda: threading.Event().wait(1)}
        with self.assertRaises(SensorError):
            env._call('pause_physics')


if __name__ == '__main__':
    unittest.main()
