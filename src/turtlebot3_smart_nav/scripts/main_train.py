#!/usr/bin/env python3
import rospy
from turtlebot_nav.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train"] + rospy.myargv()[1:]))
