#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from gdm_arm_controller.uber_controller import UberController
rospy.init_node("wiper")
wipe_pub= rospy.Publisher('/rl_erase/wipe',Point,queue_size=10)
update_pub= rospy.Publisher('/rl_erase/update',Point,queue_size=10)
numsteps = 200

uc = UberController()
joint_angles = [0.29628785835607696, 0.11011554596095237, -1.8338543122460127, -0.1799232010879831, -13.940467600087464, -1.4902528179381358, 3.0453049548764994]

# These are the experiments. The robot 
# 1) goes to start position
# 2) does the wiping motion
# 3) find the reward after executing that policy
# 4) policy update should happen when the wiper controller says so
for i in range(numsteps):
    #go to start
    uc.start_joint('r')
    uc.command_joint_pose('r',joint_angles, time=3, blocking=True)
    uc.start_cart('r')
    pt = Point()
    pt.y = 0.3*(-1)**i
    wipe_pub.publish(pt)
    rospy.sleep(5)
    update_pub.publish(pt)
    


