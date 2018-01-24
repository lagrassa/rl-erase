#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from erase_control_globals import wipe_time
from gdm_arm_controller.uber_controller import UberController
rospy.init_node("make_wipe")
wipe_pub= rospy.Publisher('/rl_erase/wipe',Point,queue_size=10)
update_pub= rospy.Publisher('/rl_erase/update',Point,queue_size=10)
numsteps = 200

uc = UberController()
start_joint_angles = [0.29628785835607696, 0.11011554596095237, -1.8338543122460127, -0.1799232010879831, -13.940467600087464, -1.4902528179381358, 3.0453049548764994]
away_joint_angles = [-0.2916858719396662, -0.06262728701834408, -1.9592517864672638, -0.2858953618207253, -13.707922310546211, -1.4222918847425698, 2.825845961894146]

# These are the experiments. The robot 
# 1) goes to start position
# 2) does the wiping motion
# 3) find the reward after executing that policy
# 4) policy update should happen when the wiper controller says so
for i in range(numsteps):
    #go to start
    uc.start_joint('r')
    uc.command_joint_pose('r',start_joint_angles, time=3, blocking=True)
    uc.start_cart('r')
    rospy.sleep(1)
    pt = Point()
    pt.y = -0.2 #*(-1)**i don't really need the alternating anymore?
    wipe_pub.publish(pt)
    rospy.sleep(wipe_time)
    uc.start_joint('r')
    uc.command_joint_pose('r',away_joint_angles, time=3, blocking=True)
    update_pub.publish(pt)
    


