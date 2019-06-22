#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from erase_control_globals import wipe_time
from gdm_arm_controller.uber_controller import UberController

CONSISTENT_STATE = True 

rospy.init_node("make_wipe")
wipe_pub= rospy.Publisher('/rl_erase/wipe',Point,queue_size=10)
update_rew_pub= rospy.Publisher('/rl_erase/update_reward',Point,queue_size=10)
numsteps = 200

uc = UberController()
start_joint_angles = [0.09482477817860147, 0.08879762533862981, -2.1107870973868064, -0.3122436312925274, 5.1535253192926955, -1.2508791706701996, 3.5850952111876673]
away_joint_angles =[-0.05432422644661594, 0.030680913165869468, -2.1242569028018767, -0.30717665639410396, 5.189853289897649, -1.3181874700642715, 3.253426713268458]

# These are the experiments. The robot 
# 1) goes to start position
# 2) does the wiping motion
# 3) find the reward after executing that policy
# 4) policy update should happen when the wiper controller says so
for i in range(numsteps):
    #go to start
    if CONSISTENT_STATE:
        uc.start_joint('r')
        uc.command_joint_pose('r',start_joint_angles, time=3, blocking=True)
        rospy.sleep(1)
    pt = Point()
    pt.y = 0.6 #*(-1)**i don't really need the alternating anymore?
    wipe_pub.publish(pt)
    rospy.sleep(wipe_time)
    if CONSISTENT_STATE:
        uc.start_joint('r')
        uc.command_joint_pose('r',away_joint_angles, time=3, blocking=True)
        rospy.sleep(2)
    update_rew_pub.publish(pt)
    


