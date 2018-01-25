#!/usr/bin/env python 
import rospy 
from geometry_msgs.msg import *
from std_msgs.msg import Header
from gdm_arm_controller.uber_controller import Uber

rospy.init_node("relative")

uc= Uber()


arm = 'l'
frame = 'base_link'
pub_l = rospy.Publisher("l_cart/command_pose", PoseStamped, queue_size=1)
pub_r = rospy.Publisher("r_cart/command_pose", PoseStamped, queue_size=1)


def get_pose():
    return uc.return_cartesian_pose(arm, frame)

def stamp_pose( (pos,quat)):
    ps = PoseStamped( 
	        Header(0,rospy.Time(0),frame),\
	        Pose(Point(*pos),\
	        Quaternion(*quat)))
    return ps

def command_delta(x,y,z):
    pos, quat = get_pose()
    pos[0] += x
    pos[1] += y
    pos[2] += z
    cmd = stamp_pose( (pos,quat))
    pub_l.publish(cmd)

pos, quat = get_pose()
joint_angles = [0.29628785835607696, 0.11011554596095237, -1.8338543122460127, -0.1799232010879831, -13.940467600087464, -1.4902528179381358, 3.0453049548764994]
new_joint_angles = [0.08877259511154134, 0.16518684090195235, -1.8341750218987525, -0.16457750568132923, -1.3685023460351986, 0.22215882822167166, -3.103032396561839]
debug_ja =[-0.19145177155507043, 0.02230601577852849, -2.1882384785234614, -0.5764501795674656, 5.152484071727588, -1.1869210581237977, 3.4316827589164953]

uc.command_joint_pose('r',debug_ja, time=3, blocking=True)
