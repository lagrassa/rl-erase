#!/usr/bin/env python
import rospy
from geometry_msgs.msg import *
from std_msgs.msg import Header, Float32
from gdm_arm_controller.uber_controller import UberController
rospy.init_node("debug")

uc = UberController()
start_joint_angles = [0.29628785835607696, 0.11011554596095237, -1.8338543122460127, -0.1799232010879831, -13.940467600087464, -1.4902528179381358, 3.0453049548764994]
away_joint_angles = [-0.7928563738626452, -0.09883391410705068, -1.958770721988154, -0.0942189399489346, -13.826103909186205, -1.5220578385309405, 2.7107648682307186]
arm = 'r'
frame = 'base_link'
pub = rospy.Publisher("r_cart/command_pose", PoseStamped, queue_size=1)
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
    pub.publish(cmd)


#print("joint angles",uc.get_joint_positions('r'))
uc.start_cart('r')
command_delta(0,0,0.4)

#uc.command_joint_pose('r',away_joint_angles, time=1, blocking=True)



