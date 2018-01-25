import rospy 
from geometry_msgs.msg import *
from std_msgs.msg import Header
from uber_controller import Uber

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

uc.command_joint_pose('r',joint_angles, time=3, blocking=True)