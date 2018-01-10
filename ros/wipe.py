import rospy 
from geometry_msgs.msg import *
from std_msgs.msg import Header
from uber_controller import Uber

rospy.init_node("relative")

uc= Uber()


arm = 'l'
frame = 'base_link'
pub = rospy.Publisher("l_cart/command_pose", PoseStamped, queue_size=1)


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

def wipe():
    #it's a move in x space
    num_micro_wipes = 60
    for i in range(num_micro_wipes):
        command_delta(0,-0.05,0)
        
wipe()
