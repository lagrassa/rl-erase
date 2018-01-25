import rospy
from geometry_msgs.msg import Point
rospy.init_node("wiper")
pub= rospy.Publisher('/rl_erase/wipe',Point,queue_size=10)
numsteps = 30
for i in range(numsteps):
    pt = Point()
    pt.y = 0.15*(-1)**i
    pub.publish(pt)
    rospy.sleep(5)

