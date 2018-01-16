import rospy
from geometry_msgs.msg import Point
rospy.init_node("wiper")

pub= rospy.Publisher('/rl_erase/wipe',Point,queue_size=10)
pt = Point()
pt.y = 0.1
pub.publish(pt)
rospy.sleep(2)


