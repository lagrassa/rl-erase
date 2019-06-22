#!/usr/bin/env python
import rospy 
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import WrenchStamped

rospy.init_node("safety")

max_downward_force = 19

scaling_factor = 0.05

#sends an increasingly worse reward depending on the magnitude of the force, probably negative 
def alert_if_needed(data):
    x = data.wrench.force.x
    y = data.wrench.force.y
    z = data.wrench.force.z
    mag = (x**2+y**2+z**2)**0.5
    if mag > max_downward_force:
        pub.publish(Float32(-mag/scaling_factor))
ft_sub = rospy.Subscriber('/ft/r_gripper_motor/', WrenchStamped, alert_if_needed)
pub = rospy.Publisher("/rl_erase/reward", Float32, queue_size=10)


rospy.spin()
