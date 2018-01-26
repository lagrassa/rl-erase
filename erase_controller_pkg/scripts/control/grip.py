#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from pr2_controllers_msgs.msg import Pr2GripperCommand

rate = rospy.Rate(3) # 3hz
class Gripper():
    def __init__(self):
	self.pub = rospy.Publisher('r_gripper_controller/command', Pr2GripperCommand, queue_size=10)
	self.grip_msg = Pr2GripperCommand()
	self.grip_msg.max_effort = -1
	self.grip_msg.position = 0.045
    
    def grip(self):
        self.pub.publish(self.grip_msg)
    def grip_constantly(self):
        while not rospy.is_shutdown():
            self.grip()
            rospy.sleep(0.5)



if __name__ == '__main__':
    try:
        rospy.init_node('gripper', anonymous=True)
        g = Gripper()
        g.grip_constantly()
    except rospy.ROSInterruptException:
        pass
