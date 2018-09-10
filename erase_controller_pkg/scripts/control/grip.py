#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from pr2_controllers_msgs.msg import Pr2GripperCommand

class Gripper():
    def __init__(self):
	self.pub = rospy.Publisher('r_gripper_controller/command', Pr2GripperCommand, queue_size=10)
    
    def grip(self, amount = 0.060, times=10):
        for i in range(times):
            self.grip_msg = Pr2GripperCommand()
	    self.grip_msg.max_effort = -1
	    self.grip_msg.position = amount
	    self.pub.publish(self.grip_msg)
    def grip_constantly(self):
        while not rospy.is_shutdown():
            self.grip(amount=0.05)
            rospy.sleep(0.6)



if __name__ == '__main__':
    rospy.init_node('gripper')
    try:
        rospy.sleep(0.2)
        g = Gripper()
        g.grip_constantly()
    except rospy.ROSInterruptException:
        pass
