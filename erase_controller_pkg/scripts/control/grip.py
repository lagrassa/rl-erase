import rospy
from std_msgs.msg import String
from pr2_controllers_msgs.msg import Pr2GripperCommand

def gripper():
    pub = rospy.Publisher('r_gripper_controller/command', Pr2GripperCommand, queue_size=10)
    rospy.init_node('gripper', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        grip_msg = Pr2GripperCommand()
        grip_msg.max_effort = -1
        grip_msg.position = 0.05
        pub.publish(grip_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        gripper()
    except rospy.ROSInterruptException:
        pass
