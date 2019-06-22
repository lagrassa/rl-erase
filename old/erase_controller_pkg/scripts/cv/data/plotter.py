import rospy
from std_msgs.msg import Float32
rospy.init_node("plotter")
f = open("reward_over_time.txt","a")

def record_data(data):
    f.write(str(data.data))
    f.write(",")

sub = rospy.Subscriber("/rl_erase/reward",Float32, record_data)
rospy.spin()
num_steps = 150

