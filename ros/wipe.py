import rospy 
from geometry_msgs.msg import *
from std_msgs.msg import Header, Float32
from uber_controller import Uber

rospy.init_node("relative")

uc= Uber()


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

def go_to_start():
    pos, quat = get_pose()
    joint_angles = [0.29628785835607696, 0.11011554596095237, -1.8338543122460127, -0.1799232010879831, -13.940467600087464, -1.4902528179381358, 3.0453049548764994]

    uc.command_joint_pose(arm,joint_angles, time=0.5, blocking=False)
    rospy.sleep(2)

class EraserController:
    def __init__(self):
        #initialize the state
        self.ft_sub = rospy.Subscriber('/ft/r_gripper_motor/', WrenchStamped, self.update_ft)
        self.joint_state_sub = rospy.Subscriber('/joint_states/', JointState, self.update_joint_state)
        self.reward_sub = rospy.Subscriber('/rl_erase/reward', Float32, self.policy_gradient_descent)
        num_params = 3+(2*45)
        self.state = np.zeros((1,num_params))
        self.params = np.zeros((1,num_params+1))
        self.reward_prev = 0
        self.epsilon = 0.01
        self.alpha = 0.01
        self.n = 0
        assert(len(self.state) == self.params)
        #self.state is comprised of forces, then joint efforts, then joint states in that order

    def update_ft(self, data):
        self.state[0:3] = data.force

    def update_joint_state(self, data):
        self.state[4:4+45+1] = data.effort
        self.state[4+45:4+(2*45)+1] = data.effort

    def policy(self):
        #sample from gaussian
        mu_of_s = self.state * self.params[:-1]
        sigma = self.params[-1]
        z_press = np.random.normal(loc = mu_of_s, scale = sigma) 
        #z_press = 0.4 
        return z_press

    def policy_gradient_descent(self, reward):
        #from the previous step, the function was nudged by epsilon in some dimension
        #update that and then
        gradient = (reward.data - self.reward_prev) / (self.epsilon) 
        self.reward_prev = reward.data
        self.params = self.params + self.alpha*gradient

        if self.n > len(self.params):
            self.n = 0 
        #nudge epsilon again
        unit_vector = np.zeros(self.params.shape)
        unit_vector[self.n] = epsilon
        self.params = self.params + unit_vector
        

    def wipe():
	#it's a move in x space
	go_to_start()
	z_press = self.policy(state)
	command_delta(0,0.1,z_press)
        
wipe()
