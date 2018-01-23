#!/usr/bin/env python
import rospy 
PR2 = False
SIMPLE = True
import pdb
from geometry_msgs.msg import *
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import JointState
if PR2:
    from uber_controller import Uber
    uc= Uber()
import numpy as np

rospy.init_node("relative")



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
        num_params = 3#+(2*45)
        self.simple_state = 0
        self.simple_param = -1
        self.state = np.matrix(np.zeros(num_params)).T
        self.params = np.matrix(np.zeros(num_params+1))
        
        self.params[:,-1] = 1
        self.reward_prev = None
        self.epsilon = 0.001
        self.alpha = 0.001
        self.discount_factor = 0.95
        self.return_val = 0
        self.n = 0
        #self.state is comprised of forces, then joint efforts, then joint states in that order
        self.ft_sub = rospy.Subscriber('/ft/r_gripper_motor/', WrenchStamped, self.update_ft)
        self.wipe_sub = rospy.Subscriber('/rl_erase/wipe', Point, self.wipe)
        #self.joint_state_sub = rospy.Subscriber('/joint_states/', JointState, self.update_joint_state)
        self.reward_sub = rospy.Subscriber('/rl_erase/reward', Float32, self.update_reward)
        self.update_sub = rospy.Subscriber('/rl_erase/update', Point, self.policy_gradient_descent)
        self.gradient_pub = rospy.Publisher("/rl_erase/gradient", Float32, queue_size=1)
        self.action_pub = rospy.Publisher("/rl_erase/action", Float32, queue_size=1)
        

    def update_ft(self, data):
        self.state[0:3] = np.matrix([data.wrench.force.x,data.wrench.force.y,data.wrench.force.z]).T
        self.simple_state = data.wrench.force.z

    def update_joint_state(self, data):
        self.state[3:3+45] = data.effort
        self.state[3+45:4+(2*45)] = data.effort

    def policy(self, state):
        if not SIMPLE:
	    #sample from gaussian
	    mu_of_s = (self.params[:,:-1]*state).item()
	    #print("Mu of s",mu_of_s)
	    sigma = abs(self.params[:,-1].item())
	    z_press = np.random.normal(loc = mu_of_s, scale = sigma) 
        else:
            z_press = self.simple_state*self.simple_param
        #z_press = 0.4 
        #print("Policy of s is", z_press)
        (safe_z_press, was_safe) = self.safe_policy(z_press)
        if not was_safe:
            self.reward_prev = -1
        return safe_z_press

    #takes policy and makes it slow enough that robot won't destroy itself
    def safe_policy(self, action):
        #definition of safe: in same direction, but no more than -5
        if abs(action) > 0.05:
            if action < 0:
                return -0.05, False
            return 0.05, False
        return action, True

    def update_reward(self, reward):
        self.return_val += self.discount_factor*reward.data

    def policy_gradient_descent(self, data, alg='PGD'):
        #from the previous step, the function was nudged by epsilon in some dimension
        #update that and then
        if self.reward_prev == None:
            gradient = 0
        else:
            gradient = (self.reward - self.reward_prev) / (self.epsilon) 
        self.gradient_pub.publish(Float32(gradient))

        self.reward_prev = self.return_val
        #print("params before", self.params)
        #nprint("Gradient",gradient)
        self.params = self.params + self.alpha*gradient
        self.simple_param = self.simple_param + self.alpha*gradient

        if self.n > len(self.params):
            self.n = 0 
        #nudge epsilon again
        if alg == "SGD":
            mu = np.zeros(self.params.shape)
            add_vector = np.random.normal(loc = mu, scale = epsilon)

        else:
            add_vector = np.zeros(self.params.shape)
            add_vector[self.n] = self.epsilon

        self.params = self.params + add_vector
        self.simple_param = self.simple_param + self.epsilon
        

    def wipe(self, pt):
	#it's a move in x space
	#go_to_start()
	z_press = self.policy(self.state)
        self.gradient_pub.publish(Float32(z_press))
        if PR2:
	    command_delta(0,pt.y,z_press)

ec = EraserController()        
rospy.spin()
