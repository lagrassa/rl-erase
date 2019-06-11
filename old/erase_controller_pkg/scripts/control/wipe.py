#!/usr/bin/env python
from __future__ import division
import rospy 
rospy.init_node("wipe")
import roslib
from grip import Gripper
from erase_control_globals import wipe_time
PR2 = True
SIMPLE = False
ACTUALLY_MOVE = True
import pdb
from geometry_msgs.msg import *
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import JointState
if PR2:
    roslib.load_manifest("gdm_arm_controller")
    from gdm_arm_controller.uber_controller import Uber
    uc= Uber()
import numpy as np




arm = 'r'
frame = 'base_link'
#pub = rospy.Publisher("r_cart/command_pose", PoseStamped, queue_size=1)


def get_pose():
    return uc.return_cartesian_pose(arm, frame)

def stamp_pose( (pos,quat)):
    ps = PoseStamped( 
	        Header(0,rospy.Time(0),frame),\
	        Pose(Point(*pos),\
	        Quaternion(*quat)))
    return ps

def command_delta(x,y,z, numsteps=1):
    pos, quat = get_pose()
    pos[0] += x/numsteps
    pos[1] += y/numsteps
    pos[2] += z/numsteps
    cmd = stamp_pose( (pos,quat))
    if ACTUALLY_MOVE:
        uc.cmd_ik_interpolated(arm, (pos, quat), wipe_time/numsteps, frame, blocking = True, use_cart=True, num_steps = 30)


class EraserController:
    def __init__(self):
        #initialize the state
        num_params = 3#+(2*45)
        self.simple_state = 0
        self.simple_param = 0.07
        self.prev_pose = get_pose()
        self.state = np.matrix(np.zeros(num_params)).T
        self.params = np.matrix([0.001,0.06,-0.03, 0.05])#pray
        #self.params = np.matrix(np.zeros(num_params+1))#uncomment when you also want to train sigma
        self.numsteps = 6
        #self.params[:,-1] = 1
        self.reward_prev = None
        self.traj_cost = 0
        self.epsilon = 0.001
        self.alpha = 0.00005 #max gradient is realistically 3200 ish
        self.discount_factor = 0.00001
        self.alg = "SGD"
        self.return_val = None
        self.n = 0
        #grip the eraser
        self.gripper = Gripper()
        #self.state is comprised of forces, then joint efforts, then joint states in that order
        self.ft_sub = rospy.Subscriber('/ft/r_gripper_motor/', WrenchStamped, self.update_ft)
        self.wipe_sub = rospy.Subscriber('/rl_erase/wipe', Point, self.wipe)
        self.update_sub = rospy.Subscriber('/rl_erase/reward', Float32, self.complete_trajectory)

        self.gradient_pub = rospy.Publisher("/rl_erase/gradient", Float32, queue_size=1)
        self.action_pub = rospy.Publisher("/rl_erase/action", Float32, queue_size=1)
        

    def update_ft(self, data):
        self.state[0:3] = np.matrix([data.wrench.force.x,data.wrench.force.y, data.wrench.force.z]).T
        self.simple_state = data.wrench.force.z

    def update_joint_state(self, data):
        self.state[3:3+45] = data.effort
        self.state[3+45:4+(2*45)] = data.effort

    def policy(self, state):
        if not SIMPLE:
	    #sample from gaussian
            print(self.params, "self.params")
            try:
	        mu_of_s = (self.params[:,:-1]*state).item()
            except:
                print self.params[:,:-1]
                print state
	    #print("Mu of s",mu_of_s)
            print("sigma val", self.params[:,-1])
            print("self.params", self.params)
	    sigma = abs(self.params[:,-1].item())
	    z_press = np.random.normal(loc = mu_of_s, scale = sigma) 
        else:
            z_press = self.simple_state*self.simple_param
            print("state",self.simple_state)
        #(safe_z_press, was_safe) = self.safe_policy(z_press)
        return z_press

    #takes policy and makes it slow enough that robot won't destroy itself
    def safe_policy(self, action):
        #definition of safe: in same direction, but no more than -5
        max_amt = 0.4
        if abs(action) > max_amt:
            if action < 0:
                return -max_amt, False
            return max_amt, False
        return action, True

    def update_reward(self, reward):
        print("updating return val to ",reward)
        self.return_val = reward.data #essentially discount 0
        #self.return_val += self.discount_factor*self.return_val + reward.data #so this is kind of weird, but I want to make the more recent rewards more important

    def compute_gradient(self):
        if self.reward_prev == None:
            gradient = 0
        else:
            delta_j = self.return_val - self.reward_prev
            if SIMPLE:
                gradient = delta_j / (self.epsilon) 
            else:
                ata = self.delta_theta.T*self.delta_theta
                if np.linalg.det(ata) == 0:
                    print("Found a zero determinant")
                    return 0
                try:
                    gradient = np.linalg.inv(ata)*self.delta_theta.T*(delta_j)
                    print("ata",ata)
                    print("invata", np.linalg.inv(ata))
                    print("dtheta", self.delta_theta.T)
                    print("deltaj",delta_j)
                    print("computed gradient to be",gradient)
                except:
                    print("A^TA",ata)
                    print("determinant",np.linalg.det(ata))
        return gradient


    def complete_trajectory(self, data):
        self.return_val = data.data-self.traj_cost
        #Done using the traj cost, so bring back to 0
        self.traj_cost = 0
        #and update the pose
        self.prev_pose = get_pose()
        self.policy_gradient_descent(self.return_val)
        self.perturb()
        
        
    def policy_gradient_descent(self, return_val):
        #from the previous step, the function was nudged by epsilon in some dimension
        #update that and then
        gradient = self.compute_gradient()
        print("#######UPDATE######")
        self.gradient_pub.publish(Float32(gradient))

        self.reward_prev = return_val
        print("params before: ", self.params)
        print("Gradient",gradient)
        self.params = self.params + self.alpha*gradient
        self.simple_param = self.simple_param + self.alpha*gradient
        print("params after: ", self.params)

        print("#######END UPDATE######")
        
    def perturb(self):
        if self.n > len(self.params):
            self.n = 0 
        #nudge epsilon again
        if self.alg == "SGD":
            mu = np.zeros(self.params.shape)
            add_vector = np.random.normal(loc = mu, scale = self.epsilon)
            add_eps = np.random.normal(loc=0, scale = self.epsilon)

        else:
            add_vector = np.zeros(self.params.shape)
            add_vector[self.n] = self.epsilon
            add_eps = self.epsilon
 
        self.delta_theta =  np.matrix(add_vector.T)
        self.params = self.params + add_vector
        self.simple_param = self.simple_param + add_eps

    def update_cost_and_pos(self):
        #update position then add to pos
        new_pose, new_quat = get_pose()
        #only care about movement in z right now
        diff_pose = abs(self.prev_pose[0][2]-new_pose[2])
        self.traj_cost += diff_pose
        print("The current cost is: ", self.traj_cost)

    def wipe(self, pt):
        self.gripper.grip()
	#it's a move in x space
	#go_to_start()
        for i in range(self.numsteps):
            if PR2:
                z_press = self.policy(self.state)
                print("Wiping with zpress", z_press)
	        command_delta(0,pt.y,z_press, numsteps = self.numsteps)
                self.update_cost_and_pos()

ec = EraserController()        
rospy.spin()
