import numpy as np
import pdb
import json
from matplotlib import pyplot as plt
import time
import rospy
rospy.init_node("collect_data")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#let's predict this function
#X = np.matrix([[2,1.1],[2,2],[2.5,2],[2,3],[1,1.5], [4,0.2],[4,3],[1.7, 0.2]])
#y = np.matrix([1,1,1,1,0,0,0,0]).T
#Xs = input forces
#apply force impedance control otherwise


def collect_traj_data():
    #collects the state and reward of a sample trajectory
    raw_input("Start trajectory?")
    forces = []
    time_to_sleep = 0.01
    for i in range(6):
        print(i, "step#")
        data = rospy.wait_for_message('/ft/r_gripper_motor/', WrenchStamped)
        force = [data.wrench.force.x,data.wrench.force.y, data.wrench.force.z]

        forces.append(force)
        time.sleep(time_to_sleep)
    x = sum(forces)/len(forces)
        
    raw_input("Press enter when arm out of way")
    reward = rospy.wait_for_message("/rl_erase/reward", Float32)
    y = 5
    return (x,y)
    
""" collects pairs of x,y,z force, state, and reward."""
def collect_data():
    size_state = 3
    num_good_trajs = 5
    num_too_light = 5
    num_too_hard = 5
    exps  = {'Good trajectory':num_good_trajs, 'Too light':num_too_light, 'Too hard':num_too_hard}
    num_trajs = num_good_trajs+num_too_light+num_too_hard
    Xs = np.zeros((num_trajs, size_state))
    ys = np.zeros((num_trajs, 1))
    i = 0
    for exp_key in exps:
        print("Collect " + exp_key)
        for _ in range(exps[exp_key]):
            x, y= collect_traj_data()
            Xs[i,:] = x
            ys[i,:] = y 
            i += 1
    
    return Xs, ys
    
X,y = collect_data()


kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X,y)

params = gp.get_params()

with open('gpParams.json', 'w') as fp:
    json.dump(params, fp)

