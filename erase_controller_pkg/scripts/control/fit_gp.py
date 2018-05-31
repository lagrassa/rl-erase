import numpy as np
import pdb
import json
from matplotlib import pyplot as plt
import time
from geometry_msgs.msg import *
from std_msgs.msg import Header, Float32
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
    time_to_sleep = 0.5
    for i in range(6):
        print(i, "step#")
        data = rospy.wait_for_message('/ft/r_gripper_motor/', WrenchStamped)
        force = [data.wrench.force.x,data.wrench.force.y, data.wrench.force.z]
        forces.append(force)
        time.sleep(time_to_sleep)
    forces = np.array(forces)
    x = np.average(forces, axis=0)
        
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
print(X, "X")
print(y, "y")


kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
X = [[  6.82363192, -16.38551153,   5.64478611],
       [  4.29724834, -13.63014662,   4.96232973],
       [  4.59071543, -11.92423152,   4.81544906],
       [  4.2949693 , -12.87675871,   5.16477966],
       [  4.10585292, -13.50090225,   4.84599718],
       [  6.91792365, -11.29063318,  -5.07800206],
       [  6.78145595, -11.1925924 ,  -7.11770544],
       [  6.70614104, -12.32381859, -14.9805829 ],
       [  7.81653159, -12.01413346,  -8.49294406],
       [  6.17192181, -12.4810872 ,  -7.79349506],
       [  4.02251458, -10.1751462 ,   3.58618972],
       [  6.08898904, -11.6674992 ,   0.28110092],
       [  6.20738109, -12.08157436,   0.05359656],
       [  5.61603719, -12.39538706,   1.1257978 ],
       [  5.84118061, -12.58242897,   1.78389973]]
y = [[0.957822501659],[0.958218336105],[0.95780223608],[0.960562646389],[0.964764118195],
[0.999685406685, 0.999685406685, 0.999685406685, 0.999675273895, 0.999685406685],
[0.985243976116, 0.968725013042, 0.958451747894, 0.954782373101, 0.958218336105]
]
gp.fit(X,y)

params = gp.get_params()

with open('gpParams.json', 'w') as fp:
    json.dump(params, fp)

