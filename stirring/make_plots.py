import matplotlib.pyplot as plt
import csv
import numpy as np
import pdb

#Import data
#get list of means and stdevs

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_line(mean, stdev, color="red", label="missing label"):
    x = mean
    #smooth  
    y_above = [mean[i]+stdev[i] for i  in range(mean.shape[0])]
    y_below = [mean[i]-stdev[i] for i  in range(mean.shape[0])]
    #plot mean
    plt.plot(x, label=label)
    coords = list(range(len(mean)))
    plt.fill_between(coords, y_below, y_above, color=color, alpha = 0.3)

def get_stdev_and_mean(exp_list, prefix):
    lengths_list = []
    for exp in exp_list:
        lengths = get_line_out_file(prefix+exp)
        lengths_list.append(lengths)
    shortest_length = min([len(l) for l in lengths_list])
    
    short_length_list = [l[:shortest_length]for l in lengths_list]
    lengths_array = np.vstack(short_length_list)
    stdevs = np.std(lengths_array, axis=0)
    means = np.mean(lengths_array, axis=0)
    return means, stdevs

"""This list of keys will appear in the legend, the list is experiment names
This should plot the average lengths, and then rewards"""
def plot_exps(exp_dict):
    #First plot average lengths
    colors = ["red", "green", "blue"]
    color_i = 0
    for exp_name in exp_dict.keys():
        means, stdevs = get_stdev_and_mean(exp_dict[exp_name], "average_reward")
        plot_line(means, stdevs, color = colors[color_i], label=exp_name)
        color_i +=1 
    plt.legend()
    plt.show()
                
    #Then rewards

def get_line_out_file(exp):
    with open("stats/"+exp, 'rb') as csvfile:
	reader = csv.reader(csvfile)
        string_list =  reader.next()
        float_list =  [float(elt) for elt in string_list if elt != ""]
        smoothed = moving_average(float_list, n = 10)
        return smoothed
        
            

forces_exp_dict = {} 
forces_exp_dict["no forces"] = ["151cd6_nonlinear_check_600_noforce_1.py","151cd6_nonlinear_check_600_noforce_2.py"] #basically random
forces_exp_dict["b5fec_600 force"] = ["b5fecf7_nonlinear_check_600_1.py", "b5fecf7_nonlinear_check_600_2.py"] #increases and converges


weight_cup_exp_dict = {}
weight_cup_exp_dict['light cup, light beads'] = ["1e95_harder_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "1e95_harder_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #okay results
weight_cup_exp_dict["heavier cup"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #does better quickly then tapers off

num_checks_exp_dict = {}
num_checks_exp_dict["check 200"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_2.py"] #doesn't get much better
num_checks_exp_dict["check 600"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #does better quickly then tapers off

linear_exp_dict = {}
linear_exp_dict["linear less restricted"] = ["6389e_linear_less_restricted.py"]
linear_exp_dict["linear"] = ['513c7.py']
linear_exp_dict["non linear less restricted"] = ["c188d_nonlinear_less_restricted_2.py"]
plot_exps(exp_dict)
 

 



