import matplotlib.pyplot as plt
import ast
import csv
import numpy as np
import pdb
plt.rcParams['font.size'] = 18

#Import data
#get list of means and stdevs

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_line(mean, stdev, color="red", label="missing label", plot_area = None):
    x = mean
    #smooth  
    y_above = [mean[i]+stdev[i] for i  in range(mean.shape[0])]
    y_below = [mean[i]-stdev[i] for i  in range(mean.shape[0])]
    #plot mean
    plot_area.plot(x, label=label, color=color)
    coords = list(range(len(mean)))
    plot_area.fill_between(coords, y_below, y_above, color=color, alpha = 0.3)


def get_stdev_and_mean(exp_list, prefix, root_dir = "No root directory", cutoff=None, lengths_array = None):

    if lengths_array is None:
	lengths_list = []
	for exp in exp_list:
	    lengths = get_line_out_file(prefix+exp, root_dir = root_dir)
	    lengths_list.append(lengths)
	shortest_length = min([len(l) for l in lengths_list])
	if cutoff is not None:
	    shortest_length = min(cutoff, shortest_length)
	short_length_list = [l[:shortest_length]for l in lengths_list]
	lengths_array = np.vstack(short_length_list)
    stdevs = np.std(lengths_array, axis=0)
    means = np.mean(lengths_array, axis=0)
    return means, stdevs

"""This list of keys will appear in the legend, the list is experiment names
This should plot the average lengths, and then rewards"""
def plot_graph(exp_dict, 
              prefix="no prefix", 
              title="No title",
              xlab = "No x label", 
              root_dir = "No root directory",
              plot_area = None,
              cutoff=None,
              lengths_array_index=None,
              ylab = "No y label"):
    #First plot average lengths
    colors = ["red", "blue","green" ]
    color_i = 0
    for exp_name in exp_dict.keys():
        means, stdevs = get_stdev_and_mean(exp_dict[exp_name], prefix, root_dir = root_dir, cutoff=cutoff, lengths_array=exp_dict[exp_name][lengths_array_index])
        plot_line(means, stdevs, color = colors[color_i], label=exp_name, plot_area = plot_area)
        color_i +=1 
    plot_area.set_xlabel(xlab)
    plot_area.set_ylabel(ylab)

"""Plots 2 plots on top of each other: average reward
and average length of episode"""
def plot_learning_curve(exp_dict, title = "No title", root_dir="No root directory", cutoff=None):
    #plot average length on top        
    f, axarr = plt.subplots(2,sharex=True)
    plt.title(title)
    plot_graph(exp_dict, prefix = "average_reward",root_dir = root_dir,  xlab = "Number of episodes", ylab = "Average episode reward", plot_area=axarr[0], cutoff=cutoff)
    plot_graph(exp_dict, prefix = "average_length",root_dir = root_dir,  xlab = "Number of episodes", ylab = "Average episode length", plot_area=axarr[1], cutoff=cutoff)
    #Then rewards
    plt.legend()
    plt.show()

def plot_policy_test(exp_dict,root_dir = "", title = "No title", cutoff=None):
    #plot average length on top        
    f, axarr = plt.subplots(2,sharex=True)
    plt.title(title)
    plot_graph(exp_dict, prefix = "average_length",root_dir = root_dir,  xlab = "Number of steps in episodes", ylab = "Hierarchical entropy", plot_area=axarr[0], cutoff=cutoff, lengths_array_index = 0)
    plot_graph(exp_dict, prefix = None,root_dir = root_dir,  xlab = "Number of steps in episode", ylab = "Ratio beads in cup", plot_area=axarr[1], cutoff=cutoff, lengths_array_index = 1)
    #Then rewards
    plt.legend()
    plt.show()

def get_line_out_file(exp, root_dir = "No root directory"):
    with open(root_dir+exp, 'rb') as csvfile:
	reader = csv.reader(csvfile)
        string_list =  reader.next()
        float_list =  [float(elt) for elt in string_list if elt != ""]
        smoothed = moving_average(float_list, n = 1)
        return smoothed
        
            
#Plotting learning curves
forces_exp_dict = {} 
forces_exp_dict["force used"] = ["c4af2082_forces_nonlinear_less_restricted_1.py", "c4af2082_forces_nonlinear_less_restricted_3.py"] #increases and converges
forces_exp_dict["force not used"] = ["c4af2082_no_forces_nonlinear_less_restricted_1.py", "c4af2082_no_forces_nonlinear_less_restricted_3.py", "c4af2082_no_forces_nonlinear_less_restricted_2.py"] #increases and converges


weight_cup_exp_dict = {}
weight_cup_exp_dict['light cup, light beads'] = ["1e95_harder_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "1e95_harder_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #okay results
weight_cup_exp_dict["heavier cup"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #does better quickly then tapers off

num_checks_exp_dict = {}
num_checks_exp_dict["200"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_2.py"] #doesn't get much better
num_checks_exp_dict["600"] = ["838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_1.py", "838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_2.py"] #does better quickly then tapers off

linear_exp_dict = {}
linear_exp_dict["linear"] = ["6389e_linear_less_restricted.py"]
#linear_exp_dict["linear"] = ['513c7.py']
linear_exp_dict["relu"] = ["c188d_nonlinear_less_restricted_2.py"]

#plot_learning_curve(linear_exp_dict, title = "Activation functions", root_dir="stats/", cutoff=170)
#plot_learning_curve(num_checks_exp_dict, title = "Number samples in actor", root_dir="stats/", cutoff=None)
#plot_learning_curve(weight_cup_exp_dict, title = "Difficulty of experimental setup", root_dir="stats/", cutoff=None)
#plot_learning_curve(forces_exp_dict, title = "Using forces in the actor", root_dir="stats/", cutoff=None)

#plot bead and entropy
exp_dict = {}

#exp dict is of the form
#dictionary to a tuple with the entropy, then the beadso

#entropy first, then beads
def beads_and_entropy_from_filename(filename):
    entropy = ast.literal_eval(open("policy_results/policies/"+filename+ "entropy_results.py", "rb").read())
    beads = ast.literal_eval(open("policy_results/policies/"+filename+ "bead_results.py", "rb").read())
    return entropy, beads
   
    
force_be_dict = {}
no_force_entropy, no_force_beads = beads_and_entropy_from_filename("c4af2082_no_forces_nonlinear_less_restricted_3_500weights")
force_entropy, force_beads = beads_and_entropy_from_filename("c4af2082_forces_nonlinear_less_restricted_3_500weights")
force_be_dict["no force"] = (no_force_entropy, no_force_beads)
force_be_dict["force"] = (force_entropy, force_beads)
#plot_policy_test(force_be_dict, title = "Force input in the policy", root_dir="")

training_iters = {}
entropy_100_iters, beads_100_iters = beads_and_entropy_from_filename("c4af2082_forces_nonlinear_less_restricted_3_100weights")
entropy_500_iters, beads_500_iters = beads_and_entropy_from_filename("c4af2082_forces_nonlinear_less_restricted_3_500weights")
training_iters["100 episodes"]  = (entropy_100_iters, beads_100_iters)
training_iters["500 episodes"] = (entropy_500_iters, beads_500_iters)
#plot_policy_test(training_iters, title = "Policy after training more iterations", root_dir="")


harder_be_dict = {}
harder_prob_entropy, harder_prob_beads = beads_and_entropy_from_filename("1e95_harder_nonlinear_less_restricted_15_rollout_5_batch_check_600_2_100weights")
harder_be_dict["lighter cup, lighter beads"] = (harder_prob_entropy, harder_prob_beads)
#plot_policy_test(harder_be_dict, title = "Trained on a lighter cup and lighter bead actors", root_dir = "")

entropy_600, beads_600 = beads_and_entropy_from_filename("838ae_nonlinear_less_restricted_15_rollout_5_batch_check_600_1_200weights")
""" Needs to be redone on flakey
sample_dict = {}
entropy_200, beads_200 = beads_and_entropy_from_filename("838ae_nonlinear_less_restricted_15_rollout_5_batch_2_200weights")
sample_dict["200"] = (entropy_200, beads_200)
sample_dict["600"] = (entropy_600, beads_600)
plot_policy_test(sample_dict, title = "Number of sampled actions in actor only during training", root_dir = "")
"""

entropy_1200, beads_1200 = beads_and_entropy_from_filename("c4af2082_forces_nonlinear_less_restricted_3_500weights1200_")
num_samples_test = {}
num_samples_test["600"] = (entropy_600, beads_600)
num_samples_test["1200"] = (entropy_1200, beads_1200)
plot_policy_test(num_samples_test, title = "Number of sampled actions in actor", root_dir = "")






