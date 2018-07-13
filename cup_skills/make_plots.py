import matplotlib.pyplot as plt
import os
import ast
import csv
import os
import numpy as np
import pdb
plt.rcParams['font.size'] = 18

#Import data
#get list of means and stdevs

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_line(mean, stdev, color="red", label="missing label", plot_area = None,xaxis=None) :
    y = mean
    #smooth  
    y_above = [mean[i]+stdev[i] for i  in range(mean.shape[0])]
    y_below = [mean[i]-stdev[i] for i  in range(mean.shape[0])]
    display_now = False
    if plot_area is None:
        display_now = True
        plot_area = plt
    #plot mean
    if xaxis is None:
        coords = list(range(len(mean)))
    else:
        coords = xaxis
    plot_area.plot(coords, y, label=label, color=color)
    plot_area.fill_between(coords, y_below, y_above, color=color, alpha = 0.3)
    if display_now:
        plt.show()


def get_stdev_and_mean(exp_list, prefix, root_dir = "No root directory", cutoff=None, lengths_array = None):

    if lengths_array is None:
	lengths_list = []
	for exp in exp_list:
	    lengths = get_line_out_file(prefix+exp, root_dir = root_dir)
	    lengths_list.append(lengths)
        try:
	    shortest_length = min([len(l) for l in lengths_list])
        except: 
            pdb.set_trace()
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
    colors = ["red", "blue","green", "purple", "gray", "yellow" ]
    color_i = 0
    for exp_name in exp_dict.keys():
        if lengths_array_index is None:
            means, stdevs = get_stdev_and_mean(exp_dict[exp_name], prefix, root_dir = root_dir, cutoff=cutoff)
        else:
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
        try:
            float_list =  [float(elt) for elt in string_list if elt != ""]
        except:
            pdb.set_trace()
        smoothed = moving_average(float_list, n = 3)
        return smoothed

def get_exps_from_root(root):
    #finds all experiments with root in the name and a number
    files = os.listdir('stats/')
    filenames = []
    for f in files:
        if root+"_"in f and "reward" in f and ".pyc" not in f:
            filenames.append(f[len("average_reward"):])
    return filenames

def generate_exp_dictionary_one_vars(root):
    exp_dict = {}
    deltas = {"0dot1":0.1, "0dot01":0.01, "0dot05":0.05}
    for delta_name in deltas:
        try:
            exp_dict["delta="+str(deltas[delta_name])] = get_exps_from_root(root+"_del_"+delta_name)
        except:
            print("Did not find file name")
            pdb.set_trace()
    return exp_dict
        
if __name__ == "__main__":            
    adagrad = {}
    adagrad["delta=0.1"] = get_exps_from_root("pdg_adagrad")
    adagrad["delta=0.05"] = get_exps_from_root("pdg_adagrad_del0dot05")
    adagrad["delta=0.5"] = get_exps_from_root("pdg_adagrad_del0dot5")
    #plot_learning_curve(adagrad, title = "Adagrad learning curves", root_dir="stats/", cutoff=None)

    pour_delta = {}
    pour_delta["delta=0.01"] = get_exps_from_root("pdg_pour_del_0dot01")
    pour_delta["delta=0.05"] = get_exps_from_root("pdg_pour_del_0dot05")
    pour_delta["delta=0.1"] = get_exps_from_root("pdg_pour_del_0dot1")
    #plot_learning_curve(pour_delta, title = "Effect of delta", root_dir="stats/", cutoff=None)

    pour_force = {}
    pour_force["force factor = 10"] = get_exps_from_root("pdg_pour_force_10")
    pour_force["force factor = 50"] = get_exps_from_root("pdg_pour_force_50")
    pour_force["force factor = 100"] = get_exps_from_root("pdg_pour_force_100")
    #plot_learning_curve(pour_force, title = "Effect of force", root_dir="stats/", cutoff=None)

    pour_big_cup = {}
    pour_big_cup["big"] = get_exps_from_root("pdg_pour_big_force_50")
    pour_big_cup["small"] = get_exps_from_root("pdg_pour_force_50")
    #plot_learning_curve(pour_big_cup, title = "Effect of cup size", root_dir="stats/", cutoff=None)

    num_params = {}
    num_params["2"] = get_exps_from_root("pdg_offset_height")
    #plot_learning_curve(num_params, title = "Finite differences on more parameters", root_dir="stats/", cutoff=None)

    one_param = {}
    one_param["offset"] = get_exps_from_root("pdg_one_var_kiwi_visual_offset")
    one_param["offset"] = get_exps_from_root("pdg_one_var_offset")
    one_param["height"] = get_exps_from_root("pdg_one_var_height")
    one_param["step size"] = get_exps_from_root("pdg_one_var_step_size")
    one_param["force"] = get_exps_from_root("pdg_one_var_force")
    one_param["dt"] = get_exps_from_root("pdg_one_var_dt")
    one_param["all"] = get_exps_from_root("pdg_all_5")
    #plot_learning_curve(one_param, title = "Finite differences one parameter at a time", root_dir="stats/", cutoff=None)
    
    gp_exps = {}
    gp_exps["more samples"] = get_exps_from_root("gp_more_samples")
    gp_exps["big alpha"] = get_exps_from_root("gp_big_alpha_bad_start")
    #gp_exps["large alpha"] = get_exps_from_root("gp_large_alpha")
    gp_exps["exploration bad start"] = get_exps_from_root("gp_more_explore")
    gp_exps["bad start"] = get_exps_from_root("gp_start_good")
    plot_learning_curve(gp_exps, title = "Select actions with GP", root_dir="stats/", cutoff=None)




    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_stepsize"), title = "Step size one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_desheight"), title = "desired height one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var"), title = "Offset one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var"), title = "Offset one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_dt"), title = "dt one variable", root_dir="stats/", cutoff=None)
    #plot_learning_curve(generate_exp_dictionary_one_vars("pdg_one_var_force"), title = "force one variable", root_dir="stats/", cutoff=None)



