from __future__ import print_function, division
import ipdb as pdb
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
import GPy as gpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


class ActiveLearner(object):
    def __init__(self):
        pass

    def query(self, context):
        pass

    def sample(self, context, N):
        pass

    def retrain(self, x, y):
        pass


class RandomSampler(ActiveLearner):
    def __init__(self, func):
        self.func = func
        self.name = 'random'

    def query(self, context):
        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        x_star = np.random.uniform(xmin, xmax)
        return np.hstack((x_star, context))

    def sample(self, context, N=1):
        return self.query(context)
    def reset_sample(self):
        pass

def run_ActiveLearner(active_learner, context, save_fnm, iters, exp_name="test"):
    '''
    Actively query a function with active learner.
    Args:
        active_learner: an ActiveLearner object.
        context: the current context we are testing for the function.
        save_fnm: a file name string to save the queries.
        iters: total number of queries.
    '''
    # Retrieve the function associated with active_learner
    EXP_NAME=exp_name
    func = active_learner.func
    # Queried x and y
    xq, yq = None, None
    # All the queries x and y
    if len(func.discrete_contexts) > 0:
        xx = np.zeros((0, func.x_range.shape[1]+len(func.discrete_contexts[0])))
    else:
        xx = np.zeros((0, func.x_range.shape[1]))
    yy = np.zeros(0)
    reward_list = []
    sample_list = []
    diversity_list = []
    SAVE_INTERVAL=2
    #recover model TODO
    pdb.set_trace()
    active_learner.restore_model(EXP_NAME=exp_name)
    # Start active queries
    for i in range(iters):
        try:
            active_learner.retrain(xq, yq)
        except:
            print("Could not retrain")
            pass
        xq = active_learner.query(context)
        yq = func(xq)
        xx = np.vstack((xx, xq))
        yy = np.hstack((yy, yq))
        sample, diversity  = active_learner.sample(context)
     
        reward = func(sample)
        reward_list.append(reward)
        sample_list.append(sample)
        diversity_list.append(diversity)
        print('i={}, xq={}, yq={}'.format(i, xq, yq))
         
        pickle.dump((xx, yy, context), open(save_fnm, 'wb'))
        if i % SAVE_INTERVAL == 0 and i > 0:
	    np.save("data/rewards_"+EXP_NAME+".npy", reward_list)
	    np.save("data/diversity_"+EXP_NAME+".npy", diversity_list)
	    np.save("data/sample_"+EXP_NAME+".npy", sample_list)
	    np.save("data/xx_"+EXP_NAME+".npy", xx)
	    np.save("data/yy_"+EXP_NAME+".npy", yy)
