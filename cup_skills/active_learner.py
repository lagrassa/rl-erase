from __future__ import print_function, division
import pdb
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


def run_ActiveLearner(active_learner, context, save_fnm, iters):
    '''
    Actively query a function with active learner.
    Args:
        active_learner: an ActiveLearner object.
        context: the current context we are testing for the function.
        save_fnm: a file name string to save the queries.
        iters: total number of queries.
    '''
    # Retrieve the function associated with active_learner
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
        sample = active_learner.sample_adaptive(context)
        reward = func(sample)
        reward_list.append(reward)
        print('i={}, xq={}, yq={}'.format(i, xq, yq))
 
        pickle.dump((xx, yy, context), open(save_fnm, 'wb'))

    np.save("rewards_1.npy", reward_list)
