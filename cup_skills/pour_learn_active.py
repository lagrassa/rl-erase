# Author: Zi Wang
import cPickle as pickle
import numpy as np
import os
import sys
import pdb
import helper as helper
from active_learner import run_ActiveLearner


def gen_data(expid, func, n_data, save_fnm):
    '''
    Generate initial data for a function associated the experiment.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        n_data: number of data points to generate.
        save_fnm: a file name string where the initial data will be
        saved.
    '''
    print('Generating data...')
    xx, yy = helper.gen_data(func, n_data, parallel=True)
    pickle.dump((xx, yy), open(save_fnm, 'wb'))

def run_exp(expid, exp, method, n_init_data, iters, exp_name='test'):
    '''
    Run the active learning experiment.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        method: learning method, including 
            'nn_classification': a classification neural network 
                based learning algorithm that queries the input that has 
                the largest output.
            'nn_regression': a regression neural network based 
                learning algorithm that queries the input that has 
                the largest output.
            'gp_best_prob': a Gaussian process based learning algorithm
                that queries the input that has the highest probability of 
                having a positive function value.
            'gp_lse': a Gaussian process based learning algorithm called
                straddle algorithm. See B. Bryan, R. C. Nichol, C. R. Genovese, 
                J. Schneider, C. J. Miller, and L. Wasserman, "Active learning for 
                identifying function threshold boundaries," in NIPS, 2006.
            'random': an algorithm that query uniformly random samples.
        n_data: number of data points to generate.
        save_fnm: a file name string where the initial data will be
        saved.
    '''
    dirnm = 'data/'
    if not os.path.isdir(dirnm):
        os.mkdir(dirnm)
    init_fnm = os.path.join(
            dirnm, '{}_init_data_{}.pk'.format(exp, expid))
    func = helper.get_func_from_exp(exp)
    gen_data(expid, func, n_init_data, init_fnm)

    initx, inity = pickle.load(open(init_fnm, 'rb'))


    active_learner = helper.get_learner_from_method(method, initx, inity, func)

    # file name for saving the learning results
    learn_fnm = os.path.join(
            dirnm, '{}_{}_{}.pk'.format(exp, method, expid))

    # get a context
    context = helper.gen_context(func)

    # start running the learner
    print('Start running the learning experiment...')
    print("exp name", exp_name)
    run_ActiveLearner(active_learner, context, learn_fnm, iters, exp_name=exp_name)

def sample_exp(expid=0, exp='pour', method='gp_lse', N=1, context=None, exp_name=None):
    '''
    Sample from the learned model.
    Args:
        expid: ID of the experiment; e.g. 0, 1, 2, ...
        exp: name of the experiment; e.g. 'pour', 'scoop'.
        method: see run_exp.
    '''
    func = helper.get_func_from_exp(exp)
    active_learner = helper.get_learner_from_method(method, None, None, func)
    active_learner.restore_model(EXP_NAME=exp_name)
    active_learner.retrain()
   
    # get a context
    if context is None:
        context = helper.gen_context(func)
    # Enable gui
    func.do_gui = True
    samples = []
    for i in range(N):
        x, diversity = active_learner.sample(context)
        samples.append(x)
    return samples
        #func(x)


def main():
    exp = 'pour'
    method = 'gp_lse'
    expid = 0
    n_init_data = 2
    iters = 2
    exp_name = sys.argv[1]
    run_exp(expid, exp, method, n_init_data, iters, exp_name=exp_name)
    context=(np.array([]), [(1.657, 0.6545)])
    #print(sample_exp(context=context, exp_name="short"))

if __name__ == '__main__':
    main()
