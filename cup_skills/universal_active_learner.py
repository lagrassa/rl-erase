#Author: Alex LaGrassa, using some of Zi Wang's code.
import helper
import numpy as np
import cPickle as pickle
import os
import sys
import ipdb as pdb
import helper as helper
from pouring_world import PouringWorld as Pour
from active_learner import run_ActiveLearner


#xrange for skill
#param_idx, cont_context_idx, 
#discrete_contexts a list of contex

class LearnableSkill():

    def __init__(self, skill):
        self.skill = skill
    
  
    '''
    Running this code should generate samples to train a model. Once active_learn is ran,  it sets self.model to an appropriate model to sample from. This calls our active learner. 

     We would fill in this code to interact with the skill
    @param skill - the Skill being learned
    @param num_iters - number of iterations to train for (could come up with something better)
    
    '''
    def active_learn(self,  num_iters, transfer_previous=False, n_initial_data=30, alg = 'gp_lse', restore_name="test"):
        dirnm = 'data/'
        exp = "learner"
        expid = 0
        if not os.path.isdir(dirnm):
            os.mkdir(dirnm)
        init_fnm = os.path.join(dirnm, '{}_init_data_{}.pk'.format(exp, expid))
        self._gen_initial_data(self.func, n_initial_data, init_fnm)
        initx, inity = pickle.load(open(init_fnm, 'rb'))
        self.active_learner = self._setup_active_learner(alg=alg, initx=initx, inity=inity)
        # file name for saving the learning results
        learn_fnm = os.path.join(dirnm, '{}_{}_{}.pk'.format(exp, alg, expid))

        # get a context
        context = helper.gen_context(self.func)
        run_ActiveLearner(self.active_learner, context, learn_fnm, num_iters, exp_name=restore_name)

    def _setup_active_learner(self, alg='gp_lse', initx=np.array([[]]), inity=np.array([[]])):
        self.func = helper.function_from_skill(self.skill) #thin wrapper to have the right names 
        return  helper.get_learner_from_method(alg, initx, inity, self.func)
 
    def restore_experience(self, restore_name = "test"):
        self.active_learner = self._setup_active_learner()
        self.active_learner.restore_model(EXP_NAME=restore_name)
       

    def _gen_initial_data(self,func, n_data, save_fnm):
        print("Generating data...")
        xx, yy = helper.gen_data(func, n_data, parallel=False)
        pickle.dump((xx, yy), open(save_fnm, 'wb'))
     
    '''
    @param fixed_params - the parameters that we condition on, of the form (cont_parameters, discrete_parameters)
    whether both are lists or tuples
    '''
    #Should call Zi's code to generate samples 
    def diverse_sampler(self,cont_params, discrete_params):
        
        fixed_params = (cont_params, discrete_params)
        sample, _ = self.active_learner.sample(fixed_params)
        return sample
      
    '''
    Return true if the parameters satisfy the constraints with high probability
    Uses the learned model to estimate the probability that the skill will be successful 
     OR
     Tests the result empirically

    '''
    def satisfies_constraints(self, params, check_sim=False):
        pass

def main():
    skill = Pour() #fill in with Push
    restore_name="short"
    ls = LearnableSkill(skill)
    #num_train = 3
    #ls.active_learn(num_train, restore_name=restore_name, n_initial_data=2)
    ls.restore_experience(restore_name=restore_name)
    discrete_params = (1.4857, 0.5964)
    cont_params = []
    sample= ls.diverse_sampler(cont_params, discrete_params)
    skill.execute(sample)
   

if __name__ == "__main__":
    main()
