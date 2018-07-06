from __future__ import division
import sys
from reward import entropy
import argparse
from scipy import misc
from random import random, randint

import numpy as np
import pdb
import pickle
from stir_env import StirEnv
import os
from keras.callbacks import CSVLogger

WINDOW_LENGTH = 1

class Learner:
    def __init__(self, env, nb_actions, input_shape, robot_dims):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.robot_dims = robot_dims
        self.env = env
        self.eps_greedy = 0.0
        self.params = [0,0,0.5,0, 500]
        self.rollout_size = 2

    """ returns a list of theta-diff, curl, period, rot"""
    def select_random_action(self):
        theta_diff = random()
        curl = -1**(randint(0,1))*3.14*random()
        period = random()
        rot = -1**(randint(0,1))*3.14*random()
        force = random()
        return (theta_diff, curl, period, rot, force)

    def select_action(self, params):
        #randomly sample actions, check their value, pick the best 
        #epsilon greedy for training:
        sigma = 0.01
        if random() <= self.eps_greedy:
            return self.select_random_action()
        else:
            return np.random.normal(params, [sigma]*len(self.params))
    
    def collect_test_batch(self):
        beads_in= []
        entropies = []
        episode_over = False
        beads_ratio = None
        entropy = None
        for i in range(self.rollout_size):
            #get current state
            img1, img2, robot_state = self.env.create_state()
            best_action = self.select_action(img1, img2, robot_state)
            #predict best action
            if not episode_over:
              
                _, beads_ratio, entropy, episode_over, _ = self.env.step(best_action)
                if episode_over:
                    self.env.reset()
            beads_in.append(beads_ratio)
            entropies.append(entropy)
        self.env.reset()
        return beads_in, entropies
            
            
                
    def collect_batch(self, params):
        img1s = np.zeros( (self.rollout_size,)+self.input_shape)
        img2s =  np.zeros( (self.rollout_size,)+ self.input_shape)
        robot_states = np.zeros((self.rollout_size, self.robot_dims))
        rewards = np.zeros(self.rollout_size)
        actions = np.zeros((self.rollout_size, self.nb_actions))
        time_since_last_reset = 0
        ep_times = []
        for i in range(self.rollout_size):
            #get current state
            img1, img2, robot_state = self.env.create_state()
            best_action = self.select_action(params)
            #predict best action
          
            _, reward, episode_over, _ = self.env.step(best_action)

            #then collect reward
            
	    img1s[i,:,:,:] = img1
	    img2s[i,:,:,:] = img2
	    robot_states[i, :] = robot_state
	    actions[i, :] = best_action
	    rewards[i] = reward
            if i > 2 and episode_over:
                ep_times.append(time_since_last_reset)
                time_since_last_reset = 0
                self.env.reset() #this is okay even though it's not a full rollout because we don't care about the state transitions, since this is super local
            if len(ep_times) == 0: #congrats you didn't have to give up
                ep_times.append(self.rollout_size)
        #log the average V for the rollout, average length episode
        average_length = sum(ep_times)/len(ep_times)
        average_reward = sum(rewards)/len(rewards)
	average_length_file = open(avg_l_fn,"a")
	average_reward_file = open(avg_r_fn,"a")
        average_length_file.write(str(average_length)+",")
        average_reward_file.write(str(average_reward)+",")
        average_length_file.close()
        average_reward_file.close()
    
        return img1s, img2s, robot_states, actions,rewards
            

    def train(self, delta):
        numsteps = 100
        SAVE_INTERVAL = 11
        PRINT_INTERVAL=5
        lr = 1
        eps = 1e-8
        gti = np.zeros((self.nb_actions,1))
     
        csv_logger = CSVLogger('log'+EXP_NAME+'.csv', append=True, separator=';')
        #self.model.load_weights("1fca5a_100weights.h5f") #uncomment if you want to start from scratch
        for i in range(numsteps):
            print("On interval",i)
            if i > 20:
                self.eps_greedy = 0.01
            delta_theta = delta*np.array(self.select_random_action())
            perturbed_params = self.params + delta_theta
            _, _, _, _, rewards_up = self.collect_batch(perturbed_params) #collect batch using this policy
            _, _, _, _, rewards_down = self.collect_batch(-1*perturbed_params) #collect batch using this policy
            delta_j = compute_j(rewards_up)-compute_j(rewards_down)
            grad = compute_gradient(delta_theta, delta_j)
            gti += np.multiply(grad, grad)
  
            difference = np.array([delta_theta[i]*grad[i].item() for i in range(delta_theta.shape[0])]) 
            adagrad_lr = lr/np.sqrt(np.diag(gti)+np.eye(self.nb_actions)*eps)
  
            self.params = self.params + np.dot(adagrad_lr,difference)
            
            if i % SAVE_INTERVAL == 0:
                print("Params:", self.params)
        print("Ending params: ", self.params)

    def test_model(self,filename):
        #do 10 rollouts, 
        #how many beads left in?
        #what is the entropy?
        #run all of the models + totall random
        self.model.load_weights(filename)       
        bead_results_file = open("policy_results/"+filename[:-4]+"control_bead_results.py","w" )
        entropy_results_file = open("policy_results/"+filename[:-4]+"control_entropy_results.py", "w")
        beads_over_time_list = []
        entropy_over_time_list = []
        numtrials = 10
        for i in range(numtrials):
            print("On trial #", i)
            beads_over_time, entropy_over_time = self.collect_test_batch()
            beads_over_time_list.append(beads_over_time)
            entropy_over_time_list.append(entropy_over_time)
        bead_results_file.write(str(beads_over_time_list))
        entropy_results_file.write(str(entropy_over_time_list))
        bead_results_file.close()
        entropy_results_file.close()
def compute_j(rewards):
    gamma = 0.9
    rewards  = rewards[::-1]
    H = len(rewards)
    J = sum([1/H*gamma**(k)*rewards[k] for k in range(len(rewards))])
    return J

    
"""assumes delta_theta is an np.array"""	
def compute_gradient(delta_theta_arr, delta_j):
    delta_theta =np.matrix(delta_theta_arr)
    ata = np.dot(delta_theta_arr, delta_theta_arr)
    gradient = ata*delta_theta.T*(delta_j)
    return gradient

         
def parse_args(args):
    parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')
    parser.add_argument('-d','--delta', help='Delta of action',required=False)
    parser.add_argument('-n','--name',help='exp name', required=False)
    args = parser.parse_args()
    arg_dict = vars(args)
    if arg_dict['delta'] is not None:
        delta = float(arg_dict['delta'])
    else:
        delta = 0.1
    if arg_dict['name'] is not None:
        name = arg_dict['name']
    else:
        name = "PGD"
    return delta, name

if __name__=="__main__":
     nb_actions = 5; #just 3 atm 6; #control period held and angle,curl, plus 3 dx dy dz
     delta, exp_name = parse_args(sys.argv[1:])
     visualize=False
     env = StirEnv(visualize=visualize)
     state_shape = list(env.world_state[0].shape)
     robot_dims = env.robot_state.shape[0]
     l = Learner(env,nb_actions, tuple(state_shape), robot_dims)
     #sets up logging
     EXP_NAME = exp_name #I'm going to be less dumb and start naming experiment names after commit hashes
     avg_l_fn = "stats/average_length"+EXP_NAME+".py"
     avg_r_fn= "stats/average_reward"+EXP_NAME+".py"
     for myfile in [avg_l_fn, avg_r_fn]:
 	 if os.path.isfile(myfile):
	     os.remove(myfile)
     l.train(delta)
     
#l.train()
