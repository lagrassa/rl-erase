from __future__ import division
import sys
from reward import entropy
import argparse
from scipy import misc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from random import random, randint

import numpy as np
import pdb
import pickle
#from stir_env import StirEnv
from pour_env import PourEnv
import os

WINDOW_LENGTH = 1

class Learner:
    def __init__(self, env, nb_actions, input_shape, robot_dims):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.robot_dims = robot_dims
        self.env = env
        self.eps_greedy = 0.0
        self.good_reward = 50
        self.bad_reward = 35
        self.exceptional_actions = [(-0.08, 0.6, 0.9, 1500), (-0.11, 0.3, 1.3, 1505)] # initialize these with 2-3 good parameters sets, preferably diverse. 
        #self.params = [0.4, 0.7, 0.2]
        self.params = [-0.08, 1.3, 0.65, 2500]
        #self.params = [-0.15, 0.8, 0.15, 0.11, 2000]
        #self.params = [-0.05, 0.6]
        self.rollout_size = 1
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(alpha=3, kernel=kernel, n_restarts_optimizer=9)

    """ returns a list of theta-diff, curl, period, rot"""
    def select_random_diff(self):
        theta_diff = []
        for i in range(self.nb_actions):
            theta_diff.append(((-1)**randint(0,1))*random())
        return theta_diff

    def select_action(self, params, sigma=0.01, uniform_random=False):
        #randomly sample actions, check their value, pick the best 
        #epsilon greedy for training:
        if uniform_random:
            return uniform_random_sample()
        return np.random.normal(params, [sigma]*len(params))

    
    """selects the best action and the corresponding score"""
    def select_best_action(self,params, obs):
        N = 400
        actions = np.zeros((N, self.nb_actions))
        c = 2
        end_sigma=0.01
        scores = []
        k = -np.log(end_sigma/c)*(1/N) 
        scalars = [1,1,1,1,2000]
        #vary sigma to get more variability 
        for i in range(N):
            sigma=c*np.e**(-k*i)
            action = self.select_action(params, sigma=sigma, uniform_random=True)
            actions[i,:] = action
            sample = np.hstack([action, obs])
            score, stdev = self.gp.predict([sample], return_std=True)
            scores.append(score.item()+stdev.item()) #mean upper bound
        best_score_i = np.argmax(scores)
        best_action = actions[best_score_i, :]
        
        return best_action, scores[best_score_i]

            
            
    
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
    #
    def plot_param_v_reward(self):
        good_params = [-0.05, 0.7, 0.174, 0.11, 1500]
        #go through each set of params and plot params + reward
        corresponding_names = ["offset", "height", "step_size", "timestep", "force"]
        #go through each set of params and plot params + reward
        corresponding_names = ["offset", "height", "step_size", "timestep", "force"]
        num_exps = 4
        num_data_points = 15
        for j in range(len(corresponding_names)):
            if j == 0:
                continue
            good_param = good_params[j]
	    params_to_test = np.linspace(good_param - good_param, good_param + good_param,num_data_points) 
            data = np.zeros((num_exps, num_data_points))
            for i in range(num_exps):
		#test params from 
		reward = []
		for param in params_to_test:
		    test_params = good_params[:]
		    test_params[j] = param
		    rw = self.collect_batch(test_params)[4][0]
		    reward.append(rw)
                data[i,:] = reward
            means = np.mean(data, axis=0)
            stdev = np.std(data, axis=0) 
            plot_line(means, stdev, xaxis = params_to_test, label = corresponding_names[i])
	      

            #plt.plot(params_to_test, reward)
            #plt.title("Rewards over varying parameter: " + corresponding_names[i])
            #plt.show()
 
        
                
    def collect_batch(self, params, avg_l_fn, avg_r_fn):
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
                break #this is okay even though it's not a full rollout because we don't care about the state transitions, since this is super local
            if len(ep_times) == 0: #congrats you didn't have to give up
                ep_times.append(self.rollout_size)
        self.env.reset()
        #log the average V for the rollout, average length episode
        average_reward = sum(rewards)/len(rewards)
        average_reward_file = open(avg_r_fn,"a")
        average_reward_file.write(str(average_reward)+",")
        average_reward_file.close()
    
        return img1s, img2s, robot_states, actions,rewards
            

    def train(self, delta, avg_l_fn,avg_r_fn):
        numsteps = 120
        SAVE_INTERVAL = 11
        PRINT_INTERVAL=5
        LESS_EPS_INTERVAL = 5
        lr = 0.05
        eps = 1e-8
        gti = np.zeros((self.nb_actions,1))
        rewards = None 
        actions = None
        #self.model.load_weights("1fca5a_100weights.h5f") #uncomment if you want to start from scratch
        for i in range(numsteps):
            obs = [] #self.world.base_world.total_bead_mass #much simpler than "world state"
            if i % LESS_EPS_INTERVAL == 0:
                self.eps_greedy = self.eps_greedy/2.0
            if i > 20:
                self.eps_greedy = 0.01
            delta_theta = delta*np.array(self.select_random_diff())
            if random() < self.eps_greedy:
                big_sigma = 2
                perturbed_params =  np.random.normal(self.params, [big_sigma]*len(self.params))
            else:
                self.params, score = self.select_best_action(self.params, obs)
                if score > self.good_reward:
                    self.exceptional_actions.append(self.params)
                elif score < self.bad_reward:
                    #pick a random good action
                    self.params = self.exceptional_actions[randint(0,len(self.exceptional_actions)-1)]

                perturbed_params = self.params + delta_theta
            #neg_perturbed_params = self.params - delta_theta
            _, _, _, _, rewards_up = self.collect_batch(perturbed_params, avg_l_fn, avg_r_fn) #collect batch using this policy
            #_, _, _, _, rewards_down = self.collect_batch(neg_perturbed_params) #collect batch using this policy
            if rewards is None:
                rewards = rewards_up
                #rewards = np.vstack([rewards, rewards_down])
                actions = perturbed_params
                #actions = np.vstack([actions, neg_perturbed_params])
            else:
                #rewards = np.vstack([rewards, np.vstack(rewards_up, rewards_down)])
                #actions = np.vstack([actions, np.vstack(actions_up, actions_down)])
                rewards = np.vstack([rewards, rewards_up])
                actions = np.vstack([actions,perturbed_params])

            #delta_j = compute_j(rewards_up)-compute_j(rewards_down)
            #grad = compute_gradient(delta_theta, delta_j)
            #gti += np.multiply(grad, grad).reshape(gti.shape)
  
            #difference = np.array([delta_theta[i]*grad[i].item() for i in range(delta_theta.shape[0])]) 
            #adagrad_lr = lr/np.sqrt(np.diag(gti)+np.eye(self.nb_actions)*eps)
            #self.params = self.params + np.dot(adagrad_lr,grad)
            if len(actions.shape) > 1:
                self.gp, score = fit_and_evaluate(self.gp, actions, rewards)
		average_length_file = open(avg_l_fn,"a")
		average_length_file.write(str(score)+",")
		average_length_file.close()
            
            if i % SAVE_INTERVAL == 0:
                print("Params:", self.params)

        print("Ending params: ", self.params)
        print(actions)
        print(rewards)

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
def compute_gradient_old(delta_theta_arr, delta_j):
    delta_theta =np.matrix(delta_theta_arr)
    ata = np.dot(delta_theta_arr, delta_theta_arr)
    gradient = ata*delta_theta.T*(delta_j)
    return gradient

def compute_gradient(delta_theta_arr, delta_j):
    return delta_j/delta_theta_arr
         
def parse_args(args):
    parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')
    parser.add_argument('-d','--delta', help='Delta of action',required=False)
    parser.add_argument('-n','--name',help='exp name', required=False)
    parser.add_argument('-v','--visualize',help='put something here to visualize', required=False)
    args = parser.parse_args()
    arg_dict = vars(args)
    visualize = False
    if arg_dict['delta'] is not None:
        delta = float(arg_dict['delta'])
    else:
        delta = 0
    if arg_dict['name'] is not None:
        name = arg_dict['name']
    else:
        name = "PGD"
    if arg_dict['visualize'] is not None:
        visualize = True
    return delta, name, visualize

def uniform_random_sample():
    lower = [-0.1,0,0.4, 1300]
    upper = [0.1,0.8,4,1600]
    sample = []
    for i in range(len(lower)):
        sample.append((upper[i] - lower[i]) * random()+ lower[i])
    return sample

"""Returns a fitted GP and a measure of how well the GP is fitting the data"""
def fit_and_evaluate(gp, actions, rewards):
    score = gp.score(actions, rewards)
    return gp.fit(actions, rewards), score 
        
def main():
     nb_actions = 4; 
     delta, exp_name, visualize = parse_args(sys.argv[1:])
     env = PourEnv(visualize=visualize)
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

     #l.plot_param_v_reward()
     l.train(delta, avg_l_fn, avg_r_fn)

if __name__=="__main__":
     main()
     
