from __future__ import division
import sys
import rospy
import argparse
#from keras.layers import Dense, Activation
#from keras.optimizers import Adam
from scipy import misc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from random import random, randint

import numpy as np
import pdb
import pickle
#from stir_env import StirEnv
from real_pour_env import PourEnv
import os

WINDOW_LENGTH = 1

class Learner:
    def __init__(self, env, nb_actions, input_length):
        self.input_length = input_length
        self.nb_actions = nb_actions
        self.env = env
        self.eps_greedy = 0.0
        self.good_reward = 50
        self.bad_reward = 35
        self.exceptional_actions = [(-0.08, 0.6, 0.9, 1500), (-0.11, 0.3, 1.3, 1505)] # initialize these with 2-3 good parameters sets, preferably diverse. 
        self.action_mean = [] #[0.3, 400, 350, 0.04, 0.07]


        self.rollout_size = 1
        kernel = C(0.01, (1e-3, 1e3)) * RBF(0.6, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=1e-1)
        """
        self.model = Sequential()
        self.model.add(Dense(8, input_dim=self.input_length, activation="relu"))
        self.model.add(Dense(4, activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        opt = Adam(lr=1)
        self.model.compile(loss="mae", optimizer=opt, metrics = ['accuracy'])
        """
 

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
    def select_best_action(self, action_mean, obs):
        N = 700
        c = 2
        end_sigma=0.01
        l = 1
        k = -np.log(end_sigma/c)*(1/N) 
        action_set = uniform_random_sample(N)
        if len(obs) > 1:
	    obs_vec = obs[0]*np.ones((N,1))
	    for i in range(1,len(obs)):
		new_obs = obs[i]*np.ones((N,1))
		obs_vec = np.hstack([obs_vec, new_obs])
            samples = np.hstack([action_set, obs_vec])
        else:
            samples = action_set
 
        scores, stdevs = self.gp.predict(samples, return_std = True)
        if len(scores.shape) > 1:
            scores += l*stdevs.reshape(stdevs.shape[0], 1)
        else:
            scores += l*stdevs
        best_score_i = np.argmax(scores)
        best_action = action_set[best_score_i, :]
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
 
        
                
    def collect_batch(self, action, avg_l_fn, avg_r_fn):
        rewards = np.zeros(self.rollout_size)
        time_since_last_reset = 0
        ep_times = []
        for i in range(self.rollout_size):
            #get current state
            #predict best action
          
            _, reward, episode_over, _ = self.env.step(action)
            #then collect reward
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
    
        return rewards
            

    def train(self, delta, avg_l_fn,avg_r_fn, exp_name="EXP"):
        numsteps = 5
        SAVE_INTERVAL = 11
        PRINT_INTERVAL=5
        LESS_EPS_INTERVAL = 5
        lr = 0.05
        eps = 1e-8
        gti = np.zeros((self.nb_actions,1))
        samples, rewards= load_datasets(exp_name)#np.load("dataset/samples.npy")
        #self.model.load_weights("1fca5a_100weights.h5f") #uncomment if you want to start from scratch
        for i in range(numsteps):
            print("on step", i)
            obs = self.env.observe_state()
            if i % LESS_EPS_INTERVAL == 0:
                self.eps_greedy = self.eps_greedy/2.0
            if i > 20:
                self.eps_greedy = 0.01
            delta_theta = delta*np.array(self.select_random_diff())
            if random() < self.eps_greedy:
                big_sigma = 2
            else:
                self.action_mean, _ = self.select_best_action(self.action_mean, obs)

            rewards_up = self.collect_batch(self.action_mean, avg_l_fn, avg_r_fn) #collect batch using this policy
            print("rewards", rewards_up)
            #_, _, _, _, rewards_down = self.collect_batch(neg_perturbed_params) #collect batch using this policy
            if rewards is None:
                rewards = rewards_up
                samples = np.hstack([self.action_mean, obs])
            else:
                rewards = np.vstack([rewards, rewards_up])
                sample = np.hstack([self.action_mean, obs])
                try:
                    samples = np.vstack([samples,sample])
                except:
                    pdb.set_trace()

            min_samples = 5
            if len(samples.shape) > 1 and samples.shape[0] > min_samples:
                #score = fit_and_evaluate(self.model, samples, rewards)
                self.gp, score = fit_and_evaluate(self.gp, samples, rewards)
		average_length_file = open(avg_l_fn,"a")
		average_length_file.write(str(score)+",")
		average_length_file.close()
            
            if i % SAVE_INTERVAL == 0:
                print("Params:", self.action_mean)

        print("Ending params: ", self.action_mean)
        np.save("dataset/samples"+exp_name+".npy",samples)
        np.save("dataset/rewards"+exp_name+".npy",rewards)

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

def uniform_random_sample(n=1):
    #close_num, close_force, lift_force, forward_offset, height, vel, grasp_height, grasp_depth
    lower = [0,0.08]
    upper = [0.2, 0.14]
    sample = np.zeros((n,len(lower)))
    for i in range(len(lower)):
        sample[:,i] = (upper[i] - lower[i]) * np.random.rand(n)+ lower[i]
    return sample

    



"""Returns a fitted GP and a measure of how well the GP is fitting the data"""
def fit_and_evaluate(gp, actions, rewards):
    score = custom_score(gp, actions, rewards)
    print("custom score", score)
    return gp.fit(actions, rewards), score 

def custom_score(gp, actions, rewards):
    predictions = gp.predict(actions)
    squared_differences = (predictions-rewards.reshape(predictions.shape))**2
    score =  sum(squared_differences)/squared_differences.shape[0]
    if len(squared_differences.shape) > 1:
        return score.item()
    else:
        return score
    


def fit_and_evaluate_nn(nn, samples, rewards):
    score = nn.evaluate(samples, rewards)
    nn.fit(samples, rewards, epochs = 100, batch_size = int(samples.shape[0]/5.0))
    return score 
#samples, rewards returned if there are some, None if there are none with that name
def load_datasets(exp_name):
    files = os.listdir('dataset')
    filenames = []
    samples = None
    rewards = None
    for f in files:
        if exp_name in f and "rewards" in f:
            filenames.append(f)      
    for fname in filenames:
        root = fname[len("rewards"):]
        samples_set = np.load("dataset/"+"samples"+root)
        rewards_set = np.load("dataset/"+"rewards"+root)
        if samples is None:
            samples = samples_set 
            rewards = rewards_set
        else:
            samples = np.vstack([samples, samples_set])
            rewards = np.vstack([rewards, rewards_set])
    if samples is None:
        print("Warning: did not find any datasets with the experiment name, resorting to training from scratch")
    return samples, rewards

def main():
     rospy.init_node("make_pour")
     nb_actions = 2; 
     delta, exp_name, visualize = parse_args(sys.argv[1:])
     env = PourEnv(visualize=visualize)
     state_length = 0# env.observe_state().shape[0]
     input_length = state_length+nb_actions
     l = Learner(env,nb_actions, input_length )
     #sets up logging
     EXP_NAME = exp_name #I'm going to be less dumb and start naming experiment names after commit hashes
     avg_l_fn = "stats/average_length"+EXP_NAME+".py"
     avg_r_fn= "stats/average_reward"+EXP_NAME+".py"
     for myfile in [avg_l_fn, avg_r_fn]:
          if os.path.isfile(myfile):
             os.remove(myfile)

     #l.plot_param_v_reward()
     l.train(delta, avg_l_fn, avg_r_fn, exp_name=EXP_NAME)

if __name__=="__main__":
     main()
     
