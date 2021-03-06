from __future__ import division
import keras
import sys
from reward import entropy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))


from scipy import misc
from random import random

import numpy as np
import pdb
import pickle
from stir_env import StirEnv
import os

from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Lambda, concatenate, Conv2D, MaxPooling2D, Convolution3D, LSTM
from keras.optimizers import Adam





WINDOW_LENGTH = 1
EXP_NAME = "7233d3_nonlinear_force_less_restricted_very_simple_1" #I'm going to be less dumb and start naming experiment names after commit hashes
avg_l_fn = "average_length"+EXP_NAME+".py"
avg_r_fn= "average_reward"+EXP_NAME+".py"
for myfile in [avg_l_fn, avg_r_fn]:
    if os.path.isfile(myfile):
	os.remove(myfile)

class Learner:
    def __init__(self, env, nb_actions, input_shape, robot_dims):
        self.env = env
        self.batch_size = 25
        self.rollout_size = 20 #5
        self.input_shape = input_shape
        self.robot_dims = robot_dims
        self.eps_greedy = 0.0
        self.nb_actions = nb_actions
        self.build_model(nb_actions,input_shape, robot_dims) 
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])

    #Let's make these images kinda small
    def build_model(self, nb_actions, input_shape, robot_dims):
        img1 = Input(shape=input_shape)
        img2 = Input(shape=input_shape)
        action = Input(shape=(nb_actions,))
        robot_state = Input(shape=(robot_dims,))
        #Conv layer and max pooling layer for img1, expecting 50x50 cup
        img1_layers = Conv2D(32, 5,5, activation='linear')(img1)
        img1_layers = MaxPooling2D((2,2), 2)(img1_layers)
        #for now, flatten im1+ 2 
        img1_layers = Flatten()(img1_layers)
        img2_layers = Conv2D(32, 5,5, activation='linear')(img2)
        img2_layers = MaxPooling2D((2,2), 2)(img2_layers)
        #for now, flatten im2+ 2 
        #img2_layers = Flatten()(img2)
        #no visual input in this one
        layer = concatenate([robot_state, action])
        layer = Dense(64, activation="relu")(layer)
        layer = Dense(32, activation="relu")(layer)
        layer = Dense(32, activation="relu")(layer)
        predictions = Dense(1, activation="linear")(layer)
        self.model = Model(inputs=[img1, img2, robot_state, action], outputs = predictions)


    """ returns a list of samples of img1, img2, robot_states, and rewards"""
    def select_random_action(self):
        theta_diff = 3.14*random()
        curl = -1**(randint(0,1))*3.14*random()
        period = random()
        rot = -1**(randint(0,1))*3.14*random()
        return (theta_diff, curl, period, rot)

    def select_action(self, img1, img2, robot_state):
        #randomly sample actions, check their value, pick the best 
        #epsilon greedy for training:
        if random() <= self.eps_greedy:
            num_to_check = 1
        else:
            num_to_check = 600
        img1s = np.array([img1]*num_to_check)
        img2s = np.array([img2]*num_to_check)
        robot_states = np.array([robot_state]*num_to_check)
        random_actions = np.array([self.select_random_action() for _ in range(num_to_check)] )
        #TODO implement epsilon greedy
        values = self.model.predict([img1s, img2s, robot_states, random_actions])
        best_value_index = np.argmax(values)
        return random_actions[best_value_index]
    
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
                print("HAS TAKEN STEP")
                if episode_over:
                    self.env.reset()
            beads_in.append(beads_ratio)
            entropies.append(entropy)
        self.env.reset()
        return beads_in, entropies
            
            
                
    def collect_batch(self):
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
            best_action = self.select_action(img1, img2, robot_state)
            #predict best action
            _, reward, episode_over, _ = self.env.step(best_action)
            #then collect reward
            
	    img1s[i,:,:,:] = img1
	    img2s[i,:,:,:] = img2
	    robot_states[i, :] = robot_state
	    actions[i, :] = best_action
	    rewards[i] = reward
            if i > 3 and episode_over:
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
            
            

    def train(self):
        numsteps = 1000
        SAVE_INTERVAL = 100
        PRINT_INTERVAL=5
        # Load dataset
        #batch_size 25, takes 25 samples of states and actions, learn what the value should be after that
        csv_logger = CSVLogger('log'+EXP_NAME+'.csv', append=True, separator=';')
        #self.model.load_weights("1fca5a_100weights.h5f") #uncomment if you want to start from scratch
        for i in range(numsteps):
            if i > 100:
                self.eps_greedy = 0.01
            img1s, img2s, robot_states, actions, rewards = self.collect_batch() #collect batch using this policy
            self.model.fit([img1s, img2s, robot_states, actions], rewards, epochs=100, batch_size=self.batch_size, callbacks=[csv_logger], verbose=0) 
            print("On interval",i)
            if i % SAVE_INTERVAL == 0:
                self.model.save_weights(EXP_NAME+'_'+str(i)+'weights.h5f')

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
         
# Finally, evaluate our algorithm for 10 episodes.
              
if __name__=="__main__":
     nb_actions = 4; #just 3 atm 6; #control period held and angle,curl, plus 3 dx dy dz
     visualize=False
     env = StirEnv(visualize=visualize)
     state_shape = list(env.world_state[0].shape)
     robot_dims = env.robot_state.shape[0]
     l = Learner(env,nb_actions, tuple(state_shape), robot_dims)
     l.test_model(sys.argv[1])
     
#l.train()
