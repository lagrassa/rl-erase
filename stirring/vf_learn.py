from __future__ import division
import keras

import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 30} )
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

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
EXP_NAME = "1fca5a_nonlinear_2" #I'm going to be less dumb and start naming experiment names after commit hashes
avg_l_fn = "average_length"+EXP_NAME+".py"
avg_r_fn= "average_reward"+EXP_NAME+".py"
for myfile in [avg_l_fn, avg_r_fn]:
    if os.path.isfile(myfile):
	os.remove(myfile)

class Learner:
    def __init__(self, env, nb_actions, input_shape, robot_dims):
        self.env = env
        self.batch_size = 25
        self.rollout_size = 50
        self.input_shape = input_shape
        self.robot_dims = robot_dims
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
        img1_layers = Conv2D(16, 5,5, activation='relu')(img1)
        img1_layers = MaxPooling2D((2,2), 2)(img1_layers)
        img1_layers = Conv2D(32, 2,2, activation='relu')(img1_layers)
        img1_layers = MaxPooling2D((2,2), 2)(img1_layers)
        #for now, flatten im1+ 2 
        img1_layers = Flatten()(img1_layers)
        img2_layers = Conv2D(16, 5,5, activation='relu')(img2)
        img2_layers = MaxPooling2D((2,2), 2)(img2_layers)
        img2_layers = Conv2D(32, 2,2, activation='relu')(img2_layers)
        img2_layers = MaxPooling2D((2,2), 2)(img2_layers)
        #for now, flatten im2+ 2 
        img2_layers = Flatten()(img2_layers)

        img2_layers = Flatten()(img2)
        layer = concatenate([img1_layers, img2_layers, robot_state, action])
        predictions = Dense(32, activation="relu")(layer)
        predictions = Dense(1, activation="linear")(layer)
        self.model = Model(inputs=[img1, img2, robot_state, action], outputs = predictions)

    """ returns a list of samples of img1, img2, robot_states, and rewards"""
    def select_random_action(self):
        theta_diff = 3.14*random()
        curl = 3.14*random()
        period = random()
        rot = 3.14*random()
        return (theta_diff, curl, period, rot)

    def select_action(self, img1, img2, robot_state):
        #randomly sample actions, check their value, pick the best 
        num_to_check = 300
        img1s = np.array([img1]*num_to_check)
        img2s = np.array([img2]*num_to_check)
        robot_states = np.array([robot_state]*num_to_check)
        random_actions = np.array([self.select_random_action() for _ in range(num_to_check)] )
        #TODO implement epsilon greedy
        values = self.model.predict([img1s, img2s, robot_states, random_actions])
        best_value_index = np.argmax(values)
        return random_actions[best_value_index]
    
        
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
        self.model.load_weights("1fca5a_100weights.h5f") #uncomment if you want to start from scratch
        for i in range(numsteps):
            img1s, img2s, robot_states, actions, rewards = self.collect_batch() #collect batch using this policy
            self.model.fit([img1s, img2s, robot_states, actions], rewards, epochs=100, batch_size=self.batch_size, callbacks=[csv_logger], verbose=0) 
            print("On interval",i)
            if i % SAVE_INTERVAL == 0:
                self.model.save_weights(EXP_NAME+'_'+str(i)+'weights.h5f')


    def test_supervised(self,env):
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])
        self.model.load_weights(EXP_NAME+'weights.h5f')
        X, Y = load_supervised_data(actionfile = "actions.pkl", statefile = "states.pkl")
        score = self.model.evaluate(X, Y);
         
        state = env.reset()
        state = state.reshape((1,state.shape[0],state.shape[1],state.shape[2]))
        numsteps = 120;
        for i in range(numsteps):
            env.render()
            action = int(round(self.model.predict(state).item()))
            state  = env.step(action)[0]
            state = state.reshape((1,state.shape[0],state.shape[1],state.shape[2]))
   

# Finally, evaluate our algorithm for 10 episodes.
              
if __name__=="__main__":
     nb_actions = 4; #just 3 atm 6; #control period held and angle,curl, plus 3 dx dy dz
     visualize=False
     env = StirEnv(visualize=visualize)
     state_shape = list(env.world_state[0].shape)
     robot_dims = env.robot_state.shape[0]
     l = Learner(env,nb_actions, tuple(state_shape), robot_dims)
     l.train()
