from __future__ import division
from scipy import misc
from random import random

import numpy as np
import pdb
import pickle
from stir_env import StirEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Lambda, concatenate, Conv2D, MaxPooling2D, Convolution3D, LSTM
from keras.optimizers import Adam
import keras.backend as K


WINDOW_LENGTH = 1
ENV_NAME = "mix_cup_5_7_2018"
class Learner:
    def __init__(self, env, nb_actions, input_shape, robot_dims):
        self.env = env
        self.batch_size = 5
        self.build_model(nb_actions,input_shape, robot_dims) 
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])

    def build_model(self, nb_actions, input_shape, robot_dims):
        img1 = Input(shape=input_shape)
        img2 = Input(shape=input_shape)
        action = Input(shape=(nb_actions,))
        robot_state = Input(shape=(robot_dims,))
        #for now, flatten im1+ 2 
        img1_layers = Flatten()(img1)
        img2_layers = Flatten()(img2)
        layer = concatenate([img1_layers, img2_layers, robot_state, action])
        predictions = Dense(1, activation="linear")(layer)
        self.model = Model(inputs=[img1, img2, robot_state, action], outputs = predictions)

    """ returns a list of samples of img1, img2, robot_states, and rewards"""
    def select_random_action(self):
        theta_diff = random()
        curl = random()
        period = random()
        rot = random()
        return (theta_diff, curl, period, rot)

    def select_action(self, img1, img2, robot_state):
        #randomly sample actions, check their value, pick the best 
        num_to_check = 10
        img1s = np.array([img1]*num_to_check)
        img2s = np.array([img2]*num_to_check)
        robot_states = np.array([robot_state]*num_to_check)
        random_actions = np.array([self.select_random_action() for _ in range(num_to_check)] )
        #TODO implement epsilon greedy
        values = self.model.predict([img1s, img2s, robot_states, random_actions])
        best_value_index = np.argmax(values)
        return random_actions[best_value_index]
        
    def collect_batch(self):
        img1s = []
        img2s = []
        robot_states = []
        rewards = []
        actions = []
        for _ in range(self.batch_size):
            #get current state
            img1, img2, robot_state = self.env.create_state()
            best_action = self.select_action(img1, img2, robot_state)
            #predict best action
            self.env.step(best_action)
   
            #then collect reward
            reward = self.env._get_reward()
            #and add to list 
            img1s.append(img1)
            img2s.append(img2)
            robot_states.append(robot_state)
            actions.append(best_action)
            rewards.append(reward)
        return img1s, img2s, robot_states, actions,rewards
            
            

    def train(self):
        numsteps = 100
        # Load dataset
        #batch_size 25, takes 25 samples of states and actions, learn what the value should be after that
        for i in range(numsteps):
            img1s, img2s, robot_states, actions, rewards = self.collect_batch() #collect batch using this policy
            self.model.fit([np.array(img1s), np.array(img2s), np.array(robot_states), np.array(actions)], np.array(rewards), epochs=2, batch_size=self.batch_size) 
        self.model.save_weights(ENV_NAME+'weights.h5f')


    def test_supervised(self,env):
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])
        self.model.load_weights(ENV_NAME+'weights.h5f')
        X, Y = load_supervised_data(actionfile = "actions.pkl", statefile = "states.pkl")
        score = self.model.evaluate(X, Y);
        print("The score is")
         
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
