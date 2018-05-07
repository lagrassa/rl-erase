from __future__ import division
from scipy import misc
import ddpg_models

import numpy as np
import pdb
import pickle
from stir_env import StirEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Lambda, concatenate, Conv2D, MaxPooling2D, Convolution3D, LSTM
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent, NAFAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 1
ENV_NAME = "mix_cup_flakey"
class Learner:
    def __init__(self, env, nb_actions, input_shape):
        env_shape = env.create_state().shape
        self.dqn = ddpg_models.construct_agent(env, env_shape, nb_actions, input_shape)

    def train(self, env, nb_actions):
        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=True, verbose=3)
           # After training is done, we save the final weights one more time.
        self.dqn.save_weights(weights_filename, overwrite=True)

    def train_supervised(self, env):
        numsteps = 1750000
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])
        # Load dataset
        X, Y = load_supervised_data(actionfile = "actions.pkl", statefile = "states.pkl")

        #load image of robot and environment in 2 channels, call that X
        #initialize tensors
        self.model.fit(X, Y, epochs=90, batch_size=2) 
        #given the environment, predict the next step from the policy
        #step according to the policy
        #back propagate
        #...I forgot that keras does this all for you
        self.model.save_weights(ENV_NAME+'weights.h5f')

    def test(self,env):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=.4), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=10000)
        amount_memory = 5000000
        memory = SequentialMemory(limit=amount_memory)
        processor = EmptyProcessor()
        weights_filename = "placeholder"
        self.dqn.load_weights(weights_filename)
        self.dqn.test(env, nb_episodes=15, visualize=True)

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
   
            
              

def load_supervised_data(actionfile=None, statefile=None):
    states = np.array(pickle.load(open(statefile)))
    actions = np.array(pickle.load(open(actionfile)))
    return states, actions

# Finally, evaluate our algorithm for 10 episodes.
              
if __name__=="__main__":
     nb_actions = 6; #control period held and angle,curl, plus 3 dx dy dz
     visualize=False
     
     env = StirEnv(visualize=visualize)
     state_shape = list(env.world_state.shape)
     robot_dims = env.robot_state.shape[0]
     state_shape[-1] +=1 
     l = Learner(env,nb_actions, tuple(state_shape))
     l.train(env, nb_actions)
