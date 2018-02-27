#mostly based on the Atari RL example given on the keras-rl github at
#https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
from __future__ import division
from scipy import misc

import numpy as np
import pdb
from board_env import BoardEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 3
ENV_NAME = "board"

#Currently implements the methods by returning what was given
class EmptyProcessor(Processor):
    def process_observation(self, observation):
        #observation here is a board state, maybe do some form of 
        #data augmentation at some point, but right now not going to
        return observation
    def process_state_batch(self, batch):
        return batch
    def process_reward(self, reward):
        return reward


class Learner():
    def __init__(self, input_shape, window_length, nb_actions):
        self.build_model(input_shape, window_length, nb_actions)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=1000000)
        amount_memory = 10000000
        memory = SequentialMemory(limit=amount_memory, window_length=WINDOW_LENGTH)
        processor = EmptyProcessor()
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,train_interval=4, delta_clip=1.)
        self.dqn.compile(Adam(lr=.00001), metrics=['mae'])

    #entirely taken from the Atari example form Mnih et al's paper
    def build_model(self, input_shape_input, window_length, nb_actions):
        input_shape = (window_length,)+input_shape_input
        self.model = Sequential()
        if K.image_dim_ordering() == 'tf':
            # (width, height, channels)
            self.model.add(Permute((2, 3, 1), input_shape=input_shape))
        elif K.image_dim_ordering() == 'th':
            # (channels, width, height)
            self.model.add(Permute((1, 2, 3), input_shape=input_shape))
        else:
            raise RuntimeError('Unknown image_dim_ordering.')
        self.model.add(Convolution2D(6, 2, 2, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        #self.model.add(Convolution2D(6, 2, 2, subsample=(1, 1)))
        #self.model.add(Activation('relu'))
        self.model.add(Convolution2D(4, 2, 2, subsample=(3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(6))
        self.model.add(Activation('relu'))
        self.model.add(Dense(nb_actions))
        self.model.add(Activation('linear'))
        print(self.model.summary())
    def train(self, env):
        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=False, verbose=2)

           # After training is done, we save the final weights one more time.
        self.dqn.save_weights(weights_filename, overwrite=True)
    def test(self,env):
        weights_filename = 'dqn_{}_weights_8000.h5f'.format(ENV_NAME)
        pdb.set_trace()
        self.dqn.load_weights(weights_filename)
        self.dqn.test(env, nb_episodes=15, visualize=True)


# Finally, evaluate our algorithm for 10 episodes.
              
if __name__=="__main__":
     actions = [[1,0],[0,1],[-1,0],[0,-1]]
     boardfile = "board.bmp"
     granularity = 10
     board = misc.imread(boardfile, flatten=True) 
     env = BoardEnv(board, granularity = granularity)
     l = Learner((2*granularity,granularity),WINDOW_LENGTH,4)
     l.train(env)
