#mostly based on the Atari RL example given on the keras-rl github at
#https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
from __future__ import division
from scipy import misc

import numpy as np
import pdb
from board_env import BoardEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Lambda, concatenate, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 2
ENV_NAME = "shallow_learning"

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
        self.picture_tensor = Input(shape=(window_length,) + input_shape, dtype='float32', name="pictureAndRobotTensor")
        #self.robot_tensor = Input(shape=((window_length,) +input_shape))
        self.build_model(self.picture_tensor, input_shape, window_length, nb_actions)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=1000)
        amount_memory = 10000000
        memory = SequentialMemory(limit=amount_memory, window_length=WINDOW_LENGTH)
        processor = EmptyProcessor()
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor, nb_steps_warmup=50, gamma=.99, target_model_update=10000,train_interval=4, delta_clip=1.)
        self.dqn.compile(Adam(lr=.001), metrics=['mae'])

    #entirely taken from the Atari example form Mnih et al's paper
    def build_model(self, picture_tensor,  input_shape, window_length, nb_actions):
        #take the number of rows, divide them in half to get the real input shape
        #input_shape = (window_length,)+ picture_shape
        try:
            assert(input_shape[0] % 2 == 0)
        except:
            raise Exception("shape of rows must be even")

        picture_shape = (window_length,) + (int(input_shape[0]/2), input_shape[1])
        grid = Lambda(lambda x: x[:,:,:10,:],  picture_shape , dtype='float32')(picture_tensor)
        robot = Lambda(lambda x: x[:,:,10:,:],  picture_shape , dtype='float32')(picture_tensor)
        #b = Dense(64, activation='linear', dtype='float32')(top)
        #now all the convolution stuff on the pic (prob not much)
        grid = Conv2D(20,(2,2), activation='relu', padding='same')(grid)
        grid = MaxPooling2D((3,3),strides=(1,1),padding='same')(grid)
        grid = Flatten(dtype='float32')(grid)
        robot_flat = Flatten(dtype='float32')(robot)
        fc1 = concatenate([robot_flat, grid])
        action_tensor = Dense(nb_actions, activation = 'sigmoid', dtype='float32')(fc1)
        self.model = Model(inputs=[picture_tensor], outputs=action_tensor)
         
        """
        self.model = Sequential()
        if K.image_dim_ordering() == 'tf':
            # (width, height, channels)
            self.model.add(Permute((2, 3, 1), input_shape=picture_shape))
        elif K.image_dim_ordering() == 'th':
            # (channels, width, height)
            self.model.add(Permute((1, 2, 3), input_shape=picture_shape))
        else:
            raise RuntimeError('Unknown image_dim_ordering.')
        self.model.add(Convolution2D(3, 2, 2, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        #self.model.add(Convolution2D(6, 2, 2, subsample=(1, 1)))
        #self.model.add(Activation('relu'))
        #self.model.add(Convolution2D(4, 2, 2, subsample=(3, 3)))
        #self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(6))
        self.model.add(Activation('relu'))
        self.model.add(Dense(nb_actions))
        self.model.add(Activation('linear'))
        """
        print(self.model.summary())

    def train(self, env):
        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=True, verbose=2)

           # After training is done, we save the final weights one more time.
        self.dqn.save_weights(weights_filename, overwrite=True)
    def test(self,env):
        weights_filename = 'dqn_board_weights_750000.h5f'.format(ENV_NAME)
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
