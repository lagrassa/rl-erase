from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from keras.regularizers import l2
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import numpy as np
import pdb
import gym


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.random import OrnsteinUhlenbeckProcess

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


def construct_agent(env, nb_actions, input_shape):
    V_model = Sequential()

    V_model.add(Flatten(input_shape=(1,) + input_shape))
    V_model.add(BatchNormalization())
    V_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    V_model.add(Activation('elu'))
    V_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    V_model.add(Activation('elu'))
    V_model.add(Activation('elu'))
    V_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    V_model.add(Activation('elu'))
    V_model.add(Activation('elu'))
    V_model.add(Dense(1,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    V_model.add(Activation('elu'))
    V_model.add(Activation('linear'))
    print(V_model.summary())

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + input_shape))
    mu_model.add(BatchNormalization())
    mu_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    mu_model.add(Activation('elu'))
    mu_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    mu_model.add(Activation('elu'))
    mu_model.add(Dense(16,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    mu_model.add(Activation('elu'))
    mu_model.add(Dense(nb_actions,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)))
    mu_model.add(Activation('linear'))
    print(mu_model.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + input_shape, name='observation_input')
    x = Concatenate()([action_input, Flatten()(observation_input)])
    x = BatchNormalization()(x)
    x = Dense(32,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(x)
    x = Activation('elu')(x)
    x = Dense(32,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(x)
    x = Activation('elu')(x)
    x = Dense(32,kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(x)
    x = Activation('elu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
    x = Activation('linear')(x)
    L_model = Model(inputs=[action_input, observation_input], outputs=x)
    print(L_model.summary())

    processor = EmptyProcessor()
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
		     memory=memory, nb_steps_warmup=100, random_process=random_process,
		     gamma=.99, target_model_update=1e-3, processor=processor)
    agent.compile(Adam(lr=.0001, clipnorm=0.99, clipvalue=0.5), metrics=['mae'])
    return agent
