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

from rl.agents import DDPGAgent 
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


def construct_agent(env_shape, nb_actions, input_shape):
    # Next, we build a very simple model.
    pdb.set_trace()
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env_shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())


    processor = EmptyProcessor()
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.0001, clipnorm=0.99, clipvalue=0.5), metrics=['mae'])
    return agent
