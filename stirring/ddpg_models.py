from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from keras.regularizers import l2
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import numpy as np
import pdb
import gym


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization,  Lambda, concatenate, Conv2D, MaxPooling2D, Convolution3D 
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


def construct_agent(env, env_shape, nb_actions, input_shape):
    dims = env.robot_state.shape[0]
    # Next, we build a very simple model.
    picture_tensor = Input(shape=(1,) + input_shape, dtype='float32', name="pictureTensor")
    #so the idea here is that we never have more than one in the window length
    #and that's just a way to work around keras-rl, so we're just going to take    #one window's length and convolve that....
    grid =  Lambda(lambda x: x[:,0,:,:,0:3],   dtype='float32')(picture_tensor)
    #horrible hack: the top left corner
    robot =  Lambda(lambda x: x[:,0,:,:,3][:,0][:,0:dims],  (dims,) , dtype='float32')(picture_tensor)
    #Convolution stuff
    actor = Dense(32, activation='relu')(robot) #shamelessly copied from an mnist tutorial architecture
    actor = Dense(32, activation='relu')(robot) #shamelessly copied from an mnist tutorial architecture
    actor = Dense(nb_actions, activation = 'sigmoid', dtype='float32')(actor) 
    actor = Model(inputs=picture_tensor, outputs=actor)
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    #observation_input = Input(shape=(1,) + env_shape, name='observation_input')
    #observation_input = picture_tensor
    x = Concatenate()([action_input, robot])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, picture_tensor], outputs=x)
    print(critic.summary())


    processor = EmptyProcessor()
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.25, mu=0., sigma=.3, size=nb_actions)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.0001, clipnorm=0.99, clipvalue=0.5), metrics=['mae'])
    return agent
