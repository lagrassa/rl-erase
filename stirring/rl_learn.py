from __future__ import division
from scipy import misc

import numpy as np
import pdb
import pickle
from stir_env import StirEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Lambda, concatenate, Conv2D, MaxPooling2D, Convolution3D, LSTM
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 1
ENV_NAME = "mix_cup"

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
        
        picture_tensor = Input(shape=input_shape, dtype='float32', name="pictureTensor")
        dims = 2
        robot_tensor = Input(shape=(dims,) , dtype='float32', name="robotTensor")
        self.build_model(picture_tensor,robot_tensor, input_shape, window_length, nb_actions)

    def build_model(self, picture_tensor,robot_tensor, input_shape, window_length, nb_actions):
        #here's where it's a bit weird. We have one part of the tuple,        #the input shape, but then the second part is the robot position 
        net_input_shape =  (None, window_length,)+input_shape
        print("net input shape", net_input_shape) 
        """ 
        if K.image_dim_ordering() == 'tf':
            # (width, height, channels)
            #self.model.add(Permute((2,3, 1), input_shape= net_input_shape))
            pass
        elif K.image_dim_ordering() == 'th':
            # (channels, width, height)
            self.model.add(Permute((1, 2, 3), input_shape=net_input_shape))
        else:
            raise RuntimeError('Unknown image_dim_ordering.')
        """
        #grid =  Lambda(lambda x: x[:,:,0],  net_input_shape , dtype='float32')(picture_tensor)
        #robot =  Lambda(lambda x: x[:,:,1],  robot_shape , dtype='float32')(picture_tensor)
        #Convolution stuff
        grid = Conv2D(4,(2,2), activation='relu', padding='same')(picture_tensor)

        grid = MaxPooling2D((3,3),strides=(1,1),padding='same')(grid)
        grid = Flatten(dtype='float32')(grid)
        #robot_flat = Flatten(dtype='float32')(robot_tensor)      
        fc1 = concatenate([robot_tensor, grid]) 
        action_tensor = Dense(nb_actions, activation = 'sigmoid', dtype='float32')(fc1)
        self.model = Model(inputs=[picture_tensor, robot_tensor], outputs=action_tensor)
        print(self.model.summary())

    def train(self, env):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=5)
        amount_memory = 1000000#0
        memory = SequentialMemory(limit=amount_memory, window_length=WINDOW_LENGTH)
        processor = EmptyProcessor()
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor, nb_steps_warmup=2, gamma=.90, target_model_update=5,train_interval=5, delta_clip=1., enable_double_dqn=False)
        self.dqn.compile(Adam(lr=0.05), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
        callbacks += [FileLogger(log_filename, interval=100)]
        #self.dqn.load_weights("dqn_shallow_learning_simple_board_imm_rew_weights_12500.h5f")
        self.dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000, visualize=True, verbose=2)
           # After training is done, we save the final weights one more time.
        self.dqn.save_weights(weights_filename, overwrite=True)

    def train_supervised(self, env):
        numsteps = 1750000
        self.model.compile(loss = "mean_absolute_error", optimizer='adam', metrics = ['accuracy'])
        # Load dataset
        X, Y = load_supervised_data(actionfile = "actions.pkl", statefile = "states.pkl")

        #load image of robot and environment in 2 channels, call that X
        #call the appropriate action to take y
        #X = np.zeros((2, 10,10,2))
        #X[0,0,0, 1] = 1;
        #Y = [1, 2] 

        #initialize tensors
        self.model.fit(X, Y, epochs=90, batch_size=2) 
        #given the environment, predict the next step from the policy
        #step according to the policy
        #back propagate
        #...I forgot that keras does this all for you
        self.model.save_weights(ENV_NAME+'weights.h5f')

    def test(self,env):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=1000)
        amount_memory = 10000000
        memory = SequentialMemory(limit=amount_memory)
        processor = EmptyProcessor()
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, policy=policy, memory=memory, processor=processor, nb_steps_warmup=50, gamma=.99, target_model_update=10000,train_interval=4, delta_clip=1., enable_double_dqn=False)
        self.dqn.compile(Adam(lr=1), metrics=['mae'])
        weights_filename = 'dqn_shallow_learning_complex_board_weights_25000.h5f'.format(ENV_NAME)
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
     actions = [[1,0],[0,1],[-1,0],[0,-1]]
     nb_actions = len(actions)
     env = StirEnv()
     world_shape = env.world_state.shape
     l = Learner((world_shape),WINDOW_LENGTH,nb_actions)
     l.train(env)
