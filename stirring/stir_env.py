import gym
import pdb
import random
from gym import error, spaces
from reward import reward_func
from gym import utils
from gym.utils import seeding
import numpy as np
from stirring_world_pybullet import World
import logging
import pygame
logger = logging.getLogger(__name__)
pygame.display.init()
actions = [[6,0],[0,6],[-6,0],[0,-6]]
RENDER = True
MAX_AIMLESS_WANDERING = 100
P_REPLAY = 0.0002 #with this probability, go back to a state you've done before, and just do that again until self.replay counter
#overflows
LENGTH_REPLAY = 15
EXP_NAME = "DDPG"
REPLAY_LENGTH = LENGTH_REPLAY
action_file = open("actions"+EXP_NAME+".py", "a") 
reward_file = open("rewards"+EXP_NAME+".py", "a")
LOG_INTERVAL = 20
class StirEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, visualize=True):
        print("Visualize=",visualize)
        self.visualize=visualize
        self.world = World(visualize=visualize)
        self.done_percentage =  300 #pretty well mixed in original example HACK
        self.baseline =  60 #just the grey
        self.world_state = self.world.world_state() 
        self.robot_state = self.world.stirrer_state()
        self.n = self.world_state.shape[1]*self.world_state.shape[0]
        self.counter = 0;
        self.replay_counter = 0;
        self.num_steps_same = 0;
        self.prev_reward = 0; 
        self.saved_robot_state = []
        self.saved_world_state = []
        self.let_beads_settle()

    def let_beads_settle(self):
        steps_to_settle = 130
        [self.progress_state() for i in range(steps_to_settle)]
        print "Settling period done"
        

    def progress_state(self, action=300):
        self.world.stir(action)
	vel_iters, pos_iters = 6, 2
	timeStep = 0.01
	self.world.step(timeStep, vel_iters, pos_iters)

    """
    does an annoying amount of functionality
     if not during experience replay:
         -moves the robot according to the policy, and adds to self.saved_world_state
         -with probability P_REPLAY, picks a random sampled batch from self.saved_world_state
          and plays that back, ignoring what the actual action was. 
    """
    #moves robot if not in replay
    def move_if_appropriate(self, action):
        if self.replay_counter == 0: #we're in normal mode, so actually move the robot
            self.progress_state(action=action)
            self.world_state = self.world.world_state()

            self.saved_world_state.append(self.world_state)
            self.saved_robot_state.append(self.world.stirrer_state())
        else:
            self.world_state = self.replay_world_states[self.replay_counter]
            self.robot_state = self.replay_robot_states[self.replay_counter]
            self.replay_counter +=1

    #returns the reward and checks if it's been the same for a while
    def process_reward(self):
        reward = self._get_reward()
        if abs(reward-self.prev_reward) < 0.0000000001:
            self.num_steps_same +=1;
        else:
            self.num_steps_same = 0
        self.prev_reward = reward; 
        return reward 

    def stop_if_necessary(self):
        if self.num_steps_same >= MAX_AIMLESS_WANDERING:
            print("Episode ended due to aimless wandering")
            self.num_steps_same = 0
            return True
        return False

    def create_state(self):
        robot_state = self.robot_state
        HACK = True
        #this is horrible: make a matrix of zeros and set the top left to be what you want
        if HACK:
            robot_state = np.zeros(self.world_state.shape[0:2])
            state_shape = list(self.world_state.shape)
            state_shape[2] +=1
            state = np.zeros(state_shape)
            robot_state[0] = 17
            robot_state[1] = 19
            state[:,:,0:3] = self.world_state
            state[:,:,3] = robot_state
        else:
            state = self.world_state
        return state
        
    def step(self, action):
        assert(action.shape == (1,))
        action = float(action[0])
        if self.counter % LOG_INTERVAL == 0:
            action_file.write(str(action) + ",")
        self.move_if_appropriate(action)
        ob = self.create_state() #self.world_state
        episode_over = False
        reward = self.process_reward()
        if self.replay_counter == 0: #never end episode during experience replay
            episode_over = (reward > self.done_percentage or reward < self.baseline or not self.world.stirrer_close()) and self.counter > 3
            episode_over = episode_over or self.stop_if_necessary()
            if episode_over:
                print("EPISODE OVER", reward) 

        self.should_replay_and_setup()

        if episode_over:
            #self.saved_robot_state = []
            self.saved_world_state = []

        return ob, reward, episode_over, {}

    #sets up replay and determines whether to replay
    def should_replay_and_setup(self):
        #start a replay counter 
        if len(self.saved_world_state) > LENGTH_REPLAY and self.replay_counter == 0 and random.random() < P_REPLAY:
            self.replay_counter =1
            self.replay_world_states, self.replay_robot_states = self.random_state_batch(LENGTH_REPLAY)
            print("Started replay")
            return True
        if self.replay_counter >= LENGTH_REPLAY:
            self.replay_counter = 0
        return False
    #returns a random sequence from self.saved_world_state of length batch_size
    def random_state_batch(self, batch_size):
        #pick a random interval between 0 and len(self.saved_world_state)
        start_index = random.randint(0,len(self.saved_world_state)-REPLAY_LENGTH)
        return self.saved_world_state[start_index:start_index+REPLAY_LENGTH][:], self.saved_robot_state[start_index:start_index+REPLAY_LENGTH][:]
        #return self.saved_world_state[start_index:start_index+REPLAY_LENGTH][:]
       

    def _get_reward(self):
        rew =  reward_func(self.world_state)
        if (self.counter % LOG_INTERVAL == 0):
            print(rew)
            reward_file.write(str(rew)+",")
        self.counter +=1
        return rew

    def reset(self):
        self.__init__(visualize=self.visualize)
        self.world.reset()
        return self.create_state()
        

    def render(self, mode='human', close=False):
        self.world.render()

def action_num_to_action(action_num):
    return tuple(action_num) #no overhead needed

if __name__=="__main__":
    be = StirEnv(visualize=False)
    be.render()
    for i in range(200):
        be.step(2)
        be.render()
 



