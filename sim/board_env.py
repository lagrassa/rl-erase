import gym
import pdb
import random
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from world import World
from robot import Robot
import numpy as np

import logging
import pygame
logger = logging.getLogger(__name__)
actions = [[1,0],[0,1],[-1,0],[0,-1]]
RENDER = True
MAX_AIMLESS_WANDERING = 100
P_REPLAY = 0.002 #with this probability, go back to a state you've done before, and just do that again until self.replay counter
#overflows
LENGTH_REPLAY = 30
REPLAY_LENGTH = LENGTH_REPLAY

class BoardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board, granularity = 10):
        self.done_percentage =  9 #chosen to be about most of the board
        self.world = World(board, granularity = granularity)
        self.res = granularity
        self.robot = Robot(self.world)
        pygame.init()
        self.screen_size = 400
        self.n = self.world.board.shape[1]*self.world.board.shape[0]
        self.counter = 0;
        self.replay_counter = 0;
        self.num_steps_same = 0;
        self.prev_reward = 0; 
        self.saved_robot_state = []
        self.saved_world_state = []
        if RENDER:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size)) 
            self.screen.fill([255,255,255])
            self.render()

    def create_state(self):
        ob = np.zeros((self.res, self.res,2))
        ob[:,:,0] = self.world.reduced_board()
        ob[:,:,1] = self.world.robot_one_hot(self.robot)
        return ob
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
            self.robot.move(action_num_to_action(action), self.world)
            self.saved_world_state.append(self.world.copy())
            self.saved_robot_state.append(self.robot.copy())
        else:
            self.world = self.replay_world_states[self.replay_counter]
            self.robot = self.replay_robot_states[self.replay_counter]
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
        
    def step(self, action):
        self.move_if_appropriate(action)
        
        ob = self.create_state()
        episode_over = False
        reward = self.process_reward()
        if self.replay_counter == 0: #never end episode during experience replay
            episode_over = ob.sum() < self.done_percentage
            episode_over = self.stop_if_necessary()

        self.should_replay_and_setup()

        if episode_over:
            self.saved_robot_state = []
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
       

    def _get_reward(self):
        rew =  self.world.reward()
        if (self.counter % 1000 == 0):
            print(rew)
        self.counter +=1
        return rew

    def reset(self):
        self.__init__(self.world.board_image, granularity = self.res)
        return self.create_state()
        

    def render(self, mode='human', close=False):
        self.world.draw(self.robot, self.screen, self.screen_size, self.n)
        pygame.display.flip()

def action_num_to_action(action_num):
    return actions[action_num]

if __name__=="__main__":
    be = BoardEnv()
    be.render()
 



