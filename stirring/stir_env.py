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
RENDER = False

MAX_AIMLESS_WANDERING = 100
P_REPLAY = 0.0002 #with this probability, go back to a state you've done before, and just do that again until self.replay counter
#overflows
LENGTH_REPLAY = 15
EXP_NAME = "value_estimation"
REPLAY_LENGTH = LENGTH_REPLAY
action_file = open("actions"+EXP_NAME+".py", "a") 
reward_file = open("rewards"+EXP_NAME+".py", "a")
LOG_INTERVAL = 20
world = World(visualize=RENDER, real_init=True)


class StirEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, visualize=True, real_init=True):
        print("Visualize=",visualize)
        self.visualize=visualize
        self.done_amount =  90 #pretty well mixed in original example HACK
        self.world = world
        self.world_state = self.world.world_state() 
        self.robot_state = self.world.stirrer_state()
        self.n = self.world_state[0].shape[1]*self.world_state[0].shape[0]
        self.counter = 0;
        self.replay_counter = 0;
        self.num_steps_same = 0;
        self.prev_val = 0; 
        self.saved_robot_state = []
        self.saved_world_state = []

        

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
        reward_val = self._get_reward()
        if abs(reward_val) < 0.11:
            self.num_steps_same +=1;
        else:
            self.num_steps_same = 0
        reward = reward_val - self.prev_val
        self.prev_val = reward_val; 
        return reward_val

    def stop_if_necessary(self):
        if self.num_steps_same >= MAX_AIMLESS_WANDERING:
            print("Episode ended due to aimless wandering")
            self.num_steps_same = 0
            return True
        return False

    def create_state(self):
        robot_state = self.robot_state
        img1, img2 = self.world_state
        return img1, img2, robot_state
        
    def step(self, action):
        if self.counter % LOG_INTERVAL == 0:
            action_file.write(str(action) + ",")
            print("action", action)
        self.move_if_appropriate(action)
        ob = self.create_state() #self.world_state
        episode_over = False
        reward= self.process_reward()
        if self.replay_counter == 0: #never end episode during experience replay
            """
            cup_knocked_over =  self.world.cup_knocked_over()
            mostly_done = val > self.done_amount
            stirrer_far = not self.world.stirrer_close()
            if cup_knocked_over:
                print("CUP KNOCKED OVER")
            if stirrer_far:
                print("STIRRER FAR")
            if mostly_done:
                print("MOSTLY DONE")
            """
            
            episode_over = val > self.done_amount or self.world.cup_knocked_over() or  not self.world.stirrer_close()
            episode_over = episode_over or self.stop_if_necessary()
            if episode_over:
		reward_file.write(str(reward)+",")
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
        if self.world.cup_knocked_over():
            return -3000
        #fun enough, world_state should now be a tuple
        rew =  reward_func(self.world_state, self.world.num_beads_out())
        self.counter +=1
        return rew

    def reset(self):
        self.__init__(visualize=self.visualize, real_init=False)
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
 



