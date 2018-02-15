import gym
import pdb
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from world import World
from robot import Robot

import logging
import pygame
logger = logging.getLogger(__name__)
actions = [[1,0],[0,1],[-1,0],[0,-1]]

class BoardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board, granularity = 10):
        print("initializing")
        self.world = World(board, granularity = granularity)
        self.res = granularity
        self.robot = Robot(self.world)
        pygame.init()
        self.screen_size = 400
        self.n = self.world.board.shape[1]*self.world.board.shape[0]
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size)) 
   
        self.screen.fill([255,255,255])
        self.render()

    def step(self, action):
        print("Acttion:",action)
        self.robot.move(action_num_to_action(action))
        reward = self._get_reward()
        ob =  self.world.reduced_board()
        episode_over = False
        return ob, reward, episode_over, {}


    def _get_reward(self):
        return self.world.reward()

    def reset(self):
        self.__init__(self.world.board_image, granularity = self.res)
        return self.world.reduced_board()

    def render(self, mode='human', close=False):
        self.world.draw(self.robot, self.screen, self.screen_size, self.n)
        pygame.display.flip()

def action_num_to_action(action_num):
    return actions[action_num]

if __name__=="__main__":
    be = BoardEnv()
    be.render()
 



