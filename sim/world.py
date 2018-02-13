import numpy as np
import math
import pygame
import pdb
import os
from scipy import misc
import math

class World:
    # an nxn board will randomly filled in values. 4x4 for now
    def __init__(self):
        #self.board = np.matrix(np.zeros((5,5)))
        #self.board[0,0] = 1
        #self.board[0,1] = 1
        #self.board[1,0] = 1
        #self.board[1,1] = 1
        #self.board[2,3] = 1
        #self.board[3,3] = 1
        self.board = draw_board_from_file('board.bmp')

        self.threshold = 0


    def draw(self,robot, screen, size, n):
        #Board with bot is the drawn board
        self.board = self.board.copy()
        width = (size/(n**0.5))
        black = (0,0,0)
        white = (255,255,255)
        red = (255,0,0)

        x = robot.state[0]
        y = robot.state[1]
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                 if self.board[i,j]:
                     #draw black square
                     color = black
                 else:
                     color = white
                     #draw white square
                 square = pygame.Rect(i*width,j*width, width, width)
                 pygame.draw.rect(screen, color, square)
        #draw the robot
        #if it goes off canvas, we don't really care
        #tricky, it's coordinate frame is the center
        robot = pygame.Rect(x*width, y*width,robot.width*width, robot.width*width)
        pygame.draw.rect(screen, red, robot)
        pygame.display.flip()
 
                 

    #uses robot to erase
    def erase(self,robot):
        x = robot.state[0]
        y = robot.state[1]
        if robot.pressure > self.threshold:
            self.board[x,y] = 0
            for i in range(robot.width):
                #erase all squares width to the right
                if self.inrange(x+i,y):
                    self.board[x+i,y] = 0
                #erase all squares to bottom
                if self.inrange(x,y+i):
                    self.board[x+i,y] = 0
            
            
    def inrange(self, x, y):
        if x < 0 or x > self.board.shape[0]:
            return False
        if y < 0 or y > self.board.shape[1]:
            return False
        return True
    #size of board - number of unerased
    def reward(self):
        num_filled = self.board.sum()
        rew =  (len(self.board)**2)-num_filled
        return rew
       
        

            
        
    
def draw_board_from_file(filename):
    image= misc.imread(filename, flatten = True)
    
    w,h = image.shape
    board = np.matrix(np.zeros((w,h)))
    for i in range(w):
        for j in range(h):
            if image[i,j] != 255:
                board[i,j] = 1
    return board
