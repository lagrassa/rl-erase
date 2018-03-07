from __future__ import division
import numpy as np
import math
import pygame
import pdb
import os
from scipy import misc
import math

class World:
    # an nxn board will randomly filled in values. 4x4 for now
    #requires some grid of boards where 255 = white and anything else is black
    def __init__(self, board_image, granularity = 10):
        self.res = granularity
        #self.board = np.matrix(np.zeros((5,5)))
        #self.board[0,0] = 1
        #self.board[0,1] = 1
        #self.board[1,0] = 1
        #self.board[1,1] = 1
        #self.board[2,3] = 1
        #self.board[3,3] = 1
        self.board = draw_board_from_file(board_image)
        self.board_image = board_image

        self.threshold = 0

    def copy(self):
        new_world = World(self.board_image, granularity=self.res)
        new_world.board = self.board.copy()
        return new_world

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
        pygame.display.update()
 
                 
    #partitions the board into a 10x10 grid and reports percentage erased
    #@requires boards to be multiples of 10 to work nicely....
    def reduced_board(self):
        #goes row by row then down res columns to compute the percentage
        reduced_board = np.zeros((self.res, self.res))
        squares_per_x = int(self.board.shape[0]/self.res) 
        squares_per_y = int(self.board.shape[1]/self.res)
        for i in range(self.res):
            for j in range(self.res):
                #add up the number of ones in the rows
                num_filled = 0
                for k in range(int(self.board.shape[0] / self.res)):
                    num_filled += self.board[i*squares_per_x +k,j*squares_per_y: j*squares_per_y+squares_per_y].sum()
                total = self.board.shape[0]/self.res * self.board.shape[1]/self.res
                percent = num_filled / total
                reduced_board[i,j] = percent
        return reduced_board

    def checkrep(self):
        if (abs(self.board.shape[0]/self.res) < 0.00001):
            raise Exception("Granularity does not divide integer into board");
        if (self.board.shape[0] != self.board.shape[1]):
            raise Exception("Breaks square board assumption: may break other things in code")


    def robot_one_hot(self, robot):
        self.checkrep()
        x = robot.state[0]
        y = robot.state[1]
        one_hot = np.zeros((self.res, self.res))
        x_transformed = int(round(x*self.res/self.board.shape[0])) 
        y_transformed = int(round(y*self.res/self.board.shape[1])) 
        one_hot[x_transformed, y_transformed] = 9 #just chose a larger number, for no good reason
        return one_hot
        

    #uses robot to erase
    def erase(self,robot):
        self.checkrep()
        x = robot.state[0]
        y = robot.state[1]
        if robot.pressure > self.threshold:
            self.board[x,y] = 0
            for i in range(robot.width):
                #erase all squares width to the right
                for j in range(robot.width):
                    if self.inrange(x+i,y+j) and self.board[x+i, y+j] != 0:
                        self.board[x+i,y+j] = 0
            
            
    def inrange(self, x, y):
        if x < 0 or x > self.board.shape[0]:
            return False
        if y < 0 or y > self.board.shape[1]:
            return False
        return True
    #size of board - number of unerased
    def reward(self):
        num_filled = self.board.sum()
        rew =  num_filled/(self.board.shape[1]**2)
        return 1-rew
       
        

            
        
    
def draw_board_from_file(image):
    w,h = image.shape
    board = np.matrix(np.zeros((w,h)))
    for i in range(w):
        for j in range(h):
            if image[i,j] != 255:
                board[i,j] = 1
    return board
