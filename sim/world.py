import numpy as np
import pygame
import pdb
class World:
    # an nxn board will randomly filled in values. 4x4 for now
    def __init__(self):
        self.board = np.matrix(np.zeros((5,5)))
        self.board[0,0] = 1
        self.board[0,1] = 1
        self.board[1,0] = 1
        self.board[1,1] = 1
        self.board[2,3] = 1
        self.board[3,3] = 1

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
                 elif i == x and j == y:
                     color = red
                 else:
                     color = white
                     #draw white square
                 square = pygame.Rect(i*width,j*width, width, width)
                 pygame.draw.rect(screen, color, square)
                 pygame.display.flip()
 
                 

    #uses robot to erase
    def erase(self,robot):
        x = robot.state[0]
        y = robot.state[1]
        if robot.pressure > self.threshold:
            self.board[x,y] = 0

    #size of board - number of unerased
    def reward(self):
        num_filled = self.board.sum()
        rew =  (len(self.board)**2)-num_filled
        print(rew)
        return rew
       
        

            
        
    
