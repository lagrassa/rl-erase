import numpy as np
import pdb
class World:
    # an nxn board will randomly filled in values. 4x4 for now
    def __init__(self):
        self.board = np.matrix(np.zeros((5,5)))
        self.board[0,0] = 1
        self.board[0,1] = 1
        self.board[1,0] = 1
        self.board[1,1] = 1

        self.threshold = 0

    def draw(self,robot):
        #Board with bot is the drawn board
        board_with_bot = self.board.copy()
        for i in range(len(board_with_bot)):
            for j in range(len(board_with_bot[0])):
                 if board_with_bot[i,j]:
                     board_with_bot[i,j] = '#'
                 else:
                     board_with_bot[i,j] = ' '
        x = robot.state[0]
        y = robot.state[1]
        board_with_bot[x,y] = "@"
        print board_with_bot

    #uses robot to erase
    def erase(self,robot):
        x = robot.state[0]
        y = robot.state[1]
        if robot.pressure > self.threshold:
            self.board[x,y] = 0

    #size of board - number of unerased
    def reward(self):
        num_filled = self.board.sum()
        return (len(self.board)**2)-num_filled
       
        

            
        
    
