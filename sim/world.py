import numpy as np
class World:
    # an nxn board will randomly filled in values. 4x4 for now
    def __init__(self):
        self.board = np.array([['#','#',' '],['#','#',' '], [' ',' ',' ']])
        self.threshold = 0
    def draw(self,robot):
        board_with_bot = self.board.copy()
        x = robot.state[0]
        y = robot.state[1]
        board_with_bot[x,y] = "@"
        print board_with_bot
    #uses robot to erase
    def erase(self,robot):
        x = robot.state[0]
        y = robot.state[1]
        if robot.pressure > self.threshold:
            self.board[x,y] = ' '
    #size of board - number of unerased
    def reward(self):
        num_filled = 0
        for row in self.board:
            for val in row:
                if val == '#':
                    num_filled +=1
        return (len(self.board)**2)-num_filled
       
        

            
        
    
