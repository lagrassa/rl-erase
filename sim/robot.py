import random
import pdb

#a 1D grid world where a robot is moving toward a cake
#s = 2 leads to a reward, s = 4 as well.
class Robot:
    def __init__(self, board):
        self.state = [0,0]
        self.limit =3
        self.pressure = 1
        self.board = board
        self.width = 15
        self.board.erase(self)

    #-1 for backward, +1 for forward
    #direction is a tuple (-1,-1) for example
    def move(self,direction):
        for i in [1,0]:
	    potential_next = self.state[i] + direction[i]
	    if potential_next < 0:
		return
	    if potential_next > self.limit:
		return
	    else:
		self.state[i] = potential_next
        print("Robot now in",self.state)
        self.board.erase(self)
       
    def reward(self,board):
        return self.board.reward()
        


