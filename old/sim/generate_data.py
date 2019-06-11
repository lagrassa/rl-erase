import numpy as np
import pdb
import pickle
from scipy import misc
import board_env

#The idea is you see the robot pose, and the board picture. 
#Then create a policy, which advances the thing forward. 
#At every step, pickle the image list and pick the action. 
#Also render

numsteps = 200
states = []
actions = []
action_space = [[1,0],[0,1],[-1,0],[0,-1]]


def main():
    boardfile = "board.bmp"
    board = misc.imread(boardfile, flatten=True)
    env = board_env.BoardEnv(board, granularity=10)
    obs = env.reset()
    for step in range(numsteps):
        env.render()
        action = manual_policy(obs)
        actions.append(action)
        states.append(obs)
        obs = env.step(action)[0]
    with open('states.pkl', 'wb') as fp:
        pickle.dump(states, fp)
    with open('actions.pkl', 'wb') as fp:
        pickle.dump(actions, fp)
#given image of board n x n returns the highest density region
def square_with_max_density(pic):
    maxIndex = None
    maxVal = -np.inf
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i,j] >= maxVal:
                maxVal = pic[i,j]
                maxIndex = (i,j)
    return maxIndex
    


#given image of robot nxn returns the robot pose as [x,y]
def robot_pose(pic):
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i,j] != 0:
                return (i,j)

#go to the region of highest density
# actions are in the list [[1,0],[0,1],[-1,0],[0,-1]]
def manual_policy(obs):
    (xd, yd) = square_with_max_density(obs[:,:,0])
    (x, y) = robot_pose(obs[:,:,1]) 
    #pick largest displacement difference 
    #go that direction. Not the most beautiful code but
    #easier to understand
    if (abs(x-xd) > abs(y-yd)): #largest different in x: go that way
        dir = 0
        if x > xd:
            sign = -1
        if x <= xd:
            sign = 1
    else:
        dir = 1
    #and go in that direction
        if y > yd:
            sign = -1
        if y <= yd:
            sign = 1
    assert(sign is not None)
    assert(dir is not None)
    policy = [0,0]
    policy[dir] = sign
    act_num =  action_space.index(policy)
    return act_num
    

if __name__=="__main__":
    main()
 





