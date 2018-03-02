import numpy
import pickle
from scipy import misc
import board_env

#The idea is you see the robot pose, and the board picture. 
#Then create a policy, which advances the thing forward. 
#At every step, pickle the image list and pick the action. 
#Also render

numsteps = 40
states = []
actions = []


def main():
    boardfile = "simple_board.bmp"
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

def manual_policy(obs):
    return 0

if __name__=="__main__":
    main()
 





