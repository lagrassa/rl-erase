import pdb
import random
from reward import reward_func, entropy
import numpy as np
from utils import simulate_for_duration
from pr2_pouring_world import PouringWorld
import logging
logger = logging.getLogger(__name__)
actions = [[6,0],[0,6],[-6,0],[0,-6]]
#RENDER =False 

MAX_AIMLESS_WANDERING = 100
GRASP_ONLY = True
P_REPLAY = 0.0000 #with this probability, go back to a state you've done before, and just do that again until self.replay counter
#overflows
LENGTH_REPLAY = 15
EXP_NAME = "value_estimation"
REPLAY_LENGTH = LENGTH_REPLAY
LOG_INTERVAL = 20
#world = PouringWorld(visualize=RENDER, real_init=True)


class PourEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, visualize=True, real_init=True):
        self.visualize=visualize
        self.done_amount =  90 #pretty well mixed in original example HACK
        if real_init:
            self.world = PouringWorld(visualize=visualize, real_init=real_init)
        self.world_state = self.world.world_state() 
        self.robot_state = []# self.world.pourer_state()
        self.n = self.world_state[0].shape[1]*self.world_state[0].shape[0]
        self.counter = 0;
        self.replay_counter = 0;
        self.num_steps_same = 0;
        self.prev_val = 0; 
        self.saved_robot_state = []
        self.saved_world_state = []

    def progress_state(self, action=300):
        close_num = action[0]
        close_force = action[1]
        lift_force = action[2]
	    grasp_height = action[3]
	    grasp_depth = action[4]
        if GRASP_ONLY:
            self.world.grasp_cup(close_num=close_num, close_force = close_force, teleport=False, grasp_height=grasp_height, grasp_depth=0.04, lift_force=lift_force, finger_close_num=finger_close_num)
        else:
	    forward_offset = action[5]
	    height = action[6]
	    vel = action[7]
            self.world.pour_pr2(close_num=close_num, close_force=close_force, lift_force=lift_force, forward_offset=forward_offset, height=height, vel = vel, grasp_height=grasp_height, grasp_depth = grasp_depth)


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

            #self.saved_world_state.append(self.world_state)
            #self.saved_robot_state.append(self.world.stirrer_state())
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

    def observe_state(self):
        return self.world.gripper_forces()

    def create_state(self):
        robot_state = self.robot_state
        img1, img2 = self.world_state
        return img1, img2, robot_state
        
    def step(self, action):
        self.move_if_appropriate(action)
        ob = self.observe_state() #self.world_state
        episode_over = False
        reward= self.process_reward()
        beads = self.world.base_world.ratio_beads_in(cup = self.world.target_cup)
        
        if self.replay_counter == 0: #never end episode during experience replay
            
            episode_over = self.world.base_world.cup_knocked_over(cup=self.world.target_cup) 
            episode_over = episode_over or self.stop_if_necessary()
            #if episode_over:
            #    reward = -1
            #    print("EPISODE OVER", reward) 

        self.should_replay_and_setup()

        if episode_over:
            #self.saved_robot_state = []
            self.saved_world_state = []

        #return ob, beads, entropies, episode_over, {}
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
        #fun enough, world_state should now be a tuple
        if not GRASP_ONLY:
            rew =  reward_func(None, self.world.base_world.ratio_beads_in(cup=self.world.target_cup))
        else:
            self.world.spawn_cup()
            rew = self.world.test_grasp()
        
        self.counter +=1
        return rew

    def reset(self, new_bead_mass=None):
        self.__init__(visualize=self.visualize, real_init=False)
        self.world.reset()
        return self.create_state()
        

    def render(self, mode='human', close=False):
        pass

def action_num_to_action(action_num):
    return tuple(action_num) #no overhead needed

if __name__=="__main__":
    be = PourEnv(visualize=False)
    be.render()
    for i in range(200):
        be.step([0.4,0.3,0.2])
        be.render()
 



