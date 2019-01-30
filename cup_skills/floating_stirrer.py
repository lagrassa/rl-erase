from __future__ import division
import ipdb
import pybullet as p
import math
import csv
import pdb
from PIL import Image
from cup_skills.cup_world import *
import numpy as np
from cup_skills.reward import reward_func
import time
import pybullet_data
from cup_skills.local_setup import path

from gym import spaces
k = 1 #scaling factor
DEMO =False 
from cup_skills.utils import add_data_path, connect, enable_gravity,  disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose, set_joint_positions, get_joint_positions

real_init = True

 
class World():
    def __init__(self, visualize=False, real_init=True, beads=True):
        #make base world 
        self.base_world = CupWorld(visualize=visualize, real_init = real_init, beads=beads)
        self.visualize=visualize
        self.unwrapped = self
        self.real_init = real_init
        self.threshold = -12 #TAU from thesis
        self.time = 0
        self.timeout = 40
        self.seed = lambda x: 17
        self.reward_range = (0,-180)
        self.metadata = {"threshold": self.threshold}
        self.setup()
        high = np.inf*np.ones(self.state().shape[0])
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        max_move = 0.8
        low_act = np.array([-max_move]*4)
        high_act = np.array([max_move]*4)
        low_act[3] = 8
        high_act[3] = 60
        self.action_space = spaces.Box(low=low_act, high=high_act, dtype=np.float32)

    #positive when good, negative when bad"
    def step(self, action):
        self.time += 1
        self.stir(action[0:3], maxForce = action[3])
        world_state = self.base_world.world_state()
        ob = self.state()
        reward_raw = reward_func(world_state, self.base_world.ratio_beads_in())
        reward = reward_raw - self.threshold 
        print("Reward", reward)
        done = reward >= self.threshold or self.time > self.timeout or self.base_world.cup_knocked_over() or self.stirrer_far()
        return ob, reward, done, {}
        

    """try doing what fetchpush does essentially"""
    def stir(self, action, maxForce = 40):
        pos, orn = p.getBasePositionAndOrientation(self.stirrer_id)
        new_pos = np.array((pos[:]))
        new_pos += action
        print(new_pos)
        p.changeConstraint(self.cid, new_pos, orn, maxForce=maxForce) #120)
        simulate_for_duration(1.8)

    def state(self):
        world_state = self.base_world.world_state()
        stirrer_state = self.stirrer_state()
        #you don't need to worry about features or invariance.....so just make it a row vector and roll with it.
        return np.hstack([np.array(world_state).flatten(),stirrer_state.flatten()]) #yolo
        
    def stirrer_far(self):
        dist = self.base_world.distance_from_cup(self.stirrer_id, -1)
        threshold = 0.5
        return dist > threshold
        
    
        
    
    #this function is now a complete lie and has not only the stirrer state but
    #also the vector from the cup
    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        #r, theta, z in pos 
        cupPos=  np.array(p.getBasePositionAndOrientation(self.base_world.cupID)[0])
        stirrerPos=  np.array(p.getBasePositionAndOrientation(self.stirrer_id)[0])
        vector_from_cup = cupPos-stirrerPos
        #forces in cup frame
        return vector_from_cup
       

    def reset(self):
        p.resetSimulation()
        self.__init__(visualize=self.visualize, real_init=False)
        return self.state()

    def setup(self, beads=True):
        start_pos = [0,0,0.3]
        start_quat = (0.0, 1, -1, 0.0) 
        self.stirrer_id = p.loadURDF(path+"urdf/green_spoon.urdf", globalScaling=1.6, basePosition=start_pos, baseOrientation=start_quat)
        stirrer_start_pose = (start_pos, start_quat)
        self.cid = p.createConstraint(self.stirrer_id, -1, -1, -1, p.JOINT_FIXED, [0,0,1], [0,0,0],[0,0,0],[0,0,0,1], [0,0,0,1])
        p.changeConstraint(self.cid, start_pos, start_quat)
        simulate_for_duration(0.1)
        num_beads = 50
        self.base_world.drop_beads_in_cup(num_beads)
        self.base_world.zoom_in_on(self.stirrer_id, 2)
        self.real_init = False

    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0) 



if __name__ == "__main__":
    import sys
    num_beads = 2
    if len(sys.argv) > 1:
        num_beads = int(sys.argv[1])
    world = World(visualize=False)
    width = 0.16
    force = 25
    actions = [[0,0,-0.7, force],[0,0,-0.7, force],  [0,0,-0.7, force],[0,width,-0.05, force],[0,-width,0, force],[0,width,0, force],[0,-width,-width, force], [width,0,0, force],[-width,0,0, force],[width,0,0, force], [-width,0,0, force]]
    #actions = [[0,0,-0.7],[0,0,-0.7], [0,0,-0.7], [0,0,-0.7],[0,0.6,0],[0,-0.6,0],[0,0.60,0],[0,-0.60,0]]
    for action in actions:
        world.step(action)


	

