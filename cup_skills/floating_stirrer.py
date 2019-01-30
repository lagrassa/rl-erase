from __future__ import division
import ipdb
import pybullet as p
import math
import csv
import pdb
from PIL import Image
from cup_world import *
import numpy as np
import utils
import time
import pybullet_data
k = 1 #scaling factor
DEMO =False 
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose, set_joint_positions, get_joint_positions

real_init = True

 
class World():
    def __init__(self, visualize=False, real_init=True, beads=True):
        #make base world 
        self.base_world = CupWorld(visualize=visualize, real_init = real_init, beads=beads)
        self.visualize=visualize
        self.real_init = real_init
        self.setup()

    """try doing what fetchpush does essentially"""
    def stir(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.stirrer_id)
        new_pos = np.array((pos[:]))
        new_pos += action
        print(new_pos)
        p.changeConstraint(self.cid, new_pos, orn, maxForce=120)
        simulate_for_duration(0.8)
        
    
    #adds something of size_step to the current angle
    def stir_circle(self, size_step=0, time_step = 0):
        #let's say it takes 1 second t
        #and find out where x,y is is 
        current_pos = p.getJointState(self.armID, 6)[0]
        desired_pos = current_pos + size_step
        if desired_pos >= 3:
            desired_pos = -3 #reset

    #this function is now a complete lie and has not only the stirrer state but
    #also the vector from the cup
    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        linkPos = p.getJointState(self.armID, 8)[0]
         
        theta_diff_pos, _, _,_ = p.getJointState(self.armID,8)
        rot_joint_pos, _, _ ,_= p.getJointState(self.armID,6)
        curl_joint_pos, _, curl_joint_forces,_ = p.getJointState(self.armID,10)

        #r, theta, z in pos 
        cupPos=  np.array(p.getBasePositionAndOrientation(self.base_world.cupID)[0])
        stirrerPos=  np.array(p.getLinkState(self.armID, 10)[0])
        vector_from_cup = cupPos-stirrerPos
        r_theta_z_pos =  cart2pol(vector_from_cup[0], vector_from_cup[1])+ (vector_from_cup[2],)
        #forces in cup frame
        r_theta_z_force =  cart2pol(curl_joint_forces[0], curl_joint_forces[1])+ (curl_joint_forces[2],)
        #return np.array([theta_diff_pos, rot_joint_pos, curl_joint_pos, r_theta_z_pos[0], r_theta_z_pos[1], r_theta_z_pos[2], r_theta_z_force[0], r_theta_z_force[1], r_theta_z_force[2]])
        return np.array([theta_diff_pos, rot_joint_pos, curl_joint_pos, r_theta_z_pos[0], r_theta_z_pos[1], r_theta_z_pos[2]])
       

    def reset(self):
        self.base_world.reset()
        self.__init__(visualize=self.visualize, real_init=False)

    def setup(self, beads=True):
        NEW = self.real_init #unfortunately
        if NEW:
            start_pos = [0,0,0.3]
            start_quat = (0.0, 1, -1, 0.0) 
            self.stirrer_id = p.loadURDF("urdf/green_spoon.urdf", globalScaling=1.6, basePosition=start_pos, baseOrientation=start_quat)
            stirrer_start_pose = (start_pos, start_quat)
            self.cid = p.createConstraint(self.stirrer_id, -1, -1, -1, p.JOINT_FIXED, [0,0,1], [0,0,0],[0,0,0],
                                          [0,0,0,1], [0, 0, 0,1])
            p.changeConstraint(self.cid, start_pos, start_quat)
            simulate_for_duration(2.0)
            self.base_world.zoom_in_on(self.stirrer_id, 2)
            self.real_init = False
        else:
            try:
                p.restoreState(self.bullet_id)
            except:
                self.real_init = True
                p.resetSimulation()
                self.setup()



	#p.resetDebugVisualizerCamera(0.5, yaw, roll, objPos)
    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0) 


if __name__ == "__main__":
    import sys
    num_beads = 2
    if len(sys.argv) > 1:
        num_beads = int(sys.argv[1])
    world = World(visualize=True)
    world.base_world.drop_beads_in_cup(num_beads)
    actions = [[0,0,-0.7],[0,0,-0.7], [0,0,-0.7], [0,0,-0.7],[0,0.1,0],[0,-0.1,0],[0,0.10,0],[0,-0.10,0]]
    for action in actions:
        world.stir(action)


	

