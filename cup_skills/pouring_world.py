from cup_world import *
import pybullet as p
import numpy as np

import reward
from utils import set_point

k = 1

class PouringWorld():
    def __init__(self, visualize=True, real_init=False, new_bead_mass=None):
        self.base_world = CupWorld(visualize=visualize, beads=False, new_bead_mass=new_bead_mass)
        self.cup_to_dims = {"cup_1.urdf":(0.5,0.5), "cup_2.urdf":(0.5, 0.2), "cup_3.urdf":(0.7, 0.3), "cup_4.urdf":(1.1,0.3), "cup_5.urdf":(1.1,0.2), "cup_6.urdf":(0.6, 0.7)}#cup name to diameter and height
        if real_init:
            self.setup()
        else:
            p.restoreState(self.bullet_id)

    def setup(self):
        #create constraint and a second cup
        self.cupStartPos = (0,-0.4,0)
        self.cupStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
        #pick random cup

        self.cup_name = np.random.choice(self.cup_to_dims.keys())
        cup_file = "urdf/cup/"+self.cup_name
        self.target_cup = p.loadURDF(cup_file,self.cupStartPos, self.cupStartOrientation, globalScaling=k*5)
        self.cid = p.createConstraint(self.base_world.cupID, -1, -1, -1, p.JOINT_FIXED, self.cupStartPos, self.cupStartOrientation, [0,0,1])
        self.bullet_id = p.saveState()
        
    def observe_cup(self):
        return np.array(self.cup_to_dims[self.cup_name])

    def reset(self, real_init=False, new_bead_mass=None):
  
        self.base_world.reset(new_bead_mass=new_bead_mass)
        self.setup()      
        #set_pose(self.target_cup, (self.cupStartPos, self.cupStartOrientation))
                

    def move_cup(self, new_loc, new_euler=None, duration=0.7, teleport=False, force = 1000):
        if new_euler is None:
            new_orn = p.getBasePositionAndOrientation(self.base_world.cupID)[1]
        else:
            new_orn = p.getQuaternionFromEuler(new_euler)
        if teleport:
            set_point(self.base_world.cupID, new_loc)
            for bead in self.base_world.droplets:
                #original_loc 
                loc = p.getBasePositionAndOrientation(bead)[0]
                new_loc = np.array(loc)+np.array(new_loc)
                set_point(bead, new_loc)
         
        p.changeConstraint(self.cid, new_loc, new_orn, maxForce = force) 
        simulate_for_duration(duration)
    
    #reactive pouring controller 
    #goes to the closest lip of the cup and decreases the pitch until the beads fall out into the right place
    def pour(self, offset=0.02, velocity=0.9, force=1500, total_diff = 4*np.pi/5.0):
        #step_size and dt come from the angular velocity it takes to make a change of 3pi/4, can set later

        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        start_point = (pourer_pos[0], pourer_pos[1]+offset, pourer_pos[2])
        self.move_cup(start_point,  duration=2,force=force) 
        start_pos, start_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        #then start decreasing the roll, pitch or yaw(whatever seems appropriate)
        current_orn = list(p.getEulerFromQuaternion(start_orn))
        numsteps = 25.0
        step_size = total_diff/numsteps; #hard to set otherwise
        dt = step_size/velocity
        for i in range(int(numsteps)):
            current_orn[0] += step_size
            self.move_cup(start_pos, current_orn, duration=dt, force=force)

    def pourer_state(self):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        target_pos, target_orn = p.getBasePositionAndOrientation(self.target_cup)
        cup_rel_pos = np.array(pourer_pos) - np.array(target_pos)
        return np.hstack([cup_rel_pos, pourer_orn])

    def world_state(self):
        return self.base_world.world_state() 
    
    def lift_cup(self, desired_height=0.7, force=1600):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        other_cup_pos, _=  p.getBasePositionAndOrientation(self.target_cup)
        desired_height = other_cup_pos[2]+desired_height
        self.move_cup((pourer_pos[0], pourer_pos[1], desired_height), duration=3.5, force=force)


    
       

if __name__ == "__main__":
    pw = PouringWorld(visualize=True, real_init = True, new_bead_mass=1.1)
    pw.lift_cup()
    pw.pour(offset=-0.2, velocity=0.9, force=1500, total_diff = 4*np.pi/5.0)
    pdb.set_trace()
    pw.pour(offset=0.02, velocity=0.02, force=1500, total_diff = np.pi/5.0)

    pw.base_world.ratio_beads_in(cup=pw.target_cup)
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.25397044e-01, -1.43723112e+00, -1.14753149e+00, -1.23676025e+00,1.99868273e+03])
    #actions = np.array([-1.16826367e-01,  6.83036833e-01,  4.13037813e-01,  9.31779934e-02,1.99998315e+03])
    #pw.parameterized_pour(offset=actions[0], desired_height=actions[1], step_size=actions[2], dt=actions[3], force=actions[4])
    #pw.parameterized_pour(offset=-0.08, velocity=1.2, force=1500, desired_height=0.6)

    print(pw.base_world.ratio_beads_in(cup=pw.target_cup), "beads in")

    pw.reset(new_bead_mass = 1.1)
    
    
