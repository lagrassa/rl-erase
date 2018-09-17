from cup_world import *
import pdb as pdb
import pybullet as p
import numpy as np

import reward
from utils import set_point

k = 1

class PouringWorld():
    def __init__(self, visualize=False, real_init=True, new_bead_mass=None, dims=None):
        self.base_world = CupWorld(visualize=visualize, beads=False, new_bead_mass=new_bead_mass)
        self.cup_to_dims = {"cup_1.urdf":(0.5,0.5), "cup_2.urdf":(0.5, 0.2), "cup_3.urdf":(0.7, 0.3), "cup_4.urdf":(1.1,0.3), "cup_5.urdf":(1.1,0.2), "cup_6.urdf":(0.6, 0.7)}#cup name to diameter and height
        lower =  [0.5, -0.3, -0.3, 0.8, 0,2*np.pi/3]
        upper = [0.75, 0.3, 0.3, 2, 3.14,np.pi]
        #height, x_offset, y_offset, velocity, yaw, total_diff = x
        self.discrete_contexts = self.cup_to_dims.values()
        self.x_range = np.array([lower, upper])
        self.nb_actions = len(lower)
        self.task_lengthscale = np.ones(self.nb_actions)*0.4
        self.lengthscale_bound = np.array([[0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.00000001], [0.3, 0.15, 0.5, 0.5, 0.2,0.3, 2]])
     
        self.context_idx = []
        
        self.param_idx = [0,1,2,3, 4, 5]
        self.dx = len(self.x_range[0])
        self.do_gui = False

        if real_init:
            self.setup(dims=dims)
        else:
            p.restoreState(self.bullet_id)

    def check_legal(self, x):
        return True

    def sampled_x(self, n):
        i = 0
        N = 300
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x
            else:
                assert(False)

    def setup(self, dims=None):
        #create constraint and a second cup
        self.cupStartPos = (0,-0.4,0)
        self.cupStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
        #pick random cup
        if dims is None:
            self.cup_name = np.random.choice(self.cup_to_dims.keys())
        else:
            found_cup = False
            for name in self.cup_to_dims.keys():
                if self.cup_to_dims[name][0] == dims[0] and self.cup_to_dims[name][1] == dims[1]:
                    found_cup = True
                    self.cup_name = name
            assert(found_cup)
        cup_file = "urdf/cup/"+self.cup_name
        self.target_cup = p.loadURDF(cup_file,self.cupStartPos, self.cupStartOrientation, globalScaling=k*5)
        self.base_world.drop_beads_in_cup()
        self.cid = p.createConstraint(self.base_world.cupID, -1, -1, -1, p.JOINT_FIXED, self.cupStartPos, self.cupStartOrientation, [0,0,1])
        self.bullet_id = p.saveState()
        
    def observe_cup(self):
        return np.array(self.cup_to_dims[self.cup_name])

    def reset(self, real_init=False, new_bead_mass=None, dims = None):
        self.base_world.reset(new_bead_mass=new_bead_mass)
        self.setup(dims=dims)      
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
    def pour(self, x_offset=0.05, y_offset=0.02, velocity=0.9, force=1500, total_diff = 4*np.pi/5.0, yaw = 0):
        #step_size and dt come from the angular velocity it takes to make a change of 3pi/4, can set later

        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        target_pos, _ = p.getBasePositionAndOrientation(self.target_cup)
        start_point = (target_pos[0]+x_offset, target_pos[1]+y_offset, pourer_pos[2])
        self.move_cup(start_point,  duration=2,force=force) 
        start_pos, start_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        #then start decreasing the roll, pitch or yaw(whatever seems appropriate)
        current_orn = list(p.getEulerFromQuaternion(start_orn))
        numsteps = 25.0
        step_size = total_diff/numsteps; #hard to set otherwise
        dt = step_size/velocity
        current_orn[2] = yaw
        for i in range(int(numsteps)):
            current_orn[0] += step_size
            self.move_cup(start_pos, current_orn, duration=dt, force=force)
        simulate_for_duration(1.2) #rest time, can be tuned

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
        #desired_height = other_cup_pos[2]+desired_height
        self.move_cup((pourer_pos[0], pourer_pos[1], desired_height), duration=3.5, force=force)

    def __call__(self, x, image_name=None, real_init=False):
        self.reset(dims=x[-2:], real_init=real_init) #discrete context 
        height, x_offset, y_offset, velocity, yaw, total_diff = x[:self.x_range.shape[1]] #last 2 are cont_context, discrete_context
        self.lift_cup(desired_height=height)
        self.pour(x_offset=x_offset, y_offset=y_offset, velocity=velocity, force=1500, yaw=yaw, total_diff = total_diff)
        #returns ratio of beads in cup over the acceptable number
        acceptable = 0.98
        beads_in_cup = self.base_world.ratio_beads_in(cup=self.target_cup) 
        return beads_in_cup - acceptable

    


    
       

if __name__ == "__main__":
    sample = [ 0.64536634, -0.17830571, -0.18541563,  0.89087883,  2.84794665,2.97115431,  1.1       ,  0.3       ]
    pw = PouringWorld(visualize=True)
    #pw.lift_cup(desired_height=sample[0])
    #pw.pour(offset=-0.2, velocity=1.4, force=1500, total_diff = 4*np.pi/5.0)
    #pw.pour(x_offset = 0.2, y_offset=-0.15, velocity=1.4, force=1500, total_diff = 2.51, yaw=np.pi)
    pw(sample, real_init=False)
    print(pw.base_world.ratio_beads_in(cup=pw.target_cup), "beads in")
    
    
