from cup_world import *
import pybullet as p
from utils import set_point

k = 1

class PouringWorld():
    def __init__(self, visualize=True, real_init=False):
        self.base_world = CupWorld(visualize=visualize, beads=False)
        if real_init:
            self.setup()
        else:
            p.restoreState(self.bullet_id)

    def setup(self):
        #create constraint and a second cup
        cupStartPos = (0,-0.4,0)
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
        self.target_cup = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*4.5)
        self.cid = p.createConstraint(self.base_world.cupID, -1, -1, -1, p.JOINT_FIXED, cupStartPos, cubeStartOrientation, [0,0,1])
        self.bullet_id = p.saveState()
        
    def reset(self, real_init=False):
        p.resetSimulation()
        self.base_world.reset()
        self.setup() 
                

    def move_cup(self, new_loc, new_euler, duration=0.7, teleport=False, force = 1000):
        if teleport:
            set_point(self.base_world.cupID, new_loc)
            for bead in self.base_world.droplets:
                #original_loc 
                loc = p.getBasePositionAndOrientation(bead)[0]
                new_loc = np.array(loc)+np.array(new_loc)
                set_point(bead, new_loc)
        else:
            new_orn = p.getQuaternionFromEuler(new_euler)
            p.changeConstraint(self.cid, new_loc, new_orn, maxForce = force) 
            simulate_for_duration(duration)
    
    #reactive pouring controller 
    #goes to the closest lip of the cup and decreases the pitch until the beads fall out into the right place
    def pour(self, offset, start_orn, step_size, dt, force):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        start_point = (pourer_pos[0], pourer_pos[1]+offset, pourer_pos[2])
        self.move_cup(start_point, start_orn, duration=2,force=1500) #don't even bother with this
        #then start decreasing the roll, pitch or yaw(whatever seems appropriate)
        start_pos, start_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        current_orn = list(p.getEulerFromQuaternion(start_orn))
        numsteps = 13
        for i in range(numsteps):
            current_orn[0] += step_size
            self.move_cup(start_pos, current_orn, duration=0.5, force=force)

    def pourer_state(self):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        target_pos, target_orn = p.getBasePositionAndOrientation(self.target_cup)
        cup_rel_pos = np.array(pourer_pos) - np.array(target_pos)
        return np.hstack([cup_rel_pos, pourer_orn])

    def world_state(self):
        return self.base_world.world_state() 

    def parameterized_pour(self, offset=0.4, desired_height = 0.8, step_size=0.2, dt = 0.11, force=2500):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        start_euler = p.getEulerFromQuaternion(pourer_orn)
        other_cup_pos, _=  p.getBasePositionAndOrientation(self.target_cup)
        desired_height = other_cup_pos[2]+desired_height
        self.move_cup((pourer_pos[0], pourer_pos[1], desired_height), start_euler, duration=5., force=1000)
        #first just straight up
        self.pour(offset, start_euler, abs(step_size), abs(dt), force)

    
       

if __name__ == "__main__":
    pw = PouringWorld(visualize=True, real_init = True)
    pw.base_world.ratio_beads_in(cup=pw.target_cup)
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.25397044e-01, -1.43723112e+00, -1.14753149e+00, -1.23676025e+00,1.99868273e+03])
    #actions = np.array([-1.16826367e-01,  6.83036833e-01,  4.13037813e-01,  9.31779934e-02,1.99998315e+03])
    #pw.parameterized_pour(offset=actions[0], desired_height=actions[1], step_size=actions[2], dt=actions[3], force=actions[4])
    pw.parameterized_pour(offset=-0.05)

    pw.reset()
    
    
