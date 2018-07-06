from cup_world import *
import pybullet as p

k = 1

class PouringWorld():
    def __init__(self, visualize=True):
        self.base_world = CupWorld(visualize=visualize)
        self.setup()

    def setup(self):
        #create constraint and a second cup
        cupStartPos = (0,0.4,0)
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
        self.target_cup = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*4.5)
        self.cid = p.createConstraint(self.base_world.cupID, -1, -1, -1, p.JOINT_FIXED, cupStartPos, cubeStartOrientation, [0,0,1])
        

    def move_cup(self, new_loc, new_euler):
        new_orn = p.getQuaternionFromEuler(new_euler)
        p.changeConstraint(self.cid, new_loc, new_orn, maxForce = 1000) 
        simulate_for_duration(0.7)
    
    #reactive pouring controller 
    #goes to the closest lip of the cup and decreases the pitch until the beads fall out into the right place
    def pour(self, start_point, start_orn):
        self.move_cup(start_point, start_orn)
        pdb.set_trace()
        #then start decreasing the roll, pitch or yaw(whatever seems appropriate)
        start_pos, start_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        current_orn = list(p.getEulerFromQuaternion(start_orn))
        numsteps = 5
        for i in range(numsteps):
            current_orn[0] += 0.4
            self.move_cup(start_pos, current_orn)
            
        
    def manual_pour(self):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        start_euler = p.getEulerFromQuaternion(pourer_orn)
        other_cup_pos, _=  p.getBasePositionAndOrientation(self.target_cup)
        desired_height = other_cup_pos[2]+0.7
        self.move_cup((pourer_pos[0], pourer_pos[1], desired_height), start_euler)
        #first just straight up
        offset = 0.4
        start_pos = (other_cup_pos[0], other_cup_pos[1]+offset, desired_height)
        self.pour(start_pos, start_euler)

    
       

if __name__ == "__main__":
    pw = PouringWorld()
    pw.base_world.ratio_beads_in(cup=pw.target_cup)
    pw.manual_pour()
    
    
