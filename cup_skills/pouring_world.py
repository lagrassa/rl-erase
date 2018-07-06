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
        

    def move_cup(self, new_loc, new_orn)
        p.changeConstraint(self.cid, new_loc, new_orn, maxForce = 200) 
    
       

if __name__ == "__main__":
    pw = PouringWorld()
