import pybullet as p
import pdb
import numpy as np
import utils
import time
import pybullet_data
k = 5 #scaling factor
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat

    

 
class World():
    def __init__(self):
	physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.resetSimulation();
	g = 9.8
	p.setGravity(0,0,-g)
	planeId = p.loadURDF("plane.urdf")
	pr2StartPos = [0,2,1]
	pr2StartOrientation = p.getQuaternionFromEuler([0,np.pi/2,np.pi/2])
	cupStartPos = [0,0,0]
	cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
	self.pr2ID = p.loadURDF("urdf/pr2/pr2_gripper.urdf",pr2StartPos, pr2StartOrientation)
	self.cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=5.0)
        self.setup()
    def stirrer_close(self):
        jointPos, jointVelocity = p.getJointState(cupId, 0)
        distance = np.linalg.norm(jointPos)
        far = 0.8
        if distance <= far:
            return True
        return False
    def stir(self, force):
        #a force to apply on the pr2 hand, failing that the gripper

        pass
        

    def render(self):
        #but actually!!
        pass

    def step(self, timeStep, vel_iters, pos_iters):
        simulate_for_duration(timeStep)

    def world_state(self):
        objPos, objQuat = p.getBasePositionAndOrientation(self.cupID)
        roll, pitch, yaw = euler_from_quat(objQuat)
        cam_distance = 0.6
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=objPos, distance=cam_distance, yaw=yaw , pitch=pitch, roll =roll, upAxisIndex=2)
        _,_,rgbPixels,_,_ = p.getCameraImage(width=200,height=200, viewMatrix=viewMatrix)
        return rgbPixels[:,:,0:3]

    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        pdb.set_trace()
        jointPos, jointVelocity = p.getJointState(cupId, 0)
        return jointPos.extend(jointVelocity)

    def reset():
        p.disconnect()
        self.__init__()
    # private methods

    def create_beads(self, color = (0,0,1,1)):
       num_droplets = 90
       radius = 0.015
       droplets = [create_sphere(radius, mass=0.01, color=color) for _ in range(num_droplets)] # kg
       cup_thickness = 0.001

       lower, upper = get_lower_upper(self.cupID)
       print(lower, upper)
       buffer = cup_thickness + radius
       lower = np.array(lower) + buffer*np.ones(len(lower))
       upper = np.array(upper) - buffer*np.ones(len(upper))
       limits = zip(lower, upper)
       x_range, y_range = limits[:2]
       z = upper[2] + 0.1
       for droplet in droplets:
	   x = np.random.uniform(*x_range)
	   y = np.random.uniform(*y_range)
	   set_point(droplet, Point(x, y, z))

       for i, droplet in enumerate(droplets):
	   x, y = np.random.normal(0, 1e-3, 2)
	   set_point(droplet, Point(x, y, z+i*(2*radius+1e-3)))

    def drop_beads_in_cup(self):
	time_to_fall = 2
	colors = [(0,0,1,1),(1,0,0,1)]
	for color in colors:
	    self.create_beads(color = color)
	    simulate_for_duration(time_to_fall, dt= 0.001)

	simulate_for_duration(3*time_to_fall, dt= 0.001)
	set_point(self.pr2ID, Point(0, 0, 0.8))

    def set_grip(self, pr2ID, width):
	right_gripper_joint_num = 2
	left_gripper_joint_num = 0
	self.set_pos(pr2ID, right_gripper_joint_num, width)
	self.set_pos(pr2ID, left_gripper_joint_num, width)
	simulate_for_duration(1.0)

    def place_stirrer_in_pr2_hand(self):
	maxForce = 10
	#set joints to be slightly open
	open_width = 0.3
	spoon_radius = 0.005
	closed_width = 2*spoon_radius-0.01
	spoon_l = 0.35
	hand_height = 0.7
	self.set_grip(self.pr2ID, open_width)
	#spawn spoon
	spoonID = create_cylinder(spoon_radius, spoon_l, color=(0, 0, 1, 1))
	set_point(spoonID, Point(0, 0, hand_height-spoon_l-0.1))

	#make joints small again
	set_point(self.pr2ID, Point(0, 0, hand_height))
	self.set_grip(self.pr2ID, closed_width)
	self.zoom_in_on(self.pr2ID)

    def set_pos(self,objID, jointIndex, pos):
	
	p.setJointMotorControl2(bodyIndex=objID, 
	jointIndex=jointIndex, 
	controlMode=p.POSITION_CONTROL,
	targetPosition=pos,
	force = 500)


    def setup(self):
	self.drop_beads_in_cup()
	self.place_stirrer_in_pr2_hand()

    def zoom_in_on(self,objID):
	objPos, objQuat = p.getBasePositionAndOrientation(objID)
	roll, pitch, yaw = euler_from_quat(objQuat)
	p.resetDebugVisualizerCamera(0.8, yaw, pitch, objPos)

if __name__ == "__main__":
    world = World()
    for i in range (10000):
        world.step(0.2, None, None)
        """
	p.stepSimulation()
	time.sleep(1./240.)
	pr2Pos = p.getBasePositionAndOrientation(self.pr2ID)[0]
        """
	

