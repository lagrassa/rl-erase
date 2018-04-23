import pybullet as p
import pdb
import numpy as np
import utils
import time
import pybullet_data
k = 5 #scaling factor
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
g = 9.8
p.setGravity(0,0,-g)
planeId = p.loadURDF("plane.urdf")
pr2StartPos = [0,2,1]
pr2StartOrientation = p.getQuaternionFromEuler([0,np.pi/2,np.pi/2])
cupStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
pr2ID = p.loadURDF("urdf/pr2/pr2_gripper.urdf",pr2StartPos, pr2StartOrientation)
cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=5.0)

def create_beads(color = (0,0,1,1)):
   num_droplets = 90
   radius = 0.015
   droplets = [create_sphere(radius, mass=0.01, color=color) for _ in range(num_droplets)] # kg
   cup_thickness = 0.001

   lower, upper = get_lower_upper(cupID)
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

def drop_beads_in_cup():
    time_to_fall = 2
    colors = [(0,0,1,1),(1,0,0,1)]
    for color in colors:
	create_beads(color = color)
	simulate_for_duration(time_to_fall, dt= 0.001)

    simulate_for_duration(3*time_to_fall, dt= 0.001)
    set_point(pr2ID, Point(0, 0, 0.8))

def set_grip(pr2ID, width):
    right_gripper_joint_num = 2
    left_gripper_joint_num = 0
    set_pos(pr2ID, right_gripper_joint_num, width)
    set_pos(pr2ID, left_gripper_joint_num, width)
    simulate_for_duration(1.0)

def place_stirrer_in_pr2_hand():
    maxForce = 10
    #set joints to be slightly open
    open_width = 0.3
    spoon_radius = 0.005
    closed_width = 2*spoon_radius-0.01
    spoon_l = 0.35
    hand_height = 0.7
    set_grip(pr2ID, open_width)
    #spawn spoon
    spoonID = create_cylinder(spoon_radius, spoon_l, color=(0, 0, 1, 1))
    set_point(spoonID, Point(0, 0, hand_height-spoon_l-0.1))

    #make joints small again
    set_point(pr2ID, Point(0, 0, hand_height))
    set_grip(pr2ID, closed_width)
    zoom_in_on(pr2ID)

def set_pos(objID, jointIndex, pos):
    
    p.setJointMotorControl2(bodyIndex=objID, 
    jointIndex=jointIndex, 
    controlMode=p.POSITION_CONTROL,
    targetPosition=pos,
    force = 500)


def setup():
    #drop_beads_in_cup()
    place_stirrer_in_pr2_hand()

def zoom_in_on(objID):
    objPos, objQuat = p.getBasePositionAndOrientation(objID)
    roll, pitch, yaw = euler_from_quat(objQuat)
    p.resetDebugVisualizerCamera(0.8, yaw, pitch, objPos)
 
setup()

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    pr2Pos = p.getBasePositionAndOrientation(pr2ID)[0]
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

