import pybullet as p
import numpy as np
import utils
import time
import pybullet_data
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, \
    enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, \
    set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
g = 9.8
p.setGravity(0,0,-g)
planeId = p.loadURDF("plane.urdf")
pr2StartPos = [0,0,1]
cupStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
pr2ID = p.loadURDF("urdf/pr2/pr2_gripper.urdf",pr2StartPos, cubeStartOrientation)
cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation)

def create_beads(color = (0,0,1,1)):
   num_droplets = 15
   radius = 0.0045
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
    create_beads()
    simulate_for_duration(time_to_fall, dt= 0.01)
    create_beads(color = (1,0,0,1))

drop_beads_in_cup()
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    pr2Pos = p.getBasePositionAndOrientation(pr2ID)[0]
    cupPos, cupQuat = p.getBasePositionAndOrientation(cupID)
    #Keep the gripper up in the air
    #p.applyExternalForce(pr2ID,-1,pr2Pos,[0,0,weight_gripper],p.WORLD_FRAME)
    #put camera on the cup
    roll, pitch, yaw = euler_from_quat(cupQuat)
    p.resetDebugVisualizerCamera(0.5, yaw, pitch, cupPos)
    
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

