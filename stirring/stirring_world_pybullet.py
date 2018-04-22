import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
g = 9.8
p.setGravity(0,0,-g)
planeId = p.loadURDF("plane.urdf")
pr2StartPos = [0,0,1]
cupStartPos = [0,0,0]
spherePos = [0,0.01,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
pr2ID = p.loadURDF("urdf/pr2/pr2_gripper.urdf",pr2StartPos, cubeStartOrientation)
cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation)
sphereID = p.loadURDF("sphere_small.urdf",spherePos, cubeStartOrientation)

mass_gripper = 0.91
weight_gripper = g*mass_gripper

#attempt to applyExternalForce
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    pr2Pos = p.getBasePositionAndOrientation(pr2ID)[0]
    #Keep the gripper up in the air
    p.applyExternalForce(pr2ID,-1,pr2Pos,[0,0,weight_gripper],p.WORLD_FRAME)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

