import pybullet as p
import math
import pdb
from PIL import Image
import numpy as np
import utils
import time
import pybullet_data
k = 1 #scaling factor
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose

real_init = True    

 
class World():
    def __init__(self, visualize=True, real_init=True):
        self.visualize=visualize
        self.real_init = real_init
        if real_init:
	    if visualize:
		print("Using GUI server")
		physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
	    else:
		print("Using Direct server")
		physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

 
        self.setup()
        print("Done with setup")

    def toggle_real_time(self):
        self.is_real_time = 1
        p.setRealTimeSimulation(self.is_real_time)
       
    def stirrer_close(self):
        
        
        jointPos = p.getJointState(self.armID, 8)[0]
        distance = np.linalg.norm(jointPos)
        far = k*0.8
        if distance <= far:
            return True
        print("distance is far at", distance)
        return False

    def stir(self, action):
        #a force to apply on the pr2 hand, failing that the gripper
        theta_diff = action[0]
        period = action[1]
        max_del = k*0.01
        x_del = max_del*np.tanh(action[2])
        y_del = max_del*np.tanh(action[3])
        z_del = max_del*np.tanh(action[4])
        direction = 1
        wrist_joint_num = 8; 
        self.delta_control(x_del, y_del, z_del)
        #place stirrer in correct location
	p.setJointMotorControl2(
	    bodyIndex=self.armID,
	    jointIndex=wrist_joint_num,
	    targetPosition = direction*theta_diff,
	    targetVelocity = 0,
	    controlMode=p.POSITION_CONTROL,
	    force=500)
	simulate_for_duration(period)
        
        

    def stir_circle(self, radius, step, depth):
        #work on a clockwise circle around (0,0)
        #try to move to the closest point on the circle, and then a circumference of step away from that
        spoon_pos,spoon_quat = p.getBasePositionAndOrientation(self.spoonID)
        roll, pitch, yaw = euler_from_quat(spoon_quat)
        r = self.spoon_l
        elevation=np.pi/2.0-pitch
        azimuth = yaw
        x_top = spoon_pos[0]+r*np.sin(elevation)*np.cos(azimuth)
        y_top = spoon_pos[1] +r*np.sin(elevation)*np.sin(azimuth)
        z_top = spoon_pos[2]  + r*np.cos(elevation)
        top_pos = [x_top, y_top, z_top]
        p.addUserDebugLine(spoon_pos,top_pos,[0.8,0,0],1,5)
        _,phi_current = cart2pol(spoon_pos[0],spoon_pos[1])
        #increase phi by step, and set rho to what you have
        target_xy = pol2cart(radius, phi_current)
        target_pos = target_xy + (float(depth),)
        # 
        diff_pos = [target_xy[i]-spoon_pos[i] for i in range(len(target_xy))]
        p.applyExternalForce(self.spoonID,0,[-17,0,0],[0,0,0],p.LINK_FRAME)
        #p.setJointMotorControl2(
        #    bodyIndex = self.spoonID,
        #    jointIndex=0,
        #    controlMode=p.POSITION_CONTROL,
        #    targetPosition=diff_pos[0],
        #    force=1.0)
        #p.setJointMotorControl2(bodyIndex = self.spoonID,jointIndex=0,controlMode=p.POSITION_CONTROL,targetPosition=190,force=500)


    def render(self):
        #but actually!!
        pass

    def step(self, timeStep, vel_iters, pos_iters):
        if not self.is_real_time:
            simulate_for_duration(timeStep, dt= 0.01)
        time.sleep(0.0001)
        return 

    def world_state(self):
        #self.showImageFromDistance(0.25)
        #crop to only relevant parts
        rgbPixels = self.getImageFromDistance(self.cupID, 0.25, z_offset=0.1)
        rgbPixels_cropped = rgbPixels[:,40:160,0:3] #maaaagic need to adjust if changing either resolution or distance....
        return rgbPixels_cropped

    def getImageFromDistance(self, objID,cam_distance, z_offset=0,y_offset=0, x_offset= 0):
        objPos, objQuat = p.getBasePositionAndOrientation(objID)
        adjustedPos = (objPos[0]+x_offset, objPos[1]+y_offset, objPos[2]+z_offset)
        roll, pitch, yaw = euler_from_quat(objQuat)
        cam_distance = k*0.25
        im_w = 200
        im_h = 200
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=adjustedPos, distance=cam_distance, yaw=yaw , pitch=pitch, roll =roll+np.pi, upAxisIndex=2)
        if self.visualize:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER
        fov = 60
        nearPlane = 0.01
        farPlane = 500
        aspect = im_w/im_h
        projectionMatrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane, farPlane)
  
        _,_,rgbPixels,_,_ = p.getCameraImage(width=im_w,height=im_h, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, shadow=0, lightDirection = [1,1,1],renderer=renderer)
        #Image.fromarray(rgbPixels).show()
        return rgbPixels
        
        
    def showImageFromDistance(self, objID,cam_distance, z_offset=0, y_offset=0, x_offset = 0):
        rgbPixels = self.getImageFromDistance(objID, cam_distance,z_offset=z_offset, y_offset=y_offset, x_offset=x_offset)
        Image.fromarray(rgbPixels[:,:,0:3]).show()

    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        linkPos = p.getJointState(self.armID, 8)[0]
        jointPos, jointVel, jointReactionForces, _ = p.getJointState(self.armID,8)
        return  np.array([linkPos, jointPos, jointVel, jointReactionForces[0], jointReactionForces[1],jointReactionForces[2],jointReactionForces[3],jointReactionForces[4],jointReactionForces[5]])

    def reset(self):
        print("In visualize self.visualize=",self.visualize)
        self.__init__(visualize=self.visualize, real_init=False)
    

    def create_beads(self, color = (0,0,1,1)):
       num_droplets = 160
       radius = k*0.011
       cup_thickness = k*0.001

       lower, upper = get_lower_upper(self.cupID)
       buffer = cup_thickness + radius
       lower = np.array(lower) + buffer*np.ones(len(lower))
       upper = np.array(upper) - buffer*np.ones(len(upper))
       limits = zip(lower, upper)
       x_range, y_range = limits[:2]
       z = upper[2] + 0.1
       droplets = [create_sphere(radius, color=color) for _ in range(num_droplets)]
       for droplet in droplets:
	   x = np.random.uniform(*x_range)
	   y = np.random.uniform(*y_range)
	   set_point(droplet, Point(x, y, z))

       for i, droplet in enumerate(droplets):
	   x, y = np.random.normal(0, 1e-3, 2)
	   set_point(droplet, Point(x, y, z+i*(2*radius+1e-3)))

    def drop_beads_in_cup(self):
	time_to_fall = k*2.5
	colors = [(0,0,1,1),(1,0,0,1)]
	for color in colors:
	    self.create_beads(color = color)
            if not self.is_real_time:
	        simulate_for_duration(time_to_fall, dt= 0.001)


    def set_grip(self, pr2ID):
	self.set_pos(pr2ID, 8,6)
        nice_joint_states = [0.0, 0.006508613619013667, 0.0, 0.19977108955651196]
        for i in range(7,11):
	    self.set_pos(pr2ID, i, nice_joint_states[i-7])
        if not self.is_real_time:
	    simulate_for_duration(1.0)

    def delta_control(self, dx, dy, dz, posGain=0.3, velGain=1):
        endEIndex = 6
        jointPos = p.getLinkState(self.armID, endEIndex)[0]
        jointOrn = p.getLinkState(self.armID, endEIndex)[1]
        desiredPos = (jointPos[0]+dx, jointPos[1]+dy, jointPos[2]+dz)
        self.move_arm_to_point(desiredPos, orn=jointOrn, posGain=posGain, velGain=velGain)        
    
    def move_arm_to_point(self, pos, orn = None, damper=0.1, posGain = 0.3, velGain=1):
        endEIndex = 6
        ikSolver = 0
        if orn is None:
	    orn = p.getQuaternionFromEuler([0,-math.pi,0])
        numJoints = p.getNumJoints(self.armID)
        
        jd=[damper]*numJoints
	jointPoses = p.calculateInverseKinematics(self.armID,endEIndex,pos,orn,jointDamping=jd,solver=ikSolver)
	for i in range (numJoints-3):
	    p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],force=500,positionGain=posGain,velocityGain=velGain, targetVelocity=0)
        if not self.is_real_time:
	    simulate_for_duration(0.2)
    

    def place_stirrer_in_pr2_hand(self):
        cup_r = k*-0.04 
        above_loc = Point(cup_r,cup_r,k*0.7)
	#set_point(self.spoonID,spoon_loc)
        for i in range(14):
            self.move_arm_to_point(above_loc)
        self.toggle_real_time()
	self.set_grip(self.armID)
        #stirring motion
        in_loc = Point(cup_r-k*0.03,cup_r,k*0.2)
        for i in range(11):
            self.move_arm_to_point(in_loc)
	self.zoom_in_on(self.cupID, k*0.2, z_offset=k*0.1)
	self.zoom_in_on(self.cupID, k*1.2, z_offset=k*0.1)


    def set_pos(self,objID, jointIndex, pos, force=500):
	
	p.setJointMotorControl2(bodyIndex=objID, 
	jointIndex=jointIndex, 
	controlMode=p.POSITION_CONTROL,
	targetPosition=pos,
	force = 500)


    def setup(self):
        NEW = self.real_init #unfortunately
        if NEW:
            #setup world
	    self.is_real_time = 0
	    p.setRealTimeSimulation(self.is_real_time)
	    #p.resetSimulation();
	    g = 9.8
	    p.setGravity(0,0,-g)
	    if self.visualize:
		self.planeId = p.loadURDF("plane.urdf")
	    else:
		self.planeId = p.loadURDF("urdf/invisible_plane.urdf")
		blacken(self.planeId)

	    best_arm_pos = [k*-0.6,0,0]
            if self.visualize:
	        self.armID = p.loadSDF("urdf/kuka_iiwa/kuka_with_gripper.sdf", globalScaling = k)[0]
            else: 
	        self.armID = p.loadSDF("urdf/kuka_iiwa/invisible_kuka_with_gripper.sdf", globalScaling = k)[0]
                blacken(self.armID, end_index=8)
	    set_pose(self.armID,(best_arm_pos,  p.getQuaternionFromEuler([0,0,-np.pi/2])))
            
            self.zoom_in_on(self.armID, 2)
	    cupStartPos = (0,0,0)
	    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
            if self.visualize:
	        self.cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*5.0)
	    else:
                self.cupID = p.loadURDF("urdf/cup/invisible_cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*5.0)
                blacken(self.cupID)

	    self.drop_beads_in_cup()
	    self.place_stirrer_in_pr2_hand()
            self.bullet_id = p.saveState()
            self.real_init = False
        else:
            try:
                p.restoreState(self.bullet_id)
            except:
                self.real_init = True
                p.resetSimulation()
                self.setup()
    def cup_knocked_over(self):
        cupPos, cupQuat =  p.getBasePositionAndOrientation(self.cupID)
        roll, pitch, yaw = euler_from_quat(cupQuat)
        thresh = 0.6
        if abs(roll) > thresh or abs(pitch) > thresh:
            return True
        return False
        


    def zoom_in_on(self,objID, dist = k*0.7, z_offset = 0):
	objPos, objQuat = p.getBasePositionAndOrientation(objID)
        adjustedPos = (objPos[0], objPos[1], objPos[2]+z_offset)
	roll, pitch, yaw = euler_from_quat(objQuat)
	p.resetDebugVisualizerCamera(dist, yaw, roll, objPos)

    def top_down_zoom_in_on(self,objID):
	objPos, objQuat = p.getBasePositionAndOrientation(objID)
	roll, pitch, yaw = euler_from_quat(objQuat)
	p.resetDebugVisualizerCamera(0.5, yaw, -70, objPos)
 

	#p.resetDebugVisualizerCamera(0.5, yaw, roll, objPos)
    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_VR_PICKING, p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0) 
#blackens all links in object
def blacken(objID, end_index =None):
    p.changeVisualShape(objID, -1, rgbaColor=(0,1,0,0))
    if end_index is None:
        end_index =  p.getNumJoints(objID)
    for link in range(end_index):
        p.changeVisualShape(objID, link, rgbaColor=(0,1,0,0))
    
    

def closest_point_circle(center, xy_pos, radius):
    A = center
    B = xy_pos
    Cx = A[0]+radius*(B[0]-A[0])/np.linalg.norm(B-A)
    Cy = A[1]+radius*(B[1]-A[1])/np.linalg.norm(B-A)
    return np.array([Cx,Cy])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
        

if __name__ == "__main__":
    world = World(visualize=False)
    world.reset()
	

