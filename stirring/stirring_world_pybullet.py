from __future__ import division
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
    def __init__(self, visualize=False, real_init=True, beads=True):
        self.visualize=visualize
        self.real_init = real_init
        if real_init:
	    if visualize or not visualize: #doing this for now to workout this weird bug where the physics doesn't work in the non-GUI version
		physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
	    else:
		physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.setup(beads=beads)

    def toggle_real_time(self):
        self.is_real_time = int(not self.is_real_time)
        p.setRealTimeSimulation(self.is_real_time)
    

    """Returns the proportion of beads that are still in the cup"""
    def ratio_beads_in(self):
        aabbMin, aabbMax = p.getAABB(self.cupID)
        num_in = len(p.getOverlappingObjects(aabbMin, aabbMax))
        total = 11+2*self.num_droplets #idk where the 11 is from, but it's always there. I'm guessing gripper, plane and cup
        return num_in/total

    def distance_from_cup(self, otherObj, otherLinkIndex):
        cupPos = np.array(p.getBasePositionAndOrientation(self.cupID)[0])
        if otherLinkIndex == -1:
            otherPos = np.array(p.getBasePositionAndOrientation(otherObj)[0])
        
        else:
            otherPos = np.array(p.getLinkState(otherObj, otherLinkIndex)[0])
        return np.linalg.norm(cupPos-otherPos)
        
       
    def stirrer_close(self):
        distance = self.distance_from_cup(self.armID, 10)
        far = k*0.1
        if distance <= far:
            return True
        print("distance is far at", distance)
        return False

    def stir(self, action):
        #a force to apply on the pr2 hand, failing that the gripper
        theta_diff = action[0]
        curl = action[1]
        period = action[2]
        rot = action[3]
        #got rid of delta control since it was a horrible idea from the start
        wrist_joint_num = 8
        self.stir_circle(time_step = 0.01, size_step = rot)
        self.set_pos(self.armID, wrist_joint_num, theta_diff)
        self.set_pos(self.armID, 10, curl) #10 = curl ID
	simulate_for_duration(period)
        self.set_pos(self.armID, wrist_joint_num, -theta_diff)
	simulate_for_duration(period)
        
    
    #adds something of size_step to the current angle
    def stir_circle(self, size_step=0, time_step = 0):
        #let's say it takes 1 second t
        #and find out where x,y is is 
        current_pos = p.getJointState(self.armID, 6)[0]
        desired_pos = current_pos + size_step
        if desired_pos >= 3:
            desired_pos = -3 #reset
	p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=6,controlMode=p.POSITION_CONTROL,force=500,positionGain=0.3,velocityGain=1, targetPosition=desired_pos)
            



    def render(self):
        #but actually!!
        pass

    def step(self, timeStep, vel_iters, pos_iters):
        if not self.is_real_time:
            simulate_for_duration(timeStep, dt= 0.01)
        time.sleep(0.0001)
        return 

    """Fun enough, this function returns a tuple of two different camera views!"""
    def world_state(self):
        #crop to only relevant parts
        views = [0, 180] #I have no idea why, but these seems to be in degrees
        images = ()
        for view in views:
            rgbPixels = self.getImageFromDistance(self.cupID, 0.25, z_offset=0.1, theta_offset = view)
            images += (rgbPixels[:,:,0:3],) # decided against cropping 
        return images

    def getImageFromDistance(self, objID,cam_distance, z_offset=0,y_offset=0, x_offset= 0, theta_offset = 0):
        objPos, objQuat = p.getBasePositionAndOrientation(objID)
        adjustedPos = (objPos[0]+x_offset, objPos[1]+y_offset, objPos[2]+z_offset)
        roll, pitch, yaw = euler_from_quat(objQuat)
        yaw = yaw + theta_offset
        cam_distance = k*0.25
        im_w = 50
        im_h = 50
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
        return rgbPixels
        
        
    def showImageFromDistance(self, objID,cam_distance, z_offset=0, y_offset=0, x_offset = 0):
        rgbPixels = self.getImageFromDistance(objID, cam_distance,z_offset=z_offset, y_offset=y_offset, x_offset=x_offset)
        Image.fromarray(rgbPixels[:,:,0:3]).show()

    #this function is now a complete lie and has not only the stirrer state but
    #also the vector from the cup
    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        linkPos = p.getJointState(self.armID, 8)[0]
         
        theta_diff_pos, _, _ = p.getJointState(self.armID,8)
        rot_joint_pos, _, _ = p.getJointState(self.armID,6)
        curl_joint_pos, _, curl_joint_forces = p.getJointState(self.armID,10)

        #r, theta, z in pos 
        cupPos=  np.array(p.getBasePositionAndOrientation(self.cupID)[0])
        stirrerPos=  np.array(p.getLinkState(self.armID, 10)[0])
        vector_from_cup = cupPos-stirrerPos
        r_theta_z_pos =  cart2pol(vector_from_cup[0], vector_from_cup[1])+ (vector_from_cup[2],)
        #forces in cup frame
        r_theta_z_force =  cart2pol(curl_joint_forces[0], curl_joint_forces[1])+ (curl_joint_forces[2],)
        return np.flatten([theta_diff_pos, rot_joint_pos, curl_joint_pos, r_theta_z_pos, r_theta_z_force]) 
 
        #np.array([linkPos, jointPos, jointVel, jointReactionForces[0], jointReactionForces[1],jointReactionForces[2],jointReactionForces[3],jointReactionForces[4],jointReactionForces[5], vector_from_cup[0], vector_from_cup[1], vector_from_cup[2]])
       

    def reset(self):
        self.__init__(visualize=self.visualize, real_init=False)
    

    def create_beads(self, color = (0,0,1,1)):
       num_droplets = 150
       self.num_droplets = num_droplets
       radius = k*0.010
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
	time_to_fall = k*3.5
	colors = [(0,0,1,1),(1,0,0,1)]
	for color in colors:
	    self.create_beads(color = color)
            if not self.is_real_time:
	        simulate_for_duration(time_to_fall, dt= 0.001)
	simulate_for_duration(time_to_fall, dt= 0.001)


    def set_grip(self, pr2ID):
	self.set_pos(pr2ID, 8,0)
        nice_joint_states = [0.0, 0.006508613619013667, 0.0, 0.19977108955651196]
        for i in range(7,11):
	    self.set_pos(pr2ID, i, nice_joint_states[i-7])
        if not self.is_real_time:
	    simulate_for_duration(1.0)

    def delta_control(self, dx, dy, dz, posGain=0.3, velGain=1):
        endEIndex = 7
        jointPos = p.getLinkState(self.armID, endEIndex)[0]
        jointOrn = p.getLinkState(self.armID, endEIndex)[1]
        desiredPos = np.array((jointPos[0]+dx, jointPos[1]+dy, jointPos[2]+dz))
        self.move_arm_to_point(desiredPos, orn=jointOrn, posGain=posGain, velGain=velGain)        
    
    def move_arm_to_point(self, pos, orn = None, damper=0.1, posGain = 0.3, velGain=1, threshold=0.03, timeout=700):
        endEIndex = 7
        actualPos =  p.getLinkState(self.armID, endEIndex)[0]
        diff = np.array(actualPos)-pos
        num_attempts = 0
        while(np.linalg.norm(diff) >= threshold and num_attempts <= timeout):
            self.move_arm_closer_to_point(pos, orn=orn, damper=damper, posGain=posGain, velGain=velGain)
	    actualPos =  p.getLinkState(self.armID, endEIndex)[0]
	    diff = np.array(actualPos)-pos
            num_attempts += 1

        if num_attempts > timeout:
            print ("Failed to move to point with a distance of ",np.linalg.norm(diff))

    def move_arm_closer_to_point(self, pos, orn = None, damper=0.1, posGain = 0.3, velGain=1):
        endEIndex = 7
        ikSolver = 0
        if orn is None:
	    orn = p.getQuaternionFromEuler([0,-np.pi,0])
        numJoints = p.getNumJoints(self.armID)
        
        jd=[damper]*numJoints
	jointPoses = p.calculateInverseKinematics(self.armID,endEIndex,pos,orn,jointDamping=jd,solver=ikSolver)
	for i in range (numJoints-3):
	    p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],force=500,positionGain=posGain,velocityGain=velGain, targetVelocity=0)
        if not self.is_real_time:
	    simulate_for_duration(0.1)
    

    def place_stirrer_in_pr2_hand(self):
        cup_r = k*-0.07 
        height_above = k*0.7
        above_loc = Point(cup_r,cup_r,height_above)
	#set_point(self.spoonID,spoon_loc)
        self.move_arm_to_point(above_loc)
        
        self.toggle_real_time()
	self.set_grip(self.armID)
        
        num_steps = 4
        desired_end_height=0.33
        dz_in_loc = height_above - desired_end_height
        step_size = dz_in_loc/num_steps
        for z in range(1,num_steps+1): #fake one indexing
            in_loc = Point(cup_r,cup_r,height_above - z*step_size)
            self.move_arm_to_point(in_loc)
        #stirring motion
        #in_loc = Point(cup_r,cup_r,k*0.35)
        self.move_arm_to_point(in_loc)
	self.zoom_in_on(self.cupID, k*0.2, z_offset=k*0.1)


    def set_pos(self,objID, jointIndex, pos, force=500):
	
	p.setJointMotorControl2(bodyIndex=objID, 
	jointIndex=jointIndex, 
	controlMode=p.POSITION_CONTROL,
	targetPosition=pos,
	force = 500)


    def setup(self, beads=True):
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

	    best_arm_pos = [k*-0.65,0,0]
            if self.visualize:
	        self.armID = p.loadSDF("urdf/kuka_iiwa/kuka_with_gripper.sdf", globalScaling = k)[0]
            else: 
	        self.armID = p.loadSDF("urdf/kuka_iiwa/invisible_kuka_with_gripper.sdf", globalScaling = k)[0]
                gripper_indices = [8,9,10]
                blacken(self.armID, end_index=8)
                greenen(self.armID, gripper_indices)
            p.enableJointForceTorqueSensor(self.armID, 8, 1)
	    set_pose(self.armID,(best_arm_pos,  p.getQuaternionFromEuler([0,0,-np.pi/2])))
            
            self.zoom_in_on(self.armID, 2)
	    cupStartPos = (0,0,0)
	    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
            if self.visualize:
	        self.cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*5.0)
	    else:
                self.cupID = p.loadURDF("urdf/cup/invisible_cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=k*5.0)
                blacken(self.cupID)
          
            if beads:
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
        thresh = 1
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
def greenen(objID, indices):
    for link in indices:
        p.changeVisualShape(objID, link, rgbaColor=(0,1,0,1))
    
    

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
	

