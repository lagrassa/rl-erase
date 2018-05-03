import pybullet as p
import math
import pdb
from PIL import Image
import numpy as np
import utils
import time
import pybullet_data
k = 5 #scaling factor
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose

real_init = True    

 
class World():
    def __init__(self, visualize=True):
        self.visualize=visualize
	physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
	p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.is_real_time = 0
        p.setRealTimeSimulation(self.is_real_time)
        p.resetSimulation();
	g = 9.8
	p.setGravity(0,0,-g)
	planeId = p.loadURDF("plane.urdf")
	pr2StartPos = [0,2,1]
	pr2StartOrientation = p.getQuaternionFromEuler([0,np.pi/2,np.pi/2])
	#self.pr2ID = p.loadURDF("urdf/pr2/pr2_gripper.urdf",pr2StartPos, pr2StartOrientation)
        self.setup()
    def toggle_real_time(self):
        self.is_real_time = 1
        p.setRealTimeSimulation(self.is_real_time)
       
    def stirrer_close(self):
        
        jointPos = p.getJointState(self.armID, 8)[0]
        distance = np.linalg.norm(jointPos)
        far = 2
        if distance <= far:
            return True
        return False

    def stir(self, theta_diff):
        #a force to apply on the pr2 hand, failing that the gripper
        num_motion = 4
        direction = 1
        wrist_joint_num = 8;
        for i in range(num_motion):
	    p.setJointMotorControl2(
		bodyIndex=self.armID,
		jointIndex=wrist_joint_num,
		targetPosition = direction*theta_diff,
		targetVelocity = 0,
		controlMode=p.POSITION_CONTROL,
		force=500)
            direction *= -1
        

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
        objPos, objQuat = p.getBasePositionAndOrientation(self.cupID)
        roll, pitch, yaw = euler_from_quat(objQuat)
        cam_distance = 0.25
        im_w = 200
        im_h = 200
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=objPos, distance=cam_distance, yaw=yaw , pitch=pitch, roll =roll+np.pi, upAxisIndex=2)
        if self.visualize:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER
        _,_,rgbPixels,_,_ = p.getCameraImage(width=im_w,height=im_h, viewMatrix=viewMatrix, renderer=renderer)
        #self.showImageFromDistance(0.25)
        #crop to only relevant parts
        rgbPixels_cropped = rgbPixels[0:115,57:135,0:3] #maaaagic need to adjust if changing either resolution or distance....
        return rgbPixels_cropped

    def showImageFromDistance(self, cam_distance):
        objPos, objQuat = p.getBasePositionAndOrientation(self.cupID)
        roll, pitch, yaw = euler_from_quat(objQuat)
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=objPos, distance=cam_distance, yaw=yaw , pitch=pitch, roll =roll, upAxisIndex=2)
        _,_,rgbPixels,_,_ = p.getCameraImage(width=200,height=200, viewMatrix=viewMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        Image.fromarray(rgbPixels[:,:,0:3]).show()

    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        linkPos = p.getJointState(self.armID, 8)[0]
        jointPos, jointVel, jointReactionForces, _ = p.getJointState(self.armID,8)
        return  np.array([linkPos, jointPos, jointVel, jointReactionForces[0], jointReactionForces[1],jointReactionForces[2],jointReactionForces[3],jointReactionForces[4],jointReactionForces[5]])

    def reset(self):
        p.disconnect()
        self.__init__()

    def create_beads(self, color = (0,0,1,1)):
       num_droplets = 90
       radius = 0.013
       droplets = [create_sphere(radius, mass=0.01, color=color) for _ in range(num_droplets)] # kg
       cup_thickness = 0.001

       lower, upper = get_lower_upper(self.cupID)
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
	time_to_fall = 1.5
	colors = [(0,0,1,1),(1,0,0,1)]
	for color in colors:
	    self.create_beads(color = color)
            if not self.is_real_time:
	        simulate_for_duration(time_to_fall, dt= 0.001)

        if not self.is_real_time:
	    simulate_for_duration(3*time_to_fall, dt= 0.001)
	#set_point(self.pr2ID, Point(0, 0, 0.8))
     

    def set_grip(self, pr2ID, width):
	self.set_pos(pr2ID, 8,6)
        nice_joint_states = [0.0, 0.006508613619013667, 0.0, 0.19977108955651196]
        for i in range(7,11):
	    self.set_pos(pr2ID, i, nice_joint_states[i-7])
        if not self.is_real_time:
	    simulate_for_duration(1.0)


    def move_arm_to_point(self, pos):
        endEIndex = 6
        ikSolver = 0
	orn = p.getQuaternionFromEuler([0,-math.pi,0])
        numJoints = p.getNumJoints(self.armID)
        
        jd=[0.1]*numJoints
	jointPoses = p.calculateInverseKinematics(self.armID,endEIndex,pos,orn,jointDamping=jd,solver=ikSolver)
	for i in range (numJoints-3):
	    p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],force=500,positionGain=0.2,velocityGain=1, targetVelocity=0)
        if not self.is_real_time:
	    simulate_for_duration(0.1)
	p.addUserDebugLine([0,0.3,0.31],pos,[0,0,0.3],1)
    

    def place_stirrer_in_pr2_hand(self):
        
        
	maxForce = 10
	#set joints to be slightly open
	open_width = 0.4
	spoon_radius = 0.005
	closed_width = 2*spoon_radius-0.01
	spoon_l = 0.4
        self.spoon_l = spoon_l
	hand_height = 0.7
	#spawn spoon
	#self.spoonID = create_cylinder(spoon_radius, spoon_l, color=(0,1, 0, 1), mass=2)
        spoon_loc =   Point(0, 0, hand_height-spoon_l-0.1)
        above_loc = Point(-0.03,-0.03,0.3)
        cup_r = -0.02 
        above_loc = Point(cup_r,cup_r,0.7)
	#set_point(self.spoonID,spoon_loc)
        for i in range(30):
            self.move_arm_to_point(above_loc)
	self.set_grip(self.armID, open_width)

        #stirring motion
        in_loc = Point(cup_r-0.03,cup_r,0.4)
        for i in range(30):
            self.move_arm_to_point(in_loc)
	self.zoom_in_on(self.cupID, 0.2)


	#self.set_grip(self.armID, closed_width)
	#self.top_down_zoom_in_on(self.cupID)

    def set_pos(self,objID, jointIndex, pos, force=500):
	
	p.setJointMotorControl2(bodyIndex=objID, 
	jointIndex=jointIndex, 
	controlMode=p.POSITION_CONTROL,
	targetPosition=pos,
	force = 500)


    def setup(self):
        global real_init
        NEW = True #unfortunately
        if NEW:
	    best_arm_pos = [-0.4,0,0]
            if self.visualize:
	        self.armID = p.loadSDF("urdf/kuka_iiwa/kuka_with_gripper.sdf")[0]
            else: 
	        self.armID = p.loadSDF("urdf/kuka_iiwa/invisible_kuka_with_gripper.sdf")[0]
                blacken(self.armID, end_index=8)
	    set_pose(self.armID,(best_arm_pos,  p.getQuaternionFromEuler([0,0,-np.pi/2])))
            
            self.zoom_in_on(self.armID, 2)
	    cupStartPos = (0,0,0)
	    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
            if self.visualize:
	        self.cupID = p.loadURDF("urdf/cup/cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=5.0)
	    else:
                self.cupID = p.loadURDF("urdf/cup/invisible_cup_small.urdf",cupStartPos, cubeStartOrientation, globalScaling=5.0)
                blacken(self.cupID)

	    self.drop_beads_in_cup()
            self.toggle_real_time()
	    self.place_stirrer_in_pr2_hand()
            #p.saveBullet("pybullet_world.bullet")
            real_init = False
            self.bullet_id = p.saveState()
        else:
            print("Restoring state")
       
            p.restoreState(self.bullet_id)

    def zoom_in_on(self,objID, dist = 0.7):
	objPos, objQuat = p.getBasePositionAndOrientation(objID)
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
    world = World()
    world.reset()
	

