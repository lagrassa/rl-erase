from __future__ import division
import pybullet as p
import math
import csv
import pdb
from PIL import Image
from cup_world import *
import numpy as np
import utils
import time
import pybullet_data
k = 1 #scaling factor
DEMO =False 
from utils import add_data_path, connect, enable_gravity, input, disconnect, create_sphere, set_point, Point, create_cylinder, enable_real_time, dump_world, load_model, wait_for_interrupt, set_camera, stable_z, set_color, get_lower_upper, wait_for_duration, simulate_for_duration, euler_from_quat, set_pose, set_joint_positions, get_joint_positions

real_init = True

 
class World():
    def __init__(self, visualize=False, real_init=True, beads=True):
        #make base world 
        self.base_world = CupWorld(visualize=visualize, real_init = real_init, beads=beads)
        self.visualize=visualize
        self.real_init = real_init
        self.setup()

    def stirrer_close(self):
        distance = self.base_world.distance_from_cup(self.armID, 10)
        far = k*0.3
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
        force = action[4]
        #got rid of delta control since it was a horrible idea from the start
        wrist_joint_num = 8
        self.stir_circle(time_step = 0.01, size_step = rot)
        self.set_pos(self.armID, wrist_joint_num, theta_diff, force=force)
        self.set_pos(self.armID, 10, curl, force=force) #10 = curl ID
	simulate_for_duration(period)
        self.set_pos(self.armID, wrist_joint_num, -theta_diff, force=force)
	simulate_for_duration(period)
        
    
    #adds something of size_step to the current angle
    def stir_circle(self, size_step=0, time_step = 0):
        #let's say it takes 1 second t
        #and find out where x,y is is 
        current_pos = p.getJointState(self.armID, 6)[0]
        desired_pos = current_pos + size_step
        if desired_pos >= 3:
            desired_pos = -3 #reset
	p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=6,controlMode=p.POSITION_CONTROL,force=900,positionGain=0.3,velocityGain=1, targetPosition=desired_pos)
            

    #this function is now a complete lie and has not only the stirrer state but
    #also the vector from the cup
    def stirrer_state(self):
        #returns position and velocity of stirrer flattened
        linkPos = p.getJointState(self.armID, 8)[0]
         
        theta_diff_pos, _, _,_ = p.getJointState(self.armID,8)
        rot_joint_pos, _, _ ,_= p.getJointState(self.armID,6)
        curl_joint_pos, _, curl_joint_forces,_ = p.getJointState(self.armID,10)

        #r, theta, z in pos 
        cupPos=  np.array(p.getBasePositionAndOrientation(self.base_world.cupID)[0])
        stirrerPos=  np.array(p.getLinkState(self.armID, 10)[0])
        vector_from_cup = cupPos-stirrerPos
        r_theta_z_pos =  cart2pol(vector_from_cup[0], vector_from_cup[1])+ (vector_from_cup[2],)
        #forces in cup frame
        r_theta_z_force =  cart2pol(curl_joint_forces[0], curl_joint_forces[1])+ (curl_joint_forces[2],)
        #return np.array([theta_diff_pos, rot_joint_pos, curl_joint_pos, r_theta_z_pos[0], r_theta_z_pos[1], r_theta_z_pos[2], r_theta_z_force[0], r_theta_z_force[1], r_theta_z_force[2]])
        return np.array([theta_diff_pos, rot_joint_pos, curl_joint_pos, r_theta_z_pos[0], r_theta_z_pos[1], r_theta_z_pos[2]])
       

    def reset(self):
        self.base_world.reset()
        self.__init__(visualize=self.visualize, real_init=False)



    def set_grip(self, pr2ID):
	self.set_pos(pr2ID, 8,0)
        nice_joint_states = [0.0, 0.006508613619013667, 0.0, 0.19977108955651196]
        for i in range(7,11):
	    self.set_pos(pr2ID, i, nice_joint_states[i-7])
        if not self.base_world.is_real_time:
	    simulate_for_duration(1.0)

    def delta_control(self, dx, dy, dz, posGain=0.3, velGain=1):
        endEIndex = 7
        jointPos = p.getLinkState(self.armID, endEIndex)[0]
        jointOrn = p.getLinkState(self.armID, endEIndex)[1]
        desiredPos = np.array((jointPos[0]+dx, jointPos[1]+dy, jointPos[2]+dz))
        self.move_arm_to_point(desiredPos, orn=jointOrn, posGain=posGain, velGain=velGain)        
    
    def move_arm_to_point(self, pos, orn = None, damper=0.1, posGain = 0.3, velGain=1, threshold=0.02, timeout=200):
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

    def move_arm_closer_to_point(self, pos, orn = None, damper=0.1, posGain = 0.3, velGain=1, teleport=True):
        endEIndex = 7
        ikSolver = 0
        if orn is None:
	    orn = p.getQuaternionFromEuler([0,-np.pi,0])
        numJoints = p.getNumJoints(self.armID)
        jd=[damper]*numJoints
        jd =[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        rp = [0, 0, 0, math.pi / 2, 0, -math.pi * 0.66, 0]
	jointPoses = p.calculateInverseKinematics(self.armID,endEIndex,pos,orn,solver=ikSolver, maxNumIterations=500, residualThreshold=0.001, restPoses=rp)
        if teleport:
	    joints = list(range(len(jointPoses)))
	    set_joint_positions(self.armID, joints, jointPoses)
        else:
	    for i in range (numJoints-3):
		p.setJointMotorControl2(bodyIndex=self.armID,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],force=500,positionGain=posGain,velocityGain=velGain, targetVelocity=0)
        if not self.is_real_time:
	    simulate_for_duration(0.1)
    
    def set_hand(self):
	good_start = [0.7355926784963512, 0.5954400794648647, 0.26945666316947803, -1.5049405693666376, -0.1823886954058032, 1.0795282294781492, 0.30963124203610953, 0.0, 0.006488946176044567, 0.0, 0.19966462229846724]
	set_joint_positions(self.armID, list(range(p.getNumJoints(self.armID))), good_start)

    def place_stirrer_in_pr2_hand(self, teleport=True):
        cup_r = k*-0.07 
        height_above = k*0.6
        start_x = 0
        start_y = 0
        above_loc = Point(start_x,start_y,height_above)
        if teleport:
            self.set_hand()
        else:
            self.move_arm_to_point(above_loc)
            p.addUserDebugLine(above_loc, [0,0,0])

        self.base_world.toggle_real_time()
	self.set_grip(self.armID)
        if not teleport:
	    num_steps = 8
	    desired_end_height=0.31
	    dz_in_loc = height_above - desired_end_height
	    step_size = dz_in_loc/num_steps
	    for z in range(1,num_steps+1): #fake one indexing
		in_loc = Point(start_x,start_y,height_above - z*step_size)
		self.move_arm_to_point(in_loc)
	    #stirring motion
	    #in_loc = Point(cup_r,cup_r,k*0.35)
	    self.move_arm_to_point(in_loc)
	    p.addUserDebugLine(in_loc, [0,0,0])



    def set_pos(self,objID, jointIndex, pos, force=500):
	
	p.setJointMotorControl2(bodyIndex=objID, 
	jointIndex=jointIndex, 
	controlMode=p.POSITION_CONTROL,
	targetPosition=pos,
	force = force)


    def setup(self, beads=True):
        NEW = self.real_init #unfortunately
        if NEW:
	    best_arm_pos = [k*-0.65,0,0]
            if self.visualize:
	        self.armID = p.loadSDF("urdf/kuka_iiwa/kuka_with_gripper.sdf", globalScaling = k)[0]
            else: 
	        self.armID = p.loadSDF("urdf/kuka_iiwa/invisible_kuka_with_gripper.sdf", globalScaling = k)[0]
                gripper_indices = [8,9,10]
                blacken(self.armID, end_index=8)
                greenen(self.armID, gripper_indices)
            p.enableJointForceTorqueSensor(self.armID, 10, 1)
	    set_pose(self.armID,(best_arm_pos,  p.getQuaternionFromEuler([0,0,-np.pi/4])))
            self.place_stirrer_in_pr2_hand() 
            self.base_world.zoom_in_on(self.armID, 2)
            self.real_init = False
        else:
            try:
                p.restoreState(self.bullet_id)
            except:
                self.real_init = True
                p.resetSimulation()
                self.setup()



	#p.resetDebugVisualizerCamera(0.5, yaw, roll, objPos)
    def simplify_viz(self):
        features_to_disable = [p.COV_ENABLE_WIREFRAME, p.COV_ENABLE_SHADOWS, p.COV_ENABLE_VR_PICKING, p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]
        for feature in features_to_disable:
            p.configureDebugVisualizer(feature, 0) 


if __name__ == "__main__":
    world = World(visualize=True)
    world.reset()
	

