from cup_world import *
import pybullet as p
import numpy as np
from pr2_utils import TOP_HOLDING_LEFT_ARM, \
    SIDE_HOLDING_LEFT_ARM, PR2_GROUPS, open_arm, get_disabled_collisions, get_gripper_link, \
    load_srdf_collisions, load_dae_collisions, REST_LEFT_ARM, rightarm_from_leftarm, get_arm_joints



import reward
from utils import set_point, create_marker,  joint_from_name
from utils import set_base_values, set_point, joint_from_name, set_joint_position, simulate_for_duration, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, input, \
    joint_controller, joint_controller_hold, dump_world, get_link_name, wait_for_interrupt, \
    get_links, get_joint_parent_frame, euler_from_quat, get_joint_inertial_pose, get_joint_info, \
    get_link_pose, VisualShapeData, get_visual_data, get_link_parent, link_from_name, sub_inverse_kinematics,\
    get_link_ancestors, get_link_children, get_link_descendants, get_joint_positions, get_movable_joints, inverse_kinematics 

ARM_JOINT_NAMES = {
   'left': ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                 'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint'],
    'right': ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                  'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint'],
}

TORSO_JOINT_NAME = 'torso_lift_joint'

k = 1

class PouringWorld():
    def __init__(self, visualize=True, real_init=False, new_bead_mass=None):
        self.base_world = CupWorld(visualize=visualize, beads=False, new_bead_mass=new_bead_mass, table=True, for_pr2=True)
        self.cup_constraint = None
        self.cup_to_dims = {"cup_1.urdf":(0.5,0.5), "cup_2.urdf":(0.5, 0.2), "cup_3.urdf":(0.7, 0.3), "cup_4.urdf":(1.1,0.3), "cup_5.urdf":(1.1,0.2), "cup_6.urdf":(0.6, 0.7)}#cup name to diameter and height
        self.torso_joint =  15
        self.constraint_force = 200
        self.torso_height = k*0.3
        self.ee_index = 54#  60
        self.target_cup = None
        if real_init:
            self.setup()
        else:
            p.restoreState(self.bullet_id)

    def setup(self):
        #create constraint and a second cup

        pr2_start_orientation = p.getQuaternionFromEuler([0,0,0])
        pr2_start_pose = [-.80*k,0,0]

        self.pr2 = p.loadURDF("urdf/pr2_description/pr2.urdf", pr2_start_pose, pr2_start_orientation, useFixedBase=True, globalScaling = k )
        self.movable_joints = get_movable_joints(self.pr2)
        self.gripper_joints = (57, 59, 58, 60)
        for joint in self.gripper_joints:
            p.enableJointForceTorqueSensor(self.pr2, joint, enableSensor=1)

        self.bullet_id = p.saveState()
        
    def observe_cup(self):
        return [] #np.array(self.cup_to_dims[self.cup_name])

    def gripper_forces(self):
        #magnitude of gripper forces
        all_forces = []
        for gripper in self.gripper_joints:
            forces = p.getJointState(self.pr2, gripper)[2]
            assert(len(forces)== 6)
            mag = np.linalg.norm(forces)
            all_forces.append(mag)
            #can average force of all the grippers
        return np.array((np.mean(all_forces),))

    ''' returns the change in force to apply to shift the force in the right direction'''
    def pid_controller(self):
        kp = 0.08
        des = 4000
        print(self.gripper_forces(), "current gripper forces") 
        error = des - self.gripper_forces()
        
        print("error", error)
        return kp*error
        
        

    def move_ee_to_point(self, pos, orn = None, damper=0.01, posGain=0.3, velGain =1, threshold = 0.03, timeout = 5, force=300, teleport=False):
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
        diff = np.array(actualPos)-pos
        num_attempts = 0
        create_marker(0.003, point=pos, color=(1,1,0,0.5))
        while(np.linalg.norm(diff) >= threshold and num_attempts <= timeout):
            self._move_arm_closer_to_point(pos, orn=orn, damper=damper, posGain=posGain, velGain=velGain, teleport=teleport, force=force)
            actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
            diff = np.array(actualPos)-pos
            num_attempts += 1

        if num_attempts > timeout:
            return False
        else:
            return True
    """ 
    helper function, see move_ee_to_point
    """
    def _move_arm_closer_to_point(self, pos, orn = None, damper=0.1, posGain = 0.03, velGain=1, force=500, teleport=False):
        ikSolver = 0
        if orn is None:
            orn = p.getQuaternionFromEuler([0,-np.pi,0])

        current_conf = get_joint_positions(self.pr2, self.movable_joints)
        left_joints = [joint_from_name(self.pr2, name) for name in ARM_JOINT_NAMES['left']]

        left_root = 42
        right_root = 65
        height_stable  = self.torso_height
        #create_marker(0.01, point=pos, color=(1,1,0,0.8))
        conf, joints = sub_inverse_kinematics(self.pr2, left_root, self.ee_index, (pos, orn))
        arm_joint_names = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                        'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
        arm_joints = [joint_from_name(self.pr2, name) for name in arm_joint_names]
         
        gripper_joints = (57, 59, 58, 60)

        moving_joints = []
        moving_confs = []
        for i in range(len(joints)):
            if joints[i] not in gripper_joints:
                moving_confs.append(conf[i])
                moving_joints.append(joints[i])
        
        heavy_joints =[]# [self.torso_joint]
        heavy_confs = []#[height_stable]
        if not teleport:
            if self.cup_constraint is not None: #needs to be small so cup constraint can be update
                num_steps = 10
            else:
                num_steps = 1
            for _ in range(num_steps):
                for i in range(len(moving_joints)):
                    p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=moving_joints[i],controlMode=p.POSITION_CONTROL,targetPosition=moving_confs[i],force=force, targetVelocity=0)
                    simulate_for_duration(0.35/num_steps)
                    if self.cup_constraint is not None:
                        left_tip_pos = p.getLinkState(self.pr2, 58)[4]
                        right_tip_pos = p.getLinkState(self.pr2, 60)[4]
                        new_loc = np.average(np.vstack([left_tip_pos, right_tip_pos]),axis=0) #the average of the left and right grippers....
                        new_loc[2] += 0.05 #hold the cup a little higher
                        new_orn = orn
                        p.changeConstraint(self.cup_constraint, new_loc, new_orn, maxForce = self.constraint_force)
        else:
            set_joint_positions(self.pr2, moving_joints, moving_confs)
            #set_joint_positions(self.pr2, heavy_joints, heavy_confs)


    def reset(self, real_init=False, new_bead_mass=None):
  
        self.base_world.reset(new_bead_mass=new_bead_mass)
        self.setup()      
                


    """Moves the robot arms to a reasonable configuration for the task"""
    def put_arms_in_useful_configuration(self, pr2):
        arm_start = SIDE_HOLDING_LEFT_ARM
        arm_goal = TOP_HOLDING_LEFT_ARM


        left_joints = [joint_from_name(pr2, name) for name in ARM_JOINT_NAMES['left']]
        right_joints = [joint_from_name(pr2, name) for name in ARM_JOINT_NAMES['right']]
        set_joint_positions(pr2, left_joints, REST_LEFT_ARM)
        starting_joint_angles =(-0.17851986801269087, -0.04107450376601436, -2.195779753219258, -0.5946310817279756, 5.122832729301108, -1.2228819639423414, 3.3251923322410066)


        set_joint_positions(pr2, right_joints, starting_joint_angles)
        set_joint_position(pr2, self.torso_joint, self.torso_height)
        force = 100
        open_num = 0.5
        finger_close_num=0.5
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=59,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=open_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=57,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=open_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=58,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=60,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)

        simulate_for_duration(0.2)
        start_pos = (-0.12, -0.13, 0.68)

        start_orn = p.getQuaternionFromEuler((0,0,3.14/2.0)) 
        self.move_ee_to_point(start_pos, start_orn, timeout=5, threshold=0.05)
       
    
    def pourer_state(self):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        target_pos, target_orn = p.getBasePositionAndOrientation(self.target_cup)
        cup_rel_pos = np.array(pourer_pos) - np.array(target_pos)
        return np.hstack([cup_rel_pos, pourer_orn])

    def world_state(self):
        return self.base_world.world_state() 
    
    def shift_cup(self, desired_height=0.7, side_offset=0, forward_offset=0, force=1600):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        other_cup_pos, _=  p.getBasePositionAndOrientation(self.target_cup)
        gripper_pose, gripper_orn =  p.getLinkState(self.pr2, self.ee_index)[0:2]
        #desired_height = other_cup_pos[2]+desired_height
        new_pose = list(pourer_pos)
        new_pose[2] += desired_height
        new_pose[1] += side_offset
        #always up after
        self.move_ee_to_point(new_pose, gripper_orn, force=force, teleport=False)
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        gripper_pose, gripper_orn =  p.getLinkState(self.pr2, self.ee_index)[0:2]
        new_pose = list(pourer_pos)
        new_pose[0] += forward_offset
        new_pose[2] =gripper_pose[2]
        self.move_ee_to_point(new_pose, gripper_orn, force=force, teleport=False)

    def open_gripper(self, open_num=0.5, finger_close_num=0.5, force=100):
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=59,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=open_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=57,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=open_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=58,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=60,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        simulate_for_duration(0.5)

    def move_gripper_to_cup(self, cup):
        cup_pose = p.getBasePositionAndOrientation(cup)[0]
        start_orn = p.getQuaternionFromEuler((0,0,0.5)) 
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
        diff = np.subtract(cup_pose, actualPos)
        total_dist = np.linalg.norm(diff)
        dy = cup_pose[1] - actualPos[1] 
        dx = cup_pose[0] - actualPos[0] 
        numsteps = 4
        dist = total_dist/numsteps
        theta = np.arctan2(dy, dx)
        for i in range(numsteps):
            new_pose = (actualPos[0] + dist*np.cos(theta), actualPos[1]+dist*np.sin(theta), actualPos[2]) 
            self.move_ee_to_point(new_pose, start_orn, threshold=0.005)
            actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
          
    def close_gripper(self, close_num=0.2, finger_close_num=0.5, force=200):
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=59,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=57,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=close_num)
        
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=58,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=60,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        simulate_for_duration(0.5)

    #grasp_depth = how far into the cup to go based on the gripper location, positive is past the cup
#grasp_height = how high above table to grasp
    def grasp_cup(self, close_num=0.35, close_force = 300 , teleport=False, grasp_height=0.1, grasp_depth=0.05, lift_force=1000, finger_close_num=0.5):
        #grasp the cup in base world
        #open gripper
       
        self.put_arms_in_useful_configuration(self.pr2)
        self.open_gripper(open_num=0.8)
        #move gripper to cup
        pourer_pos = p.getBasePositionAndOrientation(self.base_world.cupID)[0]
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
        self.base_world.drop_beads_in_cup()
        if teleport:
            set_point(self.base_world.cupID, (actualPos[0]-0.01, actualPos[1]-0.02, actualPos[2]-0.04))
        else:
            grasp_point = self.point_past_gripper(grasp_height, grasp_depth)
            start_orn = p.getQuaternionFromEuler((0,0,3.14/2.0)) 
            self.move_ee_to_point(grasp_point, start_orn, timeout=2, threshold=0.01, force=lift_force)
            
            
        simulate_for_duration(0.2)
        self.attach(self.base_world.cupID)
        self.close_gripper(close_num=close_num, force=close_force, finger_close_num=finger_close_num)
        print(self.gripper_forces())

    def attach(self, body):
        pos, orn = p.getBasePositionAndOrientation(body) 
        self.cup_constraint = p.createConstraint(body, -1, -1, -1, p.JOINT_FIXED,pos, orn, [0,0,1])
        p.changeConstraint(self.cup_constraint, pos, orn, maxForce = self.constraint_force)
        simulate_for_duration(0.1)
 
         

    def point_past_gripper(self, grasp_height, grasp_depth):
        pourer_pos = p.getBasePositionAndOrientation(self.base_world.cupID)[0]
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
	grasp_height_world =  pourer_pos[2]+ grasp_height
	#direction of 
	dx = actualPos[1]-pourer_pos[1]
	dy = actualPos[0]-pourer_pos[0]
	theta = np.arctan2(dy, dx) 
	far_point = (pourer_pos[0]-grasp_depth*np.sin(theta), pourer_pos[1]-grasp_depth*np.cos(theta), grasp_height_world)
	create_marker(0.01, color=(1,0,1,0.9), point=far_point)
        return far_point
        

    #turn the wrist link
    def turn_cup(self, amount, duration):
        current_pos = p.getJointState(self.pr2, 50)[0]        
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=50,controlMode=p.VELOCITY_CONTROL,force=800,positionGain=0.3,velocityGain=1, targetVelocity=-1*amount)
        simulate_for_duration(duration)

    '''Generates a trajectory of orientations to turn cup
    assumes the cup is at the correct yaw for the pour
    '''
    def turn_cup_traj(self, amount):
        num_steps = 10
        euler_orn = p.getLinkState(self.pr2, 58)[5] #orientation in world coordinates
        traj = []
        current_orn = euler_orn[:]
        for _ in range(num_steps):
            current_orn[0] += amount/num_steps
            traj.append(current_orn[:])
        return traj

    def turn_cup_general(self, amount, velocity):
        traj = self.turn_cup_traj(amount)
        whole_duration = amount / velocity
        step_duration = whole_duration/len(traj)
        gripper_pos = p.getLinkState(self.pr2, 58)[4] #world coordinates pose
        for i in range(len(traj)):
            self.move_ee_to_point(gripper_pos, orn = traj[i], damper=0.01, posGain=0.3, velGain =1, threshold = 0.03, timeout = 3, force=300, teleport=False)
            

    #assumes is already grasping
    def test_grasp(self, close_force, close_num):
        diff = 0.05
        empty_hand = 50
        for i in range(65):
            cup_pos = p.getBasePositionAndOrientation(self.base_world.cupID)[0]
            gripper_pos =  p.getLinkState(self.pr2, self.ee_index)[0] 
            #self.close_gripper(close_num=close_num, force=close_force)
            """
            pid_diff = self.pid_controller()
            close_force += pid_diff
            print("close force", close_force)
            if self.gripper_forces() < empty_hand:
                return i
            print("Gripper force",self.gripper_forces())
            """
            simulate_for_duration(0.08)
            if i == 10 or i == 20:
                self.shift_cup(desired_height=0.1, force=300)
        print("Woo hoo! got to the end!")
        return i 
     


    def pour_pr2(self, close_num=0.35, close_force=300, lift_force=1600, side_offset = 0, forward_offset=0, height = 0.08, vel=5, grasp_height=0.1, grasp_depth=0.04):
        #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pour_pr2_demo.mp4")
        self.grasp_cup(close_num = close_num, close_force=close_force, grasp_height=grasp_height, grasp_depth=grasp_depth, lift_force=lift_force)
        #self.spawn_cup()
        self.shift_cup(desired_height=height,side_offset=side_offset, forward_offset=forward_offset,force=lift_force)
        self.turn_cup(vel, 2)
        simulate_for_duration(3.0)

    def spawn_cup(self):
        #self.cupStartPos = (0.05,-0.10,0.63)
        self.cupStartPos = (-0.3,0,0.63)
        self.cupStartOrientation = p.getQuaternionFromEuler([0,0,0]) 
        #pick random cup
        self.cup_name = np.random.choice(self.cup_to_dims.keys())
        self.cup_name = "cup_4.urdf"
        cup_file = "urdf/cup/"+self.cup_name
        self.target_cup = p.loadURDF(cup_file,self.cupStartPos, self.cupStartOrientation, globalScaling=k*1.5)
        #self.target_cup = p.loadURDF("/home/lagrassa/git/pcakages/bullet3/data/dinnerware/cup/cup_small.urdf",self.cupStartPos, self.cupStartOrientation, globalScaling=k*1.5)
        #p.changeDynamics(self.target_cup, -1, mass=40000)


if __name__ == "__main__":
    pw = PouringWorld(visualize=False, real_init = True)
    #pw.pour_pr2(close_num=0.333, close_force=663, lift_force=328, height=0.09, forward_offset=-0.15, vel=3, grasp_height=0.04, grasp_depth=0.07)
    data = [3.30000000e-01, 6.97000000e+02, 3.26000000e+02, 2.90000000e-02,7.66003498e-01, 0.00000000e+00]
    pw.grasp_cup(close_num=0.45, close_force=20, lift_force=228, grasp_height=0.03, grasp_depth=0.02, finger_close_num=0.5)
    #pw.grasp_cup(close_num=data[0], close_force=data[1], lift_force=data[2], grasp_height=data[3], grasp_depth=data[4], finger_close_num=data[5])
    pw.spawn_cup()
    print("Test result", pw.test_grasp(700, 0.35))
    
    
