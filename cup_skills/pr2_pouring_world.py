from cup_world import *
import pybullet as p
import numpy as np
from pr2_utils import TOP_HOLDING_LEFT_ARM, \
    SIDE_HOLDING_LEFT_ARM, PR2_GROUPS, open_arm, get_disabled_collisions, get_gripper_link, \
    load_srdf_collisions, load_dae_collisions, REST_LEFT_ARM, rightarm_from_leftarm, get_arm_joints



import reward
from utils import set_point
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
        self.base_world = CupWorld(visualize=visualize, beads=False, new_bead_mass=new_bead_mass, table=True)
        self.cup_to_dims = {"cup_1.urdf":(0.5,0.5), "cup_2.urdf":(0.5, 0.2), "cup_3.urdf":(0.7, 0.3), "cup_4.urdf":(1.1,0.3), "cup_5.urdf":(1.1,0.2), "cup_6.urdf":(0.6, 0.7)}#cup name to diameter and height
        self.torso_joint =  15
        self.torso_height = k*0.3
        self.ee_index = 54#  60
        if real_init:
            self.setup()
        else:
            p.restoreState(self.bullet_id)

    def setup(self):
        #create constraint and a second cup
        self.cupStartPos = (0,-0.4,0.708)
        self.cupStartOrientation = p.getQuaternionFromEuler([0.5,0,0]) 
        #pick random cup

        self.cup_name = np.random.choice(self.cup_to_dims.keys())
        self.cup_name = "cup_1.urdf"
        cup_file = "urdf/cup/"+self.cup_name
        self.target_cup = p.loadURDF(cup_file,self.cupStartPos, self.cupStartOrientation, globalScaling=k*1.2)

        pr2_start_orientation = p.getQuaternionFromEuler([0,0,0])
        pr2_start_pose = [-.80*k,0,0]

        self.pr2 = p.loadURDF("urdf/pr2_description/pr2.urdf", pr2_start_pose, pr2_start_orientation, useFixedBase=True, globalScaling = k )
        self.movable_joints = get_movable_joints(self.pr2)

        self.bullet_id = p.saveState()
        
    def observe_cup(self):
        return np.array(self.cup_to_dims[self.cup_name])

    def move_ee_to_point(self, pos, orn = None, damper=0.01, posGain=0.3, velGain =1, threshold = 0.03, timeout = 10):
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
        diff = np.array(actualPos)-pos
        num_attempts = 0
        while(np.linalg.norm(diff) >= threshold and num_attempts <= timeout):
            #self.plot_point(pos)
            self._move_arm_closer_to_point(pos, orn=orn, damper=damper, posGain=posGain, velGain=velGain)
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
    def _move_arm_closer_to_point(self, pos, orn = None, damper=0.1, posGain = 0.03, velGain=1):
        ikSolver = 0
        if orn is None:
            orn = p.getQuaternionFromEuler([0,-np.pi,0])

        current_conf = get_joint_positions(self.pr2, self.movable_joints)
        left_joints = [joint_from_name(self.pr2, name) for name in ARM_JOINT_NAMES['left']]

        arm_joints = get_arm_joints(self.pr2, 'right')
        left_root = 42
        right_root = 65
        height_stable  = self.torso_height
        result = sub_inverse_kinematics(self.pr2, left_root, self.ee_index, (pos, orn))
        if result is not None:
            try:
                confs, joints = result
            except:
                pdb.set_trace()
            gripper_joints = (57, 59, 58, 60)
            moving_joints = []
            moving_confs = []
            for i in range(len(joints)):
                if joints[i] not in gripper_joints:
                    moving_confs.append(confs[i])
                    moving_joints.append(joints[i])
            
            heavy_joints = [self.torso_joint]
            heavy_confs = [height_stable]
            #remove finger joints
            jc = joint_controller(self.pr2, moving_joints, moving_confs, heavy_joints = heavy_joints, heavy_confs=heavy_confs)
            for i in range(6):
		try:
		    jc.next()
		    simulate_for_duration(0.8)
		except StopIteration:
                    break;


    def reset(self, real_init=False, new_bead_mass=None):
  
        self.base_world.reset(new_bead_mass=new_bead_mass)
        self.setup()      
        #set_pose(self.target_cup, (self.cupStartPos, self.cupStartOrientation))
                


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
        simulate_for_duration(0.2)
        start_pos = (-0.12, -0.13, 0.70)

        start_orn = p.getQuaternionFromEuler((0,0,3.14/2.0)) 
        self.move_ee_to_point(start_pos, start_orn, timeout=10, threshold=0.05)
        pdb.set_trace()
       
    
    #reactive pouring controller 
    #goes to the closest lip of the cup and decreases the pitch until the beads fall out into the right place
    def pour(self, offset=0.02, velocity=0.9, force=1500, total_diff = 4*np.pi/5.0):
        #step_size and dt come from the angular velocity it takes to make a change of 3pi/4, can set later

        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        start_point = (pourer_pos[0], pourer_pos[1]+offset, pourer_pos[2])
        self.move_cup(start_point,  duration=2,force=force) 
        start_pos, start_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        #then start decreasing the roll, pitch or yaw(whatever seems appropriate)
        current_orn = list(p.getEulerFromQuaternion(start_orn))
        numsteps = 25.0
        step_size = total_diff/numsteps; #hard to set otherwise
        dt = step_size/velocity
        for i in range(int(numsteps)):
            current_orn[0] += step_size
            self.move_cup(start_pos, current_orn, duration=dt, force=force)
        simulate_for_duration(1.2) #rest time, can be tuned

    def pourer_state(self):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        target_pos, target_orn = p.getBasePositionAndOrientation(self.target_cup)
        cup_rel_pos = np.array(pourer_pos) - np.array(target_pos)
        return np.hstack([cup_rel_pos, pourer_orn])

    def world_state(self):
        return self.base_world.world_state() 
    
    def lift_cup(self, desired_height=0.7, force=1600):
        pourer_pos, pourer_orn = p.getBasePositionAndOrientation(self.base_world.cupID)
        other_cup_pos, _=  p.getBasePositionAndOrientation(self.target_cup)
        gripper_pose, gripper_orn =  p.getLinkState(self.pr2, self.ee_index)[0:2]
        #desired_height = other_cup_pos[2]+desired_height
        new_pose = list(gripper_pose)
        new_pose[2] += desired_height
        self.move_ee_to_point(new_pose, gripper_orn)

    def open_gripper(self, open_num=0.5):
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=59,controlMode=p.POSITION_CONTROL,force=100,positionGain=0.3,velocityGain=1, targetPosition=open_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=57,controlMode=p.POSITION_CONTROL,force=100,positionGain=0.3,velocityGain=1, targetPosition=open_num)
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
            pdb.set_trace()
          
    def close_gripper(self, close_num=0.2, finger_close_num=0.5, force=200):
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=59,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=57,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=close_num)
        
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=58,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=60,controlMode=p.POSITION_CONTROL,force=force,positionGain=0.3,velocityGain=1, targetPosition=finger_close_num)
        simulate_for_duration(0.5)

    def grasp_cup(self):
        #grasp the cup in base world
        #open gripper
       
        self.put_arms_in_useful_configuration(self.pr2)

        self.open_gripper(0.55)
        #move gripper to cup
        actualPos =  p.getLinkState(self.pr2, self.ee_index)[0]
        set_point(self.base_world.cupID, (actualPos[0]-0.01, actualPos[1]-0.035, actualPos[2]-0.02))
        #self.move_gripper_to_cup(self.base_world.cupID)
        self.close_gripper(0.35)

    #turn the wrist link
    def turn_cup(self, amount, duration):
        current_pos = p.getJointState(self.pr2, 50)[0]        
        p.setJointMotorControl2(bodyIndex=self.pr2,jointIndex=50,controlMode=p.VELOCITY_CONTROL,force=400,positionGain=0.3,velocityGain=1, targetVelocity=amount)
        simulate_for_duration(duration)

        
        
    def pour_pr2(self):
        self.grasp_cup()
        self.lift_cup(0.05)
        self.turn_cup(0.3)




    
       

if __name__ == "__main__":
    pw = PouringWorld(visualize=True, real_init = True)
    pw.pour_pr2()
    #pw.lift_cup(desired_height=0.62)
    #pw.pour(offset=-0.2, velocity=1.4, force=1500, total_diff = 4*np.pi/5.0)
    #pw.pour(offset=-0.15, velocity=1.4, force=1500, total_diff = 2.51)

    #pw.pour(offset=0.02, velocity=0.02, force=1500, total_diff = np.pi/5.0)

    pw.base_world.ratio_beads_in(cup=pw.target_cup)
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.74658884e-01, -3.99184460e-01, -1.97149862e-01, -1.17733128e-01,-1.99983150e+03])
    #actions = np.array([-6.25397044e-01, -1.43723112e+00, -1.14753149e+00, -1.23676025e+00,1.99868273e+03])
    #actions = np.array([-1.16826367e-01,  6.83036833e-01,  4.13037813e-01,  9.31779934e-02,1.99998315e+03])
    #pw.parameterized_pour(offset=actions[0], desired_height=actions[1], step_size=actions[2], dt=actions[3], force=actions[4])
    #pw.parameterized_pour(offset=-0.08, velocity=1.2, force=1500, desired_height=0.6)

    print(pw.base_world.ratio_beads_in(cup=pw.target_cup), "beads in")

    pw.reset(new_bead_mass = 1.1)
    
    
