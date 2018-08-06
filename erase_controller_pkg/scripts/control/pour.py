#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point
from gdm_arm_controller.uber_controller import UberController
from grip import Gripper

CONSISTENT_STATE = True 

rospy.init_node("make_pour")
numsteps = 200

uc = UberController()
start_joint_angles = [0.09482477817860147, 0.08879762533862981, -2.1107870973868064, -0.3122436312925274, 5.1535253192926955, -1.2508791706701996, 3.5850952111876673]
away_joint_angles =[-0.05432422644661594, 0.030680913165869468, -2.1242569028018767, -0.30717665639410396, 5.189853289897649, -1.3181874700642715, 3.253426713268458]

class Robot:
    def __init__(self):
        self.arm = 'r'
        self.pourer_pos = (0,0,0)

    def go_to_start(self):
	uc.start_joint(self.arm)
	uc.command_joint_pose(self.arm,start_joint_angles, time=3, blocking=True)
	rospy.sleep(1)

    def go_to_away(self):
	uc.start_joint(self.arm)
	uc.command_joint_pose(self.arm,away_joint_angles, time=3, blocking=True)
	rospy.sleep(2)

    ''' currently really only actually samples a pose, and uses the original quaternion''' 
    def get_grasp(self, cup_pos):
        #grasp it from the right side, at a sort of 90 degree angle flat
        gripper_pos, gripper_quat = uc.return_cartesian_pose(self.arm, 'base_link')
        grasp_height = 0.03
        grasp_depth = 0.08
        pos = self.point_past_gripper(grasp_height, grasp_depth, cup_pos, gripper_pos)
        return pos, gripper_quat

    def point_past_gripper(self, grasp_height, grasp_depth, pourer_pos, gripper_pos):
        grasp_height_world =  pourer_pos[2]+ grasp_height
        dx = gripper_pos[1]-pourer_pos[1]
        dy = gripper_pos[0]-pourer_pos[0]
        theta = np.arctan2(dy, dx)
        far_point = (pourer_pos[0]-grasp_depth*np.sin(theta), pourer_pos[1]-grasp_depth*np.cos(theta), grasp_height_world)
        return far_point

    def update_pourer_pos(self, data):
        self.pourer_pos = None #TODO update based on data
 
    def grasp_cup(self):
        #move to good grasping point 
        grasp_time = 3.0
        frame = 'base_link'
        if self.pourer_pos is not None:
            pos,quat = self.get_grasp(pourer_pos)
            uc.cmd_ik_interpolated(self.arm, (pos, quat), grasp_time, frame, blocking = True, use_cart=True, num_steps = 30)
        
        #close gripper
        gripper = Gripper()
        gripper.grip()


    def shift_cup(self, dx=0, dy=0, dz = 0):
        shift_time = 1.0
        gripper_pos, gripper_quat = uc.return_cartesian_pose(self.arm, 'base_link')
        new_pos = (gripper_pos[0]+dx, gripper_pos[1]+dy, gripper_pos[2]+dz)
        uc.cmd_ik_interpolated(self.arm, (new_pos, gripper_quat), shift_time, frame, blocking = True, use_cart=True, num_steps = 30)
    
        


    def pour_cup(self):
        #get joint state of right turning joint
        #add to joint state

if __name__ == "__main__":
    robot = Robot()
    robot.go_to_start()
    
    for i in range(numsteps):
	#go to start
	robot.grasp_cup()
	robot.shift_cup(dz = 0.08)
	robot.pour_cup()

        
    


