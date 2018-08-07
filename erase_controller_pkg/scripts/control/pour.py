#!/usr/bin/env python
from __future__ import division
import rospy
import pdb
import numpy as np
from geometry_msgs.msg import Point
from gdm_arm_controller.uber_controller import UberController
from grip import Gripper

CONSISTENT_STATE = True 

rospy.init_node("make_pour")
numsteps = 200

uc = UberController()
start_joint_angles = [-0.8831416801644028, 0.3696527700454193, -1.5865871699836482, -1.5688534061015482, 5.913809583083913, -0.9149799482346365, 39.09787903807846]
away_joint_angles =[-0.05432422644661594, 0.030680913165869468, -2.1242569028018767, -0.30717665639410396, 5.189853289897649, -1.3181874700642715, 3.253426713268458]

class Robot:
    def __init__(self):
        print("robot init")
        self.arm = 'r'
        self.pourer_pos = (0.5, -0.01, 0.7595)
        self.go_to_start()

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
        #good values - ('gripper_pos', [0.4885209624088133, -0.1602965161312774, 0.7629772575114466])
        # ('gripper_quat', [-0.07005178434662307, -0.0861961500917268, 0.7512909881160719, 0.6505573167637076])

        grasp_height = -0.05
        grasp_depth = 0.01
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
        self.pourer_pos = self.pourer_pos #TODO update based on data
 
    def grasp_cup(self):
        #move to good grasping point 
        grasp_time = 3.0
        frame = 'base_link'
        gripper = Gripper()
        rospy.sleep(0.5)
        gripper.grip(amount=0.09, times = 40)
     
        if self.pourer_pos is not None:
            pos,quat = self.get_grasp(self.pourer_pos)
            uc.cmd_ik_interpolated(self.arm, (pos, quat), grasp_time, frame, blocking = True, use_cart=False, num_steps = 5)
        
        #close gripper
        gripper.grip(amount=0.045)


    def shift_cup(self, dx=0, dy=0, dz = 0):
        shift_time = 1.0
        gripper_pos, gripper_quat = uc.return_cartesian_pose(self.arm, 'base_link')
        new_pos = (gripper_pos[0]+dx, gripper_pos[1]+dy, gripper_pos[2]+dz)
        uc.cmd_ik_interpolated(self.arm, (new_pos, gripper_quat), shift_time, 'base_link', blocking = True, use_cart=False, num_steps = 30)
    
        


    def pour_cup(self):
        #get joint state of right turning joint, add to joint state, do a few times
        current_joint_pos = uc.get_joint_positions(self.arm)
        total_angle = 3*np.pi/4.0
        numsteps = 8
        angles = []
        total_time = 2
        times = np.linspace(0, total_time, numsteps)
        for i in range(numsteps):
            new_joint_pos = current_joint_pos[:]
            new_joint_pos[-1] += total_angle/numsteps
            current_joint_pos = new_joint_pos[:]
            angles.append(new_joint_pos)
        uc.command_joint_trajectory(self.arm, angles, times, blocking=True)
        
         
        

if __name__ == "__main__":
    robot = Robot()
    numsteps = 1    
    for i in range(numsteps):
        robot.go_to_start()        
	robot.grasp_cup()
	robot.shift_cup(dz = 0.08)
	robot.pour_cup()

        
    


