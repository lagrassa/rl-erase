#!/usr/bin/env python
from __future__ import division
import rospy
import pdb
import cv2
import numpy as np
from geometry_msgs.msg import Point, WrenchStamped

from control_tools.ros_controller import ROS_Controller
from plan_tools.execute_plan import follow_path
from pr2_ik import arm_ik
from image_geometry import PinholeCameraModel
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from grip import Gripper
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


CONSISTENT_STATE = True 

class RealScoopingWorld:
    def __init__(self):
        self.arm = 'r'
        self.target_pos = (0.8, -0.01, 0.7595)
        self.bridge = CvBridge()
        #self.uc = UberController()
        self.uc =  ROS_Controller(verbose=False)

        print(self.uc.get_arm_positions('r'))
        #self.cam = get_cam() 
        """
        for i in range(400): 
	    gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
	    print("Gripper euler", np.round(euler_from_quaternion(gripper_quat),2))
	    print("Gripper quat", gripper_quat)
            rospy.sleep(0.2)
        """
        #self.uc.command_torso(0.2, blocking=True, timeout=3)
        #self.go_to_start()

    def go_to_scooping_pose(self):
	#self.uc.start_joint(self.arm)
	start_joint_angles = [-0.9885988974424886, -0.42866951929131947, -2.0601149722539223, -2.027487019764564, -4.805082087594098, 0.019334745690731014, 21.143749123300427]
        starker_pose = [-0.6979282969888965, -0.42105597621191854, -2.2490129577176483, -1.9482974406377744, -9.747710737910838, 0.011416122232604908, 21.20988268185181]
        starker_pose2 = [-0.48361469112108824, -0.16084199274261596, -1.7257751592727093, -1.207216169065496, -13.611953993295229, -1.298637461966182, -15.485105440945112]
        starker_pose3 =[-0.20140056563790865, -0.07142515902120736, -1.461189695762397, -1.1982403849597174, -14.011619517036653, -1.6587607936305266, -11.975719454631927]

        start_joint_angles = starker_pose3
        unwound_config = self.uc.adjust_config(start_joint_angles, self.uc.get_arm_positions(self.arm))
        self.uc.command_arm(self.arm, unwound_config, 4, blocking=True)


    def change_pos(self, pos):
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        command_pose(self.uc, (pos, gripper_quat), self.arm)


    def shift(self, dx=0, dy=0, dz = 0, yaw=None, forced_gripper_pos=None):
        shift_time = 2.0
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        if forced_gripper_pos is not None:
            ds = [dx, dy, dz]
            for i in range(3):
                if ds[i] == 0:
                    gripper_pos[i] = forced_gripper_pos[i] 
                
 
        if yaw is not None:
            gripper_euler = list(euler_from_quaternion(gripper_quat))
            gripper_euler[2] = yaw
            gripper_quat = list(quaternion_from_euler(gripper_euler[0], gripper_euler[1], gripper_euler[2]))
        new_pos = (gripper_pos[0]+dx, gripper_pos[1]+dy, gripper_pos[2]+dz)
        command_pose(self.uc, (new_pos, gripper_quat), self.arm)
    
    def insert(self, bottom_force=None):
        #insert until feels bottom, moving gripper strictly down in the z position

        ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped)
        reference = ws.wrench.force.z
        self.feel_force(bottom_force, reference)

    def scrape_bottom(self, length_scrape = 0.00, theta_scrape=np.pi/2.0, scrape_force = None):
        #brings angle down along bottom
        num_steps = 1
        
        dx = length_scrape/num_steps
        ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped)
        reference = ws.wrench.force.z
        #self.feel_force(scrape_force, reference)
	#self.feel_force(scrape_force, reference)
	gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
	gripper_euler = list(euler_from_quaternion(gripper_quat))
	gripper_euler[0] = theta_scrape
        gripper_euler[1] = 0.3
	gripper_quat = list(quaternion_from_euler(gripper_euler[0], gripper_euler[1], gripper_euler[2]))
	new_pos = (gripper_pos[0]+dx, gripper_pos[1], gripper_pos[2])
	command_pose(self.uc, (new_pos, gripper_quat), self.arm)

    def set_euler(self, gripper_euler):
	gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
	gripper_quat = list(quaternion_from_euler(gripper_euler[0], gripper_euler[1], gripper_euler[2]))
  
	command_pose(self.uc, (gripper_pos, gripper_quat), self.arm)
    
    def feel_force(self, force, reference):
        #PID controller
        #push down until feels force
        kp = 0.002
	gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        force_to_feel = force-reference
        for i in range(300): 
            ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped)
            error =  force_to_feel-ws.wrench.force.z
            if force_to_feel > ws.wrench.force.z:
                break;
            correction = kp*error  
            self.shift(dz=correction, forced_gripper_pos=gripper_pos)


    def take_out(self):
        self.shift(dz=0.2)

    def scoop(self):
        self.go_to_scooping_pose()
        self.insert(bottom_force=-18)
        self.scrape_bottom(scrape_force=-7)
        self.take_out()
        
    
def command_pose(uc, pose, arm, timeout=4, unwind=True):
    angles =  arm_ik('r', pose[0], pose[1], uc.get_torso_position(), current=uc.get_arm_positions('r'))
    if unwind:
        angles = uc.adjust_config(angles, uc.get_arm_positions(arm))
    uc.command_arm('r', angles,timeout,  blocking=True)
    
    
def distance(point1, point2):
    return np.linalg.norm(np.subtract(point1, point2))

def main():
    rospy.init_node("make_scoop")
    robot = RealScoopingWorld()
    #robot.uc.command_torso(0.3, timeout=4)
    #ws = rospy.wait_for_message("/ft/r_gripper_motor",WrenchStamped)
    #reference = ws.wrench.force.z
    robot.scoop()


if __name__ == "__main__":
    main()
    


