#!/usr/bin/env python
from __future__ import division
import rospy
import pdb
import cv2
import numpy as np
from geometry_msgs.msg import Point

from control_tools.ros_controller import ROS_Controller
from plan_tools.execute_plan import follow_path
from pr2_ik import arm_ik
from image_geometry import PinholeCameraModel
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from grip import Gripper
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


CONSISTENT_STATE = True 

class RealPouringWorld:
    def __init__(self):
        print("robot init")
        self.arm = 'r'
        self.pourer_pos = (0.48, -0.03, 0.759)# (0.5, -0.12, 0.7595)
        self.target_pos = (0.8, -0.01, 0.7595)
        self.depth_image = None
        self.bridge = CvBridge()
        #self.uc = UberController()
        print("setting up uc")
        self.uc =  ROS_Controller()
        print("set up uc")

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

    def go_to_start(self):
	#self.uc.start_joint(self.arm)
	start_joint_angles = [-0.8831416801644028, 0.3696527700454193, -1.5865871699836482, -1.5688534061015482, 5.913809583083913, -0.9149799482346365, 39.09787903807846]
        #start_joint_angles = [-0.8367968810618476, 0.34334374895993397, -1.7849460902031975, -1.7724010263067882, -0.21665115067563817, 0.5939860593928067, 8.826634896625851]
        unwound_config = self.uc.adjust_config(start_joint_angles, self.uc.get_arm_positions(self.arm))
        self.uc.command_arm(self.arm, unwound_config, 4, blocking=True)
        print("Commanded joint pose")
        rospy.sleep(5)

    def world_state(self):
        return []

    def go_to_away(self):
	self.uc.start_joint(self.arm)
	self.uc.command_joint_pose(self.arm,away_joint_angles, time=3, blocking=True)
	rospy.sleep(2)

    def change_pos(self, pos):
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        command_pose(self.uc, (pos, gripper_quat), self.arm)

    ''' currently really only actually samples a pose, and uses the original quaternion''' 
    def change_quat(self, quat):
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        print("gipper quat", gripper_quat)
        pos = gripper_pos
        command_pose(self.uc, (pos, quat), self.arm)

    def get_grasp(self, cup_pos, grasp_height=0):
        #grasp it from the right side, at a sort of 90 degree angle flat
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        #good values - ('self.gripper_pos', [0.4885209624088133, -0.1602965161312774, 0.7629772575114466])
        # ('self.gripper_quat', [-0.07005178434662307, -0.0861961500917268, 0.7512909881160719, 0.6505573167637076])

        grasp_depth = -0.01
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
 
    def grasp_cup(self, grasp_height):
        #move to good grasping point 
        grasp_time = 3.0
        frame = 'base_link'
        self.gripper = Gripper()
        rospy.sleep(0.5)
        self.gripper.grip(amount=0.09, times = 40)
     
        if self.pourer_pos is not None:
            pos,quat = self.get_grasp(self.pourer_pos, grasp_height=grasp_height)
            command_pose(self.uc, (pos, quat), self.arm)
        
        #close self.gripper
        self.gripper.grip(amount=0.055)


    def shift_cup(self, dx=0, dy=0, dz = 0, yaw=None):
        shift_time = 2.0
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        if yaw is not None:
            gripper_euler = list(euler_from_quaternion(gripper_quat))
            gripper_euler[2] = yaw
            gripper_quat = list(quaternion_from_euler(gripper_euler[0], gripper_euler[1], gripper_euler[2]))
        new_pos = (gripper_pos[0]+dx, gripper_pos[1]+dy, gripper_pos[2]+dz)
        command_pose(self.uc, (new_pos, gripper_quat), self.arm)
    
    def pour_cup(self, vel = 2.0):
        #get joint state of right turning joint, add to joint state, do a few times
        current_joint_pos = self.uc.get_joint_positions(self.arm)
        print(current_joint_pos, "joint positions")
        numsteps = 5
        angles = []
        total_time = 1.5
        total_angle = vel*total_time#3*np.pi/4.0
        times = np.linspace(0, total_time, numsteps)
        print("times", times)
        for i in range(numsteps):
            new_joint_pos = current_joint_pos[:]
            new_joint_pos[-1] += total_angle/numsteps
            current_joint_pos = new_joint_pos[:]
            angles.append(new_joint_pos)
        
        self.uc.command_joint_trajectory(self.arm, angles, times, blocking=True)
        
    def pour_parameterized(self,distance_behind=None, height_up=None, speed=None, grasp_height=None):         
        self.go_to_start()        
        #self.grasp_cup(grasp_height=grasp_height)
        #self.shift_cup(dz = height_up)
        #self.shift_cup(dx = distance_behind)
        self.pour_cup(vel=speed)

    def grasp_cup_general(self, grasp_height=0.05):
        #detect point to grasp and grasp it
        point_to_grasp = self.find_general_grasp_point(self.pourer_pos, self.target_pos)
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        above_point = list(point_to_grasp)
        above_point += 0.08
        #go to just above point
        command_pose(self.uc, (above_point, gripper_quat), self.arm)
        self.gripper.grip(0.19)
        #and descend onto point        
        command_pose(self.uc, (point_to_grasp, gripper_quat), self.arm)
        self.gripper.grip(0.03)
        #and descend onto point        

    def find_general_grasp_point(self,pourer_pose, target_pose):
        #get depth image and find grasp point
        #TODO create object that does this
        pourer_point =  self.cam.project3dToPixel(pourer_pose)
        target_point =  self.cam.project3dToPixel(target_pose)
        #draw ray from camera_point to target_point until reaching a substantially higher depth than the middle
        depth_image_ros = rospy.wait_for_message("/head_mount_kinect/depth_registered/image", Image, timeout=2)
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_ros, "8UC1")
        cv2.imshow("image", 255*depth_image)
        cv2.waitKey(0)
        #find point that minimizes distance between camera_point and targeT_point
        lower = 0 
        upper = 100
        _, threshold = cv2.threshold(depth_image, lower, upper, cv2.THRESH_BINARY)
        min_point = None
        min_val = np.inf
        for i in range(depth_image.shape[0]):
            for j in range(depth_image.shape[1]):
                if threshold[i,j]  == upper: #todo fix the type of this
                    total_dist = distance((i,j), pourer_point)+distance((i,j), target_point)
                    if total_dist < min_val:
                        print("this ever updates")
                        min_point = (i,j)
                        min_val = total_dist
        grasp_point = self.cam_pixel_to_3D(min_point) 
        return min_point
    def cam_pixel_to_3D(pixel):

        ray = self.cam.projectPixelTo3dRay()


    def pour_cup_general(self, vel= 2, total_angle = 3.14*2/3):
        gripper_pos, gripper_quat = self.uc.return_cartesian_pose(self.arm, 'base_link')
        numsteps = 1
        angles = []
        total_time = total_angle / vel
        new_gripper_quat = gripper_quat[:]
        #TODO update new_gripper_quat
        new_gripper_euler = list(euler_from_quaternion(new_gripper_quat))
        new_gripper_euler[0] += total_angle
        new_gripper_quat = list(quaternion_from_euler(new_gripper_euler[0], new_gripper_euler[1], new_gripper_euler[2]))
        angles =  arm_ik('r', gripper_pos, new_gripper_quat, self.uc.get_torso_position(), current=self.uc.get_arm_positions('r'))
        angles = self.uc.adjust_config(angles, self.uc.get_arm_positions('r'))
        times=[total_time]
        angles = [angles]
        self.uc.command_arm_trajectory('r', angles, times, blocking=True, logging=False)
            
    
    def pour_parameterized_general(self,x_offset=None, y_offset=None, yaw=None, height_up=None, vel=None, grasp_height=None):         
        self.go_to_start()        
        #self.grasp_cup(grasp_height=grasp_height)
        #print("Grasp cup")
        #self.shift_cup(dz = height_up)
        #self.shift_cup(dx = x_offset, dy = y_offset, yaw=yaw)
        self.pour_cup_general(vel=vel)
    
def command_pose(uc, pose, arm, timeout=4, unwind=True):
    angles =  arm_ik('r', pose[0], pose[1], uc.get_torso_position(), current=uc.get_arm_positions('r'))
    if unwind:
        angles = uc.adjust_config(angles, uc.get_arm_positions(arm))
    uc.command_arm('r', angles,timeout,  blocking=True)
    rospy.sleep(8)
    
    
def get_cam():
    model = PinholeCameraModel()
    camera_info = rospy.wait_for_message( "/head_mount_kinect/rgb/camera_info", CameraInfo)
    model.fromCameraInfo(camera_info)
    return model
    
def distance(point1, point2):
    return np.linalg.norm(np.subtract(point1, point2))

def test_moving(robot):
    times = [1,2]
    gripper_pos, gripper_quat = robot.uc.return_cartesian_pose('r', 'base_link')
    print(gripper_pos, "gripper pos")
    pos_trajectory = [[[0.7, -0.3, 1], gripper_quat], [gripper_pos, gripper_quat]]
    robot.uc.lift_torso()
    arm_traj = [] 
    for pose in pos_trajectory:
        angles =  arm_ik('r', pose[0], pose[1], robot.uc.get_torso_position(), current=robot.uc.get_arm_positions('r'))
        assert angles is not None
        arm_traj.append(angles)
    robot.uc.command_arm_trajectory('r', arm_traj, times, True)
def main():
    rospy.init_node("make_pour")
    robot = RealPouringWorld()
    robot.uc.lift_torso()

    numsteps = 1    
    robot.pour_parameterized_general(x_offset = 0.08, y_offset= 0,height_up = 0.14, vel=0.3, grasp_height=-0.05, yaw = np.pi/2.0+0.4)


if __name__ == "__main__":
    main()
    


