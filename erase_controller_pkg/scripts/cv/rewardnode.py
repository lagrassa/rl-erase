#!/usr/bin/env python
import rospy
import pdb
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from erase_globals import board_height,board_width
from percent_erase import rectify, threshold_img
from masking import p_erased_fast

DEBUG = True

class BoardUpdate: 
    def __init__(self):
        #probability has marker
        self.belief = 0.5*np.ones((board_height, board_width))
        #self.lower_marker = np.array([108,50,160])
        #self.upper_marker = np.array([130,240,200])
        self.lower_marker = np.array([100,90,120])
        self.upper_marker = np.array([160,255,255])

        self.lower_white = np.array([0,0,200])
        self.upper_white = np.array([255,30,255])
        point_topic = "/output"
        self.corners = self.get_corners(point_topic)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/head_mount_kinect/rgb/image_rect_color", Image, self.update_belief)
        self.pub = rospy.Publisher("/rl_erase/reward", Float32,queue_size=10) 
        self.marker_threshold_pub = rospy.Publisher("/rl_erase/marker_threshold", Image,queue_size=10) 
        self.white_threshold_pub = rospy.Publisher("/rl_erase/white_threshold", Image,queue_size=10) 

    def get_corners(self, topic):
        if DEBUG:
            return np.array([[ 336.,   29.],
       [ 537.,   59.],
       [ 506.,  281.],
       [ 299.,  242.]])
 
        upper_left = rospy.wait_for_message(topic, Point, timeout =60)
        upper_right = rospy.wait_for_message(topic, Point, timeout =60)
        lower_right = rospy.wait_for_message(topic, Point, timeout =60)
        lower_left = rospy.wait_for_message(topic, Point, timeout =60)
        input_points = [upper_left, upper_right, lower_right, lower_left]
        pts =  np.array([point_to_array(pt) for pt in input_points])
        print ("Corners: ",pts)
        return pts
    

    def update_belief(self, img):
        image = self.bridge.imgmsg_to_cv2(img)
        rectified_img = rectify(image, self.corners)
        marker_img, marker_mask = threshold_img(rectified_img, self.lower_marker, self.upper_marker)
        white_img, white_mask = threshold_img(rectified_img, self.lower_white, self.upper_white)
        if DEBUG:
            #publish these messages
             marker_msg = self.bridge.cv2_to_imgmsg(marker_img, encoding="passthrough")
             white_msg = self.bridge.cv2_to_imgmsg(white_img, encoding="passthrough")
             self.marker_threshold_pub.publish(marker_msg)
             self.white_threshold_pub.publish(white_msg)

        #for i in range(board_height):
        #    for j in range(board_width):
        #        self.belief[i,j] = p_erased(marker_img[i,j],marker_img[i,j],white_img[i,j])
        self.belief = p_erased_fast(self.belief, marker_mask, white_mask)
        self.update_reward()
        
    def update_reward(self):
        cmd = Float32()
        current_reward_val = self.current_reward()
        cmd.data = current_reward_val
        self.pub.publish(cmd)

    def current_reward(self):
        #Sum over probabilities that board is completely erased 
        #scaled by size of board
        return sum(sum(self.belief))/(board_height*board_width)



def p_erased(prior, m,w,lr=0.98):
    #first case: marker and not white, set to 1
    #second case, not marker and not white, set to prior
    #third case: white and not marker, set to 0
    #fourth case: white and marker: set to 0.5
    if m>0 and w == 0: #fairly certain marker
        pnew= 1    
    if m==0 and w == 0:
        pnew =  prior #no new information
    if w>1 and m == 0: #fairly certain not marker
        pnew =  0
    if w > 0 and m > 0: #????? confusion
        pnew =  0.5
    pmarked =  (1-lr)*prior + lr*pnew
    perased = 1-pmarked
    return perased

def point_to_array(point_msg):
    return [point_msg.x, point_msg.y]
    

def listener():
    rospy.Subscriber("chatter", String, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__== "__main__":
    rospy.init_node('board_update')
    bu = BoardUpdate()
    rospy.spin()
