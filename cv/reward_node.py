import rospy
import pdb
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from erase_globals import board_height,board_width
from percent_erase import rectify, threshold_img

DEBUG = True

class BoardUpdate: 
    def __init__(self):
        #probability has marker
        self.belief = 0.5*np.ones((board_height, board_width))
        self.lower_marker = np.array([0,110,0])
        self.upper_marker = np.array([255,255,255])
        self.lower_white = np.array([0,0,0])
        self.upper_white = np.array([255,15,255])
        point_topic = "/output"
        self.corners = self.get_corners(point_topic)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/head_mount_kinect/rgb/image_rect_color", Image, self.update_belief)
        self.pub = rospy.Publisher("/rl_erase/reward", Float32,queue_size=10) 

    def get_corners(self, topic):
        if DEBUG:
            return np.array([[ 244.,   17.],
       [ 443.,   50.],
       [ 410.,  264.],
       [ 204.,  235.]])
 
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
        marker_img = threshold_img(rectified_img, self.lower_marker, self.upper_marker)
        white_img = threshold_img(rectified_img, self.lower_white, self.upper_white)
        for i in range(board_height):
            for j in range(board_width):
                self.belief[i,j] = p_marked(marker_img[i,j],marker_img[i,j],white_img[i,j])
        self.update_reward()
        
    def update_reward(self):
        cmd = Float32()
        current_reward_val = self.current_reward()
        cmd.data = current_reward_val
        self.pub.publish(cmd)

    def current_reward(self):
        #sum probability all the board is erased currently, so max reward - sum of all values on board
        return board_height*board_width - sum(sum(self.belief))

def p_marked(prior, m,w,lr=0.9):
    if m>0 and w == 0: #fairly certain marker
        pnew= 1    
    if m==0 and w == 0:
        pnew =  prior #no new information
    if w>1 and m == 0: #fairly certain not marker
        pnew =  0
    if w > 0 and m > 0: #????? confusion
        pnew =  0.5
    return (1-lr)*prior + lr*pnew

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
