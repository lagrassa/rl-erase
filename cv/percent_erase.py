import cv2
import numpy as np
import pdb
from erase_globals import board_height as h
from erase_globals import board_height as w

#rectify coordinates
#do thresholding on it
#detect portion thresholded
def rectify(im_src):
    #corners of the paper TODO find these through CV
    #pts_src = np.array([[354,410],[19,300],[483,1],[137,-35]])
    pts_src = np.array([[115,94],[474,85],[521,303],[69,311]])
    #  Hard coded destination size ( I think this is ok)
    pts_dst = np.array([[0,0],[w, 0],[w, h],[0, h]])
 
    # Calculate Homography
    H, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    #im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1],im_src.shape[0]))
    im_out = cv2.warpPerspective(im_src, H, (w,h))
    return im_out
     
    # Display images
    #cv2.imshow("Source Image", im_src)
    #cv2.imshow("Warped Source Image", im_out)
    #cv2.waitKey(0)

def threshold_img(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    cv2.imshow("thresholded", res)
    cv2.waitKey(0)
    return res


def reward(im_src):
    rectified_img = rectify(im_src)
    lower_marker = np.array([0,110,0])
    upper_marker = np.array([255,255,255])
    lower_white = np.array([0,0,0])
    upper_white = np.array([255,15,255])
    marker_img = threshold_img(rectified_img, lower_marker, upper_marker)
    #white_img = threshold(rectified_img, lower_white, upper_white)
    #percent = percent_erased(marker_img, white_img)
    return percent

def tune_hsv(img):
    #while True:
    if True:
	lower = np.array([0,110,0])
	upper = np.array([255,255,255])
	#lower[0] = int(float(input("lower: ")))
	#upper[0] = int(float(input("upper: ")))
	print lower
	print upper
	threshold_img(img, lower, upper)

if __name__ == '__main__' :
    im_src = cv2.imread('arm_in_way.png')
    tune_hsv(im_src)
    #reward(im_src)
