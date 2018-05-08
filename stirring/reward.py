#helper function: takes in a img and calculates the current reward
#reward is based on the intended number of beads and the amount mixed
import pdb
import cv2
from PIL import Image
import numpy as np
CONTOUR_DEBUG = False

small_reward = False 
def reward_func(img):
    #calculate number of beads outside of cup
    #calculate mixedness
    mixed_k = 1
    out_k = 20
    #pdb.set_trace()
    num_mixed = get_mixedness(img)
    num_out = get_out(img)
    rew =  mixed_k*num_mixed + out_k*num_out
    return rew

def get_out(img):
    return 0

def get_num_contours(hsv_filtered):
    _, binary_mask = cv2.threshold(hsv_filtered, 0.9,255, cv2.THRESH_BINARY)
    results= cv2.findContours(binary_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(results) == 3: #dealing with weird versioning stufff
        contours = results[1]
    else:
        contours = results[0]

    valid_contours = [ct for ct in contours if cv2.contourArea(ct) > 30 ]
 
    if CONTOUR_DEBUG:
	for indx in range(len(contours)):
	    print("Is convex?", cv2.isContourConvex(contours[indx]))
	    print("area", cv2.contourArea(contours[indx]))
	    cv2.drawContours(hsv_filtered[:],contours, indx, (0,255,0), 3)
	    cv2.imshow("keypoints", hsv_filtered)
	    cv2.waitKey(0)
  
    return len(valid_contours)


    
#@precondition:img.shape[0] and shape[1] on being multiples of 10
def get_mixedness(img):
    #sum up how faro ratios are from 0.5
    #Image.fromarray(img).show()
    x_num_chunks = 5.0;
    y_num_chunks = 5.0
    sum_mixed = 0
    i_chunk_size = int(img.shape[0]/x_num_chunks)
    j_chunk_size = int(img.shape[1]/y_num_chunks)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    blue  = cv2.inRange(hsv_img, np.array([115,0,0]),np.array([125,256,210]))
    red  = cv2.inRange(hsv_img, np.array([-5,0,0]),np.array([5,256,210]))
    contours_red  = get_num_contours(red)
    contours_blue  = get_num_contours(blue)
    return contours_red+contours_blue


def mixedness_region(red_section, blue_section):
    num_red = 0.0
    num_blue = 0.0
    try:
        num_red = sum(sum(red_section))
        num_blue = sum(sum(blue_section))
    except:
        pdb.set_trace()
    if num_red == 0 and num_blue == 0:
        #this region isn't worth bothering with
        return 0
    ratio = num_red/(num_red+num_blue+0.0)
    #want to penalize distance from 0.5 more
    k = 1.9 #some number between 1 and 2
    return 1-k*abs(0.5-ratio)
            
if __name__ == "__main__":
    ims = ["all_blue.png","all_red.png", "nearly_blue.png", "nearly_red.png", "mix.png"]
    ims = ["not-mixed.png", "stirred.png"]
    for im_name in ims:
        #im = Image.open(im_name).resize((10,10))
        im = cv2.imread(im_name)
        im = cv2.resize(im, (200,200))
    
        print(get_mixedness(im))

