from __future__ import division
from math import log, sqrt
#helper function: takes in a img and calculates the current reward
#reward is based on the intended number of beads and the amount mixed
import pdb
import cv2
from PIL import Image
import numpy as np
CONTOUR_DEBUG = False

small_reward = False 
def stir_reward(imgs, num_in):
    #calculate number of beads outside of cup
    #calculate mixedness
    rew = 0
    #k should try to keep the mixness proportions approx equal to the ratios, so like 10
    k = 50
    if imgs is not None:
        assert(len(imgs) == 2)
        for img in imgs:
            rew += get_mixedness(img)

    total_rew =  rew+k*num_in
    #print(total_rew)
    #print("num in", num_in)
    return total_rew

def entropy(imgs):
    assert(len(imgs) == 2)
    rew = 0
    for img in imgs:
        rew += get_mixedness(img)
    return rew
    

def get_num_contours(hsv_filtered):
    _, binary_mask = cv2.threshold(hsv_filtered, 0.9,255, cv2.THRESH_BINARY)
    results= cv2.findContours(binary_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(results) == 3: #dealing with weird versioning stufff
        contours = results[1]
    else:
        contours = results[0]

    valid_contours = [ct for ct in contours if cv2.contourArea(ct) > 6]
    #max contour
 
    if CONTOUR_DEBUG:
        for indx in range(len(contours)):
            print("Is convex?", cv2.isContourConvex(contours[indx]))
            print("area", cv2.contourArea(contours[indx]))
            cv2.drawContours(hsv_filtered[:],contours, indx, (0,255,0), 3)
            cv2.imshow("keypoints", hsv_filtered)
            cv2.waitKey(0)
  
    return len(valid_contours), cv2.contourArea(max(valid_contours, key = lambda x: cv2.contourArea(x)))


    
#@precondition:img.shape[0] and shape[1] on being multiples of 10
def get_mixedness(img):
    #Image.fromarray(img).show()
    #filter out the green stirrer! 
    img[:,:,1] = np.zeros(img[:,:,1].shape)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    blue  = cv2.inRange(hsv_img, np.array([115,200,0]),np.array([125,256,210]))/255.0 #hack, this is okay because there are only two colors
    red  = cv2.inRange(hsv_img, np.array([-5,200,0]),np.array([5,256,210]))/255.0
    return hierarchical_entropy(red, blue)

def hierarchical_entropy(red, blue):
    #partition into 2, 4, 8 regions and compute entropy of subregions
    #@requires 32x32 image or 64x64 image to work well, some multiple of 8
    #I continue to think this is reasonable.
    total_entropy = 0
    for n in [2, 4, 8]:
        total_entropy += n_entropy(red, blue, n)
    return total_entropy

def showImage(red_section, blue_section):
    img = np.zeros(red_section.shape+(3,), dtype = np.uint8)
    img[:,:,0] = red_section*255
    img[:,:,2] = blue_section*255
    Image.fromarray(img).show()
        

def n_entropy(red, blue, n):
    #split into n on the x and y axis
    i_chunk_size = int(red.shape[0]/n)
    j_chunk_size = int(red.shape[1]/n)
    sum_entropy = 0
    for i in range(n):
        for j in range(n):
            red_section = red[i*i_chunk_size:i*i_chunk_size+i_chunk_size, j*j_chunk_size:j*j_chunk_size+j_chunk_size]
            blue_section = blue[i*i_chunk_size:i*i_chunk_size+i_chunk_size, j*j_chunk_size:j*j_chunk_size+j_chunk_size]
            entropy = entropy_region(red_section, blue_section)
            sum_entropy += entropy
    return sum_entropy

def entropy_region(red_section, blue_section):
    num_red = 0.0
    num_blue = 0.0
    num_red = sum(sum(red_section))
    num_blue = sum(sum(blue_section))
    if num_red ==  0 or num_blue == 0:
        return 0
    p_red = num_red/(num_red+num_blue)
    p_blue = num_blue/(num_red+num_blue)
    #avoid math domain errors this way
    entropy_individual = 0
    if p_red > 0:
        entropy_individual += p_red*log(p_red)
    if p_blue > 0:
        entropy_individual += p_blue*log(p_blue)

    entropy = -(entropy_individual) 
    return entropy
            
if __name__ == "__main__":
    ims = ["all_blue.png","all_red.png", "nearly_blue.png", "nearly_red.png", "mix.png"]
    ims = ["pink.png"]
    for im_name in ims:
        #im = Image.open(im_name).resize((10,10))
        im = cv2.imread(im_name)
        #im = cv2.resize(im, (20,20))
    
        print(get_mixedness(im))

