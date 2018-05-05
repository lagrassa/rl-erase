#helper function: takes in a img and calculates the current reward
#reward is based on the intended number of beads and the amount mixed
import pdb
import cv2
from PIL import Image
import numpy as np

small_reward = False 
def reward_func(img):
    #calculate number of beads outside of cup
    #calculate mixedness
    mixed_k = 4
    out_k = 20
    num_mixed = get_mixedness(img)
    num_out = get_out(img)
    rew =  mixed_k*num_mixed + out_k+num_out
    return rew

def get_out(img):
    return 0
    
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
    blue  = cv2.inRange(hsv_img, np.array([115,0,0]),np.array([125,255,200]))/255.0
    red  = cv2.inRange(hsv_img, np.array([-5,0,0]),np.array([5,255,200]))/255.0
    for i in range(int(x_num_chunks)):
        for j in range(int(y_num_chunks)):
            start_i = i_chunk_size*i
            start_j = j_chunk_size*j
            red_section = red[start_i:start_i+i_chunk_size, start_j:start_j+j_chunk_size]
            blue_section = blue[start_i:start_i+i_chunk_size, start_j:start_j+j_chunk_size]
            #don't bother computing if it's just white
            sum_mixed += mixedness_region(red_section, blue_section)
    return sum_mixed


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
    ims = ["stirred.png", "not_stirred.png"]
    for im_name in ims:
        #im = Image.open(im_name).resize((10,10))
        im = cv2.imread(im_name)
        im = cv2.resize(im, (200,200))
    
        print(get_mixedness(im))

