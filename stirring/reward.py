#helper function: takes in a img and calculates the current reward
#reward is based on the intended number of beads and the amount mixed
import pdb

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
    #sum up how far ratios are from 0.5
    res = 10.0;
    sum_mixed = 0
    assert(img.shape[0] % res == 0) 
    assert(img.shape[1] % res == 0) 
    i_chunk_size = int(round(img.shape[0]/res))
    j_chunk_size = int(round(img.shape[1]/res))
    for i in range(int(res)):
        for j in range(int(res)):
            start_i = i_chunk_size*i
            start_j = j_chunk_size*j
            section = img[start_i:start_i+i_chunk_size, start_j:start_j+j_chunk_size,:]
            #don't bother computing if it's just white
            if section.min() == 255:
                continue;
             
            sum_mixed += mixedness_region(section)
    return sum_mixed


def mixedness_region(img):
    num_red = 0.0
    num_blue = 0.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] != 255:
                num_red +=1
            elif img[i,j,1] != 255:
                num_blue +=1
    ratio = num_red/(num_red+num_blue+0.0)
    return 1-abs(0.5-ratio)
            
            

