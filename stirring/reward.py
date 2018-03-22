#helper function: takes in a img and calculates the current reward
#reward is based on the intended number of beads and the amount mixed

def reward_func(img):
    #calculate number of beads outside of cup
    #calculate mixedness
    mixed_k = 4
    out_k = 20
    num_mixed = get_mixedness(img)
    num_out = get_out(img)
    return mixed_k*num_mixed + out_k+num_out

def get_out(img):
    return 0
    
def get_mixedness(img):
    #sum up how far ratios are from 0.5
    res = 10.0;
    sum_mixed = 0
    for i in range(int(res)):
        i_chunk_size = int(round(img.shape[0]/res))
        for j in range(int(res)):
            j_chunk_size = int(round(img.shape[1]/res))
            mixedness_region(img[i:i+i_chunk_size, j:j+j_chunk_size])
    return sum_mixed

def mixedness_region(img):
    num_red = 0.0
    num_blue = 0.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pass
    ratio = num_red/(num_red+num_blue+1.0)
    return abs(0.5-ratio)
            
            

