#first case: marker and not white, set to 1
#second case, not marker and not white, set to prior
#third case: white and not marker, set to 0
#fourth case: white and marker: set to 0.5
import numpy as np
import cv2
import pdb


def p_marked(prior,marker, white, lr =1):
    not_white = cv2.bitwise_not(white)
    not_marker = cv2.bitwise_not(marker)

    uncertain = prior*cv2.bitwise_and(not_white, not_marker)/255
    def_white = cv2.bitwise_and(white, not_marker)/255
    confused = 0.5*cv2.bitwise_and(marker, white)/255
    
    new =  uncertain + def_white + confused
    return new*lr + prior*(1-lr)

if __name__ == "__main__":
    #run test
    #test case:
    # prior  0.5
    # marker white
    marker = np.array([[0,255],[255,0]]).astype(np.uint8)
    white = np.array([[0,255],[0,255]]).astype(np.uint8)
    prior = np.array([[0.42,0],[0,0]])
    result = p_marked(prior, marker,white)
    correct_result = np.array([[0.42,0.5],[0,1]])
    if np.array_equal(result, correct_result):
        print("lr = 1 test passed!")
    else:
        print "Your array ",result 
        print "Correct array ",correct_result 

    
    


    


