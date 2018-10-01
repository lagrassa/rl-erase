#!/usr/bin/env python

import rospy 
import numpy
from uber_controller import UberController

rospy.init_node("record_and_replay") 

uc = UberController()

def record(total_time = 15):
    step_size = 0.4
    torso = uc.command_torso(0.1, 4, True)
    times = []
    time = total_time
    angles = []
    while time >= 0:
        angle = uc.get_joint_positions('r') 
        angles.append(angle)
        times.append(total_time-time)
        rospy.sleep(step_size)
        time -= step_size
        
    numpy.save("recordings/angles.npy", angles)
    numpy.save("recordings/torso.npy", torso)
    numpy.save("recordings/times.npy", times)

def replay():
    angles = numpy.load("recordings/angles.npy")
    times = numpy.load("recordings/times.npy")
    torso = numpy.load("recordings/torso.npy")
    uc.command_joint_trajectory('r', angles, times, True)
        
        

replay()
