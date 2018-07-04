# rl-erase
Reinforcement learning board erasing simulator (in sim/ and ros/ directories)
and code to train a primitive in stirring/

To run the simulation, use sim\_stir.py The code for running the model is in the poorly named vf\_learn.py.  train() trains a new model, the name of which can be specified in the EXP\_NAME variable, and test() runs a policy, which takes in the argument of the model of the policy. The corresponding model must be in build\model() in vf\_learn.py

naf\_models.py and ddpg\_models.py can be used to train NAF/DDPG, but using the keras-rl framework, which can be run from the even more poorly-named rl\_learn.py. 
