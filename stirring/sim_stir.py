import pdb
import time
import stirring_world_pybullet
from stirring_world_pybullet import World
timeStep = 1
vel_iters, pos_iters = 6,2
world = World(visualize=True, beads=True)
from reward import reward_func
direction = 1
for i in range(80):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    #world.stir(0.3)
    step = 0.005
    theta_diff = 0.1
    pdb.set_trace()
    world.stir([0.2, 0.3, 0.3, 0.3])

   
    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()
