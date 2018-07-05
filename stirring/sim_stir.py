import pdb
import time
import stirring_world_pybullet
from stirring_world_pybullet import World
timeStep = 1
world = World(visualize=False, beads=True)
from reward import reward_func
direction = 1
for i in range(40):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    #world.stir(0.3)
    step = 0.005
    theta_diff = 0.1
    world.stir([0.4, 0.5, 0.6, 0.01, 500])
    print(reward_func(world.world_state(), world.ratio_beads_in()))
    world.render()
