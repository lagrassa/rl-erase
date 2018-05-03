import pdb
import time
import stirring_world_pybullet
from stirring_world_pybullet import World
timeStep = 1
vel_iters, pos_iters = 6,2
world = World(visualize=False)
for i in range(100000):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    #world.stir(0.3)
    depth = 0.25
    step = 0.005
    radius = 0.01
    #world.stir_circle(radius, step, depth)
    world.stirrer_close()
    world.stirrer_state()
    world.world_state()
    world.step(step, vel_iters, pos_iters)
    world.stir(1.5)
   
    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()
    time.sleep(0.01)
