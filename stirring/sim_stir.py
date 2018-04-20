from  stirring_world import world
import pdb
timeStep = 1/60.0
vel_iters, pos_iters = 6,2
for i in range(100000):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    world.stirrer.stir(force=[-62.82,-0.011])
    world.step(timeStep, vel_iters, pos_iters)
    world.world.ClearForces()
     


    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()

