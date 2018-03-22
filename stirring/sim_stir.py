from  stirring_world import world
timeStep = 1/30.0
vel_iters, pos_iters = 6, 2
for i in range(10000):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    world.step(timeStep, vel_iters, pos_iters)

    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()

