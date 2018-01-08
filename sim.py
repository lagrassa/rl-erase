from world import World
from robot import Robot

world = World()
robot = Robot(world)

world.draw(robot)
robot.move([1,0])
world.draw(robot)

