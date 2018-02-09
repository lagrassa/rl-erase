from world import World
from robot import Robot
from rl_learn import Learner

world = World()
robot = Robot(world)

world.draw(robot)
robot.move([1,0])
world.draw(robot)

