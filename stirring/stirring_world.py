from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2World,
                   b2_pi)
import time
from math import sqrt
import numpy as np
import pdb
import random 
import pygame
pygame.init()
screen_size = 800
screen = pygame.display.set_mode((screen_size, screen_size))
screen.fill([255,255,255])
ppm = 500 #pixels_per_meter

#1) make a bowl (circle)
#2) make beads

def vec_to_pygame_pos(vec):
    return tuple(int(ppm*x) for x in vec)

def create_box(origin, l, world):
    #wallShapes = b2EdgeShape(vertices=[(0,0),(0,l),(l,0),(l,l)]),
    top = b2EdgeShape(vertices=[(0,0),(0,l)])
    bottom = b2EdgeShape(vertices=[(0,l),(l,l)])
    left = b2EdgeShape(vertices=[(0,0),(l,0)])
    right = b2EdgeShape(vertices=[(l,0),(l,l)])
    wall = world.CreateStaticBody(
                shapes=[top, bottom, left, right],
                position=origin
            )
    return wall

def adjusted_vertex(v,origin, scale=1):
    return (scale*(v[0]+origin[0]), scale*(v[1]+origin[1]))

class Stirrer(object):
    def __init__(self, world, origin):
        self.l = 0.07;
        self.w = 0.02;
        self.pose = origin
        box = b2PolygonShape(vertices = ((0,0),(0,self.w),(self.l,0),(self.l,self.w)))
        stirrer = world.CreateStaticBody(
                shapes=[box],
                position=origin
            )
    def render(self):
        pygame.draw.rect(screen, (0,0,0), (self.pose[0],self.pose[1],self.w,self.l), 1)
    

class Box(object):
    def __init__(self, world, box_length = 0.5, center = (0.5,0.5)):
        self.color = (50,50,50)
        self.box_length = box_length
        self.center = center
        self.walls = create_box(center,box_length, world)

    def render(self):
        for f in self.walls.fixtures:
            edge = []
            for v in f.shape.vertices:
                pygame_pose = adjusted_vertex(v, self.center, scale = ppm)
                edge.append(pygame_pose)
            pygame.draw.lines(screen, self.color, True, edge,4)


class Beads(object):
    def __init__(self, world, poses=None, radius=None, colors =None ):
        assert(poses is not None and radius is not None and colors is not None)
        self.radius = radius
        self.colors = colors
        self.beads = [world.CreateDynamicBody(
        position=pos,
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=self.radius),
            density=2,
        )
        ) for pos in poses]
    def render(self):
        for i in range(len(self.beads)):
            bead = self.beads[i]
            color = self.colors[i]
            self.render_bead(bead,color)

    def render_bead(self, bead, color):
        pygame.draw.circle(screen, color, vec_to_pygame_pos(bead.position), int(ppm*self.radius))

def random_bead_poses_and_colors(length, origin, numbeads, bead_radius):
    beads = []
    bead_colors = []
    color_choices = [(255,0,0), (0,0,255)]
    adj_l = length-bead_radius;
    for i in range(numbeads):
	pos = (bead_radius + adj_l*random.random()+origin[0], bead_radius + adj_l*random.random()+origin[1])
        beads.append(pos)
        random_color = color_choices[random.randint(0,1)]
        bead_colors.append(random_color)
    return beads, bead_colors
    
def dist(x, y):
    return sqrt(sum((xi - yi)**2 for (xi, yi) in zip(x, y)))

class World(object):
    def __init__(self):
	self.world = b2World()
	box_pos = (0.9,0.9)
	box_length = 0.4
	bead_radius = 0.01
	stirrer_pos = box_pos
	self.bowl = Box(self.world, box_length = box_length, center = box_pos)
	self.stirrer = Stirrer(self.world, stirrer_pos)
	bead_poses, bead_colors = random_bead_poses_and_colors(box_length, box_pos, 300, bead_radius)
	self.beads = Beads(self.world, poses = bead_poses, colors = bead_colors, radius=bead_radius)
    def render(self):
	self.bowl.render()
	self.beads.render()
	self.stirrer.render()
        pygame.display.flip()
        

world = World()
world.render()
