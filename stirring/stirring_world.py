from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2World,
                   b2_pi)
import time
from math import sqrt
import numpy as np
import pickle
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
    top = b2PolygonShape(vertices = ((0,0),(0,l),(0.01,0.01),(0.01,0.01+l)))
    bottom = b2EdgeShape(vertices=[(0,l),(l,l)])
    left = b2EdgeShape(vertices=[(0,0),(l,0)])
    right = b2EdgeShape(vertices=[(l,0),(l,l)])
    right = b2PolygonShape(vertices = ((l,0),(l,l),(l+0.01,0.01),(l+0.01,0.01+l)))
    center = origin
    wall = world.CreateStaticBody(
                shapes=[top, bottom, left, right],
                position=center
            )
    return wall

def adjusted_vertex(v,origin, scale=1):
    return (scale*(v[0]+origin[0]), scale*(v[1]+origin[1]))

class Stirrer(object):
    def __init__(self, world, origin):
        self.l = 0.08;
        self.w = 0.03;
        box = b2PolygonShape(box=(self.w/2.0, self.l/2.0))
        self.origin = origin
        self.world = world
        #center = (origin[0]-(self.w/2),origin[1](self.l/2))
        center = origin
        self.stirrer = world.CreateDynamicBody(
                shapes=[box],
                position=center
            )
    def render(self):
        pos = self.stirrer.position
        v = (ppm*(pos[0]-self.w/2), ppm*(pos[1]-self.l/2))
        pygame.draw.rect(screen, (0,0,0), (v[0],v[1],ppm*self.w,ppm*self.l), 6)
    #pick a aandom direction: go in it
    def policy(self):
        l = 2
        random_thing =  ((-1)**random.randint(0,1)*l*random.random(),(-1)**random.randint(0,1)*l*random.random())
        print(random_thing)
        return random_thing

    def stir(self):
        force = self.policy()
        self.stirrer.ApplyForce(force=force, point=self.stirrer.position,wake = True)
    

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
            density=0.03,
        )
        ) for pos in poses]
    def render(self):
        for i in range(len(self.beads)):
            bead = self.beads[i]
            color = self.colors[i]
            self.render_bead(bead,color)

    def render_bead(self, bead, color):
        pygame.draw.circle(screen, color, vec_to_pygame_pos(bead.position), int(ppm*self.radius))

def random_bead_poses_and_colors(length, origin, numbeads, bead_radius, new= False):
    
    if not new:
        beads  = pickle.load(open("beads.pkl"))
        bead_colors  = pickle.load(open("colors.pkl"))
        print("Loaded beads from file")
    if new:
	beads = []
	bead_colors = []
	color_choices = [(255,0,0), (0,0,255)]
	adj_l = length-bead_radius;
	#color depends on x coord
	for i in range(numbeads):
	    pos = (bead_radius + adj_l*random.random()+origin[0], bead_radius + adj_l*random.random()+origin[1])
	    beads.append(pos)
	    #random_color = color_choices[random.randint(0,1)]
	    if pos[0] < origin[0]+length/2.0:
		color = color_choices[0]
	    else:
		color = color_choices[1]
      
	    bead_colors.append(color)
	    with open('beads.pkl', 'wb') as fp:
		pickle.dump(beads, fp)
	    with open('colors.pkl', 'wb') as fp:
		pickle.dump(bead_colors, fp)

    return beads, bead_colors
    
def dist(x, y):
    return sqrt(sum((xi - yi)**2 for (xi, yi) in zip(x, y)))

class World(object):
    def __init__(self):
	self.world = b2World(gravity=(0,0))
	self.box_pos = (0.9,0.9)
	self.box_length = 0.4
	bead_radius = 0.006
	stirrer_pos =( self.box_pos[0] + 0.2, self.box_pos[1]+0.1)
	self.bowl = Box(self.world, box_length = self.box_length, center = self.box_pos)
	self.stirrer = Stirrer(self.world, stirrer_pos)
	bead_poses, bead_colors = random_bead_poses_and_colors(self.box_length, self.box_pos, 1500, bead_radius, new =True)
	self.beads = Beads(self.world, poses = bead_poses, colors = bead_colors, radius=bead_radius)
    def render(self):
        screen.fill((255,255,255))
	self.bowl.render()
	self.beads.render()
	self.stirrer.render()
        pygame.display.flip()

    def step(self, timeStep, vel_iters, pos_iters):
        self.world.Step(timeStep, vel_iters, pos_iters)
        self.world.ClearForces()
        self.stirrer.stir()

    def reward(self):
        #checks how well mixed board is based on pygame screen
        #the portion of beads in each section should be roughly 50/50
        #penalize based on deviation from this and add
        #section board into 36 pieces
        n = 6
        m = 6
        total_meters = self.box_length
        #for i in range(n):
        #    for j in range(m):
        #    #sample 
        
        
             
 
        

world = World()
timeStep = 1/30.0
vel_iters, pos_iters = 6, 2
for i in range(10000):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    world.step(timeStep, vel_iters, pos_iters)

    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()
