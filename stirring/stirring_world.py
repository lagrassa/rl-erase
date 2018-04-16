from __future__ import division
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape, b2World,
                   b2_pi)
import time
from math import sqrt
import numpy as np
import pickle
import pdb
import random 
import pygame
pygame.display.init()
screen = pygame.display.set_mode((280, 280))
screen.fill([255,255,255])
ppm = 50 #pixels_per_meter
eps = 0.0001

#1) make a bowl (circle)
#2) make beads

def vec_to_pygame_pos(vec):
    return tuple(int(round(ppm*x)) for x in vec)

def create_box(origin, l,w, world):
    #wallShapes = b2EdgeShape(vertices=[(0,0),(0,l),(l,0),(l,l)]),
    left = b2PolygonShape(vertices = ((0,0),(0,l),(eps,l),(eps,0)))
    bottom = b2PolygonShape(vertices=[(0,l),(w,l), (w,l+eps),(0,l+eps)])
    right = b2PolygonShape(vertices = ((w-eps,0),(w-eps,l),(w,l),(w,0)))
    center = origin
    left_corner = b2CircleShape(pos=(0, l), radius=0.1)
    right_corner = b2CircleShape(pos=(w, l), radius=0.1)

    walls = [ left, bottom,right, left_corner, right_corner]
    fixture_list = [b2FixtureDef(shape=shape, density = 10, friction = 0.9) for shape in walls]
    STATIC = False 
    if STATIC:
        wall = world.CreateStaticBody(
                    fixtures=fixture_list,
                    position=center,
                )
    else:
        wall = world.CreateDynamicBody(
                    fixtures=fixture_list,
                    position=center,
                )
    return wall
class Floor(object):

    def __init__(self, origin, l, world):
        #origin = (0.2,0.2)
        floor_length = l;
        bottom = b2EdgeShape(vertices=[(0,floor_length),(floor_length,floor_length)])

        bottom = b2PolygonShape(vertices = ((0,0),(l,0),(l+eps,eps),(eps,eps)))
        fixture_list = b2FixtureDef(shape=bottom, density = 8, friction = 0.9)
        self.floor = world.CreateStaticBody(
                    position=origin)
        self.floor.CreateFixture(fixture_list) 
        self.color = (90,90,90)

    def destroy(self):
        pass
    def render(self):
        for f in self.floor.fixtures:
            edge = []
            for v in f.shape.vertices:
                pygame_pose = adjusted_vertex(v, self.floor.position, scale = ppm)
                edge.append(pygame_pose)
            pygame.draw.lines(screen, self.color, True, edge,4)
        
        

def adjusted_vertex(v,origin, scale=1):
    return (int(round(scale*(v[0]+origin[0]))), int(round(scale*(v[1]+origin[1]))))

class Stirrer(object):
    def __init__(self, world, origin):
        self.l = 2.5;
        self.w = 0.15;
        box = b2PolygonShape(box=(self.w/2.0, self.l/2.0))
        self.origin = origin
        self.world = world
        #center = (origin[0]-(self.w/2),origin[1](self.l/2))
        center = origin
        self.stirrer = world.CreateDynamicBody(
                shapes=[box],
                position=center
            )
    def state(self):
        st = self.stirrer
        state =  np.array([st.angle, st.angularDamping, st.angularVelocity, st.linearDamping, st.linearVelocity[0], st.linearVelocity[1],st.inertia, st.position[0], st.position[1]])
        return state

    def destroy(self):
        pass
    def render(self):
        pos = self.stirrer.position
        v = (ppm*(pos[0]-self.w/2.0), ppm*(pos[1]-self.l/2.0))
        pygame.draw.rect(screen, (0,255,0), (v[0],v[1],ppm*self.w,ppm*self.l), 0)
    #pick a aandom direction: go in it
    def policy(self):
        l = 2
        random_thing =  ((-1)**random.randint(0,1)*l*random.random(),(-1)**random.randint(0,1)*l*random.random())
        return [3*i for i in random_thing]

    def stir(self, force=(0,0)):
        force = np.array((float(force[0]), float(force[1])))
        input_max = 150;
        reasonable_force = input_max*sigmoid(force)
        self.stirrer.ApplyForce(force=reasonable_force, point=self.stirrer.position,wake = True)
    
def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

class Box(object):
    def __init__(self, world, box_length = 0.5,box_width=0.25, center = (0.5,0.5)):
        self.color = (50,50,50)
        self.box_length = box_length
        self.box_width = box_width
        self.center = center
        self.walls = create_box(center,box_length, box_width, world)

    def destroy(self):
        pass
    def render(self):
        for f in self.walls.fixtures:
            if isinstance(f.shape, b2CircleShape):
                pygame_pose = adjusted_vertex(f.shape.pos, self.walls.position, scale = ppm)
                pygame.draw.circle(screen, self.color, pygame_pose, int(f.shape.radius*ppm))
            else:
                edge = []
                for v in f.shape.vertices:
                    pygame_pose = adjusted_vertex(v, self.walls.position, scale = ppm)
                    edge.append(pygame_pose)
                pygame.draw.lines(screen, self.color, True, edge,8)


class Beads(object):
    def __init__(self, world, poses=None, radius=None, colors =None ):
        assert(poses is not None and radius is not None and colors is not None)
        self.radius = radius
        self.colors = colors
        tolerance = 0.001
        self.world = world
        self.beads = [world.CreateDynamicBody(
        position=pos,
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=self.radius+tolerance),
            density=0.3,
        )
        ) for pos in poses]
    def render(self):
        for i in range(len(self.beads)):
            bead = self.beads[i]
            color = self.colors[i]
            self.render_bead(bead,color)

    def destroy(self):
        for bead in self.beads:
            self.world.DestroyBody(bead) 

    def render_bead(self, bead, color):
        if not np.isnan(bead.position).all():
            pygame.draw.circle(screen, color, vec_to_pygame_pos(bead.position), int(round(ppm*self.radius)))
        else:
            print("skipped drawing a bead because it's position was unknown")

def random_bead_poses_and_colors(length,width, origin, numbeads, bead_radius, new= True):
    
    if not new:
        beads  = pickle.load(open("beads.pkl"))
        bead_colors  = pickle.load(open("colors.pkl"))
        print("Loaded beads from file")
    if new:
        beads = []
        bead_colors = []
        color_choices = [(255,0,0), (0,0,255)]
        adj_l = length-bead_radius-eps;
        adj_w = width-bead_radius-eps;
        #color depends on x coord
        for i in range(numbeads):
            pos = (bead_radius + adj_w*random.random()+origin[0], bead_radius + adj_l*random.random()+origin[1])
            beads.append(pos)
            #random_color = color_choices[random.randint(0,1)]
            if pos[1] < origin[1]+length/2.0:
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
        self.world = b2World(gravity=(0,9.8))
        self.box_pos = (2,0.4)
        self.box_length = 4
        self.box_width = 2
        floor_pos = (self.box_pos[0]-0.1,self.box_pos[1]+(self.box_length/1.0)+eps+0.02)
        bead_radius = 0.1
        stirrer_pos =( self.box_pos[0] + self.box_width/2.0, self.box_pos[1]+self.box_length/2.0)
        self.bowl = Box(self.world, box_width=self.box_width, box_length = self.box_length, center = self.box_pos)
        self.stirrer = Stirrer(self.world, stirrer_pos)
        self.floor = Floor(floor_pos, 60, self.world)
        numbeads = 120
        bead_poses, bead_colors = random_bead_poses_and_colors(self.box_length, self.box_width, self.box_pos, numbeads, bead_radius, new =True)
        self.beads = Beads(self.world, poses = bead_poses, colors = bead_colors, radius=bead_radius)
        self.objects = [self.beads,self.floor, self.bowl, self.stirrer, ]
            
    def stirrer_close(self):
        far = 6
        if abs(self.stirrer.stirrer.position[0]) > far or abs(self.stirrer.stirrer.position[1]) > far:
            print("Stirrer is far")
            return False
        return True
    def reset(self):
        print("RESET")
        for obj in self.objects:
            obj.destroy()
        self.__init__()
    def destroy(self):
        pass

    def render(self):
        screen.fill((255,255,255))
        [obj.render() for obj in self.objects]
        pygame.display.flip()

    def step(self, timeStep, vel_iters, pos_iters):
        self.world.Step(timeStep, vel_iters, pos_iters)
        self.world.ClearForces()
        self.stirrer.stir()

    def world_state(self):
        num_colors = 3
        state = np.zeros((screen.get_width(), screen.get_height(), num_colors))
        red = pygame.surfarray.pixels_red(screen)
        blue = pygame.surfarray.pixels_blue(screen)
        green = pygame.surfarray.pixels_green(screen)
        state[:,:,0] = red
        state[:,:,1] =blue 
        state[:,:,2] =green 
    
        return state

    def stirrer_state(self):
        return self.stirrer.state()
        
world = World()
"""
timeStep = 1/30.0
vel_iters, pos_iters = 6, 2
for i in range(10000):
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    world.step(timeStep, vel_iters, pos_iters)

    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.render()
"""
